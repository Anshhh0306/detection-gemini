"""
AI Voice Detection API
Competition-compliant REST API for detecting AI-generated vs Human voice
"""

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Literal
from auth import verify_api_key
from model import predict_voice, is_model_trained, get_model_info
import tempfile
import base64
import os
import time
from pathlib import Path
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from google import genai
from google.genai import types
import sys

client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))

class GeminiAgentSchema(BaseModel):
    corrected_transcript: str
    detected_language: str
    intent: str
    actionable_summary: str

class FullVoiceResponse(BaseModel):
    classification: str
    confidenceScore: float
    corrected_transcript: str
    detected_language: str
    intent: str
    actionable_summary: str

def process_voice_with_gemini(audio_bytes: bytes) -> GeminiAgentSchema:
    # Fallback chain: try 2.5-flash first, then 1.5-flash
    models_to_try = ['gemini-2.5-flash', 'gemini-1.5-flash']
    last_error = None

    # Build multimodal content: actual audio + text prompt
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg")
    prompt = (
        "Listen to this audio carefully. "
        "Transcribe what is being said, detect the actual language spoken, "
        "identify the speaker's intent, and provide an actionable summary. "
        "Return the result as JSON matching the schema."
    )

    for model_name in models_to_try:
        for attempt in range(3):  # Up to 3 retries per model
            try:
                print(f"\n[Gemini] Trying model={model_name}, attempt={attempt + 1}", flush=True)
                response = client.models.generate_content(
                    model=model_name,
                    contents=[audio_part, prompt],
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': GeminiAgentSchema,
                    },
                )

                print("\n" + "="*40, flush=True)
                print("🤖 GEMINI AGENT RESPONSE:", flush=True)
                print(response.text, flush=True)
                print("="*40 + "\n", flush=True)

                return response.parsed

            except Exception as e:
                last_error = e
                err_str = str(e)
                if '503' in err_str or 'UNAVAILABLE' in err_str or 'high demand' in err_str:
                    wait = 2 * (attempt + 1)
                    print(f"[Gemini] 503 on {model_name} attempt {attempt + 1} — waiting {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    # Non-503 error: skip retries for this model
                    print(f"[Gemini] Non-503 error on {model_name}: {err_str}", flush=True)
                    break

    raise Exception(f"All Gemini models failed. Last error: {last_error}")

app = FastAPI(title="AI Voice Detection API")

# Custom exception handlers for competition-compliant error format
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": "Invalid API key or malformed request"}
    )

# Setup templates
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Supported languages
SUPPORTED_LANGUAGES = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}


class VoiceDetectionRequest(BaseModel):
    """Request body for voice detection endpoint."""
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"] = Field(
        ..., description="Language of the audio"
    )
    audioFormat: Literal["mp3"] = Field(
        ..., description="Audio format (must be mp3)"
    )
    audioBase64: str = Field(
        ..., description="Base64-encoded MP3 audio"
    )


class VoiceDetectionResponse(BaseModel):
    """Success response for voice detection."""
    status: Literal["success"] = "success"
    language: str
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str


class ErrorResponse(BaseModel):
    """Error response."""
    status: Literal["error"] = "error"
    message: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main upload page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/model-status")
async def model_status():
    """Check if the model is trained and ready."""
    return get_model_info()


@app.post("/api/voice-detection", response_model=FullVoiceResponse)
async def detect_voice(
    request: VoiceDetectionRequest,
    auth=Depends(verify_api_key)
):
    """
    Detect if an audio file contains AI-generated or human voice.
    
    Accepts: Base64-encoded MP3 audio
    Returns: Classification (AI_GENERATED/HUMAN) with confidence score and explanation
    """
    # Validate language
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {request.language}. Supported: {', '.join(SUPPORTED_LANGUAGES)}"
        )
    
    # Validate audio format
    if request.audioFormat != "mp3":
        raise HTTPException(
            status_code=400,
            detail="Only MP3 format is supported"
        )
    
    # Decode Base64 audio
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Base64 encoding: {str(e)}"
        )
    
    # Validate that we got actual data
    if len(audio_bytes) < 100:
        raise HTTPException(
            status_code=400,
            detail="Audio data too small or empty"
        )
    
    # Save to temp file and process
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Get prediction
        result = predict_voice(tmp_path, request.language)
        
        if result.get("error"):
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Unknown error")
            )
        
        # Send actual audio to Gemini for real transcription & language detection
        gemini_result = process_voice_with_gemini(audio_bytes)

        # ── Terminal Logging: Final combined response ──
        print("\n" + "─"*50, flush=True)
        print("📡 FINAL RESPONSE TO FRONTEND:", flush=True)
        print(f"  classification     : {result['classification']}", flush=True)
        print(f"  confidenceScore    : {result['confidenceScore']}", flush=True)
        print(f"  corrected_transcript: {gemini_result.corrected_transcript}", flush=True)
        print(f"  detected_language  : {gemini_result.detected_language}", flush=True)
        print(f"  intent             : {gemini_result.intent}", flush=True)
        print(f"  actionable_summary : {gemini_result.actionable_summary}", flush=True)
        print("─"*50 + "\n", flush=True)

        return FullVoiceResponse(
            classification=result["classification"],
            confidenceScore=result["confidenceScore"],
            corrected_transcript=gemini_result.corrected_transcript,
            detected_language=gemini_result.detected_language,
            intent=gemini_result.intent,
            actionable_summary=gemini_result.actionable_summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
