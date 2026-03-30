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
from pathlib import Path
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from google import genai

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

def process_voice_with_gemini(raw_transcript: str) -> GeminiAgentSchema:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Please fix transcription errors, detect the language, and extract user intent from this transcript: {raw_transcript}",
        config={
            'response_mime_type': 'application/json',
            'response_schema': GeminiAgentSchema,
        },
    )
    
    # Add this right before you return the data to the frontend
    print("\n" + "="*40)
    print("🤖 GEMINI AGENT RESPONSE:")
    print(response.text) # This will print the raw JSON 
    print("="*40 + "\n")

    return response.parsed

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
        
        raw_transcript_text = result.get("explanation", "")
        gemini_result = process_voice_with_gemini(raw_transcript_text)
        
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
