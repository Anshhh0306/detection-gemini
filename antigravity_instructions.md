# Hackathon Next Steps: Gen-AI Audio Agent

## 🎯 Context
This project is a FastAPI-based **Gen-AI Corrective Audio Agent**. It takes audio input (mp3), runs it through a local ML model for "AI vs Human" voice detection, and then uses the `gemini-2.5-flash` model via the `google-genai` SDK and Pydantic structured outputs to extract:
- `corrected_transcript`
- `detected_language`
- `intent`
- `actionable_summary`

The frontend (`templates/index.html`) is updated to show both the ML confidence bar and the Gemini agent fields.

## 🚀 Priority Tasks (Execute sequentially)

### Step 1: Database & History Setup
- **Goal:** Allow users to see a history of their past analyses.
- **Action:** Integrate SQLAlchemy (with SQLite for simplicity). Create a `VoiceAnalysis` model mapping the inputs and Gemini outputs. Add an endpoint `GET /api/history` and update the frontend to show a "Recent Analyses" sidebar or section.

### Step 2: Cloud Deployment Preparation
- **Goal:** Get this ready to host live so judges can test it.
- **Action:** Standardize the `Procfile` and create a `Dockerfile` optimized for Python/librosa (needs system audio libs like `libsndfile1` or `ffmpeg`).

### Step 3: Frontend Polish & Error Handling
- **Goal:** Bulletproof the UX for the live demo.
- **Action:** Add better audio waveform visualization before uploading. Ensure mobile responsiveness. Add a "copy to clipboard" button next to the Gemini action summary.

### Step 4: Add Unit Tests
- **Goal:** Prove backend stability.
- **Action:** Create `test_main.py` using `pytest` and `httpx`. Mock the Gemini API call and the local ML `predict_voice` function to ensure endpoints behave correctly.

## 🛠 Instructions for Antigravity (Agent)
1. Read `main.py` to understand the `FullVoiceResponse` schema and route logic.
2. Read `templates/index.html` to see the current UI structure.
3. Start with Step 1 and ask before committing major database changes.