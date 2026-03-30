 # OmniVoice: Gen-AI Corrective Audio Agent
 **Google Gen AI Academy APAC - Hackathon Submission**
 
 ## The Problem
 Traditional open-source Voice ML models max out at ~70% accuracy and lack contextual understanding, especially in multilingual environments. They can detect human vs. AI audio, but cannot reason about the content.
 
 ## The Solution
 A **Hybrid Edge-to-Cloud Architecture**. 
 1. We use a lightweight local ML model for audio feature extraction (breathing, phase, smoothness) to determine if the audio is human or AI-generated.
 2. We route the raw audio heuristics into **Google Gemini 2.5 Flash** (via the new `google-genai` SDK).
 3. Gemini acts as an orchestration agent to logically interpret the features, correct multilingual transcripts (e.g., Hindi, Malayalam), detect the language, and extract actionable user intent in real-time.
 
 ## Tech Stack
 * **Frontend:** HTML/CSS/JS
 * **Backend:** Python (FastAPI)
 * **Gen-AI:** Google Gemini 2.5 Flash (Pydantic Structured Output)
