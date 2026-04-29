# MediVoice AI

MediVoice AI is a Streamlit-based patient intake assistant that captures symptoms through voice or text, asks empathetic follow-up questions with Google Gemini 2.5 Flash on Vertex AI, and generates a structured clinical intake summary for export.

## Features

- Medical-themed Streamlit interface with blue and white styling
- Voice intake using `streamlit-audiorecorder`
- Typed fallback input for accessibility and resilience
- Guided 3 to 5 question intake flow powered by Vertex AI Gemini
- Structured clinical summary with severity assessment and red-flag review
- Downloadable plain-text summary export
- Session state preservation for ongoing intake conversations
- gTTS audio playback for assistant responses

## Project Structure

```text
medivoice-ai/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
└── utils/
    ├── __init__.py
    ├── config.py
    ├── gemini_client.py
    └── audio_utils.py
```

## Prerequisites

- Python 3.10 or newer
- A Google Cloud project with Vertex AI enabled
- Application default credentials configured locally, or another supported Google auth mechanism

## Configuration

Copy `.env.example` to `.env` and set the values if needed:

```bash
VERTEX_PROJECT_ID=project-d33cb38a-e8d2-46c0-972
VERTEX_REGION=us-central1
VERTEX_MODEL=gemini-2.5-flash-002
VERTEX_MODEL_OPTIONS=gemini-2.5-flash-002,gemini-2.5-pro,gemini-2.0-flash-001
```

You can also copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` when deploying with Streamlit-compatible secret management.

To switch models, you now have three easy options:

- Update `VERTEX_MODEL` in `.env` or Streamlit secrets
- Set `VERTEX_MODEL_OPTIONS` to control the selectable models shown in the UI
- Use the sidebar model selector at runtime without changing code

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## How It Works

1. The patient records symptoms or types them into the app.
2. Speech recognition attempts to transcribe recorded audio. If it fails, the patient can continue with text input.
3. Gemini reviews the intake conversation and asks one follow-up question at a time.
4. After sufficient detail is collected, Gemini returns a structured intake summary.
5. The summary is displayed in-app and can be exported as a `.txt` file.

## Error Handling

- Voice transcription gracefully falls back to text input
- Clear UI messages are shown when recognition or model calls fail
- Session state is retained across Streamlit reruns
- A manual summary button is available if the patient wants to finish early

## Important Note

MediVoice AI is for intake assistance only. It does not diagnose conditions, prescribe treatment, or replace licensed medical care. If a patient reports severe or rapidly worsening symptoms, they should seek urgent medical attention immediately.
