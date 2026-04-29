"""Main Streamlit application for MediVoice AI."""

from __future__ import annotations

import html as _html
import os
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

try:
    from st_audiorec import st_audiorec
except ImportError:  # pragma: no cover - dependency issue surfaced in UI
    st_audiorec = None

from utils import (
    AppConfig,
    AudioProcessingError,
    ClinicalSummary,
    ConversationTurn,
    GeminiClient,
    GeminiConfig,
    build_audio_download_html,
    generate_tts_audio,
    transcribe_audio_bytes,
)


load_dotenv()

APP_TITLE = "MediVoice AI - Patient Intake Assistant"
APP_SUBTITLE = "Powered by Google Gemini 2.5 Flash"
WELCOME_MESSAGE = (
    "Hello, I’m here to help gather information before your visit. "
    "Please describe what’s bothering you today, either by speaking or typing."
)
MIN_FOLLOW_UPS_FOR_SUMMARY = 3


def _safe_secrets() -> Optional[Mapping[str, object]]:
    """Return st.secrets if a secrets file exists, otherwise None."""
    try:
        # `__contains__` triggers the lazy file parse; raises if no secrets.toml
        _ = "VERTEX_PROJECT_ID" in st.secrets
        return st.secrets  # type: ignore[return-value]
    except Exception:
        return None


def hydrate_environment_from_secrets() -> None:
    """Load config values from Streamlit secrets if they are present."""
    secrets = _safe_secrets()
    if secrets is None:
        return
    for key in ("VERTEX_PROJECT_ID", "VERTEX_REGION", "VERTEX_MODEL", "VERTEX_MODEL_OPTIONS"):
        if key not in os.environ and key in secrets:
            os.environ[key] = str(secrets[key])


def initialize_session_state() -> None:
    """Initialize application state used across reruns."""
    defaults: Dict[str, object] = {
        "session_id": str(uuid4()),
        "conversation": [
            ConversationTurn(role="assistant", content=WELCOME_MESSAGE),
        ],
        "clinical_summary": None,
        "last_transcript": "",
        "last_audio_hash": None,
        "last_error": "",
        "questions_asked_so_far": 0,
        "audio_response": None,
        "last_input_source": "text",
        "session_started_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": None,
        "app_config": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_config() -> AppConfig:
    """Return the AppConfig cached in session state; reload only when the model selection changes."""
    if st.session_state.get("app_config") is None:
        st.session_state.app_config = AppConfig.load(
            secrets=_safe_secrets(),
            model_override=st.session_state.get("selected_model"),
        )
    return st.session_state.app_config  # type: ignore[return-value]


@st.cache_resource(show_spinner=False)
def get_gemini_client(project_id: str, region: str, model_name: str) -> GeminiClient:
    """Create or return a cached Gemini client."""
    return GeminiClient(
        config=GeminiConfig(
            project_id=project_id,
            region=region,
            model_name=model_name,
        )
    )


@st.cache_data(show_spinner=False)
def build_summary_export(summary_text: str) -> bytes:
    """Prepare the summary for download as UTF-8 text."""
    return summary_text.encode("utf-8")


def render_sidebar() -> None:
    """Render instructions and environment status."""
    with st.sidebar:
        app_config = _get_config()
        st.subheader("Instructions")
        st.markdown(
            """
            1. Record your symptoms or type them in the text box.
            2. Answer a few follow-up questions one at a time.
            3. Review the generated intake summary before sharing it with a clinician.
            """
        )
        st.info(
            "This tool supports intake documentation only and is not a substitute for emergency care."
        )
        chosen_model = st.selectbox(
            "Vertex AI model",
            options=app_config.model_options,
            index=app_config.model_options.index(app_config.vertex.model_name),
            help="Change the model here without editing code.",
        )
        if chosen_model != st.session_state.selected_model:
            st.session_state.selected_model = chosen_model
            st.session_state.app_config = None
            get_gemini_client.clear()
        st.caption(f"Project: `{app_config.vertex.project_id}`")
        st.caption(f"Region: `{app_config.vertex.region}`")
        st.caption(f"Model: `{st.session_state.selected_model or app_config.vertex.model_name}`")
        st.caption(f"Session ID: `{st.session_state.session_id}`")
        st.caption(f"Started: `{st.session_state.session_started_at}`")


def render_conversation() -> None:
    """Render the chat history."""
    st.subheader("Conversation")
    for turn in st.session_state.conversation:
        avatar = "🩺" if turn.role == "assistant" else "🧑"
        with st.chat_message(turn.role, avatar=avatar):
            st.markdown(turn.content)


def append_turn(role: str, content: str) -> None:
    """Append a new conversation turn if content is not empty."""
    cleaned = content.strip()
    if cleaned:
        st.session_state.conversation.append(ConversationTurn(role=role, content=cleaned))


def handle_patient_message(message: str) -> None:
    """Send a patient message to Gemini and update UI state."""
    cleaned = message.strip()
    if not cleaned:
        st.warning("Please provide a symptom description before continuing.")
        return

    append_turn("user", cleaned)
    st.session_state.last_error = ""
    app_config = _get_config()

    try:
        with st.spinner("Analyzing your intake response..."):
            result = get_gemini_client(
                app_config.vertex.project_id,
                app_config.vertex.region,
                app_config.vertex.model_name,
            ).continue_intake(
                conversation=st.session_state.conversation,
                session_id=st.session_state.session_id,
            )
    except Exception as exc:
        st.session_state.last_error = str(exc)
        st.error(
            "The assistant could not process that response right now. "
            "Please try again in a moment."
        )
        return

    append_turn("assistant", result.assistant_message)
    st.session_state.questions_asked_so_far = result.questions_asked_so_far

    try:
        st.session_state.audio_response = generate_tts_audio(result.assistant_message)
    except AudioProcessingError:
        st.session_state.audio_response = None

    if result.should_generate_summary and result.summary is not None:
        st.session_state.clinical_summary = result.summary


def transcribe_uploaded_audio(audio_bytes: bytes) -> Optional[str]:
    """Transcribe browser-recorded audio while avoiding duplicate processing."""
    audio_signature = hash(audio_bytes)
    if st.session_state.last_audio_hash == audio_signature:
        return None

    st.session_state.last_audio_hash = audio_signature
    try:
        transcript = transcribe_audio_bytes(audio_bytes)
    except AudioProcessingError as exc:
        st.session_state.last_error = str(exc)
        st.warning(str(exc))
        return None

    st.session_state.last_transcript = transcript
    st.session_state.last_error = ""
    return transcript


def render_input_panel() -> None:
    """Render audio and text intake controls."""
    st.subheader("Describe Symptoms")
    st.caption("Use voice input when available, or fall back to typed text.")

    audio_col, text_col = st.columns(2, gap="large")

    with audio_col:
        st.markdown("**Voice input**")
        if st_audiorec is None:
            st.info("Audio recorder is unavailable. Install dependencies and use text input.")
        else:
            audio_bytes = st_audiorec()
            if audio_bytes:
                transcript = transcribe_uploaded_audio(audio_bytes)
                if transcript:
                    st.success("Audio captured successfully.")
                    st.text_area(
                        "Transcribed message",
                        value=transcript,
                        height=120,
                        key="voice_transcript_preview",
                    )
                    if st.button("Send transcribed message", use_container_width=True):
                        st.session_state.last_input_source = "voice"
                        handle_patient_message(transcript)
                        st.rerun()

    with text_col:
        st.markdown("**Text input**")
        default_text = st.session_state.last_transcript if st.session_state.last_input_source == "voice" else ""
        with st.form("text_input_form", clear_on_submit=True):
            typed_message = st.text_area(
                "Describe your symptoms or answer the latest question",
                value=default_text,
                height=160,
                placeholder="Example: I have had a dull headache for three days and it is getting worse.",
            )
            submitted = st.form_submit_button("Send message", use_container_width=True)

        if submitted:
            st.session_state.last_input_source = "text"
            handle_patient_message(typed_message)
            st.rerun()

    if st.session_state.audio_response:
        st.markdown("**Latest audio response**")
        st.markdown(
            build_audio_download_html(st.session_state.audio_response),
            unsafe_allow_html=True,
        )


def render_summary(summary: ClinicalSummary) -> None:
    """Render the generated clinical summary and download controls."""
    st.subheader("Clinical Summary")

    severity_color = {
        "URGENT": "#dc2626",
        "MODERATE": "#d97706",
        "ROUTINE": "#15803d",
    }.get(summary.severity_assessment, "#1d4ed8")

    e = _html.escape
    st.markdown(
        f"""
        <div style="padding: 1rem 1.25rem; border-radius: 12px; background: #f8fbff; border: 1px solid #cfe3ff;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a;">Patient Intake Summary</div>
            <div style="font-size: 0.9rem; color: #475569; margin-bottom: 0.8rem;">Generated by MediVoice AI</div>
            <div><strong>Chief Complaint:</strong> {e(summary.chief_complaint)}</div>
            <div style="margin-top: 0.7rem;"><strong>Present Illness:</strong> {e(summary.present_illness)}</div>
            <div style="margin-top: 0.7rem;"><strong>Symptoms Timeline:</strong></div>
            <div>Onset: {e(summary.onset)}</div>
            <div>Duration: {e(summary.duration)}</div>
            <div>Progression: {e(summary.progression)}</div>
            <div style="margin-top: 0.7rem;"><strong>Associated Symptoms:</strong> {e(", ".join(summary.associated_symptoms))}</div>
            <div style="margin-top: 0.7rem;"><strong>Severity Assessment:</strong> <span style="color: {severity_color}; font-weight: 700;">{e(summary.severity_assessment)}</span></div>
            <div style="margin-top: 0.7rem;"><strong>Recommended Actions:</strong> {e("; ".join(summary.recommended_actions))}</div>
            <div style="margin-top: 0.7rem;"><strong>Red Flags Assessed:</strong> {e(summary.red_flags_assessed)}</div>
            <div style="margin-top: 0.7rem;"><strong>Clinical Notes:</strong> {e(summary.clinical_notes)}</div>
            <div style="margin-top: 0.8rem; font-size: 0.85rem; color: #64748b;">Timestamp: {e(summary.generated_at)}</div>
            <div style="font-size: 0.85rem; color: #64748b;">Session ID: {e(summary.session_id)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_text = summary.to_display_text()
    st.text_area("Structured summary export preview", summary_text, height=360)
    st.download_button(
        "Export Summary as Text",
        data=build_summary_export(summary_text),
        file_name=f"medivoice-summary-{summary.session_id}.txt",
        mime="text/plain",
        use_container_width=True,
    )


def render_footer_actions() -> None:
    """Render utility actions for session management."""
    left_col, right_col = st.columns(2)
    with left_col:
        if st.button("Generate Summary Now", use_container_width=True):
            if st.session_state.questions_asked_so_far < MIN_FOLLOW_UPS_FOR_SUMMARY:
                st.info("A few more details may improve the summary, but I will try now.")
            handle_patient_message(
                "Please finalize the intake and generate the structured clinical summary now."
            )
            st.rerun()
    with right_col:
        if st.button("Start New Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            initialize_session_state()
            st.rerun()


def inject_custom_styles() -> None:
    """Apply a clean medical-themed UI."""
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            h1, h2, h3 {
                color: #0f3d75;
            }
            div[data-testid="stChatMessage"] {
                border-radius: 14px;
                border: 1px solid #dbeafe;
                background-color: rgba(255, 255, 255, 0.8);
            }
            div[data-testid="stSidebar"] {
                background: #ffffff;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run the MediVoice AI Streamlit application."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    hydrate_environment_from_secrets()
    inject_custom_styles()
    initialize_session_state()

    st.title(APP_TITLE)
    st.subheader(APP_SUBTITLE)
    st.caption(
        "Securely collect symptom details with guided follow-up questions and a structured clinical summary."
    )

    render_sidebar()

    if st.session_state.last_error:
        st.error(f"Last issue: {st.session_state.last_error}")

    render_input_panel()
    render_conversation()

    if st.session_state.clinical_summary is not None:
        render_summary(st.session_state.clinical_summary)

    render_footer_actions()

    st.markdown(
        """
        <hr style="margin-top: 2rem; margin-bottom: 1rem;">
        <small>
            MediVoice AI is an intake support assistant and does not provide diagnosis or emergency services.
            If symptoms are severe, worsening rapidly, or include chest pain, severe shortness of breath,
            confusion, or uncontrolled bleeding, seek urgent medical attention immediately.
        </small>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
