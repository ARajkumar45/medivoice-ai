"""Main Streamlit application for MediVoice AI."""

from __future__ import annotations

import base64
import html as _html
import os
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

try:
    from audiorecorder import audiorecorder
except ImportError:  # pragma: no cover - dependency issue surfaced in UI
    audiorecorder = None

try:
    from components.web_speech import web_speech_input
except Exception:  # pragma: no cover
    web_speech_input = None  # type: ignore[assignment]

from utils import (
    AppConfig,
    AudioProcessingError,
    ClinicalSummary,
    ConversationTurn,
    GeminiAPIClient,
    GeminiClient,
    GeminiConfig,
    NvidiaClient,
    build_audio_download_html,
    generate_tts_audio,
    transcribe_audio_bytes,
)


load_dotenv()

APP_TITLE = "MediVoice AI - Patient Intake Assistant"
APP_SUBTITLE = "Powered by Google Gemini"
WELCOME_MESSAGE = (
    "Hello, I'm here to help gather information before your visit. "
    "Please describe what's bothering you today, either by speaking or typing."
)
MIN_FOLLOW_UPS_FOR_SUMMARY = 3


def _safe_secrets() -> Optional[Mapping[str, object]]:
    """Return st.secrets if a secrets file exists, otherwise None."""
    try:
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
        "last_web_speech_hash": None,
        "last_voice_audio_hash": None,
        "pending_voice_transcript": None,
        "user_gemini_api_key": "",
        "session_started_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": None,
        "app_config": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_config() -> AppConfig:
    """Return the AppConfig cached in session state."""
    if st.session_state.get("app_config") is None:
        st.session_state.app_config = AppConfig.load(
            secrets=_safe_secrets(),
            model_override=st.session_state.get("selected_model"),
        )
    return st.session_state.app_config  # type: ignore[return-value]


@st.cache_resource(show_spinner=False)
def get_gemini_client(project_id: str, region: str, model_name: str) -> GeminiClient:
    """Create or return a cached Vertex AI Gemini client."""
    return GeminiClient(
        config=GeminiConfig(
            project_id=project_id,
            region=region,
            model_name=model_name,
        )
    )


@st.cache_resource(show_spinner=False)
def get_api_key_client(api_key: str, model_name: str) -> GeminiAPIClient:
    """Create or return a cached Gemini API-key client."""
    return GeminiAPIClient(api_key=api_key, model_name=model_name)


def _get_active_client() -> "GeminiClient | GeminiAPIClient":
    """Return the right client based on configuration."""
    api_key = st.session_state.get("user_gemini_api_key", "").strip()
    app_config = _get_config()
    if api_key:
        model = st.session_state.get("selected_model") or app_config.vertex.model_name
        return get_api_key_client(api_key, model)
    return get_gemini_client(
        app_config.vertex.project_id,
        app_config.vertex.region,
        app_config.vertex.model_name,
    )


@st.cache_data(show_spinner=False)
def build_summary_export(summary_text: str) -> bytes:
    """Prepare the summary for download as UTF-8 text."""
    return summary_text.encode("utf-8")


def _section_heading(text: str) -> None:
    """Render a styled section heading with a horizontal rule."""
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;"
        "margin-bottom:12px;'>"
        "<span style='font-size:1.15rem;font-weight:700;"
        "color:var(--on-surface);'>{text}</span>"
        "<div style='flex:1;height:1px;"
        "background:var(--surface-variant);'></div></div>".format(text=text),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Render professional sidebar with branding, config, and status."""
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:1rem 0 0.5rem;'>"
            "<div style='display:inline-flex;align-items:center;"
            "justify-content:center;width:52px;height:52px;"
            "border-radius:14px;"
            "background:var(--primary-container);"
            "margin-bottom:8px;'>"
            "<span style='font-size:26px;'>&#x1FA7A;</span></div>"
            "<div style='font-size:1.15rem;font-weight:700;"
            "color:var(--on-surface);letter-spacing:-0.02em;'>MediVoice AI</div>"
            "<div style='font-size:0.78rem;color:var(--on-surface-variant);"
            "margin-top:2px;'>Intelligent Patient Intake</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ---- Configure AI (model + key) ------------------------------------
        st.markdown(
            "<p style='font-size:0.8rem;font-weight:600;color:var(--on-surface-variant);"
            "text-transform:uppercase;letter-spacing:0.08em;"
            "margin-bottom:8px;'>&#9881; Configure AI</p>",
            unsafe_allow_html=True,
        )

        # Model selector
        app_config = _get_config()
        st.markdown(
            "<p style='font-size:0.75rem;font-weight:500;color:var(--on-surface-variant);"
            "margin-bottom:3px;'>Gemini Model</p>",
            unsafe_allow_html=True,
        )
        chosen_model = st.selectbox(
            "Gemini model",
            options=app_config.model_options,
            index=app_config.model_options.index(app_config.vertex.model_name),
            help="Select the Gemini model to use for this session.",
            label_visibility="collapsed",
        )
        if chosen_model != st.session_state.selected_model:
            st.session_state.selected_model = chosen_model
            st.session_state.app_config = None
            get_gemini_client.clear()
            get_api_key_client.clear()

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        # API key
        st.markdown(
            "<p style='font-size:0.75rem;font-weight:500;color:var(--on-surface-variant);"
            "margin-bottom:3px;'>Google AI API Key "
            "<span style='font-weight:400;color:var(--outline);'>(optional)</span></p>",
            unsafe_allow_html=True,
        )
        entered_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.get("user_gemini_api_key", ""),
            type="password",
            placeholder="AIzaSy...",
            help="Your personal Gemini API key. Uses your own free quota and is stored only in this browser session.",
            label_visibility="collapsed",
        )
        if entered_key != st.session_state.get("user_gemini_api_key", ""):
            st.session_state.user_gemini_api_key = entered_key
            get_api_key_client.clear()

        current_key = st.session_state.get("user_gemini_api_key", "").strip()

        if current_key:
            if not current_key.startswith("AIza"):
                st.markdown(
                    "<div style='display:flex;align-items:center;gap:6px;"
                    "padding:8px 12px;border-radius:8px;"
                    "background:rgba(239,68,68,0.1);"
                    "border:1px solid rgba(239,68,68,0.25);margin:6px 0;'>"
                    "<span style='color:#EF4444;font-size:14px;'>&#9888;</span>"
                    "<span style='color:#EF4444;font-size:0.82rem;"
                    "font-weight:500;'>Key format looks invalid</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='display:flex;align-items:center;gap:6px;"
                    "padding:8px 12px;border-radius:8px;"
                    "background:rgba(16,185,129,0.1);"
                    "border:1px solid rgba(16,185,129,0.25);margin:6px 0;'>"
                    "<span style='color:#10B981;font-size:14px;'>&#10003;</span>"
                    "<span style='color:#10B981;font-size:0.85rem;"
                    f"font-weight:500;'>Personal key active &middot; {chosen_model}</span></div>",
                    unsafe_allow_html=True,
                )
            if st.button("Clear API key", use_container_width=True):
                st.session_state.user_gemini_api_key = ""
                get_api_key_client.clear()
                st.rerun()
        else:
            st.markdown(
                "<div style='display:flex;align-items:center;gap:6px;"
                "padding:8px 12px;border-radius:8px;"
                "background:var(--surface-container-low);"
                "border:1px solid var(--surface-variant);margin:6px 0;'>"
                "<span style='color:var(--primary);font-size:0.85rem;"
                f"font-weight:500;'>Shared quota &middot; {chosen_model}</span></div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"`{app_config.vertex.project_id}` "
                f"&middot; `{app_config.vertex.region}`"
            )

        # Step-by-step key guide (collapsed by default)
        with st.expander("&#128273; How to get your free API key"):
            st.markdown(
                "<div style='font-size:0.82rem;line-height:1.65;"
                "color:var(--on-surface-variant);'>"
                "<b style='color:var(--on-surface);'>Get a free Gemini API key in 3 steps:</b>"
                "<ol style='margin:8px 0 0 16px;padding:0;'>"
                "<li style='margin-bottom:6px;'>Go to "
                "<a href='https://aistudio.google.com/apikey' target='_blank' "
                "style='color:var(--secondary);font-weight:500;'>"
                "aistudio.google.com/apikey</a> and sign in with your Google account.</li>"
                "<li style='margin-bottom:6px;'>Click <b>Create API key</b> &rarr; "
                "choose <b>Create API key in new project</b>.</li>"
                "<li style='margin-bottom:6px;'>Copy the key (starts with <code>AIzaSy</code>) "
                "and paste it in the field above.</li>"
                "</ol>"
                "<div style='margin-top:10px;padding:8px 10px;border-radius:6px;"
                "background:rgba(0,100,148,0.07);"
                "border:1px solid rgba(0,100,148,0.15);font-size:0.78rem;'>"
                "&#128274; The key is stored only in your browser session "
                "and never sent to our servers.</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        st.markdown(
            "<p style='font-size:0.8rem;font-weight:600;color:var(--on-surface-variant);"
            "text-transform:uppercase;letter-spacing:0.08em;"
            "margin-bottom:4px;'>How it works</p>",
            unsafe_allow_html=True,
        )
        steps = [
            ("1", "Describe symptoms via voice or text"),
            ("2", "Answer follow-up questions"),
            ("3", "Review your clinical summary"),
        ]
        for num, desc in steps:
            st.markdown(
                f"<div style='display:flex;align-items:center;"
                f"gap:10px;margin:8px 0;'>"
                f"<div style='width:26px;height:26px;border-radius:8px;"
                f"background:var(--primary-container);color:var(--on-primary-container);"
                f"font-weight:700;font-size:0.8rem;display:flex;"
                f"align-items:center;justify-content:center;"
                f"flex-shrink:0;'>{num}</div>"
                f"<span style='color:var(--on-surface-variant);"
                f"font-size:0.88rem;'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div style='margin-top:10px;padding:10px 12px;"
            "border-radius:8px;background:var(--surface-container-low);"
            "border:1px solid var(--surface-variant);"
            "font-size:0.8rem;color:var(--on-surface-variant);line-height:1.5;'>"
            "Intake documentation only &mdash; "
            "not a substitute for emergency care.</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p style='font-size:0.8rem;font-weight:600;color:var(--on-surface-variant);"
            "text-transform:uppercase;letter-spacing:0.08em;"
            "margin-bottom:4px;margin-top:8px;'>Session</p>",
            unsafe_allow_html=True,
        )
        sid_short = st.session_state.session_id[:12]
        ts_short = st.session_state.session_started_at[:19].replace("T", " ")
        st.markdown(
            f"<div style='font-size:0.78rem;color:var(--on-surface-variant);"
            f"font-family:monospace;background:var(--surface-container-low);"
            f"border-radius:6px;padding:8px 10px;"
            f"border:1px solid var(--surface-variant);'>"
            f"{sid_short}&hellip;<br>{ts_short}</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

def render_conversation() -> None:
    """Render the chat history with modern styling."""
    turn_count = len(st.session_state.conversation)
    suffix = "s" if turn_count != 1 else ""
    label = f"Conversation &middot; {turn_count} message{suffix}"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;"
        f"margin-bottom:12px;'>"
        f"<span style='font-size:1.15rem;font-weight:700;"
        f"color:var(--on-surface);'>{label}</span>"
        f"<div style='flex:1;height:1px;"
        f"background:var(--surface-variant);'></div></div>",
        unsafe_allow_html=True,
    )
    for turn in st.session_state.conversation:
        avatar = "\U0001FA7A" if turn.role == "assistant" else "\U0001F9D1"
        with st.chat_message(turn.role, avatar=avatar):
            st.markdown(turn.content)
            ts = getattr(turn, "created_at", "")
            if ts:
                short_ts = ts[:19].replace("T", " ")
                st.markdown(
                    f"<div style='font-size:0.7rem;color:var(--on-surface-variant);"
                    f"margin-top:6px;'>{short_ts}</div>",
                    unsafe_allow_html=True,
                )


def append_turn(role: str, content: str) -> None:
    """Append a new conversation turn if content is not empty."""
    cleaned = content.strip()
    if cleaned:
        st.session_state.conversation.append(
            ConversationTurn(role=role, content=cleaned)
        )


def handle_patient_message(message: str) -> None:
    """Send a patient message to Gemini and update UI state."""
    cleaned = message.strip()
    if not cleaned:
        st.warning("Please provide a symptom description before continuing.")
        return

    append_turn("user", cleaned)
    st.session_state.last_error = ""

    try:
        with st.spinner("Analyzing your intake response..."):
            result = _get_active_client().continue_intake(
                conversation=st.session_state.conversation,
                session_id=st.session_state.session_id,
            )
    except Exception as exc:
        # Roll back the user turn so it doesn't accumulate on repeated failures
        if st.session_state.conversation and st.session_state.conversation[-1].role == "user":
            st.session_state.conversation.pop()
        err_str = str(exc)
        st.session_state.last_error = err_str
        has_key = bool(st.session_state.get("user_gemini_api_key", "").strip())
        if not has_key:
            st.error(
                "**Could not reach the AI assistant.** "
                "No API key is configured — add your free Gemini API key in the sidebar "
                "under **⚙ Configure AI** to get started. "
                "Click *How to get your free API key* there for step-by-step instructions."
            )
        else:
            st.error(f"**AI error:** {err_str}")
        return

    append_turn("assistant", result.assistant_message)
    st.session_state.questions_asked_so_far = result.questions_asked_so_far

    try:
        st.session_state.audio_response = generate_tts_audio(
            result.assistant_message,
        )
    except AudioProcessingError:
        st.session_state.audio_response = None

    if result.should_generate_summary and result.summary is not None:
        st.session_state.clinical_summary = result.summary


# ---------------------------------------------------------------------------
# Audio transcription
# ---------------------------------------------------------------------------

def transcribe_uploaded_audio(audio_bytes: bytes) -> Optional[str]:
    """Transcribe browser-recorded audio, avoiding duplicate processing."""
    audio_signature = hash(audio_bytes)
    if st.session_state.last_audio_hash == audio_signature:
        return None

    st.session_state.last_audio_hash = audio_signature
    transcript = None
    try:
        transcript = _get_active_client().transcribe_audio(audio_bytes)
    except Exception:
        try:
            transcript = transcribe_audio_bytes(audio_bytes)
        except AudioProcessingError as exc:
            st.session_state.last_error = str(exc)
            st.warning(str(exc))
            return None

    if not transcript:
        st.warning(
            "Could not understand the recording. "
            "Please try again or type your message."
        )
        return None

    st.session_state.last_transcript = transcript
    st.session_state.last_error = ""
    return transcript


def _process_voice_audio(voice_result: dict) -> None:
    """Decode base64 audio and transcribe via Gemini."""
    audio_hash = hash(voice_result["audio_b64"][:300])
    if st.session_state.last_voice_audio_hash == audio_hash:
        return

    st.session_state.last_voice_audio_hash = audio_hash
    st.session_state.pending_voice_transcript = None
    mime_type = voice_result.get("mime_type", "audio/webm")

    try:
        audio_bytes = base64.b64decode(voice_result["audio_b64"])
        with st.spinner("Transcribing with Gemini..."):
            transcript = _get_active_client().transcribe_audio(
                audio_bytes, mime_type=mime_type,
            )
        st.session_state.pending_voice_transcript = transcript
    except Exception:
        preview = voice_result.get("preview", "").strip()
        if preview:
            st.session_state.pending_voice_transcript = preview
        else:
            st.warning("Could not transcribe. Please try again or type.")


# ---------------------------------------------------------------------------
# Input panel
# ---------------------------------------------------------------------------

def render_input_panel() -> None:
    """Render voice and text intake controls."""
    _section_heading("Describe symptoms")
    st.caption(
        "Record your voice (Chrome/Edge) or type below "
        "-- the assistant will ask follow-up questions."
    )

    voice_col, text_col = st.columns(2, gap="large")

    with voice_col:
        st.markdown("**Voice input**")
        st.caption("Records with noise suppression + Gemini transcription.")

        if web_speech_input is not None:
            voice_result = web_speech_input(
                transcript_ready=bool(
                    st.session_state.get("pending_voice_transcript"),
                ),
                key="web_speech",
            )
            if isinstance(voice_result, dict) and voice_result.get("audio_b64"):
                _prev_hash = st.session_state.get("last_voice_audio_hash")
                _curr_hash = hash(voice_result["audio_b64"][:300])
                _process_voice_audio(voice_result)
                if _prev_hash != _curr_hash:
                    st.rerun()

            pending = st.session_state.get("pending_voice_transcript")
            if pending:
                edited = st.text_area(
                    "Edit transcript before sending:",
                    value=pending, height=90, key="voice_edit_area",
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(
                        "Send voice message",
                        use_container_width=True, type="primary",
                    ):
                        st.session_state.pending_voice_transcript = None
                        st.session_state.last_input_source = "voice"
                        handle_patient_message(edited)
                        st.rerun()
                with c2:
                    if st.button("Discard", use_container_width=True):
                        st.session_state.pending_voice_transcript = None
                        st.rerun()
        else:
            st.info("Voice component could not load.")

        with st.expander("Alternative: upload an audio clip"):
            st.caption("Supports WAV, MP3, M4A, OGG, WebM, FLAC.")
            uploaded_file = st.file_uploader(
                "Audio file",
                type=["wav", "mp3", "m4a", "ogg", "webm", "flac"],
                label_visibility="collapsed",
                key="audio_upload",
            )
            if uploaded_file is not None:
                audio_bytes_up = uploaded_file.read()
                with st.spinner("Transcribing..."):
                    transcript = transcribe_uploaded_audio(audio_bytes_up)
                if transcript:
                    st.success("Transcribed successfully.")
                    edited_fallback = st.text_area(
                        "Review transcript",
                        value=transcript, height=90,
                        key="upload_transcript_preview",
                    )
                    if st.button(
                        "Send transcribed message",
                        use_container_width=True,
                    ):
                        st.session_state.last_input_source = "voice"
                        handle_patient_message(edited_fallback)
                        st.rerun()

    with text_col:
        st.markdown("**Text input**")
        default_text = (
            st.session_state.last_transcript
            if st.session_state.last_input_source == "voice"
            else ""
        )
        with st.form("text_input_form", clear_on_submit=True):
            typed_message = st.text_area(
                "Describe your symptoms or answer the latest question",
                value=default_text, height=160,
                placeholder=(
                    "Example: I have had a dull headache for three days "
                    "and it is getting worse."
                ),
            )
            submitted = st.form_submit_button(
                "Send message", use_container_width=True,
            )
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


# ---------------------------------------------------------------------------
# Clinical summary
# ---------------------------------------------------------------------------

def render_summary(summary: ClinicalSummary) -> None:
    """Render the generated clinical summary with modern card design."""
    _section_heading("Clinical summary")

    severity_cfg = {
        "URGENT": ("#EF4444", "rgba(239,68,68,0.15)", "rgba(239,68,68,0.4)"),
        "MODERATE": ("#F59E0B", "rgba(245,158,11,0.15)", "rgba(245,158,11,0.4)"),
        "ROUTINE": ("#10B981", "rgba(16,185,129,0.15)", "rgba(16,185,129,0.4)"),
    }
    sev_color, sev_bg, sev_border = severity_cfg.get(
        summary.severity_assessment,
        ("#8B5CF6", "rgba(108,58,237,0.15)", "rgba(108,58,237,0.4)"),
    )

    e = _html.escape

    symptoms_parts = []
    for s in summary.associated_symptoms:
        symptoms_parts.append(
            "<span style='display:inline-block;padding:4px 10px;"
            "border-radius:6px;background:var(--surface-container-low);"
            "border:1px solid var(--surface-variant);"
            "font-size:0.82rem;color:var(--on-surface-variant);"
            "margin:3px 4px 3px 0;'>{e(s)}</span>"
        )
    symptoms_html = "".join(symptoms_parts)

    actions_parts = []
    for a in summary.recommended_actions:
        actions_parts.append(
            "<div style='display:flex;align-items:flex-start;"
            "gap:8px;margin:5px 0;'>"
            "<span style='color:var(--primary);font-size:0.85rem;"
            "margin-top:1px;'>&#10095;</span>"
            "<span style='color:var(--on-surface-variant);"
            "font-size:0.88rem;'>{e(a)}</span></div>"
        )
    actions_html = "".join(actions_parts)

    ts_display = e(summary.generated_at[:19].replace("T", " "))
    sid_display = e(summary.session_id[:16])

    _lbl = "font-size:0.72rem;font-weight:600;color:var(--on-surface-variant);"
    _lbl += "text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;"
    _cell = ("background:var(--surface-container-lowest);border-radius:10px;"
             "padding:12px 14px;border:1px solid var(--surface-variant);")

    card = []
    card.append(
        "<div style='background:var(--surface-container-lowest);"
        "border:1px solid var(--surface-variant);"
        "border-radius:16px;padding:1.5rem;"
        "animation:fadeInUp 0.4s ease-out;'>"
    )
    # Header row
    card.append(
        "<div style='display:flex;align-items:center;"
        "justify-content:space-between;margin-bottom:1.2rem;'>"
        "<div>"
        "<div style='font-size:1.15rem;font-weight:700;"
        "color:var(--on-surface);'>Patient Intake Summary</div>"
        "<div style='font-size:0.8rem;color:var(--on-surface-variant);"
        "margin-top:2px;'>Generated by MediVoice AI</div>"
        "</div>"
        f"<div style='padding:6px 16px;border-radius:8px;"
        f"background:{sev_bg};border:1px solid {sev_border};"
        f"font-weight:700;font-size:0.85rem;color:{sev_color};"
        f"letter-spacing:0.04em;'>{e(summary.severity_assessment)}</div>"
        "</div>"
    )
    # Chief complaint + red flags
    card.append(
        "<div style='display:grid;grid-template-columns:1fr 1fr;"
        "gap:12px;margin-bottom:1rem;'>"
        f"<div style='{_cell}'>"
        f"<div style='{_lbl}'>Chief complaint</div>"
        f"<div style='color:var(--on-surface);font-size:0.92rem;"
        f"font-weight:500;'>{e(summary.chief_complaint)}</div></div>"
        f"<div style='{_cell}'>"
        f"<div style='{_lbl}'>Red flags</div>"
        f"<div style='color:var(--on-surface);font-size:0.92rem;"
        f"font-weight:500;'>{e(summary.red_flags_assessed)}</div></div>"
        "</div>"
    )
    # Present illness
    card.append(
        f"<div style='{_cell}margin-bottom:12px;'>"
        f"<div style='{_lbl}'>Present illness</div>"
        f"<div style='color:var(--on-surface-variant);font-size:0.88rem;"
        f"line-height:1.55;'>{e(summary.present_illness)}</div></div>"
    )
    # Timeline
    card.append(
        "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
        "gap:12px;margin-bottom:12px;'>"
        f"<div style='{_cell}'><div style='{_lbl}'>Onset</div>"
        f"<div style='color:var(--on-surface);font-size:0.88rem;'>"
        f"{e(summary.onset)}</div></div>"
        f"<div style='{_cell}'><div style='{_lbl}'>Duration</div>"
        f"<div style='color:var(--on-surface);font-size:0.88rem;'>"
        f"{e(summary.duration)}</div></div>"
        f"<div style='{_cell}'><div style='{_lbl}'>Progression</div>"
        f"<div style='color:var(--on-surface);font-size:0.88rem;'>"
        f"{e(summary.progression)}</div></div>"
        "</div>"
    )
    # Symptoms
    card.append(
        "<div style='margin-bottom:12px;'>"
        f"<div style='{_lbl}margin-bottom:6px;'>Associated symptoms</div>"
        f"<div style='display:flex;flex-wrap:wrap;'>{symptoms_html}</div>"
        "</div>"
    )
    # Actions
    card.append(
        "<div style='margin-bottom:12px;'>"
        f"<div style='{_lbl}margin-bottom:6px;'>Recommended actions</div>"
        f"{actions_html}</div>"
    )
    # Clinical notes
    card.append(
        f"<div style='{_cell}margin-bottom:12px;'>"
        f"<div style='{_lbl}'>Clinical notes</div>"
        f"<div style='color:var(--on-surface-variant);font-size:0.88rem;"
        f"line-height:1.55;'>{e(summary.clinical_notes)}</div></div>"
    )
    # Footer
    card.append(
        "<div style='display:flex;justify-content:space-between;"
        "padding-top:10px;border-top:1px solid var(--surface-variant);"
        "font-size:0.75rem;color:var(--on-surface-variant);font-family:monospace;'>"
        f"<span>{ts_display}</span>"
        f"<span>{sid_display}&hellip;</span></div>"
    )
    card.append("</div>")

    st.markdown("".join(card), unsafe_allow_html=True)

    summary_text = summary.to_display_text()
    st.text_area("Export preview", summary_text, height=300)
    st.download_button(
        "Export Summary",
        data=build_summary_export(summary_text),
        file_name=f"medivoice-summary-{summary.session_id}.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def render_footer_actions() -> None:
    """Render utility actions for session management."""
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    left_col, right_col = st.columns(2)
    with left_col:
        if st.button(
            "Generate Summary Now",
            use_container_width=True, type="primary",
        ):
            if st.session_state.questions_asked_so_far < MIN_FOLLOW_UPS_FOR_SUMMARY:
                st.info("A few more details may help, but generating now.")
            handle_patient_message(
                "Please finalize the intake and generate "
                "the structured clinical summary now."
            )
            st.rerun()
    with right_col:
        if st.button("Start New Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            initialize_session_state()
            st.rerun()


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """\
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

:root {
    --primary: #006a63;
    --on-primary: #ffffff;
    --primary-container: #4fd1c5;
    --on-primary-container: #005750;
    --secondary: #006494;
    --on-secondary: #ffffff;
    --secondary-container: #74c3fe;
    --on-secondary-container: #005077;
    --tertiary: #555f71;
    --background: #f7f9fb;
    --on-background: #191c1e;
    --surface: #f7f9fb;
    --on-surface: #191c1e;
    --surface-variant: #e0e3e5;
    --on-surface-variant: #3c4947;
    --outline: #6c7a77;
    --surface-container-lowest: #ffffff;
    --surface-container-low: #f2f4f6;
    --surface-container: #eceef0;
    --surface-container-high: #e6e8ea;
    --error: #ba1a1a;
    --on-error: #ffffff;
    --radius-lg: 0.25rem;
    --radius-xl: 0.5rem;
    --radius-full: 0.75rem;
}

.material-symbols-outlined {
    font-family: 'Material Symbols Outlined' !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
}

.stApp {
    background-color: var(--background) !important;
    color: var(--on-background) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: var(--surface-container-lowest) !important;
    border-right: 1px solid var(--surface-variant) !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--on-surface-variant) !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: var(--on-surface) !important;
    font-weight: 600 !important;
}

h1 {
    font-size: 32px !important;
    letter-spacing: -0.02em !important;
}

/* Chat Messages / Cards */
div[data-testid="stChatMessage"] {
    background-color: var(--surface-container-lowest) !important;
    border: 1px solid var(--surface-variant) !important;
    border-radius: var(--radius-xl) !important;
    box-shadow: 0px 10px 15px -3px rgba(148, 163, 184, 0.05) !important;
}

/* Buttons */
.stButton>button {
    background-color: var(--primary-container) !important;
    color: var(--on-primary-container) !important;
    border-radius: var(--radius-xl) !important;
    border: none !important;
    font-weight: 500 !important;
    transition: opacity 0.2s !important;
}

.stButton>button:hover {
    opacity: 0.9 !important;
    color: var(--on-primary-container) !important;
}

/* Primary Button Override */
.stButton>button[kind="primary"] {
    background-color: var(--primary) !important;
    color: var(--on-primary) !important;
}

/* Forms and Inputs */
[data-testid="stForm"] {
    background-color: var(--surface-container-lowest) !important;
    border: 1px solid var(--surface-variant) !important;
    border-radius: var(--radius-xl) !important;
}

.stTextInput>div>div>input, .stTextArea>div>div>textarea {
    background-color: var(--surface-container-low) !important;
    border: 1px solid var(--surface-variant) !important;
    border-radius: var(--radius-xl) !important;
    color: var(--on-surface) !important;
}

/* Summary Card Styling */
[data-testid="stExpander"] {
    background-color: var(--surface-container-lowest) !important;
    border: 1px solid var(--surface-variant) !important;
    border-radius: var(--radius-xl) !important;
}

/* Custom UI Components from app.py */
div[style*="background:rgba(255,255,255,0.03)"] {
    background-color: var(--surface-container-lowest) !important;
    border: 1px solid var(--surface-variant) !important;
    border-radius: var(--radius-xl) !important;
    color: var(--on-surface) !important;
}

div[style*="color:#F1F1F4"] {
    color: var(--on-surface) !important;
}

div[style*="color:#D1D5DB"] {
    color: var(--on-surface-variant) !important;
}

div[style*="background:linear-gradient(135deg,#6C3AED,#06D6A0)"] {
    background: var(--primary-container) !important;
}

</style>
"""


def inject_custom_styles() -> None:
    """Apply a bold, modern production-grade UI theme."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the MediVoice AI Streamlit application."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="\U0001FA7A",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    hydrate_environment_from_secrets()
    inject_custom_styles()
    initialize_session_state()

    st.title(APP_TITLE)
    st.markdown(
        "<p style='font-size:1.05rem;color:#9CA3AF;"
        "margin-top:-8px;margin-bottom:4px;'>"
        "Powered by Google Gemini</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.88rem;color:#6B7280;"
        "margin-bottom:1.2rem;'>"
        "Securely collect symptom details with guided follow-up "
        "questions and a structured clinical summary.</p>",
        unsafe_allow_html=True,
    )

    render_sidebar()

    # Warn if no API key and no Vertex AI credentials are likely to work
    if not st.session_state.get("user_gemini_api_key", "").strip():
        st.warning(
            "**No API key set.** Add your free Gemini API key in the sidebar under "
            "**⚙ Configure AI** — then expand *How to get your free API key* for "
            "step-by-step instructions. Without a key the assistant cannot respond.",
            icon="⚠️",
        )

    if st.session_state.last_error:
        st.error(f"Last error: {st.session_state.last_error}")

    render_input_panel()
    render_conversation()

    if st.session_state.clinical_summary is not None:
        render_summary(st.session_state.clinical_summary)

    render_footer_actions()

    st.markdown(
        "<div style='margin-top:2.5rem;padding:16px 20px;"
        "border-radius:12px;background:var(--surface-container-low);"
        "border:1px solid var(--surface-variant);'>"
        "<div style='font-size:0.78rem;color:var(--on-surface-variant);line-height:1.6;'>"
        "MediVoice AI is an intake support assistant and does not "
        "provide diagnosis or emergency services. If symptoms are "
        "severe, worsening rapidly, or include chest pain, severe "
        "shortness of breath, confusion, or uncontrolled bleeding, "
        "<span style='color:var(--error);font-weight:600;'>"
        "seek urgent medical attention immediately</span>."
        "</div></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
