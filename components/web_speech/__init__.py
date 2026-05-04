"""Streamlit custom component wrapping the browser Web Speech API."""

from pathlib import Path

import streamlit.components.v1 as components

_FRONTEND = Path(__file__).parent / "frontend"

_component_func = components.declare_component(
    "web_speech_input",
    path=str(_FRONTEND),
)


def web_speech_input(transcript_ready: bool = False, key: str = "web_speech") -> "dict | None":
    """High-quality voice recorder component.

    Captures audio via MediaRecorder with explicit noise-suppression constraints,
    then returns a dict ``{"audio_b64": ..., "mime_type": ..., "preview": ...}``
    when the user stops recording.

    Pass ``transcript_ready=True`` once Python has finished transcribing so the
    in-component "Sending to Gemini…" spinner is dismissed automatically.
    """
    return _component_func(key=key, default=None, transcript_ready=transcript_ready, height=320)
