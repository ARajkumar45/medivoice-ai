"""Audio helpers for recording, transcription, and text-to-speech."""

from __future__ import annotations

import base64
import io
from typing import Optional

import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import normalize as _pydub_normalize


class AudioProcessingError(RuntimeError):
    """Raised when audio processing cannot be completed."""


def preprocess_audio(audio_bytes: bytes) -> bytes:
    """Normalize volume of WAV audio before transcription."""
    try:
        audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        audio = _pydub_normalize(audio)
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        return buf.getvalue()
    except Exception:
        return audio_bytes


def transcribe_audio_bytes(audio_bytes: bytes, language: str = "en-US") -> str:
    """Transcribe WAV audio bytes into text using SpeechRecognition."""
    if not audio_bytes:
        raise AudioProcessingError("No audio data was provided.")

    audio_bytes = preprocess_audio(audio_bytes)

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
    except Exception as exc:
        raise AudioProcessingError(
            "The recording could not be parsed. "
            "Please try again or use text input."
        ) from exc

    try:
        transcript = recognizer.recognize_google(audio_data, language=language)
    except sr.UnknownValueError as exc:
        raise AudioProcessingError(
            "I could not clearly understand the recording. "
            "Please try again or type your message."
        ) from exc
    except sr.RequestError as exc:
        raise AudioProcessingError(
            "Speech recognition is temporarily unavailable. "
            "Please use text input for now."
        ) from exc

    cleaned = transcript.strip()
    if not cleaned:
        raise AudioProcessingError(
            "The recording did not contain usable speech."
        )
    return cleaned


def generate_tts_audio(
    text: str, lang: str = "en", slow: bool = False,
) -> bytes:
    """Generate MP3 speech bytes from plain text using gTTS."""
    if not text.strip():
        raise AudioProcessingError("Cannot generate speech for empty text.")

    buffer = io.BytesIO()
    try:
        tts = gTTS(text=text.strip(), lang=lang, slow=slow)
        tts.write_to_fp(buffer)
    except Exception as exc:
        raise AudioProcessingError(
            "Audio playback could not be generated at the moment."
        ) from exc
    return buffer.getvalue()


def build_audio_download_html(
    audio_bytes: Optional[bytes],
    filename: str = "response.mp3",
) -> str:
    """Create an inline HTML audio player with a download link."""
    if not audio_bytes:
        return ""

    encoded = base64.b64encode(audio_bytes).decode("utf-8")
    src = "data:audio/mp3;base64," + encoded
    return (
        "<div style='background:var(--surface-container-low);"
        "border:1px solid var(--surface-variant);"
        "border-radius:12px;padding:12px 16px;margin:4px 0;'>"
        "<audio controls style='width:100%;height:36px;border-radius:8px;'>"
        "<source src='" + src + "' type='audio/mpeg'>"
        "Your browser does not support audio playback.</audio>"
        "<div style='text-align:right;margin-top:6px;'>"
        "<a href='" + src + "' download='" + filename + "' "
        "style='font-size:0.78rem;color:var(--primary);"
        "text-decoration:none;font-weight:500;'>"
        "Download audio</a></div></div>"
    )
