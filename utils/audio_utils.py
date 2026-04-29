"""Audio helpers for recording, transcription, and text-to-speech."""

from __future__ import annotations

import base64
import io
from typing import Optional

import speech_recognition as sr
from gtts import gTTS


class AudioProcessingError(RuntimeError):
    """Raised when audio processing cannot be completed."""


def transcribe_audio_bytes(audio_bytes: bytes, language: str = "en-US") -> str:
    """Transcribe WAV audio bytes into text using SpeechRecognition.

    Args:
        audio_bytes: Raw WAV audio bytes captured from the recorder widget.
        language: Recognition language code.

    Returns:
        The recognized transcript.

    Raises:
        AudioProcessingError: If transcription fails or audio cannot be parsed.
    """
    if not audio_bytes:
        raise AudioProcessingError("No audio data was provided.")

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
    except Exception as exc:  # pragma: no cover - depends on browser/audio source
        raise AudioProcessingError(
            "The recording could not be parsed. Please try again or use text input."
        ) from exc

    try:
        transcript = recognizer.recognize_google(audio_data, language=language)
    except sr.UnknownValueError as exc:
        raise AudioProcessingError(
            "I could not clearly understand the recording. Please try again or type your message."
        ) from exc
    except sr.RequestError as exc:
        raise AudioProcessingError(
            "Speech recognition is temporarily unavailable. Please use text input for now."
        ) from exc

    cleaned = transcript.strip()
    if not cleaned:
        raise AudioProcessingError("The recording did not contain usable speech.")
    return cleaned


def generate_tts_audio(text: str, lang: str = "en", slow: bool = False) -> bytes:
    """Generate MP3 speech bytes from plain text using gTTS.

    Args:
        text: Text to synthesize.
        lang: Output language code.
        slow: Whether to slow the playback speed.

    Returns:
        MP3 audio bytes.

    Raises:
        AudioProcessingError: If speech synthesis fails.
    """
    if not text.strip():
        raise AudioProcessingError("Cannot generate speech for empty text.")

    buffer = io.BytesIO()
    try:
        tts = gTTS(text=text.strip(), lang=lang, slow=slow)
        tts.write_to_fp(buffer)
    except Exception as exc:  # pragma: no cover - network dependent
        raise AudioProcessingError(
            "Audio playback could not be generated at the moment."
        ) from exc
    return buffer.getvalue()


def build_audio_download_html(audio_bytes: Optional[bytes], filename: str = "response.mp3") -> str:
    """Create an inline HTML audio player with a download link.

    Args:
        audio_bytes: MP3 audio bytes.
        filename: Suggested download filename.

    Returns:
        Safe HTML string that renders an audio player and download anchor.
    """
    if not audio_bytes:
        return ""

    encoded = base64.b64encode(audio_bytes).decode("utf-8")
    return f"""
    <audio controls style="width: 100%;">
        <source src="data:audio/mp3;base64,{encoded}" type="audio/mpeg">
        Your browser does not support audio playback.
    </audio>
    <a href="data:audio/mp3;base64,{encoded}" download="{filename}">Download audio response</a>
    """
