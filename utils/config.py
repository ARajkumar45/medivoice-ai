"""Application configuration helpers for MediVoice AI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional


DEFAULT_PROJECT_ID = "project-d33cb38a-e8d2-46c0-972"
DEFAULT_REGION = "us-central1"
DEFAULT_MODEL = "gemini-2.5-flash-002"
DEFAULT_MODEL_OPTIONS = (
    "gemini-2.5-flash-002",
    "gemini-2.5-pro",
    "gemini-2.0-flash-001",
)


@dataclass(slots=True)
class VertexSettings:
    """Runtime settings for Vertex AI access."""

    project_id: str
    region: str
    model_name: str


@dataclass(slots=True)
class AppConfig:
    """All high-level configuration used by the Streamlit app."""

    vertex: VertexSettings
    model_options: List[str]

    @classmethod
    def load(
        cls,
        secrets: Optional[Mapping[str, object]] = None,
        model_override: Optional[str] = None,
    ) -> "AppConfig":
        """Load config from environment and optionally Streamlit secrets.

        Args:
            secrets: Optional secret mapping, usually `st.secrets`.
            model_override: Optional runtime-selected model value.
        """
        project_id = _read_setting("VERTEX_PROJECT_ID", DEFAULT_PROJECT_ID, secrets)
        region = _read_setting("VERTEX_REGION", DEFAULT_REGION, secrets)
        model_name = model_override or _read_setting("VERTEX_MODEL", DEFAULT_MODEL, secrets)
        model_options = _read_model_options(secrets)

        if model_name not in model_options:
            model_options.insert(0, model_name)

        return cls(
            vertex=VertexSettings(
                project_id=project_id,
                region=region,
                model_name=model_name,
            ),
            model_options=model_options,
        )


def _read_setting(
    key: str,
    default: str,
    secrets: Optional[Mapping[str, object]] = None,
) -> str:
    """Read a config value from environment first, then secrets, then default."""
    env_value = os.getenv(key)
    if env_value:
        return env_value

    if secrets and key in secrets:
        secret_value = str(secrets[key]).strip()
        if secret_value:
            return secret_value

    return default


def _read_model_options(secrets: Optional[Mapping[str, object]] = None) -> List[str]:
    """Read the list of selectable model options."""
    raw = os.getenv("VERTEX_MODEL_OPTIONS", "")
    if not raw and secrets and "VERTEX_MODEL_OPTIONS" in secrets:
        raw = str(secrets["VERTEX_MODEL_OPTIONS"])

    if raw.strip():
        parsed = [item.strip() for item in raw.split(",") if item.strip()]
        return _dedupe(parsed)

    return list(DEFAULT_MODEL_OPTIONS)


def _dedupe(values: Iterable[str]) -> List[str]:
    """Preserve order while removing duplicates."""
    seen = set()
    result: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
