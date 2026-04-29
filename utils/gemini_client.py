"""Vertex AI Gemini client helpers for the MediVoice AI application."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

from .config import VertexSettings


SYSTEM_PROMPT = """You are an empathetic AI medical intake assistant. Your role is to gather comprehensive patient information before their doctor visit. Ask clear, compassionate questions to understand:
1. Chief complaint (main reason for visit)
2. Symptom details (what, when, where, severity)
3. Timeline (onset, duration, progression)
4. Associated symptoms
5. Previous treatments tried

Be conversational but professional. Ask ONE question at a time. Show empathy. Use simple medical terminology. After gathering information, generate a structured clinical summary.

Follow-up question logic:
- If patient mentions pain: Ask about location, severity (1-10), type (sharp/dull/throbbing)
- If patient mentions duration: Ask if symptoms are constant or intermittent
- If patient mentions previous treatments: Ask what worked/didn't work
- Always ask: "Is there anything else you think the doctor should know?"

Return valid JSON only with this exact schema:
{
  "assistant_message": "string",
  "should_generate_summary": true,
  "questions_asked_so_far": 0,
  "summary_ready_reason": "string",
  "summary": {
    "chief_complaint": "string",
    "present_illness": "string",
    "onset": "string",
    "duration": "string",
    "progression": "string",
    "associated_symptoms": ["string"],
    "severity_assessment": "URGENT | MODERATE | ROUTINE",
    "recommended_actions": ["string"],
    "red_flags_assessed": "string",
    "clinical_notes": "string"
  }
}

Rules:
- Ask 3 to 5 follow-up questions total before generating a summary unless the user asks to stop or enough detail is already available.
- `assistant_message` should be the next conversational reply shown to the patient.
- When enough information has been gathered, `assistant_message` must begin with: "Let me make sure I have this right..."
- If `should_generate_summary` is false, set `summary` to null.
- Keep summaries factual and non-diagnostic. Do not prescribe medications.
- If symptoms sound emergent, mark severity as URGENT and recommend immediate medical attention.
"""


GeminiConfig = VertexSettings


@dataclass(slots=True)
class ConversationTurn:
    """Represents a single chat turn."""

    role: str
    content: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass(slots=True)
class ClinicalSummary:
    """Structured intake summary returned by Gemini."""

    chief_complaint: str
    present_illness: str
    onset: str
    duration: str
    progression: str
    associated_symptoms: List[str]
    severity_assessment: str
    recommended_actions: List[str]
    red_flags_assessed: str
    clinical_notes: str
    generated_at: str
    session_id: str

    @classmethod
    def from_model_payload(
        cls, payload: Dict[str, Any], session_id: Optional[str] = None
    ) -> "ClinicalSummary":
        """Create a summary from model JSON."""
        return cls(
            chief_complaint=str(payload.get("chief_complaint", "Not provided")).strip(),
            present_illness=str(payload.get("present_illness", "Not provided")).strip(),
            onset=str(payload.get("onset", "Unknown")).strip(),
            duration=str(payload.get("duration", "Unknown")).strip(),
            progression=str(payload.get("progression", "Unknown")).strip(),
            associated_symptoms=[
                str(item).strip()
                for item in payload.get("associated_symptoms", [])
                if str(item).strip()
            ]
            or ["None reported"],
            severity_assessment=str(
                payload.get("severity_assessment", "ROUTINE")
            ).strip().upper(),
            recommended_actions=[
                str(item).strip()
                for item in payload.get("recommended_actions", [])
                if str(item).strip()
            ]
            or ["Follow up with a licensed clinician for formal evaluation."],
            red_flags_assessed=str(
                payload.get("red_flags_assessed", "No red flags documented.")
            ).strip(),
            clinical_notes=str(payload.get("clinical_notes", "None")).strip(),
            generated_at=datetime.now(timezone.utc).isoformat(),
            session_id=session_id or str(uuid4()),
        )

    def to_display_text(self) -> str:
        """Render the summary in the requested text format."""
        severity_map = {
            "URGENT": "🔴 URGENT",
            "MODERATE": "🟡 MODERATE",
            "ROUTINE": "🟢 ROUTINE",
        }
        severity = severity_map.get(self.severity_assessment, self.severity_assessment)
        symptoms = "\n".join(f"- {item}" for item in self.associated_symptoms)
        actions = "\n".join(f"- {item}" for item in self.recommended_actions)
        return f"""═══════════════════════════════════════════
PATIENT INTAKE SUMMARY
Generated by MediVoice AI
═══════════════════════════════════════════
CHIEF COMPLAINT:
{self.chief_complaint}

PRESENT ILLNESS:
{self.present_illness}

SYMPTOMS TIMELINE:
Onset: {self.onset}
Duration: {self.duration}
Progression: {self.progression}

ASSOCIATED SYMPTOMS:
{symptoms}

SEVERITY ASSESSMENT:
{severity}

RECOMMENDED ACTIONS:
{actions}

RED FLAGS ASSESSED:
{self.red_flags_assessed}

CLINICAL NOTES:
{self.clinical_notes}
═══════════════════════════════════════════
Timestamp: {self.generated_at}
Session ID: {self.session_id}
═══════════════════════════════════════════"""


@dataclass(slots=True)
class IntakeResponse:
    """Parsed response from the LLM."""

    assistant_message: str
    should_generate_summary: bool
    questions_asked_so_far: int
    summary_ready_reason: str
    summary: Optional[ClinicalSummary]


class GeminiClient:
    """Small wrapper around Vertex AI Gemini for intake conversations."""

    def __init__(self, config: GeminiConfig) -> None:
        self.config = config
        vertexai.init(project=config.project_id, location=config.region)
        self.model = GenerativeModel(config.model_name, system_instruction=SYSTEM_PROMPT)

    def continue_intake(
        self, conversation: List[ConversationTurn], session_id: str
    ) -> IntakeResponse:
        """Generate the next assistant response and optional clinical summary."""
        prompt = self._build_conversation_prompt(conversation)
        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.3,
                top_p=0.9,
                response_mime_type="application/json",
            ),
        )
        raw_text = getattr(response, "text", "") or ""
        payload = self._parse_json_response(raw_text)
        summary_payload = payload.get("summary")
        summary = None
        if payload.get("should_generate_summary") and isinstance(summary_payload, dict):
            summary = ClinicalSummary.from_model_payload(summary_payload, session_id=session_id)

        return IntakeResponse(
            assistant_message=str(payload.get("assistant_message", "")).strip()
            or "Could you tell me a bit more about what you're experiencing?",
            should_generate_summary=bool(payload.get("should_generate_summary", False)),
            questions_asked_so_far=int(payload.get("questions_asked_so_far", 0)),
            summary_ready_reason=str(payload.get("summary_ready_reason", "")).strip(),
            summary=summary,
        )

    @staticmethod
    def _build_conversation_prompt(conversation: List[ConversationTurn]) -> str:
        """Flatten the conversation into a compact prompt for the model."""
        transcript = "\n".join(
            f"{turn.role.upper()}: {turn.content}" for turn in conversation if turn.content.strip()
        )
        return (
            "Review the intake conversation below and respond with valid JSON only.\n\n"
            f"{transcript}\n\n"
            "Determine whether enough information exists to produce the structured summary."
        )

    @staticmethod
    def _parse_json_response(raw_text: str) -> Dict[str, Any]:
        """Parse the model response, tolerating fenced JSON blocks."""
        candidate = raw_text.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            candidate = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The AI returned an unexpected response format. Please try again."
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError("The AI response was not a JSON object.")
        return payload
