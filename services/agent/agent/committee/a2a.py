"""Agent-to-Agent (A2A) Protocol & Interoperability Envelope.

Standardizes communication across disparate agent runtimes (Google ADK and CrewAI).
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class A2AMessage(BaseModel):
    """Standardized envelope for cross-runtime agent communication."""

    message_id: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    )
    source_agent: str
    target_agent: str = "all"
    runtime: str = "crewai"  # "google-adk" | "crewai"
    message_type: str = "thesis"  # "thesis" | "critique" | "synthesis" | "telemetry"
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class A2ABridge:
    """Coordinates message passing between Google ADK and CrewAI agents."""

    def __init__(self) -> None:
        self.message_log: list[A2AMessage] = []

    def transmit(self, message: A2AMessage) -> None:
        """Log and route an A2A message."""
        self.message_log.append(message)

    def get_messages_for(self, target_agent: str) -> list[A2AMessage]:
        """Fetch messages intended for a specific agent or broadcast to all."""
        return [m for m in self.message_log if m.target_agent in (target_agent, "all")]
