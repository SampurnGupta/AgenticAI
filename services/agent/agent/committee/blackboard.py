"""Hover Blackboard Architecture for concurrent multi-agent context sharing."""

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MessagingTopology(str, Enum):
    DIRECT_SEQUENTIAL = "direct_sequential"
    PUB_SUB = "pub_sub"
    HOVER_BLACKBOARD = "hover_blackboard"


class BlackboardState(BaseModel):
    """Centralized shared state store across all committee agents."""

    session_id: str
    query: str
    investor_profile: str = "Moderate"
    horizon: int = 10
    capital: float = 1000000.0
    portfolio_context: dict[str, Any] = Field(default_factory=dict)
    evidence_items: list[dict[str, Any]] = Field(default_factory=list)
    theses: dict[str, dict[str, Any]] = Field(default_factory=dict)
    critiques: list[dict[str, Any]] = Field(default_factory=list)
    consensus_reached: bool = False
    verdict: Optional[dict[str, Any]] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HoverBlackboard:
    """Thread-safe Blackboard coordinating ADK and CrewAI agents."""

    def __init__(
        self,
        state: BlackboardState,
        topology: MessagingTopology = MessagingTopology.HOVER_BLACKBOARD,
    ):
        self._state = state
        self._topology = topology
        self._lock = asyncio.Lock()
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    @property
    def topology(self) -> MessagingTopology:
        return self._topology

    async def get_state(self) -> BlackboardState:
        async with self._lock:
            return self._state.model_copy(deep=True)

    async def post_thesis(self, agent_id: str, thesis_data: dict[str, Any]) -> None:
        async with self._lock:
            self._state.theses[agent_id] = {
                **thesis_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            # If in pub-sub mode, notify subscribers
            for q in self._subscribers:
                await q.put({"type": "thesis", "agent_id": agent_id, "data": thesis_data})

    async def post_critique(self, critique_data: dict[str, Any]) -> None:
        async with self._lock:
            self._state.critiques.append(
                {
                    **critique_data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            for q in self._subscribers:
                await q.put({"type": "critique", "data": critique_data})

    async def set_consensus(self, verdict_data: dict[str, Any]) -> None:
        async with self._lock:
            self._state.verdict = verdict_data
            self._state.consensus_reached = True
            for q in self._subscribers:
                await q.put({"type": "verdict", "data": verdict_data})

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a queue for pub-sub topology events."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.append(q)
        return q
