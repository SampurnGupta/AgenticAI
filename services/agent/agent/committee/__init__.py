"""Tangent Standardized Multi-Agent Investment Committee package (CrewAI + Google ADK)."""

from agent.committee.blackboard import BlackboardState, HoverBlackboard
from agent.committee.coordinator import CommitteeCoordinator
from agent.committee.mcp_tools import market_mcp_tool, quant_mcp_tool

__all__ = [
    "BlackboardState",
    "HoverBlackboard",
    "CommitteeCoordinator",
    "quant_mcp_tool",
    "market_mcp_tool",
]
