"""CrewAI multi-agent committee for Valuation, Growth, Risk, and Synthesis Arbiter."""

import os

from crewai import LLM, Agent
from pydantic import BaseModel

from agent.committee.mcp_tools import market_mcp_tool, quant_mcp_tool


class CrewTurnSchema(BaseModel):
    headline: str
    argument: str
    verifiableData: str


class DebateConclusionSchema(BaseModel):
    consensusVerdict: str
    verdictTone: str
    synthesisText: str
    keyPitfalls: list[str]
    actionableConsensus: list[str]


class CommitteeCrewOutputSchema(BaseModel):
    valuation: CrewTurnSchema
    growth: CrewTurnSchema
    risk: CrewTurnSchema
    arbiter: CrewTurnSchema
    debateConclusion: DebateConclusionSchema


def build_crewai_committee(
    gemini_api_key: str = "",
) -> tuple[Agent, Agent, Agent, Agent, LLM]:
    """Construct the 4 CrewAI committee agents (Valuation, Growth, Risk, Arbiter)."""
    key = gemini_api_key or os.getenv("GEMINI_API_KEY", "mock-key")
    llm = LLM(
        model="gemini/gemini-flash-lite-latest",
        api_key=key,
        temperature=0.25,
    )

    # 1. Valuation Analyst
    valuation_agent = Agent(
        role="Valuation Multiples & Cycles Analyst",
        goal="Evaluate entry multiples (P/E, EV/EBITDA), margin of safety, and cyclical reversion risk.",
        backstory="An institutional value specialist demanding tangible margin of safety and normalized cash flows.",
        llm=llm,
        tools=[market_mcp_tool],
        verbose=False,
    )

    # 2. Growth Equity Analyst
    growth_agent = Agent(
        role="Growth Equity & Moats Analyst",
        goal="Assess corporate compounding, revenue reinvestment runways, pricing power, and ROCE > 18%.",
        backstory="A growth compounder focused on durable economic moats and compounding capital efficiency.",
        llm=llm,
        tools=[market_mcp_tool],
        verbose=False,
    )

    # 3. Quantitative Risk Analyst
    risk_agent = Agent(
        role="Quantitative Risk & Covariance Analyst",
        goal="Enforce Ledoit-Wolf covariance shrinkage constraints, 95% 1-year VaR limits, and single-asset caps.",
        backstory="A mathematical risk officer who enforces asset caps (10-15%) and non-correlated buffers.",
        llm=llm,
        tools=[quant_mcp_tool],
        verbose=False,
    )

    # 4. Chief Synthesis Arbiter
    arbiter_agent = Agent(
        role="Chief Synthesis Arbiter",
        goal="Synthesize conflicting ideologies into an actionable, traceable institutional consensus verdict.",
        backstory="The institutional referee presiding over Tangent's investment committee.",
        llm=llm,
        verbose=False,
    )

    return valuation_agent, growth_agent, risk_agent, arbiter_agent, llm
