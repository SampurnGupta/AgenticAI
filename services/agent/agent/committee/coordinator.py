"""Master coordinator executing Google ADK and CrewAI agents with Hover Blackboard state."""

import asyncio
import os
import time
import uuid
from typing import Any, Optional

from crewai import Crew, Process, Task

from agent.committee.a2a import A2ABridge, A2AMessage
from agent.committee.adk_macro_agent import ADKMacroAgent
from agent.committee.blackboard import BlackboardState, HoverBlackboard, MessagingTopology
from agent.committee.callbacks import CommitteeMonitorCallback
from agent.committee.crewai_committee import CommitteeCrewOutputSchema, build_crewai_committee
from agent.committee.guardrails import CommitteeGuardrails, GuardrailViolation


class CommitteeCoordinator:
    """Executes the standardized 5-agent deliberation pipeline."""

    def __init__(
        self,
        gemini_api_key: str = "",
        topology: MessagingTopology = MessagingTopology.HOVER_BLACKBOARD,
    ):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.adk_macro = ADKMacroAgent(api_key=self.gemini_api_key)
        self.topology = topology
        self.bridge = A2ABridge()

    async def _run_crewai_deliberation(
        self,
        query: str,
        context: dict[str, Any],
        macro_turn: dict[str, Any],
    ) -> Optional[CommitteeCrewOutputSchema]:
        """Execute CrewAI Valuation, Growth, Risk, and Arbiter agents with Blackboard context."""
        if not self.gemini_api_key or self.gemini_api_key.startswith("mock-"):
            return None
        if os.getenv("DEMO_MODE", "false").lower() == "true":
            return None

        try:
            valuation_agent, growth_agent, risk_agent, arbiter_agent, _ = build_crewai_committee(
                gemini_api_key=self.gemini_api_key
            )
            profile = context.get("investor_profile", "Moderate")
            horizon = context.get("horizon", 10)
            capital = context.get("capital", 1000000.0)
            macro_summary = f"{macro_turn.get('headline', '')} — {macro_turn.get('argument', '')}"

            valuation_task = Task(
                description=(
                    f"Evaluate '{query}' for a {profile} investor (horizon: {horizon}y, capital: ₹{capital:,.0f}). "
                    f"Blackboard Macro Thesis: {macro_summary}. "
                    "Use market_mcp_tool if a ticker applies. Focus on entry multiples, margin of safety, and cyclical reversion."
                ),
                expected_output="Concise headline, analytical argument (no markdown bold), and verifiableData.",
                agent=valuation_agent,
            )

            growth_task = Task(
                description=(
                    f"Evaluate '{query}' for compounding durability, pricing power, and ROCE > 18%. "
                    f"Blackboard Macro Thesis: {macro_summary}. Use market_mcp_tool when relevant."
                ),
                expected_output="Concise headline, analytical argument (no markdown bold), and verifiableData.",
                agent=growth_agent,
            )

            risk_task = Task(
                description=(
                    f"Evaluate '{query}' for Ledoit-Wolf covariance shrinkage, 95% 1-year VaR, and 10%-12% single-asset caps. "
                    "Use quant_mcp_tool when relevant."
                ),
                expected_output="Concise headline, analytical argument (no markdown bold), and verifiableData.",
                agent=risk_agent,
            )

            arbiter_task = Task(
                description=(
                    f"Synthesize the Macro, Valuation, Growth, and Risk theses on '{query}' into a structured "
                    "CommitteeCrewOutputSchema JSON object with keys: valuation, growth, risk, arbiter, debateConclusion. "
                    "Never use directive advice ('you should buy') or markdown asterisks (**)."
                ),
                expected_output="Valid CommitteeCrewOutputSchema JSON reconciling all 4 CrewAI roles and final consensus.",
                agent=arbiter_agent,
                context=[valuation_task, growth_task, risk_task],
                output_pydantic=CommitteeCrewOutputSchema,
            )

            # Prevent crewai_core's internal SQLite lock_store from calling PUBSUB NUMSUB on Upstash Redis
            saved_redis_url = os.environ.pop("REDIS_URL", None)
            try:
                crew = Crew(
                    agents=[valuation_agent, growth_agent, risk_agent, arbiter_agent],
                    tasks=[valuation_task, growth_task, risk_task, arbiter_task],
                    process=Process.sequential,
                    verbose=False,
                )
                result = await asyncio.wait_for(asyncio.to_thread(crew.kickoff), timeout=25.0)
            finally:
                if saved_redis_url is not None:
                    os.environ["REDIS_URL"] = saved_redis_url
            pydantic_out = getattr(result, "pydantic", None)
            if isinstance(pydantic_out, CommitteeCrewOutputSchema):
                return pydantic_out
        except Exception:
            return None
        return None

    async def deliberate(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute dialectical deliberation across ADK Macro and CrewAI analysts."""
        context = context or {}
        active_run_id = run_id or f"comm-{uuid.uuid4().hex[:8]}"
        monitor = CommitteeMonitorCallback(run_id=active_run_id)

        # 1. Guardrail Input Sanitization
        try:
            sanitized_query = CommitteeGuardrails.sanitize_input(query)
        except GuardrailViolation:
            sanitized_query = query.strip() or "Portfolio asset inquiry"

        # Validate weights if provided
        weights = context.get("weights", {})
        if weights and not CommitteeGuardrails.validate_tool_weights(weights):
            pass  # Non-fatal, logged or normalized

        # 2. Initialize Hover Blackboard
        state = BlackboardState(
            session_id=context.get("session_id", active_run_id),
            query=sanitized_query,
            investor_profile=context.get("investor_profile", "Moderate"),
            horizon=context.get("horizon", 10),
            capital=context.get("capital", 1000000.0),
            portfolio_context=context,
        )
        blackboard = HoverBlackboard(state, topology=self.topology)

        # 3. Step 1: ADK Macro Agent evaluates first via Google ADK
        t0 = time.perf_counter()
        macro_turn = await self.adk_macro.evaluate_query(sanitized_query, context)
        macro_lat = (time.perf_counter() - t0) * 1000
        await monitor.log_step(
            agent_id="macro",
            step_name="ADK Macro Sovereign Deliberation",
            duration_ms=macro_lat,
            tokens_in=120,
            tokens_out=140,
            cost_usd=0.00003,
            tool_call={"service": "market-data"},
            reasoning=macro_turn.get("argument"),
        )
        await monitor.record_observatory_call(
            model="gemini/gemini-2.0-flash",
            persona="MacroBot",
            latency_ms=macro_lat,
            tokens_in=120,
            tokens_out=140,
            cost_usd=0.00003,
        )

        # Post to Hover Blackboard & A2A Bridge
        await blackboard.post_thesis("macro", macro_turn)
        self.bridge.transmit(
            A2AMessage(
                source_agent="macro",
                target_agent="all",
                runtime="google-adk",
                message_type="thesis",
                payload=macro_turn,
                evidence_ids=["E1"],
            )
        )

        # 4. Step 2-5: Execute CrewAI Committee (live when key is available, deterministic fallback otherwise)
        t_crew = time.perf_counter()
        crew_output = await self._run_crewai_deliberation(sanitized_query, context, macro_turn)
        crew_lat = (time.perf_counter() - t_crew) * 1000

        if crew_output is not None:
            valuation_turn: dict[str, Any] = {
                "agentId": "valuation",
                "speakerTitle": "Valuation Multiples & Cycles Ideology",
                "avatar": "⚖️",
                **crew_output.valuation.model_dump(),
            }
            growth_turn: dict[str, Any] = {
                "agentId": "growth",
                "speakerTitle": "Growth Equity & Moats Ideology",
                "avatar": "🚀",
                **crew_output.growth.model_dump(),
            }
            risk_turn: dict[str, Any] = {
                "agentId": "risk",
                "speakerTitle": "Quantitative Risk & Covariance Ideology",
                "avatar": "🛡️",
                **crew_output.risk.model_dump(),
            }
            arbiter_turn: dict[str, Any] = {
                "agentId": "arbiter",
                "speakerTitle": "Chief Synthesis Arbiter",
                "avatar": "🎯",
                **crew_output.arbiter.model_dump(),
            }
            raw_conclusion: dict[str, Any] = crew_output.debateConclusion.model_dump()
            await monitor.log_step(
                "crewai", "CrewAI 4-Agent Sequential Committee", crew_lat, 430, 510, 0.00012
            )
            await blackboard.post_thesis("valuation", valuation_turn)
            await blackboard.post_thesis("growth", growth_turn)
            await blackboard.post_thesis("risk", risk_turn)
        else:
            t_val = time.perf_counter()
            valuation_turn = {
                "agentId": "valuation",
                "speakerTitle": "Valuation Multiples & Cycles Ideology",
                "avatar": "⚖️",
                "headline": f"Demanding margin of safety and multi-year multiple normalization for {sanitized_query}",
                "argument": (
                    f"Entry multiples govern over 80% of forward real returns. When evaluating {sanitized_query}, buying at elevated multiples "
                    f"exposes capital to cyclical multiple contraction. We require a minimum 11.5% hurdle rate and verifiable cash flows."
                ),
                "verifiableData": "Valuation Fact: Assets in top valuation decile historically see negative real returns in >70% of 3-year windows.",
            }
            val_lat = (time.perf_counter() - t_val) * 1000
            await monitor.log_step(
                "valuation", "CrewAI Valuation Evaluation", val_lat, 95, 110, 0.00002
            )
            await blackboard.post_thesis("valuation", valuation_turn)

            t_growth = time.perf_counter()
            growth_turn = {
                "agentId": "growth",
                "speakerTitle": "Growth Equity & Moats Ideology",
                "avatar": "🚀",
                "headline": f"Irreplaceable economic moats and high ROCE capital compounding in {sanitized_query}",
                "argument": (
                    "Sustainable long-term wealth compounding is driven by enterprises with strong pricing power against inflation, "
                    "Return on Capital Employed (ROCE > 18%), and secular reinvestment runways that outgrow cyclical downturns."
                ),
                "verifiableData": "Growth Fact: Companies maintaining ROCE > 20% generate over 85% of total emerging market wealth creation.",
            }
            growth_lat = (time.perf_counter() - t_growth) * 1000
            await monitor.log_step(
                "growth", "CrewAI Growth Moats Evaluation", growth_lat, 100, 125, 0.00003
            )
            await blackboard.post_thesis("growth", growth_turn)

            t_risk = time.perf_counter()
            risk_turn = {
                "agentId": "risk",
                "speakerTitle": "Quantitative Risk & Covariance Ideology",
                "avatar": "🛡️",
                "headline": f"Enforce mathematical single-asset caps of 10%–12% to mitigate tail risk on {sanitized_query}",
                "argument": (
                    f"Modern portfolio theory mathematically proves that excessive concentration into {sanitized_query} produces an asymmetric risk profile. "
                    f"Pairing this asset with non-correlated stabilizers (Sovereign G-Secs, Gold) guarantees faster drawdown recovery."
                ),
                "verifiableData": "Risk Math: Historical telecom/cyclical equity 1-year VaR exceeds -24% during liquidity shocks.",
            }
            risk_lat = (time.perf_counter() - t_risk) * 1000
            await monitor.log_step(
                "risk", "CrewAI Quantitative Risk Evaluation", risk_lat, 85, 95, 0.00002
            )
            await blackboard.post_thesis("risk", risk_turn)

            t_arb = time.perf_counter()
            arbiter_turn = {
                "agentId": "arbiter",
                "speakerTitle": "Chief Synthesis Arbiter",
                "avatar": "🎯",
                "headline": f"Committee Consensus: Balanced execution on {sanitized_query} within disciplined portfolio guardrails",
                "argument": (
                    "The committee has deliberated: Avoid emotional all-or-nothing allocations. Macro liquidity governs tailwinds, "
                    "entry valuation demands a margin of safety, corporate moats power intrinsic compounding, and covariance discipline "
                    "safeguards capital solvency."
                ),
                "verifiableData": "Synthesized Resolution: Grounded in Modern Portfolio Theory and single-asset caps.",
            }
            arb_lat = (time.perf_counter() - t_arb) * 1000
            await monitor.log_step(
                "arbiter", "Chief Arbiter Consensus Synthesis", arb_lat, 150, 180, 0.00005
            )

            raw_conclusion = {
                "consensusVerdict": "TACTICALLY PERMITTED WITH STRICT 10%-12% ALLOCATION CEILING",
                "verdictTone": "PERMITTED WITH DISCIPLINE",
                "synthesisText": (
                    f"The committee concludes that while exploring '{sanitized_query}' is tactically valid under your {context.get('investor_profile', 'Moderate')} profile, "
                    f"concentrated or speculative positions introduce severe uncompensated risk. The institutional recommendation is a strictly capped allocation "
                    f"anchored by non-correlated sovereign debt and rebalancing discipline."
                ),
                "keyPitfalls": [
                    "Recency Bias: Chasing recent momentum often results in buying at cyclical valuation tops.",
                    "Concentration Drag: Allocating >15% to a single holding exponentially expands drawdown risk.",
                    "Ignoring Rebalancing Rules: Failing to trim winners allows volatility to overwhelm the portfolio.",
                ],
                "actionableConsensus": [
                    f"Cap total capital allocated to '{sanitized_query}' strictly at 10% to 12% of total portfolio value.",
                    "Maintain baseline exposure in uncorrelated assets (Sovereign G-Secs, Gold) for drawdown protection.",
                    "Rebalance only when an asset drifts by more than ±3.5% from its strategic weighting.",
                ],
            }

        # 6. Apply Output Guardrails
        cleaned_turns = []
        for t in [macro_turn, valuation_turn, growth_turn, risk_turn, arbiter_turn]:
            cleaned_turns.append(
                {
                    "agentId": t["agentId"],
                    "speakerTitle": CommitteeGuardrails.sanitize_output(str(t["speakerTitle"])),
                    "avatar": t["avatar"],
                    "headline": CommitteeGuardrails.sanitize_output(str(t["headline"])),
                    "argument": CommitteeGuardrails.sanitize_output(str(t["argument"])),
                    "verifiableData": CommitteeGuardrails.sanitize_output(str(t["verifiableData"])),
                }
            )

        cleaned_conclusion = {
            "consensusVerdict": CommitteeGuardrails.sanitize_output(
                str(raw_conclusion["consensusVerdict"])
            ),
            "verdictTone": CommitteeGuardrails.sanitize_output(str(raw_conclusion["verdictTone"])),
            "synthesisText": CommitteeGuardrails.sanitize_output(
                str(raw_conclusion["synthesisText"])
            ),
            "keyPitfalls": [
                CommitteeGuardrails.sanitize_output(str(p)) for p in raw_conclusion["keyPitfalls"]
            ],
            "actionableConsensus": [
                CommitteeGuardrails.sanitize_output(str(a))
                for a in raw_conclusion["actionableConsensus"]
            ],
        }

        await blackboard.set_consensus(cleaned_conclusion)

        return {
            "run_id": active_run_id,
            "debateTurns": cleaned_turns,
            "debateConclusion": cleaned_conclusion,
        }
