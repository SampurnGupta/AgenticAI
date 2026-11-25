# Tangent Agent Service (`tangent-agent` / `tangent-committee-agent`)

Evidence-first AI Decision Studio & Standardized Investment Committee service powered by **Google ADK (`google-genai`)** and **CrewAI (`crewai`)**.

## Architecture

### 1. Standardized Investment Committee (`agent/committee/`)
- **Hover Blackboard (`blackboard.py`):** Thread-safe shared state (`BlackboardState`) coordinating concurrent context across ADK and CrewAI agents (supports Hover Blackboard, Direct Sequential, and Pub-Sub topologies).
- **Google ADK Macro Analyst (`adk_macro_agent.py`):** Evaluates sovereign yield curves, RBI/Fed policy spreads, and currency depreciation using `google-genai`.
- **CrewAI Analysts & Arbiter (`crewai_committee.py`):**
  - **Valuation Analyst:** Evaluates entry multiples, margin of safety, and cyclical reversion using `market_mcp_tool`.
  - **Growth Equity Analyst:** Assesses compounding, pricing power, and ROCE > 18% using `market_mcp_tool`.
  - **Quantitative Risk Analyst:** Enforces Ledoit-Wolf covariance constraints, 95% 1-year VaR, and single-asset caps using `quant_mcp_tool`.
  - **Chief Synthesis Arbiter:** Reconciles all inputs into a traceable `DebateConclusion`.
- **MCP Tool Adapters (`mcp_tools.py`):** Model Context Protocol adapters connecting agents to `quant:8002` (`QuantMCPTool`) and `market-data:8001` (`MarketDataMCPTool`).
- **A2A Protocol (`a2a.py`):** `A2AMessage` envelope and `A2ABridge` for cross-runtime interoperability between Google ADK and CrewAI.
- **Guardrails (`guardrails.py`):** Input prompt-injection sanitization, portfolio weight/metric bounds validation, and output filtering against directive advice (`DISCLAIMER.md`).
- **Callbacks & Observability (`callbacks.py`):** `CommitteeMonitorCallback` streaming step latencies, reasoning, and token/cost telemetry to Redis (`tangent:telemetry:committee:*`) and the LLM Observatory.

### 2. Decision Studio & Draww Co-Pilot (`orchestrator.py`)
- **Evidence-First:** Collects deterministic metrics (`[E1]`–`[E9]`) from `quant`, `market-data`, and `sentiment` before LLM execution.
- **Critic Agent (`critic.py`):** Verifies every claim against deterministic evidence and calculates a Groundedness Score.
- **SSE Streaming:** Real-time event streaming (`text/event-stream`) via `/api/v1/agent/runs/stream/{ticker}`.

## Key Endpoints
- `POST /api/v1/agent/committee/debate` — Run the 5-agent CrewAI + Google ADK Investment Committee deliberation.
- `POST /api/v1/agent/runs` — Trigger an evidence-first Decision Studio brief run.
- `GET /api/v1/agent/runs/{run_id}` — Fetch completed run details.
- `POST /api/v1/agent/chat` — Conversational Draww portfolio co-pilot.
- `GET /api/v1/agent/runs/stream/{ticker}` — Stream SSE progress events.
- `GET /health` & `GET /metrics` — Service health and committee metrics.
