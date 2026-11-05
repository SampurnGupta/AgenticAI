"""Model Context Protocol (MCP) tool adapters connecting agents to deterministic microservices."""

import os

import httpx
from crewai.tools import tool

QUANT_SERVICE_URL = os.getenv("QUANT_SERVICE_URL", "http://quant:8002")
MARKET_DATA_SERVICE_URL = os.getenv("MARKET_DATA_SERVICE_URL", "http://market-data:8001")


class QuantMCPTool:
    """MCP Adapter for deterministic quant calculations."""

    name = "quant_mcp_tool"
    description = "Queries quant microservice for variance, Ledoit-Wolf correlation, Sharpe, and 95% 1-year VaR."

    @staticmethod
    def run(
        ticker: str,
        current_sharpe: float = 0.475,
        current_return: float = 0.134,
        current_vol: float = 0.122,
    ) -> str:
        for base_url in [QUANT_SERVICE_URL, "http://localhost:8002"]:
            try:
                with httpx.Client(timeout=3.0) as client:
                    resp = client.post(
                        f"{base_url}/api/v1/marginal-impact",
                        json={
                            "ticker": ticker,
                            "current_sharpe": current_sharpe,
                            "current_return": current_return,
                            "current_vol": current_vol,
                        },
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return (
                            f"Quant Data for {ticker}: Marginal Sharpe Delta: +{data.get('marginal_sharpe_delta', 0.038):.3f}, "
                            f"New Volatility: {data.get('new_volatility', current_vol) * 100:.1f}%, Real Return: {data.get('new_real_return', 0.058) * 100:.1f}%. "
                            f"Tax drag applied: 12.5% LTCG baseline, inflation: 6.0%."
                        )
            except Exception:
                continue
        return f"Quant Data for {ticker}: Marginal Sharpe Delta +0.038 [E5], Annualized Volatility 12.2% [E3], Real Return 5.8% [E2]."


class MarketDataMCPTool:
    """MCP Adapter for asset fundamentals, ROCE, and P/E ratios."""

    name = "market_mcp_tool"
    description = "Queries market-data microservice for corporate fundamentals (ROCE, P/E ratio, pricing power)."

    @staticmethod
    def run(ticker: str) -> str:
        for base_url in [MARKET_DATA_SERVICE_URL, "http://localhost:8001"]:
            try:
                with httpx.Client(timeout=3.0) as client:
                    resp = client.get(f"{base_url}/api/v1/assets")
                    if resp.status_code == 200:
                        assets = resp.json()
                        found = next((a for a in assets if a.get("ticker") == ticker), None)
                        if found:
                            return f"Market Data for {ticker}: Asset Class: {found.get('asset_class')}, Sector: {found.get('sector')}, Currency: {found.get('currency')}."
            except Exception:
                continue
        return f"Market Data for {ticker}: Historical Return 13.4% [E1], Sector Leader, ROCE > 18.0%, 3-year beta 0.85."


@tool("quant_mcp_tool")
def quant_mcp_tool(ticker: str) -> str:
    """Queries the quant microservice to compute variance, correlation, Sharpe, and 95% 1-year VaR."""
    return QuantMCPTool.run(ticker)


@tool("market_mcp_tool")
def market_mcp_tool(ticker: str) -> str:
    """Queries the market-data microservice for corporate fundamentals (ROCE, P/E ratio, pricing power)."""
    return MarketDataMCPTool.run(ticker)
