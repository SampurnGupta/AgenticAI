"""Macro & Sovereign Rates Analyst powered exclusively by Google ADK / GenAI SDK."""

import json
import os
from typing import Any, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel


class MacroThesis(BaseModel):
    headline: str
    argument: str
    verifiableData: str


class ADKMacroAgent:
    """Google ADK agent evaluating sovereign yield curves, Fed/RBI policy spreads, and currency depreciation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.client: Optional[genai.Client] = None
        if self.api_key and not self.api_key.startswith("mock-"):
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    async def evaluate_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        system_instruction = (
            "You are Tangent's Macro & Sovereign Rates Analyst. "
            "Analyze portfolio inquiries strictly through central bank policy (RBI repo rate 6.50% vs US Fed funds 4.25%-4.50%), "
            "sovereign yield curves (GOI 10Y Benchmark G-Sec at ~6.85%), and structural USD/INR currency depreciation (~3.2%-3.5% annually). "
            "Never give directive advice ('you should buy'). Output must be plain text with NO markdown bold asterisks (**)."
        )

        user_content = (
            f"USER QUERY: {query}\n"
            f"INVESTOR CONTEXT: Profile {context.get('investor_profile', 'Moderate')}, "
            f"Horizon: {context.get('horizon', 10)} Years, Capital: ₹{context.get('capital', 1000000):,.0f}.\n"
            "Return a JSON object with keys: 'headline', 'argument', 'verifiableData'."
        )

        if self.client:
            try:
                response = self.client.models.generate_content(
                    model="gemini-flash-lite-latest",
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                if response.text:
                    parsed = json.loads(response.text)
                    return {
                        "agentId": "macro",
                        "speakerTitle": "Macro & Sovereign Rates Ideology",
                        "avatar": "🏛️",
                        "headline": str(parsed.get("headline", ""))
                        .replace("**", "")
                        .replace("*", "")
                        .strip(),
                        "argument": str(parsed.get("argument", ""))
                        .replace("**", "")
                        .replace("*", "")
                        .strip(),
                        "verifiableData": str(
                            parsed.get(
                                "verifiableData",
                                "Macro Fact: RBI Repo Rate 6.50%, GOI 10Y Yield 6.85% [E1].",
                            )
                        )
                        .replace("**", "")
                        .replace("*", "")
                        .strip(),
                    }
            except Exception:
                pass

        # Deterministic institutional fallback
        return {
            "agentId": "macro",
            "speakerTitle": "Macro & Sovereign Rates Ideology",
            "avatar": "🏛️",
            "headline": f"Evaluating sovereign interest rate cycles and monetary liquidity for {query}",
            "argument": (
                f"Analyzing this proposition through sovereign macroeconomic realities: The central bank monetary environment "
                f"(RBI repo rate 6.50% vs US Federal Reserve 4.25%-4.50%), domestic headline CPI inflation (~5.0%), and sovereign yield curve "
                f"dynamics set the hurdle rate. Any deployment into {query} must outpace the risk-free rate (~6.85% GOI 10Y) plus an appropriate risk premium."
            ),
            "verifiableData": "Macro Fact: Domestic institutional SIP liquidity (~₹23,000 Cr/month) supports equity floors against FPI outflows.",
        }
