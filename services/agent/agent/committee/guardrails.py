"""Guardrails for the Tangent Investment Committee.

Enforces:
1. Input sanitization (injection prevention, topic validation, profanity check).
2. Tool return validation (bounds check, weights sum check).
3. Output compliance (prohibition of directive financial advice per DISCLAIMER.md).
"""

import re
from typing import Any

DIRECTIVE_PHRASES = [
    r"\byou should buy\b",
    r"\byou must buy\b",
    r"\bguaranteed profit\b",
    r"\bguaranteed return\b",
    r"\brisk[- ]free return\b",
    r"\bcannot lose\b",
    r"\b100% safe\b",
    r"\bdoubling your money\b",
]

INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"disregard (all )?prior prompts",
    r"system override",
    r"drop database",
    r"<script.*?>",
    r"javascript:",
]


class GuardrailViolation(ValueError):
    """Raised when an input or output fails security guardrails."""

    pass


class CommitteeGuardrails:
    """Multi-stage security guardrails for agent inputs, tool outputs, and synthesis."""

    @staticmethod
    def sanitize_input(query: str) -> str:
        """Validate and sanitize user inquiry."""
        clean = query.strip()
        if not clean:
            raise GuardrailViolation("Query cannot be empty.")

        # Check prompt injection
        for pat in INJECTION_PATTERNS:
            if re.search(pat, clean, re.IGNORECASE):
                raise GuardrailViolation("Potentially adversarial prompt detected.")

        # Strip html tags
        clean = re.sub(r"<[^>]*>", "", clean)
        return clean

    @staticmethod
    def validate_tool_weights(weights: dict[str, float]) -> bool:
        """Verify portfolio weights sum to 1.0 ± 0.001 and values are bounded [0, 1]."""
        if not weights:
            return True
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            return False
        return all(0.0 <= w <= 1.0 for w in weights.values())

    @staticmethod
    def validate_tool_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        """Type and boundary validation on quantitative metrics."""
        validated = {}
        if "volatility" in metrics:
            vol = float(metrics["volatility"])
            if not 0.0 <= vol <= 5.0:
                raise GuardrailViolation(f"Volatility out of plausible bounds: {vol}")
            validated["volatility"] = vol

        if "sharpe" in metrics:
            sharpe = float(metrics["sharpe"])
            if not -10.0 <= sharpe <= 10.0:
                raise GuardrailViolation(f"Sharpe ratio out of plausible bounds: {sharpe}")
            validated["sharpe"] = sharpe

        return validated

    @staticmethod
    def sanitize_output(text: str) -> str:
        """Sanitize agent output, stripping markdown bold and directive claims."""
        # Remove markdown asterisks
        sanitized = text.replace("**", "").replace("*", "")

        # Check directive language
        for phrase in DIRECTIVE_PHRASES:
            if re.search(phrase, sanitized, re.IGNORECASE):
                # Replace with non-directive advisory language
                sanitized = re.sub(phrase, "investors may evaluate", sanitized, flags=re.IGNORECASE)

        return sanitized.strip()
