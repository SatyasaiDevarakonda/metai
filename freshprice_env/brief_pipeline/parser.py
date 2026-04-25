"""Parses raw LLM text output into a structured OperatingBrief.

Uses regex for section extraction — never uses LLM to parse LLM output.
Robust to minor formatting variations (extra whitespace, case differences).
Never raises on parse failure — returns ParseResult with success=False.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from freshprice_env.enums import BriefConfidence, BriefEngineType

# ---------------------------------------------------------------------------
# Section extraction patterns (all used with re.DOTALL | re.IGNORECASE)
# ---------------------------------------------------------------------------
_RE_FLAGS = re.DOTALL | re.IGNORECASE

_SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "situation": re.compile(
        r"(?:#+\s*)?SITUATION\s*:?\s*(.*?)(?=(?:#+\s*)?(?:SIGNAL\s*ANALYSIS|VIABILITY\s*CHECK|RECOMMENDATION|DIRECTIVE|CONFIDENCE)|$)",
        _RE_FLAGS,
    ),
    "signal_analysis": re.compile(
        r"(?:#+\s*)?SIGNAL\s*ANALYSIS\s*:?\s*(.*?)(?=(?:#+\s*)?(?:VIABILITY\s*CHECK|RECOMMENDATION|DIRECTIVE|CONFIDENCE)|$)",
        _RE_FLAGS,
    ),
    "viability_check": re.compile(
        r"(?:#+\s*)?VIABILITY\s*CHECK\s*:?\s*(.*?)(?=(?:#+\s*)?(?:RECOMMENDATION|DIRECTIVE|CONFIDENCE)|$)",
        _RE_FLAGS,
    ),
    "recommendation": re.compile(
        r"(?:#+\s*)?RECOMMENDATION\s*:?\s*(.*?)(?=(?:#+\s*)?(?:DIRECTIVE|CONFIDENCE)|$)",
        _RE_FLAGS,
    ),
    "directive": re.compile(
        r"(?:#+\s*)?DIRECTIVE\s*:?\s*(.*?)(?=(?:#+\s*)?CONFIDENCE|$)",
        _RE_FLAGS,
    ),
    "confidence": re.compile(
        r"(?:#+\s*)?CONFIDENCE\s*:?\s*(HIGH|MEDIUM|LOW)",
        re.IGNORECASE,
    ),
}

_REQUIRED_SECTIONS: list[str] = ["situation", "recommendation", "directive", "confidence"]

_CONFIDENCE_MAP: dict[str, BriefConfidence] = {
    "HIGH": BriefConfidence.HIGH,
    "MEDIUM": BriefConfidence.MEDIUM,
    "LOW": BriefConfidence.LOW,
}


@dataclass
class ParseResult:
    """Result of parsing raw LLM output."""

    success: bool
    brief: dict | None  # Structured brief fields, not the domain entity
    raw_text: str
    failure_reason: str | None  # None if success


class BriefParser:
    """Parses raw LLM text into structured Operating Brief sections."""

    @staticmethod
    def parse(raw_text: str, engine_type: BriefEngineType) -> ParseResult:
        """Extract all 6 sections from raw_text.

        Returns ParseResult with success=True and brief dict on success,
        or success=False with failure_reason on failure.
        """
        extracted: dict[str, str] = {}

        # Extract each section
        for section_name, pattern in _SECTION_PATTERNS.items():
            match = pattern.search(raw_text)
            if match:
                extracted[section_name] = match.group(1).strip()
            else:
                extracted[section_name] = ""

        # Check required sections
        for section in _REQUIRED_SECTIONS:
            value = extracted.get(section, "")
            if not value:
                return ParseResult(
                    success=False,
                    brief=None,
                    raw_text=raw_text,
                    failure_reason=f"MISSING_SECTION_{section.upper()}",
                )

        # Parse DIRECTIVE JSON
        raw_directive = extracted["directive"]
        directive_dict = BriefParser._extract_directive_json(raw_directive)
        if directive_dict is None:
            return ParseResult(
                success=False,
                brief=None,
                raw_text=raw_text,
                failure_reason="DIRECTIVE_JSON_INVALID",
            )

        # Validate directive has correct engine and actions
        if not BriefParser.validate_directive_schema(directive_dict, engine_type):
            return ParseResult(
                success=False,
                brief=None,
                raw_text=raw_text,
                failure_reason="DIRECTIVE_SCHEMA_INVALID",
            )

        # Parse CONFIDENCE
        confidence_raw = extracted["confidence"].strip().upper()
        confidence = _CONFIDENCE_MAP.get(confidence_raw)
        if confidence is None:
            return ParseResult(
                success=False,
                brief=None,
                raw_text=raw_text,
                failure_reason="INVALID_CONFIDENCE_VALUE",
            )

        # Handle optional sections — "N/A" treated as None
        signal_analysis = extracted.get("signal_analysis", "")
        if not signal_analysis or signal_analysis.upper().strip() in ("N/A", "NA", "NONE", ""):
            signal_analysis_out = None
        else:
            signal_analysis_out = signal_analysis

        viability_raw = extracted.get("viability_check", "")
        if not viability_raw or viability_raw.upper().strip() in ("N/A", "NA", "NONE", ""):
            viability_check_out = None
        else:
            viability_check_out = _parse_viability_text(viability_raw)

        brief = {
            "engine_type": engine_type,
            "situation": extracted["situation"],
            "signal_analysis": signal_analysis_out,
            "viability_check": viability_check_out,
            "recommendation": extracted["recommendation"],
            "directive": directive_dict,
            "confidence": confidence,
        }

        return ParseResult(
            success=True,
            brief=brief,
            raw_text=raw_text,
            failure_reason=None,
        )

    @staticmethod
    def _extract_directive_json(raw_directive: str) -> dict | None:
        """Extract JSON object from the directive section.

        Handles common LLM formatting issues:
          - JSON wrapped in ```json ... ``` code fences
          - Trailing commas
          - Single quotes instead of double quotes
        """
        text = raw_directive.strip()

        # Strip code fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Try direct parse first
        parsed = _try_json_parse(text)
        if parsed is not None:
            return parsed

        # Try extracting just the JSON object from surrounding text
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            parsed = _try_json_parse(brace_match.group(0))
            if parsed is not None:
                return parsed

        # Try fixing single quotes
        fixed = text.replace("'", '"')
        parsed = _try_json_parse(fixed)
        if parsed is not None:
            return parsed

        # Try removing trailing commas before } or ]
        fixed = re.sub(r",\s*([}\]])", r"\1", text)
        parsed = _try_json_parse(fixed)
        if parsed is not None:
            return parsed

        # All attempts failed
        return None

    @staticmethod
    def validate_directive_schema(
        directive: dict, engine_type: BriefEngineType,
    ) -> bool:
        """Validate directive matches the expected schema for its engine type.

        Does not raise.
        """
        # Must have "engine" matching engine_type
        engine_val = directive.get("engine", "").upper()
        if engine_val != engine_type.value:
            return False

        # Must have "actions" as a list
        actions = directive.get("actions")
        if not isinstance(actions, list):
            return False

        if not actions:
            return True  # Empty actions list is valid (no-op brief)

        if engine_type == BriefEngineType.PRICING:
            return all(_validate_pricing_action(a) for a in actions)
        if engine_type == BriefEngineType.FARMER:
            return all(_validate_farmer_action(a) for a in actions)
        if engine_type == BriefEngineType.TREND:
            return all(_validate_trend_action(a) for a in actions)

        return False


# ---------------------------------------------------------------------------
# Directive action validators
# ---------------------------------------------------------------------------

def _validate_pricing_action(action: dict) -> bool:
    if "batch_id" not in action:
        return False
    # Blinkit-style LIQUIDATE action ({"action": "LIQUIDATE", "batch_id": ...,
    # "channel": "B2B"}) takes a different shape from a standard pricing
    # action: it has no price_multiplier because it removes the batch
    # entirely. The LiquidationEngine validates urgency / anti-hack
    # downstream, so the parser only needs to confirm the shape is a
    # known one.
    if str(action.get("action", "")).upper() == "LIQUIDATE":
        return True
    pm = action.get("price_multiplier")
    if pm is None or not isinstance(pm, (int, float)):
        return False
    if pm < 0.25 or pm > 1.0:
        return False
    return True


def _validate_farmer_action(action: dict) -> bool:
    if "offer_id" not in action:
        return False
    decision = action.get("decision", "").upper()
    if decision not in ("ACCEPT", "COUNTER", "DECLINE"):
        return False
    if decision == "COUNTER":
        cp = action.get("counter_price")
        if cp is None or not isinstance(cp, (int, float)):
            return False
    return True


def _validate_trend_action(action: dict) -> bool:
    if "category" not in action:
        return False
    decision = action.get("decision", "").upper()
    if decision not in ("APPROVE", "DECLINE"):
        return False
    if decision == "APPROVE":
        qty = action.get("order_quantity_kg")
        if qty is None or not isinstance(qty, (int, float)):
            return False
        if qty <= 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_json_parse(text: str) -> dict | None:
    """Attempt json.loads, return dict or None."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _parse_viability_text(text: str) -> dict:
    """Parse viability check free text into a structured dict.

    Extracts lines like "Shelf life: PASS — adequate clearance window"
    into {"shelf_life": {"outcome": "PASS", "reason": "adequate clearance window"}}
    """
    factors: dict[str, dict[str, str]] = {}

    # Match patterns like "Factor name: PASS/FLAG/FAIL — reason" or "Factor name: PASS/FLAG/FAIL - reason"
    factor_pattern = re.compile(
        r"([A-Za-z_\s]+?)\s*:\s*(PASS|FLAG|FAIL)\s*[-—]\s*(.*?)(?:\n|$)",
        re.IGNORECASE,
    )

    for match in factor_pattern.finditer(text):
        factor_name = match.group(1).strip().lower().replace(" ", "_")
        outcome = match.group(2).strip().upper()
        reason = match.group(3).strip()
        factors[factor_name] = {"outcome": outcome, "reason": reason}

    if not factors:
        # Fallback: return the raw text as a single entry
        return {"raw": text}

    return factors
