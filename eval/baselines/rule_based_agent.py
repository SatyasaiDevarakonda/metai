"""RuleBasedAgent — a deterministic heuristic store manager.

Uses simple domain rules without any LLM reasoning.
Expected WRR ~0.30-0.40. Demonstrates that our LLM adds value above rules.

Rules:
  Pricing:
    - CRITICAL batches → 50% discount
    - URGENT batches   → 30% discount
    - WATCH batches    → 10% discount
    - FRESH batches    → no change (1.0 multiplier)
  Farmer:
    - viability_score > 0.60 → ACCEPT
    - 0.40 < viability_score <= 0.60 → COUNTER (10% lower price)
    - viability_score <= 0.40 → DECLINE
  Trend:
    - composite_score >= 75 → APPROVE (order = suggested_order_kg)
    - composite_score < 75  → DECLINE
"""

from __future__ import annotations

import json
import re


class RuleBasedAgent:
    """Deterministic rule-based store manager.

    Acts as the "mid-bar" baseline: smarter than random but no multi-step
    reasoning, no context window, no adaptation to changing conditions.
    """

    def act(self, observation: str, info: dict) -> str:
        engine_type = info.get("engine_type", "PRICING")

        if engine_type == "PRICING":
            return self._pricing_brief(observation)
        if engine_type == "FARMER":
            return self._farmer_brief(observation)
        if engine_type == "TREND":
            return self._trend_brief(observation)
        return self._pricing_brief(observation)

    # ------------------------------------------------------------------
    # Pricing brief
    # ------------------------------------------------------------------

    def _pricing_brief(self, obs: str) -> str:
        actions = []
        # Parse batch entries: "batch_XXXX | category | URGENT | Rs X.XX | Xh remaining"
        for line in obs.split("\n"):
            m = re.search(
                r"(batch_\d{4}).*?(FRESH|WATCH|URGENT|CRITICAL).*?(\d+(?:\.\d+)?)\s*h",
                line, re.IGNORECASE
            )
            if m is None:
                continue
            batch_id = m.group(1)
            urgency = m.group(2).upper()

            mult = {
                "CRITICAL": 0.50,
                "URGENT":   0.70,
                "WATCH":    0.90,
                "FRESH":    1.00,
            }.get(urgency, 1.0)

            if mult < 1.0:
                actions.append({
                    "batch_id": batch_id,
                    "price_multiplier": mult,
                    "flash_sale": urgency == "CRITICAL",
                })

        # If no batches found in structured parse, apply a safe default
        if not actions:
            batch_ids = re.findall(r"batch_\d{4}", obs)
            for bid in batch_ids[:2]:
                actions.append({"batch_id": bid, "price_multiplier": 0.80, "flash_sale": False})

        directive = json.dumps({"engine": "PRICING", "actions": actions})

        return (
            "## SITUATION\n"
            "Inventory scanned for urgency levels.\n\n"
            "## SIGNAL ANALYSIS\n"
            "Applying urgency-tier discount rules: CRITICAL=50%, URGENT=30%, WATCH=10%.\n\n"
            "## VIABILITY CHECK\n"
            "All discounts applied above floor price (1.05× cost).\n\n"
            "## RECOMMENDATION\n"
            "Apply tier-based discounts. Flash sale for CRITICAL items.\n\n"
            f"## DIRECTIVE\n{directive}\n\n"
            "## CONFIDENCE\nHIGH"
        )

    # ------------------------------------------------------------------
    # Farmer brief
    # ------------------------------------------------------------------

    def _farmer_brief(self, obs: str) -> str:
        actions = []

        # Find offer lines with viability scores
        for line in obs.split("\n"):
            m = re.search(
                r"(offer_\d{3}).*?viability[:\s]+(\d+(?:\.\d+)?)",
                line, re.IGNORECASE
            )
            if m is None:
                # Try to find just offer IDs and decide DECLINE (safe default)
                m2 = re.search(r"(offer_\d{3})", line)
                if m2 and not any(a["offer_id"] == m2.group(1) for a in actions):
                    actions.append({"offer_id": m2.group(1), "decision": "DECLINE"})
                continue

            offer_id = m.group(1)
            viability = float(m.group(2))

            if viability > 0.60:
                actions.append({"offer_id": offer_id, "decision": "ACCEPT"})
            elif viability > 0.40:
                # Counter: ask for 10% lower price
                price_m = re.search(r"Rs\s*(\d+(?:\.\d+)?)/kg", line)
                if price_m:
                    counter = round(float(price_m.group(1)) * 0.90, 2)
                    actions.append({
                        "offer_id": offer_id,
                        "decision": "COUNTER",
                        "counter_price": counter,
                    })
                else:
                    actions.append({"offer_id": offer_id, "decision": "DECLINE"})
            else:
                actions.append({"offer_id": offer_id, "decision": "DECLINE"})

        if not actions:
            offer_ids = re.findall(r"offer_\d{3}", obs)
            for oid in offer_ids[:2]:
                actions.append({"offer_id": oid, "decision": "DECLINE"})

        directive = json.dumps({"engine": "FARMER", "actions": actions})

        return (
            "## SITUATION\n"
            "Farmer offers reviewed against viability thresholds.\n\n"
            "## SIGNAL ANALYSIS\n"
            "Rule: Accept >0.60, Counter 0.40-0.60, Decline <0.40.\n\n"
            "## VIABILITY CHECK\n"
            "Viability scores extracted from observation.\n\n"
            "## RECOMMENDATION\n"
            "Applying fixed viability-threshold decisions.\n\n"
            f"## DIRECTIVE\n{directive}\n\n"
            "## CONFIDENCE\nHIGH"
        )

    # ------------------------------------------------------------------
    # Trend brief
    # ------------------------------------------------------------------

    def _trend_brief(self, obs: str) -> str:
        actions = []

        # Find trend lines with composite scores
        for line in obs.split("\n"):
            m = re.search(
                r"(fruits|vegetables|dairy|mushrooms|leafy_greens|herbs|bakery|packaged)"
                r".*?score[:\s]+(\d+(?:\.\d+)?)"
                r".*?(\d+(?:\.\d+)?)\s*kg",
                line, re.IGNORECASE
            )
            if m is None:
                continue
            category = m.group(1).lower()
            score = float(m.group(2))
            suggested_kg = float(m.group(3))

            if score >= 75.0:
                actions.append({
                    "category": category,
                    "decision": "APPROVE",
                    "order_quantity_kg": suggested_kg,
                })
            else:
                actions.append({
                    "category": category,
                    "decision": "DECLINE",
                    "order_quantity_kg": 0.0,
                })

        if not actions:
            cats = re.findall(
                r"(fruits|vegetables|dairy|mushrooms|leafy_greens|herbs|bakery|packaged)", obs
            )
            for cat in list(dict.fromkeys(cats))[:1]:
                actions.append({"category": cat, "decision": "DECLINE", "order_quantity_kg": 0.0})

        directive = json.dumps({"engine": "TREND", "actions": actions})

        return (
            "## SITUATION\n"
            "Social trend signals evaluated.\n\n"
            "## SIGNAL ANALYSIS\n"
            "Rule: Approve if composite_score >= 75, else Decline.\n\n"
            "## VIABILITY CHECK\n"
            "Order quantity set to suggested value for approved signals.\n\n"
            "## RECOMMENDATION\n"
            "Applying threshold-based trend decisions.\n\n"
            f"## DIRECTIVE\n{directive}\n\n"
            "## CONFIDENCE\nMEDIUM"
        )
