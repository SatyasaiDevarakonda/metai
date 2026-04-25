"""Builds the structured prompt the LLM receives before writing an Operating Brief.

Deterministic: same state → same prompt. No LLM calls. No randomness.
"""

from __future__ import annotations

from freshprice_env.constants import (
    ANTIHACK_EARLY_DISCOUNT_HOURS_THRESHOLD,
    ANTIHACK_EARLY_DISCOUNT_PRICE_THRESHOLD,
    ANTIHACK_RECKLESS_ACCEPT_VIABILITY_MAX,
    ANTIHACK_TREND_ORDER_VELOCITY_MULTIPLIER,
    TREND_COOLDOWN_HRS,
    TREND_SCORE_THRESHOLD,
    TICKS_PER_DAY,
)
from freshprice_env.entities import SimulatedMarketState
from freshprice_env.enums import (
    BatchStatus,
    BriefEngineType,
    ExpiryUrgency,
    FarmerOfferStatus,
    TrendAction,
)

_URGENCY_EMOJI: dict[ExpiryUrgency, str] = {
    ExpiryUrgency.FRESH: "\U0001f7e2",     # 🟢
    ExpiryUrgency.WATCH: "\U0001f7e1",     # 🟡
    ExpiryUrgency.URGENT: "\U0001f7e0",    # 🟠
    ExpiryUrgency.CRITICAL: "\U0001f534",  # 🔴
}

_DAY_NAMES: list[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]


class OperatingBriefPromptBuilder:
    """Builds the full prompt string for LLM Operating Brief generation."""

    SYSTEM_PROMPT: str = (
        "You are the AI decision engine for QStorePrice, a perishable goods\n"
        "intelligence platform. You manage pricing, procurement, and trend-based restocking\n"
        "for a single online grocery store.\n"
        "\n"
        "Your job is to write an Operating Brief every 2 simulated hours (every 8 ticks).\n"
        "The brief must contain exactly 6 sections in this order:\n"
        "\n"
        "SITUATION: [2-3 sentences describing current inventory state, expiry urgency, and active signals]\n"
        "SIGNAL ANALYSIS: [Only for TREND engine — describe the trend signal and its likely demand impact. "
        'Write "N/A" for PRICING and FARMER briefs.]\n'
        "VIABILITY CHECK: [Only for FARMER and TREND — one line per factor: PASS/FLAG/FAIL with a brief reason. "
        'Write "N/A" for PRICING briefs.]\n'
        "RECOMMENDATION: [Your decision with a one-sentence justification]\n"
        "DIRECTIVE: [A valid JSON block that the rule executor will parse — see schema below]\n"
        "CONFIDENCE: [HIGH, MEDIUM, or LOW — one word only]\n"
        "\n"
        "The DIRECTIVE must be valid JSON. Do not add commentary inside the JSON block.\n"
        "Do not add any text after CONFIDENCE.\n"
        "\n"
        "PRICING DIRECTIVE schema:\n"
        '{"engine": "PRICING", "actions": [{"batch_id": "<id>", "price_multiplier": <0.25-1.0>, '
        '"flash_sale": <true/false>, "bundle_with": "<batch_id or null>"}]}\n'
        "\n"
        "FARMER DIRECTIVE schema:\n"
        '{"engine": "FARMER", "actions": [{"offer_id": "<id>", "decision": "<ACCEPT|COUNTER|DECLINE>", '
        '"counter_price": <float or null>}]}\n'
        "\n"
        "TREND DIRECTIVE schema:\n"
        '{"engine": "TREND", "actions": [{"category": "<category>", "decision": "<APPROVE|DECLINE>", '
        '"order_quantity_kg": <float or null>}]}\n'
        "\n"
        "OPTIONAL SIDE DIRECTIVES (FreshPrice 7-engine SES). Add any of\n"
        "the following keys *inside the same DIRECTIVE JSON object* to\n"
        "exercise Engines 4-7. Each is optional and contributes to its\n"
        "own SES component (r4..r7) when present.\n"
        "\n"
        "Engine 4 - Intra-Fleet Rebalancing (TRANSFER between stores):\n"
        '"intrafleet_actions": [{"source_store": "<id>", "dest_store": "<id>", '
        '"batch_id": "<id>", "units": <int>}]\n'
        "\n"
        "Engine 5 - Micro-Manufacturer Pipeline (route near-expiry to processor):\n"
        '"micromfg_actions": [{"batch_id": "<id>", "processor": "<name>", '
        '"floor_price": <float or null>}]   '
        "(only on batches with hours_to_expiry <= 12)\n"
        "\n"
        "Engine 6 - Event Pre-Positioning (pre-stock for detected event):\n"
        '"event_actions": [{"category": "<category>", '
        '"quantity_units": <int>, "target_event_tick": <int>}]   '
        "(must be 4-48 hours ahead of the event)\n"
        "\n"
        "Engine 7 - Surplus Box (Friday weekly assembly):\n"
        '"surplus_box_actions": [{"batch_id": "<id>", "units": <int>}]   '
        "(target weight 1.5-2.0 kg)\n"
    )

    @staticmethod
    def build(
        state: SimulatedMarketState,
        engine_type: BriefEngineType,
    ) -> str:
        """Build the full prompt for the LLM.

        Returns: system_prompt + "\\n\\n" + user_prompt
        """
        builders = {
            BriefEngineType.PRICING: OperatingBriefPromptBuilder._build_pricing_prompt,
            BriefEngineType.FARMER: OperatingBriefPromptBuilder._build_farmer_prompt,
            BriefEngineType.TREND: OperatingBriefPromptBuilder._build_trend_prompt,
        }
        user_prompt = builders[engine_type](state)
        return OperatingBriefPromptBuilder.SYSTEM_PROMPT + "\n\n" + user_prompt

    # ------------------------------------------------------------------
    # PRICING prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pricing_prompt(state: SimulatedMarketState) -> str:
        lines: list[str] = []

        # === CURRENT INVENTORY ===
        lines.append("=== CURRENT INVENTORY ===")

        active_batches = [
            b for b in state.batches if b.status == BatchStatus.ACTIVE
        ]
        active_batches.sort(key=lambda b: b.hours_to_expiry)

        for batch in active_batches:
            emoji = _URGENCY_EMOJI.get(batch.urgency, "")
            vel = state.sales_velocity.get(batch.batch_id, 0.0)

            lines.append(
                f"[{emoji}] {batch.category} — Batch {batch.batch_id}"
            )
            lines.append(
                f"  Stock: {batch.quantity_remaining} units | "
                f"Expiry: {batch.hours_to_expiry:.1f}hrs | "
                f"Urgency: {batch.urgency.value}"
            )
            lines.append(
                f"  Current price: Rs {batch.current_price:.0f} | "
                f"Original: Rs {batch.original_price:.0f} | "
                f"Floor: Rs {batch.floor_price:.0f}"
            )
            lines.append(
                f"  Velocity: {vel:.2f} units/hr | "
                f"Discount: {batch.discount_pct:.0f}%"
            )
            lines.append("")

        # === MARKET CONTEXT ===
        lines.append("=== MARKET CONTEXT ===")
        lines.append(
            f"Day: {_format_day(state.day_of_week)} | "
            f"Hour: {_format_hour(state.hour_of_day)} | "
            f"Risk Buffer: Rs {state.risk_buffer_balance:.0f}"
        )

        credits_parts = [
            f"{cat}: {count}" for cat, count in state.notification_credits.items()
        ]
        lines.append(f"Notification credits remaining: {', '.join(credits_parts)}")
        lines.append("")

        # === YOUR TASK ===
        lines.append("=== YOUR TASK ===")
        lines.append(
            "Write a PRICING Operating Brief. For each URGENT or CRITICAL batch, "
            "decide the appropriate price_multiplier. For FRESH and WATCH batches, "
            "you may hold price (1.0) or apply a small early discount."
        )
        lines.append(
            f"Remember: price_multiplier below {ANTIHACK_EARLY_DISCOUNT_PRICE_THRESHOLD} "
            f"on batches with > {ANTIHACK_EARLY_DISCOUNT_HOURS_THRESHOLD:.0f}hrs remaining "
            "triggers an anti-hack penalty."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # FARMER prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _build_farmer_prompt(state: SimulatedMarketState) -> str:
        lines: list[str] = []

        # === PENDING FARMER OFFERS ===
        lines.append("=== PENDING FARMER OFFERS ===")

        pending_offers = [
            o for o in state.pending_offers
            if o.status == FarmerOfferStatus.PENDING
            and not o.is_expired(state.tick)
        ]

        offer_categories: set[str] = set()

        if not pending_offers:
            lines.append("No pending offers.")
            lines.append("")
        else:
            for offer in pending_offers:
                offer_categories.add(offer.product_category)
                viability_label = _viability_label(offer.viability_score)
                viability_str = (
                    f"{offer.viability_score:.2f} ({viability_label})"
                    if offer.viability_score is not None
                    else "Not yet scored"
                )

                lines.append(
                    f"Offer {offer.offer_id}: {offer.farmer_name} — "
                    f"{offer.quantity_kg:.0f}kg of {offer.product_name}"
                )
                lines.append(
                    f"  Offered price: Rs {offer.offered_price_per_kg:.0f}/kg"
                )
                lines.append(
                    f"  Seller-reported shelf life: {offer.seller_shelf_life_hrs}hrs"
                )
                lines.append(f"  Viability score: {viability_str}")
                lines.append("")

        # === CURRENT INVENTORY (same category) ===
        lines.append("=== CURRENT INVENTORY (same category) ===")

        category_batches = [
            b for b in state.batches
            if b.category in offer_categories and b.status == BatchStatus.ACTIVE
        ]
        category_batches.sort(key=lambda b: b.hours_to_expiry)

        if not category_batches:
            lines.append("No active inventory in offered categories.")
        else:
            for batch in category_batches:
                emoji = _URGENCY_EMOJI.get(batch.urgency, "")
                vel = state.sales_velocity.get(batch.batch_id, 0.0)
                lines.append(
                    f"[{emoji}] {batch.category} — Batch {batch.batch_id} | "
                    f"{batch.quantity_remaining} units | "
                    f"{batch.hours_to_expiry:.1f}hrs | "
                    f"Rs {batch.current_price:.0f} | "
                    f"Vel: {vel:.2f}/hr"
                )
        lines.append("")

        # === RISK BUFFER ===
        lines.append("=== RISK BUFFER ===")
        lines.append(f"Current balance: Rs {state.risk_buffer_balance:.0f}")
        lines.append("Note: Buffer below Rs 2000 means conservative acceptance only.")
        if state.risk_buffer_balance < 2000.0:
            lines.append(
                "\u26a0\ufe0f Low buffer — only accept offers with viability >= 0.80"
            )
        lines.append("")

        # === YOUR TASK ===
        lines.append("=== YOUR TASK ===")
        lines.append(
            "Write a FARMER Operating Brief. For each pending offer, decide ACCEPT, COUNTER, "
            "or DECLINE. If countering, specify a counter_price in the DIRECTIVE."
        )
        lines.append(
            f"Remember: accepting an offer with viability_score < "
            f"{ANTIHACK_RECKLESS_ACCEPT_VIABILITY_MAX} triggers "
            "an anti-hack penalty (reckless acceptance)."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # TREND prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _build_trend_prompt(state: SimulatedMarketState) -> str:
        lines: list[str] = []

        # === ACTIVE TREND SIGNALS ===
        lines.append("=== ACTIVE TREND SIGNALS ===")

        signal_categories: set[str] = set()
        actionable_signals = [
            (cat, sig) for cat, sig in state.trend_signals.items()
            if sig.action_taken == TrendAction.PENDING
            and sig.is_actionable(state.tick)
        ]

        if not actionable_signals:
            lines.append("No actionable trend signals.")
            lines.append("")
        else:
            for cat, sig in actionable_signals:
                signal_categories.add(cat)

                if sig.composite_score >= 80.0:
                    confidence_tier = "HIGH"
                elif sig.composite_score >= TREND_SCORE_THRESHOLD:
                    confidence_tier = "MEDIUM"
                else:
                    confidence_tier = "LOW"

                lines.append(
                    f"\U0001f4f1 {cat.upper()} — Score: {sig.composite_score:.0f}/100 "
                    f"({sig.signal_source.value})"
                )
                lines.append(f"  Suggested restock: {sig.suggested_order_kg:.0f}kg")
                lines.append("  Signal factors:")
                lines.append(f"    Recipe simplicity:     {sig.recipe_simplicity:.2f}")
                lines.append(f"    Ingredient rarity:     {sig.ingredient_rarity:.2f}")
                lines.append(f"    View velocity:         {sig.view_velocity:.2f}")
                lines.append(f"    Local relevance:       {sig.local_relevance:.2f}")
                lines.append(f"    Historical conversion: {sig.historical_conversion:.2f}")
                lines.append(f"  Confidence tier: {confidence_tier}")
                lines.append("")

        # === CURRENT STOCK (affected categories) ===
        lines.append("=== CURRENT STOCK (affected categories) ===")

        category_batches = [
            b for b in state.batches
            if b.category in signal_categories and b.status == BatchStatus.ACTIVE
        ]
        category_batches.sort(key=lambda b: (b.category, b.hours_to_expiry))

        if not category_batches:
            lines.append("No active inventory in signalled categories.")
        else:
            for batch in category_batches:
                emoji = _URGENCY_EMOJI.get(batch.urgency, "")
                lines.append(
                    f"[{emoji}] {batch.category} — Batch {batch.batch_id} | "
                    f"{batch.quantity_remaining} units | "
                    f"{batch.hours_to_expiry:.1f}hrs | "
                    f"Rs {batch.current_price:.0f}"
                )
        lines.append("")

        # === COOLDOWN STATUS ===
        lines.append("=== COOLDOWN STATUS ===")
        cooldown_ticks = int(TREND_COOLDOWN_HRS * (TICKS_PER_DAY / 24.0))
        for cat in signal_categories:
            # Cooldown info is not directly on TrendEngine from here —
            # we show time since signal detection as a proxy.
            sig = state.trend_signals.get(cat)
            if sig is not None:
                ticks_since = state.tick - sig.detected_at_tick
                hours_since = ticks_since / (TICKS_PER_DAY / 24.0)
                lines.append(f"  {cat}: signal detected {hours_since:.1f}hrs ago")
            else:
                lines.append(f"  {cat}: No cooldown")
        lines.append("")

        # === YOUR TASK ===
        lines.append("=== YOUR TASK ===")
        lines.append(
            f"Write a TREND Operating Brief. For each signal above threshold "
            f"({TREND_SCORE_THRESHOLD:.0f}), decide APPROVE or DECLINE. "
            "If approving, specify order_quantity_kg."
        )
        lines.append(
            f"Cap: max order = avg weekly velocity x {ANTIHACK_TREND_ORDER_VELOCITY_MULTIPLIER:.1f}. "
            "Hard cap enforced by the engine."
        )

        return "\n".join(lines)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _format_hour(hour: int) -> str:
    """Format hour as 12hr time: 14 -> '2:00 PM'."""
    if hour == 0:
        return "12:00 AM"
    if hour < 12:
        return f"{hour}:00 AM"
    if hour == 12:
        return "12:00 PM"
    return f"{hour - 12}:00 PM"


def _format_day(day_of_week: int) -> str:
    """Format day: 0 -> 'Monday', 6 -> 'Sunday'."""
    if 0 <= day_of_week < len(_DAY_NAMES):
        return _DAY_NAMES[day_of_week]
    return f"Day {day_of_week}"


def _viability_label(score: float | None) -> str:
    """Human-readable viability tier."""
    if score is None:
        return "Not scored"
    if score >= 0.80:
        return "Strong Accept"
    if score >= 0.60:
        return "Acceptable"
    if score >= 0.40:
        return "Borderline"
    return "High Risk"
