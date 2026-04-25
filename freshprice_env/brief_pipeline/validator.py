"""Validates a parsed OperatingBrief before it is acted on.

A brief can parse successfully but still be invalid for business reasons.
Errors → brief rejected, last valid directive used.
Warnings → brief proceeds, issues logged for analysis.

Additionally, the *active schema version* is consulted via
``schema_registry.default_registry()``. The RegulatorAgent mutates that
registry mid-episode. A brief written against an outdated schema (e.g.
``price_multiplier`` after a v3 cap-and-rename event) fails validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from freshprice_env.brief_pipeline.schema_registry import default_registry
from freshprice_env.constants import (
    ANTIHACK_RECKLESS_ACCEPT_VIABILITY_MAX,
    ANTIHACK_TREND_ORDER_VELOCITY_MULTIPLIER,
    PRICE_MULTIPLIER_MAX,
    PRICE_MULTIPLIER_MIN,
    TREND_SIGNAL_EXPIRY_HRS,
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
from freshprice_env.market_state import get_base_demand_velocity

# Farmer counter-offer plausibility ceiling: 150% of ask
_COUNTER_PRICE_MAX_RATIO: float = 1.50

# Ticks per brief interval (used for imminent-expiry warning)
_BRIEF_INTERVAL_TICKS: int = 8

# Trend signal near-expiry warning threshold (ticks before signal expires)
_TREND_NEAR_EXPIRY_TICKS: int = 20


@dataclass
class ValidationResult:
    """Result of validating an Operating Brief."""

    valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class BriefValidator:
    """Validates parsed Operating Briefs against business rules."""

    @staticmethod
    def validate(
        brief: dict,
        state: SimulatedMarketState,
    ) -> ValidationResult:
        """Run all validations for the brief's engine type.

        Args:
            brief: Parsed brief dict from BriefParser (contains engine_type, directive, etc.)
            state: Current market state.
        """
        engine_type = brief["engine_type"]

        # Schema-drift check first: the active schema version may have
        # been mutated by the RegulatorAgent. Briefs that don't match
        # the new schema fail before business validation runs.
        registry = default_registry()
        directive = brief.get("directive") or {}
        schema_ok, schema_errors = registry.validate(directive, engine_type)

        validators = {
            BriefEngineType.PRICING: BriefValidator._validate_pricing,
            BriefEngineType.FARMER: BriefValidator._validate_farmer,
            BriefEngineType.TREND: BriefValidator._validate_trend,
        }
        biz_result = validators[engine_type](brief, state)

        # Merge schema errors as the highest-priority failure surface
        merged_errors = list(schema_errors) + list(biz_result.errors)
        active_v = registry.active_version(engine_type)
        if not schema_ok:
            biz_result.warnings.append(
                f"directive must conform to active schema "
                f"{engine_type.value} {active_v}"
            )
        return ValidationResult(
            valid=schema_ok and biz_result.valid,
            warnings=biz_result.warnings,
            errors=merged_errors,
        )

    # ------------------------------------------------------------------
    # PRICING validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_pricing(brief: dict, state: SimulatedMarketState) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        directive = brief["directive"]

        # Engine match
        if directive.get("engine", "").upper() != "PRICING":
            errors.append("Directive engine is not PRICING")
            return ValidationResult(valid=False, warnings=warnings, errors=errors)

        actions = directive.get("actions", [])

        # Empty actions = legitimate "hold all prices" decision (parity with
        # FARMER/TREND validators). Warn instead of erroring so the brief
        # proceeds and gets quality-scored rather than falling back.
        if not actions:
            warnings.append("Directive has no actions — interpreting as hold-all-prices")
            return ValidationResult(valid=True, warnings=warnings, errors=errors)

        # Build batch lookup
        batch_ids = {
            b.batch_id: b for b in state.batches if b.status == BatchStatus.ACTIVE
        }

        seen_multipliers: list[float] = []
        critical_batches_addressed = set()
        critical_batch_ids = {
            b.batch_id for b in state.batches
            if b.status == BatchStatus.ACTIVE and b.urgency == ExpiryUrgency.CRITICAL
        }

        for action in actions:
            bid = action.get("batch_id")

            # Batch must exist
            if bid not in batch_ids:
                errors.append(f"batch_id '{bid}' not found in active inventory")
                continue

            batch = batch_ids[bid]
            pm = action.get("price_multiplier", 1.0)

            # Multiplier range
            if pm < PRICE_MULTIPLIER_MIN or pm > PRICE_MULTIPLIER_MAX:
                errors.append(
                    f"price_multiplier {pm} for {bid} outside "
                    f"[{PRICE_MULTIPLIER_MIN}, {PRICE_MULTIPLIER_MAX}]"
                )

            seen_multipliers.append(pm)

            if bid in critical_batch_ids:
                critical_batches_addressed.add(bid)

            # Flash sale with no notification credits
            if action.get("flash_sale", False):
                credits = state.notification_credits.get(batch.category, 0)
                if credits <= 0:
                    warnings.append(
                        f"Flash sale for {bid} ({batch.category}) but no notification "
                        "credits remaining — will be silently ignored"
                    )

        # Warnings: all CRITICAL batches held at 1.0
        unaddressed_critical = critical_batch_ids - critical_batches_addressed
        addressed_at_full = [
            a for a in actions
            if a.get("batch_id") in critical_batches_addressed
            and a.get("price_multiplier", 1.0) == 1.0
        ]
        if critical_batch_ids and len(addressed_at_full) == len(critical_batches_addressed):
            warnings.append(
                "All CRITICAL batches held at price_multiplier 1.0 — "
                "agent may not be responding to urgency"
            )

        # Warning: all same multiplier
        if len(seen_multipliers) > 1 and len(set(seen_multipliers)) == 1:
            warnings.append(
                f"All {len(seen_multipliers)} actions use the same "
                f"price_multiplier ({seen_multipliers[0]}) — possible lazy brief"
            )

        return ValidationResult(
            valid=len(errors) == 0,
            warnings=warnings,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # FARMER validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_farmer(brief: dict, state: SimulatedMarketState) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        directive = brief["directive"]

        # Engine match
        if directive.get("engine", "").upper() != "FARMER":
            errors.append("Directive engine is not FARMER")
            return ValidationResult(valid=False, warnings=warnings, errors=errors)

        actions = directive.get("actions", [])

        # Build offer lookup
        offer_map = {
            o.offer_id: o for o in state.pending_offers
            if o.status == FarmerOfferStatus.PENDING
        }

        # Track which offers have actions
        acted_on: set[str] = set()

        for action in actions:
            oid = action.get("offer_id")

            # Offer must exist
            if oid not in offer_map:
                errors.append(f"offer_id '{oid}' not found in pending offers")
                continue

            offer = offer_map[oid]
            decision = action.get("decision", "").upper()
            acted_on.add(oid)

            if decision == "COUNTER":
                cp = action.get("counter_price")
                if cp is None or cp <= 0:
                    errors.append(
                        f"COUNTER for {oid} has invalid counter_price: {cp}"
                    )
                elif cp > offer.offered_price_per_kg * _COUNTER_PRICE_MAX_RATIO:
                    errors.append(
                        f"COUNTER for {oid} at Rs {cp:.0f}/kg is > 150% of ask "
                        f"(Rs {offer.offered_price_per_kg:.0f}/kg) — likely hallucination"
                    )

            # Warnings
            if decision == "ACCEPT" and offer.viability_score is not None:
                if offer.viability_score < ANTIHACK_RECKLESS_ACCEPT_VIABILITY_MAX:
                    warnings.append(
                        f"ACCEPT on {oid} with viability {offer.viability_score:.2f} "
                        "— anti-hack will penalise (reckless acceptance)"
                    )

            if decision == "DECLINE" and offer.viability_score is not None:
                if offer.viability_score > 0.85:
                    warnings.append(
                        f"DECLINE on {oid} with viability {offer.viability_score:.2f} "
                        "— very conservative decision"
                    )

        # Warning: offer expiring within next brief interval with no action
        for oid, offer in offer_map.items():
            if oid in acted_on:
                continue
            if offer.is_expired(state.tick + _BRIEF_INTERVAL_TICKS):
                warnings.append(
                    f"Offer {oid} expires within next {_BRIEF_INTERVAL_TICKS} ticks "
                    "but has no action in this brief"
                )

        return ValidationResult(
            valid=len(errors) == 0,
            warnings=warnings,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # TREND validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_trend(brief: dict, state: SimulatedMarketState) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        directive = brief["directive"]

        # Engine match
        if directive.get("engine", "").upper() != "TREND":
            errors.append("Directive engine is not TREND")
            return ValidationResult(valid=False, warnings=warnings, errors=errors)

        actions = directive.get("actions", [])

        ticks_per_hour = TICKS_PER_DAY / 24.0
        signal_expiry_ticks = TREND_SIGNAL_EXPIRY_HRS * ticks_per_hour

        for action in actions:
            category = action.get("category")
            decision = action.get("decision", "").upper()

            # Category must have a signal
            if category not in state.trend_signals:
                errors.append(f"Category '{category}' not in trend_signals")
                continue

            signal = state.trend_signals[category]

            if decision == "APPROVE":
                qty = action.get("order_quantity_kg")
                if qty is None or qty <= 0:
                    errors.append(
                        f"APPROVE for {category} has invalid order_quantity_kg: {qty}"
                    )
                    continue

                # Warning: approaching cap
                base_hourly = get_base_demand_velocity(category, 12, 3)
                weekly_velocity = base_hourly * 24.0 * 7.0
                soft_warn_cap = weekly_velocity * (ANTIHACK_TREND_ORDER_VELOCITY_MULTIPLIER * 0.75)
                if qty > soft_warn_cap:
                    warnings.append(
                        f"APPROVE for {category} at {qty:.0f}kg is approaching "
                        f"the hard cap ({weekly_velocity * ANTIHACK_TREND_ORDER_VELOCITY_MULTIPLIER:.0f}kg)"
                    )

                # Warning: low confidence approval
                if signal.composite_score < 70.0:
                    warnings.append(
                        f"APPROVE for {category} with composite score "
                        f"{signal.composite_score:.0f} (below 70) — low confidence"
                    )

            # Warning: signal near expiry
            ticks_since_detection = state.tick - signal.detected_at_tick
            ticks_remaining = signal_expiry_ticks - ticks_since_detection
            if 0 < ticks_remaining < _TREND_NEAR_EXPIRY_TICKS:
                warnings.append(
                    f"Signal for {category} expires in {ticks_remaining:.0f} ticks "
                    "— decision window nearly closed"
                )

        return ValidationResult(
            valid=len(errors) == 0,
            warnings=warnings,
            errors=errors,
        )
