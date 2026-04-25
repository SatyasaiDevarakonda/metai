"""ConsumerCohortAgent — Blinkit/Zepto-style heterogeneous consumers.

The original ``ConsumerAgent`` treated demand as one elastic blob.
Real quick-commerce stores serve at least three very different cohorts:

  - PREMIUM         (Blinkit Plus / Zepto Pass) — low elasticity,
                    intolerant of slow ETAs, walks on URGENT stock
  - MASS            — moderate elasticity, accepts URGENT, ETA-sensitive
  - BARGAIN_HUNTER  — high elasticity, happily buys CRITICAL at 60%
                    off, willing to wait

The hero must learn to balance discount depth against delivery
promise: too steep a discount draws bargain hunters and saturates the
rider pool, which then loses the premium cohort.

This agent is additive: ``MultiAgentFreshPriceEnv`` /
``MarketCommonsEnv`` can opt-in by holding one alongside (or instead
of) the legacy ``ConsumerAgent``. The output shape is a superset of the
legacy one, so the env's downstream code continues to work.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from freshprice_env.agents.consumer_agent import ConsumerAgent
from freshprice_env.constants import (
    COHORT_ELASTICITY_BARGAIN,
    COHORT_ELASTICITY_MASS,
    COHORT_ELASTICITY_PREMIUM,
    COHORT_ETA_TOLERANCE_BARGAIN_MIN,
    COHORT_ETA_TOLERANCE_MASS_MIN,
    COHORT_ETA_TOLERANCE_PREMIUM_MIN,
    COHORT_FRESHNESS_TOL_BARGAIN,
    COHORT_FRESHNESS_TOL_MASS,
    COHORT_FRESHNESS_TOL_PREMIUM,
    COHORT_WEIGHT_BARGAIN,
    COHORT_WEIGHT_MASS,
    COHORT_WEIGHT_PREMIUM,
)
from freshprice_env.entities import SimulatedMarketState
from freshprice_env.enums import ExpiryUrgency


@dataclass(frozen=True)
class Cohort:
    name: str
    weight: float                # share of footfall (sums to 1.0 across cohorts)
    price_elasticity: float
    freshness_tolerance: float   # max fraction of URGENT/CRITICAL units they buy
    eta_tolerance_minutes: float


DEFAULT_COHORTS: tuple[Cohort, ...] = (
    Cohort("PREMIUM",  COHORT_WEIGHT_PREMIUM,
           COHORT_ELASTICITY_PREMIUM, COHORT_FRESHNESS_TOL_PREMIUM,
           COHORT_ETA_TOLERANCE_PREMIUM_MIN),
    Cohort("MASS",     COHORT_WEIGHT_MASS,
           COHORT_ELASTICITY_MASS,    COHORT_FRESHNESS_TOL_MASS,
           COHORT_ETA_TOLERANCE_MASS_MIN),
    Cohort("BARGAIN",  COHORT_WEIGHT_BARGAIN,
           COHORT_ELASTICITY_BARGAIN, COHORT_FRESHNESS_TOL_BARGAIN,
           COHORT_ETA_TOLERANCE_BARGAIN_MIN),
)


class ConsumerCohortAgent:
    """Multi-cohort wrapper around the legacy ``ConsumerAgent``."""

    def __init__(
        self,
        rng: random.Random,
        cohorts: tuple[Cohort, ...] = DEFAULT_COHORTS,
    ) -> None:
        self._rng = rng
        self._cohorts = cohorts
        # Re-use the legacy agent for global weather / event multipliers.
        self._legacy = ConsumerAgent(rng, price_sensitivity=1.5)
        # Per-cohort retention metrics, updated each call to act().
        self.last_retention: dict[str, float] = {c.name: 1.0 for c in cohorts}
        self.last_walk_aways: dict[str, int] = {c.name: 0 for c in cohorts}

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def act(
        self,
        state: SimulatedMarketState,
        *,
        avg_eta_minutes: float = 0.0,
        urgency_share: float | None = None,
    ) -> dict[str, float]:
        """Compute aggregate per-batch demand multipliers across cohorts.

        Args:
            state: SimulatedMarketState the env is about to apply.
            avg_eta_minutes: Most recent average ETA from the rider pool
                (0 means we don't penalize ETA at all).
            urgency_share: Pre-computed fraction of active units that are
                URGENT or CRITICAL. If None, computed locally.

        Returns:
            dict[batch_id -> demand multiplier]. The multiplier is the
            weighted sum of per-cohort multipliers, accounting for
            cohort retention (cohorts that walk away contribute 0).
        """
        if urgency_share is None:
            urgency_share = self._urgency_share(state)

        # Update retention based on ETA / freshness tolerance.
        retention = {}
        walk_aways = {}
        for c in self._cohorts:
            eta_ok = c.eta_tolerance_minutes <= 0 or avg_eta_minutes <= c.eta_tolerance_minutes
            fresh_ok = urgency_share <= c.freshness_tolerance
            r = 1.0
            if not eta_ok:
                # Linear walk-away curve up to 80% loss.
                slip_ratio = (avg_eta_minutes / max(c.eta_tolerance_minutes, 1.0)) - 1.0
                r *= max(0.20, 1.0 - 0.80 * min(slip_ratio, 1.0))
            if not fresh_ok:
                excess = (urgency_share - c.freshness_tolerance) / max(
                    1.0 - c.freshness_tolerance, 1e-3
                )
                r *= max(0.10, 1.0 - excess)
            retention[c.name] = round(r, 3)
            walk_aways[c.name] = round((1.0 - r) * c.weight * 100, 1)
        self.last_retention = retention
        self.last_walk_aways = walk_aways

        # Aggregate per-batch boost across cohorts.
        boosts: dict[str, float] = {}
        for batch in state.batches:
            if batch.status.value != "ACTIVE":
                continue
            agg = 0.0
            for c in self._cohorts:
                price_mult = self._cohort_price_mult(batch, c.price_elasticity)
                freshness_mult = self._freshness_mult(batch, c)
                agg += c.weight * retention[c.name] * price_mult * freshness_mult
            agg *= self._rng.uniform(0.94, 1.06)  # cohort heterogeneity noise
            boosts[batch.batch_id] = max(0.0, min(agg, 3.0))
        return boosts

    def observe(self, state: SimulatedMarketState, *, avg_eta_minutes: float = 0.0) -> dict:
        """Structured observation for the brief prompt."""
        return {
            "cohorts": [
                {
                    "name": c.name,
                    "weight": c.weight,
                    "price_elasticity": c.price_elasticity,
                    "eta_tolerance_min": c.eta_tolerance_minutes,
                    "retention_pct": self.last_retention.get(c.name, 1.0) * 100,
                    "walked_away_pct": self.last_walk_aways.get(c.name, 0.0),
                }
                for c in self._cohorts
            ],
            "avg_eta_minutes": avg_eta_minutes,
            "urgency_share": self._urgency_share(state),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _cohort_price_mult(batch, elasticity: float) -> float:
        if batch.original_price <= 0:
            return 1.0
        ratio = batch.original_price / max(batch.current_price, 0.01)
        return min(ratio ** elasticity, 3.0)

    @staticmethod
    def _freshness_mult(batch, cohort: Cohort) -> float:
        # Premium customers physically walk away from CRITICAL stock.
        if batch.urgency == ExpiryUrgency.CRITICAL:
            if cohort.freshness_tolerance < 0.10:
                return 0.05    # essentially never buy
            return 0.55
        if batch.urgency == ExpiryUrgency.URGENT:
            if cohort.freshness_tolerance < 0.20:
                return 0.40
            return 0.85
        return 1.0

    @staticmethod
    def _urgency_share(state: SimulatedMarketState) -> float:
        active = [b for b in state.batches if b.status.value == "ACTIVE"]
        if not active:
            return 0.0
        urgent_units = sum(
            b.quantity_remaining for b in active
            if b.urgency in (ExpiryUrgency.URGENT, ExpiryUrgency.CRITICAL)
        )
        total_units = sum(b.quantity_remaining for b in active)
        return urgent_units / max(total_units, 1)
