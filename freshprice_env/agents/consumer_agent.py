"""ConsumerAgent — a reactive buyer that responds to price and event signals.

In MultiAgentFreshPriceEnv, this agent runs alongside the LLM store-manager.
It models the collective buying behaviour of real customers:

  - Discount signals  → demand boost (price elasticity)
  - Festival events   → category-specific surge
  - Weather signals   → footfall and preference shifts
  - Urgency urgency   → reduced trust in CRITICAL items (freshness concern)

The agent is NOT trained. It is a fixed rule-based policy that makes the
environment more realistic and challenges the LLM to anticipate consumer
reactions — this is the "theory-of-mind" element for Theme #1.
"""

from __future__ import annotations

import random

from freshprice_env.constants import (
    FESTIVAL_DEMAND_MULTIPLIERS,
    WEATHER_HOT_FRUITS_MULTIPLIER,
    WEATHER_RAIN_DEMAND_MULTIPLIER,
)
from freshprice_env.entities import SimulatedBatch, SimulatedMarketState
from freshprice_env.enums import ExpiryUrgency, ExternalEvent, WeatherCondition


class ConsumerAgent:
    """Simulates a heterogeneous pool of price-sensitive consumers.

    Each tick the agent:
      1. Observes the market state (prices, weather, events)
      2. Computes a per-batch demand multiplier
      3. Writes the multiplier into state.consumer_demand_boost

    The PricingEngine already reads sales_velocity; the demand boost is an
    additive signal layered on top so existing engine maths stay intact.
    """

    PRICE_ELASTICITY: float = 1.5      # demand ∝ (original/current)^elasticity
    CRITICAL_TRUST_PENALTY: float = 0.70  # consumers buy 30% less from CRITICAL batches
    MAX_DEMAND_BOOST: float = 3.0         # hard cap to prevent unrealistic spikes

    def __init__(self, rng: random.Random, price_sensitivity: float = 1.5) -> None:
        self._rng = rng
        self._price_sensitivity = price_sensitivity

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def act(self, state: SimulatedMarketState) -> dict[str, float]:
        """Compute per-batch demand multipliers for this tick.

        Returns a dict {batch_id: multiplier} written into
        state.consumer_demand_boost by the calling environment.
        """
        boosts: dict[str, float] = {}

        weather_global = self._weather_global_mult(state.weather_condition)
        event_global = self._event_global_mult(state.active_event)

        for batch in state.batches:
            if batch.status.value != "ACTIVE":
                continue

            # Price-driven demand elasticity
            price_mult = self._price_elasticity_mult(batch)

            # Freshness trust: consumers are wary of CRITICAL items
            freshness_mult = (
                self.CRITICAL_TRUST_PENALTY
                if batch.urgency == ExpiryUrgency.CRITICAL
                else 1.0
            )

            # Category-specific event boost
            cat_event_mult = self._category_event_mult(batch.category, state.active_event)

            # Category-specific weather boost
            cat_weather_mult = self._category_weather_mult(batch.category, state.weather_condition)

            combined = (
                price_mult
                * freshness_mult
                * weather_global
                * event_global
                * cat_event_mult
                * cat_weather_mult
            )

            # Add ±8% consumer heterogeneity noise
            combined *= self._rng.uniform(0.92, 1.08)
            boosts[batch.batch_id] = min(combined, self.MAX_DEMAND_BOOST)

        return boosts

    def observe(self, state: SimulatedMarketState) -> dict:
        """Return a structured observation of what consumers see.

        Used for logging and multi-agent prompt building.
        """
        active = [b for b in state.batches if b.status.value == "ACTIVE"]
        return {
            "weather": state.weather_condition.value,
            "event": state.active_event.value,
            "visible_discounts": [
                {
                    "batch_id": b.batch_id,
                    "category": b.category,
                    "discount_pct": round(b.discount_pct, 1),
                    "urgency": b.urgency.value,
                    "current_price": b.current_price,
                }
                for b in active
                if b.discount_pct > 5.0  # only visible discounts
            ],
            "high_demand_signals": [
                cat for cat, v in state.sales_velocity.items()
                if v > 3.0
            ],
        }

    # ------------------------------------------------------------------
    # Internal multiplier helpers
    # ------------------------------------------------------------------

    def _price_elasticity_mult(self, batch: SimulatedBatch) -> float:
        if batch.original_price <= 0.0:
            return 1.0
        ratio = batch.original_price / batch.current_price
        return min(ratio ** self._price_sensitivity, self.MAX_DEMAND_BOOST)

    @staticmethod
    def _weather_global_mult(weather: WeatherCondition) -> float:
        if weather == WeatherCondition.RAINY:
            return WEATHER_RAIN_DEMAND_MULTIPLIER
        if weather == WeatherCondition.SUNNY:
            return 1.1   # sunny increases footfall slightly
        return 1.0

    @staticmethod
    def _event_global_mult(event: ExternalEvent) -> float:
        if event == ExternalEvent.LOCAL_HOLIDAY:
            return 1.3
        return 1.0

    @staticmethod
    def _category_event_mult(category: str, event: ExternalEvent) -> float:
        if event == ExternalEvent.FESTIVAL:
            return FESTIVAL_DEMAND_MULTIPLIERS.get(category, 1.3)
        if event == ExternalEvent.SPORTS_EVENT:
            return 1.4 if category == "packaged" else 1.0
        return 1.0

    @staticmethod
    def _category_weather_mult(category: str, weather: WeatherCondition) -> float:
        if weather == WeatherCondition.HOT and category == "fruits":
            return WEATHER_HOT_FRUITS_MULTIPLIER
        if weather == WeatherCondition.COLD and category in ("bakery", "dairy"):
            return 1.15
        return 1.0
