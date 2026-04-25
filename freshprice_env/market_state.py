"""Factory for building the initial SimulatedMarketState per curriculum scenario.

No randomness generated here — accepts a random.Random instance for reproducibility.
"""

from __future__ import annotations

import random
from dataclasses import replace

from freshprice_env.constants import (
    NOTIFICATION_CREDITS_PER_CATEGORY_PER_DAY,
    RISK_BUFFER_INITIAL_SEED_RS,
    TREND_SCORE_THRESHOLD,
)
from freshprice_env.entities import (
    SimulatedBatch,
    SimulatedFarmerOffer,
    SimulatedMarketState,
    SimulatedTrendSignal,
)
from freshprice_env.enums import (
    BatchStatus,
    BatchType,
    CurriculumScenario,
    FarmerOfferStatus,
    SignalSource,
    TrendAction,
)

# ---------------------------------------------------------------------------
# Category specs: (min_qty, max_qty, min_cost, max_cost, min_price, max_price,
#                   min_shelf_hrs, max_shelf_hrs)
# ---------------------------------------------------------------------------
_CATEGORY_SPECS: dict[str, tuple[int, int, float, float, float, float, float, float]] = {
    "vegetables": (15, 25, 12.0, 18.0, 30.0, 45.0, 48.0, 72.0),
    "dairy":      (10, 20, 25.0, 35.0, 60.0, 80.0, 24.0, 48.0),
    "fruits":     (15, 25, 20.0, 30.0, 55.0, 75.0, 48.0, 96.0),
    "bakery":     (20, 30, 15.0, 22.0, 40.0, 60.0, 18.0, 36.0),
    "packaged":   (30, 50, 40.0, 60.0, 90.0, 120.0, 144.0, 240.0),
    "herbs":      (8, 15, 8.0, 12.0, 20.0, 30.0, 24.0, 48.0),
}

ALL_CATEGORIES: list[str] = list(_CATEGORY_SPECS.keys())

# ---------------------------------------------------------------------------
# Base demand velocities (units/hour at full price)
# ---------------------------------------------------------------------------
_BASE_VELOCITY: dict[str, float] = {
    "vegetables": 2.5,
    "dairy":      1.8,
    "fruits":     2.2,
    "bakery":     3.0,
    "packaged":   1.2,
    "herbs":      0.8,
}

# ---------------------------------------------------------------------------
# Time-of-day demand multipliers
# ---------------------------------------------------------------------------
_TIME_MULTIPLIERS: list[tuple[range, float]] = [
    (range(7, 10),  1.4),   # morning rush
    (range(12, 15), 1.6),   # lunch
    (range(17, 21), 1.8),   # evening peak
]
_NIGHT_HOURS: set[int] = {22, 23, 0, 1, 2, 3, 4, 5, 6}
_NIGHT_MULTIPLIER: float = 0.3
_DEFAULT_TIME_MULTIPLIER: float = 1.0

# ---------------------------------------------------------------------------
# Day-of-week demand multipliers (Monday=0 .. Sunday=6)
# ---------------------------------------------------------------------------
_DAY_MULTIPLIERS: dict[int, float] = {
    4: 1.3,  # Friday
    5: 1.5,  # Saturday
    6: 1.4,  # Sunday
}
_DEFAULT_DAY_MULTIPLIER: float = 1.0


def get_base_demand_velocity(category: str, hour_of_day: int, day_of_week: int) -> float:
    """Return expected units/hour demand for a category at the given time.

    The environment adds ±20% noise on top of this value.
    """
    base = _BASE_VELOCITY.get(category, 1.0)

    # Time-of-day multiplier
    time_mult = _DEFAULT_TIME_MULTIPLIER
    if hour_of_day in _NIGHT_HOURS:
        time_mult = _NIGHT_MULTIPLIER
    else:
        for hour_range, mult in _TIME_MULTIPLIERS:
            if hour_of_day in hour_range:
                time_mult = mult
                break

    # Day-of-week multiplier
    day_mult = _DAY_MULTIPLIERS.get(day_of_week, _DEFAULT_DAY_MULTIPLIER)

    return base * time_mult * day_mult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_batches(
    rng: random.Random,
    min_shelf_override: float | None = None,
    max_shelf_override: float | None = None,
    include_urgent: bool = False,
) -> tuple[list[SimulatedBatch], dict[str, float]]:
    """Generate 2-3 batches per category. Returns (batches, sales_velocity_dict)."""
    batches: list[SimulatedBatch] = []
    velocities: dict[str, float] = {}
    batch_counter = 0

    for cat, (qmin, qmax, cmin, cmax, pmin, pmax, smin, smax) in _CATEGORY_SPECS.items():
        num_batches = rng.randint(2, 3)
        for i in range(num_batches):
            batch_counter += 1
            bid = f"batch_{batch_counter:04d}"

            qty = rng.randint(qmin, qmax)
            cost = round(rng.uniform(cmin, cmax), 2)
            price = round(rng.uniform(pmin, pmax), 2)

            # Shelf life: use override range if provided, else category defaults
            shelf_min = min_shelf_override if min_shelf_override is not None else smin
            shelf_max = max_shelf_override if max_shelf_override is not None else smax

            if include_urgent and i == 0 and cat in ("dairy", "bakery", "herbs"):
                # First batch of select categories starts URGENT (6-24 hrs)
                hours = round(rng.uniform(7.0, 20.0), 1)
            else:
                # Distribute between FRESH and WATCH — no URGENT/CRITICAL at init.
                # FRESH requires > 72h; only attempt if category max shelf life > 72h,
                # otherwise fall back to WATCH to avoid an inverted rng.uniform range.
                fresh_min = 73.0
                can_be_fresh = shelf_max > fresh_min
                if can_be_fresh and rng.random() < 0.6:
                    hours = round(rng.uniform(fresh_min, shelf_max), 1)
                else:
                    watch_max = min(shelf_max, 71.0)
                    hours = round(rng.uniform(25.0, max(25.0, watch_max)), 1)

            batch = SimulatedBatch(
                batch_id=bid,
                sku_id=f"sku_{cat}_{i}",
                store_id="store_001",
                category=cat,
                quantity_remaining=qty,
                unit_cost=cost,
                current_price=price,
                original_price=price,
                hours_to_expiry=hours,
                batch_type=BatchType.REGULAR,
                status=BatchStatus.ACTIVE,
            )
            batches.append(batch)

            # Initial velocity: base demand with a small rng perturbation
            base_vel = get_base_demand_velocity(cat, hour_of_day=8, day_of_week=0)
            velocities[bid] = round(base_vel * rng.uniform(0.8, 1.2), 3)

    return batches, velocities


def _empty_notification_credits() -> dict[str, int]:
    return {cat: NOTIFICATION_CREDITS_PER_CATEGORY_PER_DAY for cat in ALL_CATEGORIES}


def _base_state(
    batches: list[SimulatedBatch],
    velocities: dict[str, float],
    offers: list[SimulatedFarmerOffer] | None = None,
    trends: dict[str, SimulatedTrendSignal] | None = None,
    risk_buffer: float = RISK_BUFFER_INITIAL_SEED_RS,
) -> SimulatedMarketState:
    return SimulatedMarketState(
        tick=0,
        batches=batches,
        pending_offers=offers or [],
        trend_signals=trends or {},
        sales_velocity=velocities,
        risk_buffer_balance=risk_buffer,
        notification_credits=_empty_notification_credits(),
        at_risk_cost_accumulator=0.0,
        revenue_recovered_accumulator=0.0,
    )


# ---------------------------------------------------------------------------
# MarketStateBuilder
# ---------------------------------------------------------------------------

class MarketStateBuilder:
    """Factory that creates the initial SimulatedMarketState per curriculum scenario."""

    @staticmethod
    def build(scenario: CurriculumScenario, rng: random.Random) -> SimulatedMarketState:
        builders = {
            CurriculumScenario.STABLE_WEEK:     MarketStateBuilder._build_stable_week,
            CurriculumScenario.BUSY_WEEKEND:    MarketStateBuilder._build_busy_weekend,
            CurriculumScenario.FARMER_WEEK:     MarketStateBuilder._build_farmer_week,
            CurriculumScenario.TREND_WEEK:      MarketStateBuilder._build_trend_week,
            CurriculumScenario.CRISIS_WEEK:     MarketStateBuilder._build_crisis_week,
            # REGULATORY_WEEK reuses the crisis baseline; the regulator
            # adds the difficulty by mutating the directive schema.
            CurriculumScenario.REGULATORY_WEEK: MarketStateBuilder._build_crisis_week,
        }
        builder = builders.get(scenario)
        if builder is None:
            raise ValueError(f"Unknown curriculum scenario: {scenario}")
        return builder(rng)

    @staticmethod
    def _build_stable_week(rng: random.Random) -> SimulatedMarketState:
        """Engine 1 only. Predictable demand. No farmer offers. No trends."""
        batches, velocities = _generate_batches(rng)
        return _base_state(batches, velocities)

    @staticmethod
    def _build_busy_weekend(rng: random.Random) -> SimulatedMarketState:
        """Engine 1 + Engine 3. Weekend demand surge. 1 trend signal active."""
        batches, velocities = _generate_batches(rng)

        # 1 active trend signal: fruits, composite 72.0
        # 0.80*0.25 + 0.725*0.30 + 0.75*0.25 + 0.55*0.10 + 0.60*0.10 = 0.72
        trend = SimulatedTrendSignal(
            category="fruits",
            composite_score=72.0,
            signal_source=SignalSource.INSTAGRAM,
            detected_at_tick=0,
            action_taken=TrendAction.PENDING,
            suggested_order_kg=8.0,
            recipe_simplicity=0.80,
            ingredient_rarity=0.725,
            view_velocity=0.75,
            local_relevance=0.55,
            historical_conversion=0.60,
        )

        return _base_state(batches, velocities, trends={"fruits": trend})

    @staticmethod
    def _build_farmer_week(rng: random.Random) -> SimulatedMarketState:
        """Engine 1 + Engine 2. 3 farmer offers across the week (2 at start, 1 mid-episode)."""
        batches, velocities = _generate_batches(rng)

        offer_1 = SimulatedFarmerOffer(
            offer_id="offer_001",
            farmer_name="Ramesh Kumar",
            product_category="fruits",
            product_name="mangoes",
            quantity_kg=40.0,
            offered_price_per_kg=35.0,
            seller_shelf_life_hrs=48,
            offered_at_tick=0,
            status=FarmerOfferStatus.PENDING,
        )
        offer_2 = SimulatedFarmerOffer(
            offer_id="offer_002",
            farmer_name="Lakshmi Devi",
            product_category="vegetables",
            product_name="spinach",
            quantity_kg=15.0,
            offered_price_per_kg=12.0,
            seller_shelf_life_hrs=36,
            offered_at_tick=0,
            status=FarmerOfferStatus.PENDING,
        )

        return _base_state(batches, velocities, offers=[offer_1, offer_2])

    @staticmethod
    def _build_trend_week(rng: random.Random) -> SimulatedMarketState:
        """All 3 engines. 2 trend signals. 1 farmer offer."""
        batches, velocities = _generate_batches(rng)

        # Trend 1: mushrooms, composite 78.0
        # 0.85*0.25 + 0.775*0.30 + 0.80*0.25 + 0.65*0.10 + 0.70*0.10 = 0.78
        trend_mushrooms = SimulatedTrendSignal(
            category="mushrooms",
            composite_score=78.0,
            signal_source=SignalSource.YOUTUBE,
            detected_at_tick=0,
            action_taken=TrendAction.PENDING,
            suggested_order_kg=10.0,
            recipe_simplicity=0.85,
            ingredient_rarity=0.775,
            view_velocity=0.80,
            local_relevance=0.65,
            historical_conversion=0.70,
        )

        # Trend 2: dairy, composite 68.0
        # 0.75*0.25 + 0.675*0.30 + 0.70*0.25 + 0.55*0.10 + 0.60*0.10 = 0.68
        trend_dairy = SimulatedTrendSignal(
            category="dairy",
            composite_score=68.0,
            signal_source=SignalSource.GOOGLE_TRENDS,
            detected_at_tick=0,
            action_taken=TrendAction.PENDING,
            suggested_order_kg=6.0,
            recipe_simplicity=0.75,
            ingredient_rarity=0.675,
            view_velocity=0.70,
            local_relevance=0.55,
            historical_conversion=0.60,
        )

        offer = SimulatedFarmerOffer(
            offer_id="offer_001",
            farmer_name="Priya Sharma",
            product_category="herbs",
            product_name="coriander",
            quantity_kg=8.0,
            offered_price_per_kg=10.0,
            seller_shelf_life_hrs=24,
            offered_at_tick=0,
            status=FarmerOfferStatus.PENDING,
        )

        return _base_state(
            batches, velocities,
            offers=[offer],
            trends={"mushrooms": trend_mushrooms, "dairy": trend_dairy},
        )

    @staticmethod
    def _build_crisis_week(rng: random.Random) -> SimulatedMarketState:
        """All 3 engines simultaneously. Maximum complexity. Depleted risk buffer."""
        batches, velocities = _generate_batches(rng, include_urgent=True)

        # Offer 1: reasonable viability
        offer_1 = SimulatedFarmerOffer(
            offer_id="offer_001",
            farmer_name="Anil Reddy",
            product_category="vegetables",
            product_name="tomatoes",
            quantity_kg=35.0,
            offered_price_per_kg=18.0,
            seller_shelf_life_hrs=36,
            offered_at_tick=0,
            status=FarmerOfferStatus.PENDING,
            viability_score=0.72,
        )

        # Offer 2: borderline viability — tests the agent's judgement
        offer_2 = SimulatedFarmerOffer(
            offer_id="offer_002",
            farmer_name="Meena Bai",
            product_category="dairy",
            product_name="paneer",
            quantity_kg=20.0,
            offered_price_per_kg=45.0,
            seller_shelf_life_hrs=18,
            offered_at_tick=0,
            status=FarmerOfferStatus.PENDING,
            viability_score=0.38,
        )

        # Trend 1: strong signal, composite 82.0
        # 0.90*0.25 + 0.825*0.30 + 0.85*0.25 + 0.70*0.10 + 0.65*0.10 = 0.82
        trend_herbs = SimulatedTrendSignal(
            category="herbs",
            composite_score=82.0,
            signal_source=SignalSource.INSTAGRAM,
            detected_at_tick=0,
            action_taken=TrendAction.PENDING,
            suggested_order_kg=12.0,
            recipe_simplicity=0.90,
            ingredient_rarity=0.825,
            view_velocity=0.85,
            local_relevance=0.70,
            historical_conversion=0.65,
        )

        # Trend 2: just above threshold, composite 67.0
        # 0.70*0.25 + 0.70*0.30 + 0.68*0.25 + 0.60*0.10 + 0.55*0.10 = 0.67
        trend_bakery = SimulatedTrendSignal(
            category="bakery",
            composite_score=67.0,
            signal_source=SignalSource.ZOMATO,
            detected_at_tick=0,
            action_taken=TrendAction.PENDING,
            suggested_order_kg=5.0,
            recipe_simplicity=0.70,
            ingredient_rarity=0.70,
            view_velocity=0.68,
            local_relevance=0.60,
            historical_conversion=0.55,
        )

        return _base_state(
            batches, velocities,
            offers=[offer_1, offer_2],
            trends={"herbs": trend_herbs, "bakery": trend_bakery},
            risk_buffer=RISK_BUFFER_INITIAL_SEED_RS * 0.6,
        )
