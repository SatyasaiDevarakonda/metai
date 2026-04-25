"""Simulation dataclasses for the FreshPrice RL environment.

Pure Python only. No SQLAlchemy. No Pydantic. No ORM.
All thresholds imported from constants — no hardcoded numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from freshprice_env.constants import (
    FLOOR_PRICE_MARGIN,
    TREND_SCORE_THRESHOLD,
    TREND_SIGNAL_EXPIRY_HRS,
    TICKS_PER_DAY,
    URGENCY_CRITICAL_HRS,
    URGENCY_URGENT_HRS,
    URGENCY_WATCH_HRS,
)
from freshprice_env.enums import (
    BatchStatus,
    BatchType,
    ExpiryUrgency,
    ExternalEvent,
    FarmerOfferStatus,
    SignalSource,
    TrendAction,
    WeatherCondition,
)

# ---------------------------------------------------------------------------
# Farmer offer expiry: 8 simulated hours = 32 ticks at 15-min resolution
# ---------------------------------------------------------------------------
FARMER_OFFER_EXPIRY_TICKS: int = 32


@dataclass(frozen=True)
class SimulatedBatch:
    """A single inventory batch inside the simulation."""

    batch_id: str
    sku_id: str
    store_id: str
    category: str
    quantity_remaining: int
    unit_cost: float
    current_price: float
    original_price: float
    hours_to_expiry: float
    batch_type: BatchType
    status: BatchStatus

    @property
    def urgency(self) -> ExpiryUrgency:
        if self.hours_to_expiry > URGENCY_WATCH_HRS:
            return ExpiryUrgency.FRESH
        if self.hours_to_expiry > URGENCY_URGENT_HRS:
            return ExpiryUrgency.WATCH
        if self.hours_to_expiry > URGENCY_CRITICAL_HRS:
            return ExpiryUrgency.URGENT
        return ExpiryUrgency.CRITICAL

    @property
    def floor_price(self) -> float:
        return self.unit_cost * (1.0 + FLOOR_PRICE_MARGIN)

    @property
    def discount_pct(self) -> float:
        if self.original_price == 0.0:
            return 0.0
        return (self.original_price - self.current_price) / self.original_price * 100.0

    @property
    def is_at_risk(self) -> bool:
        return self.urgency in (ExpiryUrgency.URGENT, ExpiryUrgency.CRITICAL)


@dataclass(frozen=True)
class SimulatedFarmerOffer:
    """A farmer surplus offer inside the simulation."""

    offer_id: str
    farmer_name: str
    product_category: str
    product_name: str
    quantity_kg: float
    offered_price_per_kg: float
    seller_shelf_life_hrs: int
    offered_at_tick: int
    status: FarmerOfferStatus
    viability_score: float | None = None
    counter_price: float | None = None

    @property
    def is_pending(self) -> bool:
        return self.status == FarmerOfferStatus.PENDING

    def is_expired(self, current_tick: int) -> bool:
        return current_tick - self.offered_at_tick > FARMER_OFFER_EXPIRY_TICKS


@dataclass(frozen=True)
class SimulatedTrendSignal:
    """A social trend signal inside the simulation."""

    category: str
    composite_score: float
    signal_source: SignalSource
    detected_at_tick: int
    action_taken: TrendAction
    suggested_order_kg: float
    recipe_simplicity: float
    ingredient_rarity: float
    view_velocity: float
    local_relevance: float
    historical_conversion: float

    def is_actionable(self, current_tick: int) -> bool:
        ticks_until_expiry = TREND_SIGNAL_EXPIRY_HRS * (TICKS_PER_DAY / 24.0)
        return (
            self.action_taken == TrendAction.PENDING
            and self.composite_score >= TREND_SCORE_THRESHOLD
            and (current_tick - self.detected_at_tick) < ticks_until_expiry
        )

    @property
    def is_high_confidence(self) -> bool:
        return self.composite_score >= 80.0


@dataclass
class SimulatedMarketState:
    """Mutable shared state that all engines read from and the env mutates each tick."""

    tick: int
    batches: list[SimulatedBatch]
    pending_offers: list[SimulatedFarmerOffer]
    trend_signals: dict[str, SimulatedTrendSignal]
    sales_velocity: dict[str, float]
    risk_buffer_balance: float
    notification_credits: dict[str, int]
    at_risk_cost_accumulator: float
    revenue_recovered_accumulator: float
    # External shocks — set by ExternalShockEngine each tick
    weather_condition: WeatherCondition = WeatherCondition.NORMAL
    active_event: ExternalEvent = ExternalEvent.NONE
    # Consumer demand boost — set by ConsumerAgent each tick (multi-agent mode)
    consumer_demand_boost: dict[str, float] = field(default_factory=dict)

    @property
    def day_of_week(self) -> int:
        return (self.tick // TICKS_PER_DAY) % 7

    @property
    def hour_of_day(self) -> int:
        return (self.tick % TICKS_PER_DAY) // 4

    @property
    def wrr(self) -> float:
        if self.at_risk_cost_accumulator <= 0.0:
            return 0.0
        return self.revenue_recovered_accumulator / self.at_risk_cost_accumulator
