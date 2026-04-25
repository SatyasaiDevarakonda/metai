"""Enumerations for the FreshPrice RL environment."""

from enum import Enum


class ExpiryUrgency(str, Enum):
    """Batch urgency tier based on hours remaining to expiry."""
    FRESH = "FRESH"         # > 72 hours remaining
    WATCH = "WATCH"         # 24-72 hours
    URGENT = "URGENT"       # 6-24 hours
    CRITICAL = "CRITICAL"   # <= 6 hours


class BatchStatus(str, Enum):
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    CLEARED = "CLEARED"
    DONATED = "DONATED"
    LIQUIDATED = "LIQUIDATED"   # Blinkit-style B2B firesale (see LiquidationEngine)


class BatchType(str, Enum):
    REGULAR = "REGULAR"
    FARMER_SURPLUS = "FARMER_SURPLUS"
    TREND_RESTOCK = "TREND_RESTOCK"


class FarmerOfferStatus(str, Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    COUNTERED = "COUNTERED"
    DECLINED = "DECLINED"
    EXPIRED = "EXPIRED"


class TrendAction(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DECLINED = "DECLINED"
    EXPIRED = "EXPIRED"


class BriefEngineType(str, Enum):
    PRICING = "PRICING"
    FARMER = "FARMER"
    TREND = "TREND"


class BriefConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SellerAction(str, Enum):
    AUTO_APPROVED = "AUTO_APPROVED"
    SELLER_APPROVED = "SELLER_APPROVED"
    SELLER_OVERRIDDEN = "SELLER_OVERRIDDEN"


class ViabilityOutcome(str, Enum):
    PASS = "PASS"
    FLAG = "FLAG"
    FAIL = "FAIL"


class CompensationPolicy(str, Enum):
    FULL_PAY = "FULL_PAY"
    PARTIAL_PAY = "PARTIAL_PAY"
    SHARED_LOSS = "SHARED_LOSS"


class SignalSource(str, Enum):
    INSTAGRAM = "INSTAGRAM"
    GOOGLE_TRENDS = "GOOGLE_TRENDS"
    ZOMATO = "ZOMATO"
    YOUTUBE = "YOUTUBE"


class CurriculumScenario(int, Enum):
    """Training curriculum levels 0-5 per FreshPrice strategy Section 8.

    Engine activations per scenario (strategy Section 8 "Five Training
    Scenarios"):

      STABLE_WEEK     = Engine 1 only. Predictable demand baseline.
      BUSY_WEEKEND    = Engines 1 + 4 + 6. Demand surge + multi-store +
                        nearby event.
      FARMER_WEEK     = Engines 1 + 2 + 5. Three farmer offers + micro-
                        manufacturer routing needed.
      TREND_WEEK      = Engines 1 + 3 + 7. Two social trends + surplus
                        box + a festival day.
      CRISIS_WEEK     = All 7 engines simultaneously. The benchmark.
      REGULATORY_WEEK = Continuous schema drift (Patronus AI sub-prize)
                        on top of CRISIS_WEEK loadings.
    """
    STABLE_WEEK = 0
    BUSY_WEEKEND = 1
    FARMER_WEEK = 2
    TREND_WEEK = 3
    CRISIS_WEEK = 4
    REGULATORY_WEEK = 5


# Per-scenario active engine map -- used by the prompt builder, the
# curriculum manager, and the eval scripts. Keys are 1..7 matching the
# strategy's r1..r7 numbering.
ACTIVE_ENGINES_BY_SCENARIO: dict[int, frozenset[int]] = {
    CurriculumScenario.STABLE_WEEK.value:     frozenset({1}),
    CurriculumScenario.BUSY_WEEKEND.value:    frozenset({1, 4, 6}),
    CurriculumScenario.FARMER_WEEK.value:     frozenset({1, 2, 5}),
    CurriculumScenario.TREND_WEEK.value:      frozenset({1, 3, 7}),
    CurriculumScenario.CRISIS_WEEK.value:     frozenset({1, 2, 3, 4, 5, 6, 7}),
    CurriculumScenario.REGULATORY_WEEK.value: frozenset({1, 2, 3, 4, 5, 6, 7}),
}


def active_engines(scenario: "CurriculumScenario") -> frozenset[int]:
    """Return the set of engine IDs (1..7) active in this scenario."""
    return ACTIVE_ENGINES_BY_SCENARIO.get(int(scenario), frozenset())


class WeatherCondition(str, Enum):
    """External weather conditions that affect store footfall and demand."""
    NORMAL = "NORMAL"
    SUNNY = "SUNNY"
    RAINY = "RAINY"   # Reduces footfall 25%
    HOT = "HOT"       # Spikes cold-fruit demand, suppresses heavy dairy
    COLD = "COLD"     # Drives bakery/comfort foods


class ExternalEvent(str, Enum):
    """Local events that create demand spikes."""
    NONE = "NONE"
    FESTIVAL = "FESTIVAL"         # 2-2.5x demand on produce + dairy
    SPORTS_EVENT = "SPORTS_EVENT" # Packaged + dairy spike
    LOCAL_HOLIDAY = "LOCAL_HOLIDAY"  # General 1.3x footfall boost
