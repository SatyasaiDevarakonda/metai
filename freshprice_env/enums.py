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
    """Training curriculum levels 0-4."""
    STABLE_WEEK = 0     # Engine 1 only. Predictable demand.
    BUSY_WEEKEND = 1    # Engine 1 + Engine 3. Weekend demand surge. 1 trend signal.
    FARMER_WEEK = 2     # Engine 1 + Engine 2. 3 farmer offers. No trends.
    TREND_WEEK = 3      # All 3 engines. 2 trend signals. 1 festival day demand spike.
    CRISIS_WEEK = 4     # All 3 engines simultaneously. The benchmark.


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
