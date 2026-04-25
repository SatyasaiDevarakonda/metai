# FreshPrice AI — Spec-Driven Development (Claude Code Edition)
> **How to use this file with Claude Code:**
> Feed one section at a time. Each section has a `<!-- GENERATE -->` instruction block.
> Start with Section 03 (Domain), then 04 (DB), then 06 (Services), then 05 (API), then 08 (Workers), then 10 (Frontend).
> Never feed the whole file at once.

---

## Project Summary

**Product:** FreshPrice AI — Perishable goods intelligence for a single online grocery / dark store seller.

**Three core engines (v1 only):**
1. Dynamic Pricing — auto-discount items as they approach expiry
2. Farmer Offer Engine — accept / counter / decline surplus procurement with risk management
3. Social Trend Engine — detect viral food signals, restock before demand arrives

**Unified metric:** Weekly Waste Recovery Rate (WRR) = revenue from at-risk inventory / cost of at-risk inventory. Target: 61% → 89%.

**Stack:** FastAPI (Python 3.11) · PostgreSQL · Redis · Celery · SQLAlchemy Core · Next.js 14 · TypeScript · Supabase · HuggingFace TRL · Unsloth

---

## Section 01 — Project Structure

<!-- GENERATE: Run this first. Generate the full folder scaffold with empty files and __init__.py where needed. -->

```
freshprice-api/                     # Backend root
├── alembic/                        # DB migrations
│   ├── versions/                   # Migration files go here
│   └── env.py
├── freshprice/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app factory
│   ├── api/
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── routers/
│   │       │   ├── __init__.py
│   │       │   ├── inventory.py
│   │       │   ├── farmer_offers.py
│   │       │   ├── trends.py
│   │       │   ├── briefs.py
│   │       │   └── analytics.py
│   │       └── schemas/
│   │           ├── __init__.py
│   │           ├── common.py       # SuccessResponse, ErrorResponse, PaginationMeta
│   │           ├── inventory.py
│   │           ├── farmer_offers.py
│   │           ├── trends.py
│   │           └── briefs.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── entities/
│   │   │   ├── __init__.py
│   │   │   ├── inventory_batch.py
│   │   │   ├── farmer_offer.py
│   │   │   ├── trend_signal.py
│   │   │   └── operating_brief.py
│   │   ├── value_objects/
│   │   │   ├── __init__.py
│   │   │   ├── expiry_urgency.py
│   │   │   ├── viability_result.py
│   │   │   └── pricing_curve.py
│   │   └── commands/
│   │       ├── __init__.py
│   │       ├── accept_farmer_offer.py
│   │       ├── counter_farmer_offer.py
│   │       ├── decline_farmer_offer.py
│   │       ├── approve_trend_order.py
│   │       └── update_batch_price.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pricing_service.py
│   │   ├── farmer_service.py
│   │   ├── trend_service.py
│   │   ├── brief_service.py
│   │   └── wrr_service.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── inventory_repo.py
│   │   ├── farmer_offer_repo.py
│   │   ├── trend_repo.py
│   │   ├── brief_repo.py
│   │   └── wrr_repo.py
│   ├── workers/
│   │   ├── __init__.py
│   │   ├── celery_app.py
│   │   ├── brief_worker.py
│   │   ├── price_worker.py
│   │   ├── trend_worker.py
│   │   └── wrr_worker.py
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── llm_client.py
│   │   ├── google_trends.py
│   │   ├── instagram.py
│   │   └── whatsapp.py
│   └── core/
│       ├── __init__.py
│       ├── config.py               # pydantic-settings
│       ├── database.py             # SQLAlchemy async engine + session
│       ├── redis.py                # Redis connection pool
│       ├── circuit_breaker.py
│       └── exceptions.py           # All domain exceptions

freshprice-web/                     # Frontend root
├── app/
│   ├── (auth)/
│   │   └── login/
│   │       └── page.tsx
│   └── (dashboard)/
│       ├── layout.tsx
│       ├── operations/
│       │   └── page.tsx
│       ├── briefs/
│       │   ├── page.tsx
│       │   └── [id]/page.tsx
│       ├── farmer-offers/
│       │   ├── page.tsx
│       │   └── [id]/page.tsx
│       ├── trends/
│       │   ├── page.tsx
│       │   └── [id]/page.tsx
│       └── analytics/
│           └── page.tsx
├── components/
│   ├── ui/                         # shadcn/ui primitives
│   ├── inventory/
│   │   ├── BatchCard.tsx
│   │   ├── ExpiryBar.tsx
│   │   └── PriceDisplay.tsx
│   ├── briefs/
│   │   ├── BriefFeed.tsx
│   │   ├── BriefCard.tsx
│   │   └── BriefDetail.tsx
│   ├── farmer/
│   │   ├── OfferCard.tsx
│   │   └── ViabilityCheckTable.tsx
│   ├── trends/
│   │   ├── TrendRadar.tsx
│   │   └── ScoreBreakdown.tsx
│   └── analytics/
│       ├── WRRChart.tsx
│       └── EngineBreakdown.tsx
├── hooks/
│   ├── useRealtimeInventory.ts
│   ├── useRealtimeBriefs.ts
│   └── useRealtimeTrends.ts
├── lib/
│   ├── api-client.ts               # Typed fetch wrapper
│   ├── supabase.ts                 # Supabase client factory
│   └── types/
│       └── api.ts                  # Generated from OpenAPI spec
└── middleware.ts                   # Auth token refresh
```

---

## Section 02 — Configuration & Core

<!-- GENERATE: Generate freshprice/core/config.py, core/database.py, core/redis.py, core/exceptions.py, core/circuit_breaker.py -->

### config.py — All settings from environment variables

```python
# freshprice/core/config.py
from pydantic_settings import BaseSettings
from pydantic import Field
from decimal import Decimal

class Settings(BaseSettings):
    # App
    environment: str = "development"
    log_level: str = "INFO"
    secret_key: str

    # Database
    database_url: str  # postgresql+asyncpg://...

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Supabase
    supabase_url: str
    supabase_service_key: str
    supabase_anon_key: str

    # LLM
    hf_api_key: str
    hf_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_timeout_seconds: int = 40
    llm_max_retries: int = 3

    # External APIs
    google_trends_api_key: str = ""
    instagram_access_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_access_token: str = ""

    # Business Rules
    risk_buffer_initial_seed: Decimal = Decimal("5000")
    trend_score_threshold: float = 65.0
    max_trend_order_multiplier: float = 2.0
    floor_price_margin: float = 0.05  # 5% above unit cost
    notification_credits_per_day: int = 3  # per SKU category

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### database.py — Async SQLAlchemy

```python
# freshprice/core/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from freshprice.core.config import settings
from typing import AsyncGenerator

engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Reconnect on stale connections
    echo=settings.environment == "development",
)

AsyncSessionFactory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionFactory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
```

### exceptions.py — All domain exceptions

```python
# freshprice/core/exceptions.py
# RULE: Never raise HTTPException from services or repositories.
# Only raise these domain exceptions. Routers convert them to HTTP responses.
from uuid import UUID
from decimal import Decimal

class FreshPriceError(Exception):
    """Base exception for all domain errors."""
    pass

# Farmer Offer Errors
class FarmerOfferNotFoundError(FreshPriceError):
    def __init__(self, offer_id: UUID):
        self.offer_id = offer_id
        super().__init__(f"Farmer offer {offer_id} not found")

class InvalidOfferStatusError(FreshPriceError):
    def __init__(self, current: str, expected: str):
        super().__init__(f"Offer status is {current}, expected {expected}")

class ConflictingInventoryError(FreshPriceError):
    def __init__(self, batch_id: UUID):
        self.batch_id = batch_id
        super().__init__(f"Conflicting urgent batch {batch_id} already exists for this category")

class InsufficientRiskBufferError(FreshPriceError):
    def __init__(self, required: Decimal, available: Decimal):
        super().__init__(f"Risk buffer insufficient: need {required}, have {available}")

# Pricing Errors
class BelowFloorPriceError(FreshPriceError):
    def __init__(self, proposed: Decimal, floor: Decimal):
        super().__init__(f"Proposed price {proposed} is below floor price {floor}")

class AntiHackViolationError(FreshPriceError):
    def __init__(self, violation_type: str, detail: str):
        self.violation_type = violation_type
        super().__init__(f"Anti-hack violation [{violation_type}]: {detail}")

# Trend Errors
class TrendSignalNotFoundError(FreshPriceError):
    def __init__(self, trend_id: UUID):
        super().__init__(f"Trend signal {trend_id} not found")

class TrendOrderCapExceededError(FreshPriceError):
    def __init__(self, requested: float, cap: float):
        super().__init__(f"Requested {requested}kg exceeds order cap of {cap}kg")

# Brief Errors
class BriefValidationError(FreshPriceError):
    def __init__(self, missing_section: str):
        self.missing_section = missing_section
        super().__init__(f"Operating brief missing required section: {missing_section}")

class BriefGenerationTimeoutError(FreshPriceError):
    pass

# LLM / External API Errors
class LLMTimeoutError(FreshPriceError):
    pass

class CircuitBreakerOpenError(FreshPriceError):
    def __init__(self, service: str):
        self.service = service
        super().__init__(f"Circuit breaker open for service: {service}")
```

---

## Section 03 — Domain Model

<!-- GENERATE: Generate all files in freshprice/domain/entities/, freshprice/domain/value_objects/, and freshprice/domain/commands/ -->

### Enums (define in a shared enums.py)

```python
# freshprice/domain/enums.py
from enum import Enum

class ExpiryUrgency(str, Enum):
    FRESH    = "FRESH"     # > 72 hours remaining
    WATCH    = "WATCH"     # 24–72 hours
    URGENT   = "URGENT"    # 6–24 hours
    CRITICAL = "CRITICAL"  # <= 6 hours

class BatchStatus(str, Enum):
    ACTIVE  = "ACTIVE"
    EXPIRED = "EXPIRED"
    CLEARED = "CLEARED"
    DONATED = "DONATED"

class BatchType(str, Enum):
    REGULAR         = "REGULAR"
    FARMER_SURPLUS  = "FARMER_SURPLUS"
    TREND_RESTOCK   = "TREND_RESTOCK"

class FarmerOfferStatus(str, Enum):
    PENDING   = "PENDING"
    ACCEPTED  = "ACCEPTED"
    COUNTERED = "COUNTERED"
    DECLINED  = "DECLINED"
    EXPIRED   = "EXPIRED"

class TrendAction(str, Enum):
    PENDING  = "PENDING"
    APPROVED = "APPROVED"
    DECLINED = "DECLINED"
    EXPIRED  = "EXPIRED"

class BriefEngineType(str, Enum):
    PRICING = "PRICING"
    FARMER  = "FARMER"
    TREND   = "TREND"

class BriefConfidence(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"

class SellerAction(str, Enum):
    AUTO_APPROVED    = "AUTO_APPROVED"
    SELLER_APPROVED  = "SELLER_APPROVED"
    SELLER_OVERRIDDEN = "SELLER_OVERRIDDEN"

class ViabilityOutcome(str, Enum):
    PASS = "PASS"
    FLAG = "FLAG"
    FAIL = "FAIL"

class CompensationPolicy(str, Enum):
    FULL_PAY    = "FULL_PAY"
    PARTIAL_PAY = "PARTIAL_PAY"
    SHARED_LOSS = "SHARED_LOSS"

class SignalSource(str, Enum):
    INSTAGRAM    = "INSTAGRAM"
    GOOGLE_TRENDS = "GOOGLE_TRENDS"
    ZOMATO       = "ZOMATO"
    YOUTUBE      = "YOUTUBE"
```

### InventoryBatch entity

```python
# freshprice/domain/entities/inventory_batch.py
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from decimal import Decimal
from datetime import datetime
from freshprice.domain.enums import BatchStatus, BatchType, ExpiryUrgency

@dataclass
class InventoryBatch:
    id: UUID
    sku_id: UUID
    store_id: UUID
    quantity: int
    quantity_remaining: int
    unit_cost: Decimal          # What we paid per unit
    listed_price: Decimal       # Current price on website
    original_price: Decimal     # Full price at stocking time
    expiry_at: datetime
    batch_type: BatchType
    status: BatchStatus = BatchStatus.ACTIVE
    source_id: UUID | None = None   # FK to FarmerOffer or TrendSignal
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def hours_remaining(self) -> float:
        delta = self.expiry_at - datetime.utcnow()
        return max(0.0, delta.total_seconds() / 3600)

    @property
    def urgency(self) -> ExpiryUrgency:
        h = self.hours_remaining
        if h > 72:   return ExpiryUrgency.FRESH
        if h > 24:   return ExpiryUrgency.WATCH
        if h > 6:    return ExpiryUrgency.URGENT
        return ExpiryUrgency.CRITICAL

    @property
    def floor_price(self) -> Decimal:
        """Minimum price — 5% above unit cost. Never sell below this."""
        return self.unit_cost * Decimal("1.05")

    @property
    def discount_pct(self) -> float:
        if self.original_price == 0:
            return 0.0
        return float((self.original_price - self.listed_price) / self.original_price * 100)

    def apply_price(self, new_price: Decimal) -> "InventoryBatch":
        """Returns a new instance with updated price. Does NOT validate — caller must check floor."""
        return InventoryBatch(
            **{**self.__dict__, "listed_price": new_price, "updated_at": datetime.utcnow()}
        )

    def record_sale(self, units_sold: int) -> "InventoryBatch":
        new_qty = max(0, self.quantity_remaining - units_sold)
        new_status = BatchStatus.CLEARED if new_qty == 0 else self.status
        return InventoryBatch(
            **{**self.__dict__,
               "quantity_remaining": new_qty,
               "status": new_status,
               "updated_at": datetime.utcnow()}
        )

    def mark_expired(self) -> "InventoryBatch":
        return InventoryBatch(
            **{**self.__dict__, "status": BatchStatus.EXPIRED, "updated_at": datetime.utcnow()}
        )

    @classmethod
    def create(cls, sku_id: UUID, store_id: UUID, quantity: int,
               unit_cost: Decimal, original_price: Decimal,
               expiry_at: datetime, batch_type: BatchType = BatchType.REGULAR,
               source_id: UUID | None = None) -> "InventoryBatch":
        return cls(
            id=uuid4(),
            sku_id=sku_id,
            store_id=store_id,
            quantity=quantity,
            quantity_remaining=quantity,
            unit_cost=unit_cost,
            listed_price=original_price,
            original_price=original_price,
            expiry_at=expiry_at,
            batch_type=batch_type,
            source_id=source_id,
        )
```

### FarmerOffer entity

```python
# freshprice/domain/entities/farmer_offer.py
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from decimal import Decimal
from datetime import datetime
from freshprice.domain.enums import FarmerOfferStatus, CompensationPolicy

@dataclass
class ViabilityCheckResult:
    shelf_life:        str  # PASS | FLAG | FAIL
    inventory_conflict: str
    break_even:        str
    worst_case_pl:     str
    demand_timing:     str
    justifications:    dict[str, str]  # factor -> plain English reason

    @property
    def is_viable(self) -> bool:
        outcomes = [self.shelf_life, self.inventory_conflict,
                    self.break_even, self.worst_case_pl, self.demand_timing]
        return "FAIL" not in outcomes

    @property
    def composite_score(self) -> float:
        weights = {"PASS": 1.0, "FLAG": 0.5, "FAIL": 0.0}
        outcomes = [self.shelf_life, self.inventory_conflict,
                    self.break_even, self.worst_case_pl, self.demand_timing]
        return sum(weights[o] for o in outcomes) / len(outcomes)

@dataclass
class FarmerOffer:
    id: UUID
    farmer_name: str
    farmer_phone_hash: str          # SHA-256 hash — never store raw
    product_category: str
    product_name: str
    quantity_kg: Decimal
    offered_price_per_kg: Decimal
    seller_shelf_life_hrs: int
    offered_at: datetime
    status: FarmerOfferStatus = FarmerOfferStatus.PENDING
    counter_price: Decimal | None = None
    final_intake_price: Decimal | None = None
    viability_score: Decimal | None = None
    expected_profit_min: Decimal | None = None
    expected_profit_max: Decimal | None = None
    operating_brief_id: UUID | None = None
    inventory_batch_id: UUID | None = None
    resolved_at: datetime | None = None

    def accept(self, final_price: Decimal) -> "FarmerOffer":
        return FarmerOffer(**{
            **self.__dict__,
            "status": FarmerOfferStatus.ACCEPTED,
            "final_intake_price": final_price,
            "resolved_at": datetime.utcnow(),
        })

    def counter(self, counter_price: Decimal) -> "FarmerOffer":
        return FarmerOffer(**{
            **self.__dict__,
            "status": FarmerOfferStatus.COUNTERED,
            "counter_price": counter_price,
        })

    def decline(self) -> "FarmerOffer":
        return FarmerOffer(**{
            **self.__dict__,
            "status": FarmerOfferStatus.DECLINED,
            "resolved_at": datetime.utcnow(),
        })

    @classmethod
    def create(cls, farmer_name: str, farmer_phone_hash: str,
               product_category: str, product_name: str,
               quantity_kg: Decimal, offered_price_per_kg: Decimal,
               seller_shelf_life_hrs: int) -> "FarmerOffer":
        return cls(
            id=uuid4(),
            farmer_name=farmer_name,
            farmer_phone_hash=farmer_phone_hash,
            product_category=product_category,
            product_name=product_name,
            quantity_kg=quantity_kg,
            offered_price_per_kg=offered_price_per_kg,
            seller_shelf_life_hrs=seller_shelf_life_hrs,
            offered_at=datetime.utcnow(),
        )

@dataclass
class FarmerOfferOutcome:
    id: UUID
    farmer_offer_id: UUID
    actual_sell_through_pct: Decimal
    actual_revenue: Decimal
    actual_cost: Decimal
    net_profit: Decimal
    risk_buffer_impact: Decimal     # Positive = added, Negative = drawn
    compensation_policy: CompensationPolicy
    resolved_at: datetime = field(default_factory=datetime.utcnow)
```

### TrendSignal entity

```python
# freshprice/domain/entities/trend_signal.py
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from datetime import datetime
from freshprice.domain.enums import TrendAction, SignalSource

@dataclass
class TrendSignal:
    id: UUID
    product_category: str
    signal_source: SignalSource
    source_url: str | None
    raw_score: int                      # 0-100 from source
    composite_score: float              # Weighted 0-100

    # The 5 scoring factors (each 0-1)
    recipe_simplicity: float
    ingredient_rarity: float
    view_velocity: float
    local_relevance: float
    historical_conversion: float

    detected_at: datetime
    action_taken: TrendAction = TrendAction.PENDING
    purchase_order_id: UUID | None = None
    operating_brief_id: UUID | None = None
    suggested_order_kg: float | None = None
    expires_at: datetime | None = None  # Signal expires 48hrs after detection

    @property
    def is_actionable(self) -> bool:
        return (
            self.action_taken == TrendAction.PENDING
            and self.composite_score >= 65.0
            and (self.expires_at is None or self.expires_at > datetime.utcnow())
        )

    def approve(self, purchase_order_id: UUID) -> "TrendSignal":
        return TrendSignal(**{
            **self.__dict__,
            "action_taken": TrendAction.APPROVED,
            "purchase_order_id": purchase_order_id,
        })

    def decline(self) -> "TrendSignal":
        return TrendSignal(**{**self.__dict__, "action_taken": TrendAction.DECLINED})

    @staticmethod
    def calculate_composite(recipe_simplicity: float, ingredient_rarity: float,
                             view_velocity: float, local_relevance: float,
                             historical_conversion: float) -> float:
        return (
            recipe_simplicity    * 0.25 +
            ingredient_rarity    * 0.30 +
            view_velocity        * 0.25 +
            local_relevance      * 0.10 +
            historical_conversion * 0.10
        ) * 100

    @classmethod
    def create(cls, product_category: str, signal_source: SignalSource,
               raw_score: int, source_url: str | None,
               recipe_simplicity: float, ingredient_rarity: float,
               view_velocity: float, local_relevance: float,
               historical_conversion: float,
               suggested_order_kg: float | None = None) -> "TrendSignal":
        composite = cls.calculate_composite(
            recipe_simplicity, ingredient_rarity, view_velocity,
            local_relevance, historical_conversion
        )
        now = datetime.utcnow()
        from datetime import timedelta
        return cls(
            id=uuid4(),
            product_category=product_category,
            signal_source=signal_source,
            source_url=source_url,
            raw_score=raw_score,
            composite_score=composite,
            recipe_simplicity=recipe_simplicity,
            ingredient_rarity=ingredient_rarity,
            view_velocity=view_velocity,
            local_relevance=local_relevance,
            historical_conversion=historical_conversion,
            detected_at=now,
            expires_at=now + timedelta(hours=48),
            suggested_order_kg=suggested_order_kg,
        )
```

### OperatingBrief entity

```python
# freshprice/domain/entities/operating_brief.py
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from datetime import datetime
from freshprice.domain.enums import BriefEngineType, BriefConfidence, SellerAction

REQUIRED_SECTIONS = ["situation", "recommendation", "directive", "confidence"]
CONDITIONAL_SECTIONS = {
    BriefEngineType.FARMER: ["viability_check"],
    BriefEngineType.TREND:  ["signal_analysis", "viability_check"],
    BriefEngineType.PRICING: [],
}

@dataclass
class OperatingBrief:
    id: UUID
    engine_type: BriefEngineType
    situation: str
    recommendation: str
    directive: dict                  # Machine-readable: {"action": ..., "params": {...}}
    confidence: BriefConfidence
    signal_analysis: str | None      # Only for TREND briefs
    viability_check: dict | None     # Only for FARMER and TREND briefs
    quality_score: float | None = None
    model_version: str | None = None
    generation_ms: int | None = None
    seller_action: SellerAction = SellerAction.AUTO_APPROVED
    created_at: datetime = field(default_factory=datetime.utcnow)

    def validate(self) -> None:
        """Raise BriefValidationError if required sections are missing."""
        from freshprice.core.exceptions import BriefValidationError
        for section in REQUIRED_SECTIONS:
            if not getattr(self, section, None):
                raise BriefValidationError(section)
        for section in CONDITIONAL_SECTIONS.get(self.engine_type, []):
            if not getattr(self, section, None):
                raise BriefValidationError(section)

    @classmethod
    def create(cls, engine_type: BriefEngineType, situation: str,
               recommendation: str, directive: dict, confidence: BriefConfidence,
               signal_analysis: str | None = None,
               viability_check: dict | None = None,
               model_version: str | None = None,
               generation_ms: int | None = None) -> "OperatingBrief":
        brief = cls(
            id=uuid4(),
            engine_type=engine_type,
            situation=situation,
            recommendation=recommendation,
            directive=directive,
            confidence=confidence,
            signal_analysis=signal_analysis,
            viability_check=viability_check,
            model_version=model_version,
            generation_ms=generation_ms,
        )
        brief.validate()
        return brief
```

### Commands

```python
# freshprice/domain/commands/accept_farmer_offer.py
from dataclasses import dataclass, field
from uuid import UUID
from decimal import Decimal
from datetime import datetime

@dataclass(frozen=True)
class AcceptFarmerOfferCommand:
    offer_id: UUID
    final_intake_price: Decimal
    seller_id: UUID
    store_id: UUID
    confirmed_at: datetime = field(default_factory=datetime.utcnow)

# freshprice/domain/commands/counter_farmer_offer.py
@dataclass(frozen=True)
class CounterFarmerOfferCommand:
    offer_id: UUID
    counter_price: Decimal
    seller_id: UUID

# freshprice/domain/commands/decline_farmer_offer.py
@dataclass(frozen=True)
class DeclineFarmerOfferCommand:
    offer_id: UUID
    seller_id: UUID
    reason: str  # Required for model learning

# freshprice/domain/commands/approve_trend_order.py
@dataclass(frozen=True)
class ApproveTrendOrderCommand:
    trend_id: UUID
    final_quantity_kg: float
    seller_id: UUID
    store_id: UUID

# freshprice/domain/commands/update_batch_price.py
@dataclass(frozen=True)
class UpdateBatchPriceCommand:
    batch_id: UUID
    new_price: Decimal
    trigger: str  # "scheduled_recalc" | "brief_directive" | "seller_override"
    seller_id: UUID | None = None  # Only set for seller overrides
```

---

## Section 04 — Database Schema (PostgreSQL)

<!-- GENERATE: Generate alembic/versions/20260423_000001_initial_schema.py with the full migration. -->

```sql
-- Full schema in SQL for reference — Alembic migration generates equivalent Python

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Product SKUs
CREATE TABLE product_sku (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                  VARCHAR(100) NOT NULL,
    category              VARCHAR(50) NOT NULL,
    unit                  VARCHAR(20) NOT NULL,
    standard_shelf_life_hrs INTEGER NOT NULL,
    cost_price_avg        DECIMAL(8,2) NOT NULL DEFAULT 0,
    avg_daily_velocity    DECIMAL(8,3) NOT NULL DEFAULT 0,
    price_elasticity      DECIMAL(5,3) NOT NULL DEFAULT 0,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Store locations
CREATE TABLE store_location (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        VARCHAR(100) NOT NULL,
    address     TEXT,
    lat         DECIMAL(9,6),
    lng         DECIMAL(9,6),
    is_primary  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Inventory batches
CREATE TABLE inventory_batch (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sku_id              UUID NOT NULL REFERENCES product_sku(id),
    store_id            UUID NOT NULL REFERENCES store_location(id),
    quantity            INTEGER NOT NULL CHECK (quantity > 0),
    quantity_remaining  INTEGER NOT NULL CHECK (quantity_remaining >= 0),
    unit_cost           DECIMAL(10,2) NOT NULL,
    listed_price        DECIMAL(10,2) NOT NULL,
    original_price      DECIMAL(10,2) NOT NULL,
    expiry_at           TIMESTAMPTZ NOT NULL,
    batch_type          VARCHAR(20) NOT NULL DEFAULT 'REGULAR',
    source_id           UUID,
    status              VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Farmer offers
CREATE TABLE farmer_offer (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farmer_name             VARCHAR(100) NOT NULL,
    farmer_phone_hash       VARCHAR(64) NOT NULL,
    product_category        VARCHAR(50) NOT NULL,
    product_name            VARCHAR(100) NOT NULL,
    quantity_kg             DECIMAL(8,2) NOT NULL,
    offered_price_per_kg    DECIMAL(8,2) NOT NULL,
    seller_shelf_life_hrs   INTEGER NOT NULL,
    offered_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status                  VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    counter_price           DECIMAL(8,2),
    final_intake_price      DECIMAL(8,2),
    viability_score         DECIMAL(4,2),
    expected_profit_min     DECIMAL(10,2),
    expected_profit_max     DECIMAL(10,2),
    operating_brief_id      UUID,
    inventory_batch_id      UUID REFERENCES inventory_batch(id),
    resolved_at             TIMESTAMPTZ
);

-- Farmer offer outcomes (append-only)
CREATE TABLE farmer_offer_outcome (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farmer_offer_id         UUID NOT NULL REFERENCES farmer_offer(id),
    actual_sell_through_pct DECIMAL(5,2) NOT NULL,
    actual_revenue          DECIMAL(10,2) NOT NULL,
    actual_cost             DECIMAL(10,2) NOT NULL,
    net_profit              DECIMAL(10,2) NOT NULL,
    risk_buffer_impact      DECIMAL(10,2) NOT NULL,
    compensation_policy     VARCHAR(20) NOT NULL,
    resolved_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Trend signals
CREATE TABLE trend_signal (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_category        VARCHAR(50) NOT NULL,
    signal_source           VARCHAR(30) NOT NULL,
    source_url              VARCHAR(500),
    raw_score               INTEGER NOT NULL,
    composite_score         DECIMAL(5,2) NOT NULL,
    recipe_simplicity       DECIMAL(3,2) NOT NULL,
    ingredient_rarity       DECIMAL(3,2) NOT NULL,
    view_velocity           DECIMAL(3,2) NOT NULL,
    local_relevance         DECIMAL(3,2) NOT NULL,
    historical_conversion   DECIMAL(3,2) NOT NULL,
    detected_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action_taken            VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    purchase_order_id       UUID,
    operating_brief_id      UUID,
    suggested_order_kg      DECIMAL(8,2),
    expires_at              TIMESTAMPTZ
);

-- Operating briefs
CREATE TABLE operating_brief (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    engine_type      VARCHAR(20) NOT NULL,
    situation        TEXT NOT NULL,
    signal_analysis  TEXT,
    viability_check  JSONB,
    recommendation   TEXT NOT NULL,
    directive        JSONB NOT NULL,
    confidence       VARCHAR(10) NOT NULL,
    quality_score    DECIMAL(3,2),
    model_version    VARCHAR(50),
    generation_ms    INTEGER,
    seller_action    VARCHAR(30) NOT NULL DEFAULT 'AUTO_APPROVED',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- WRR snapshots (append-only)
CREATE TABLE wrr_snapshot (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_date               DATE NOT NULL UNIQUE,
    at_risk_inventory_cost      DECIMAL(12,2) NOT NULL,
    revenue_recovered           DECIMAL(12,2) NOT NULL,
    units_wasted                INTEGER NOT NULL,
    units_sold                  INTEGER NOT NULL,
    wrr                         DECIMAL(5,4) NOT NULL,
    engine_breakdown            JSONB NOT NULL,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Risk buffer ledger (append-only)
CREATE TABLE risk_buffer_ledger (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_type VARCHAR(10) NOT NULL,
    amount           DECIMAL(10,2) NOT NULL,
    source_type      VARCHAR(30) NOT NULL,
    source_id        UUID NOT NULL,
    balance_after    DECIMAL(10,2) NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- INDEXES
CREATE INDEX idx_inventory_batch_expiry_active
    ON inventory_batch (store_id, expiry_at, status)
    WHERE status = 'ACTIVE';

CREATE INDEX idx_inventory_critical
    ON inventory_batch (store_id, expiry_at)
    WHERE status = 'ACTIVE' AND expiry_at <= NOW() + INTERVAL '6 hours';

CREATE INDEX idx_farmer_offer_pending
    ON farmer_offer (status, offered_at DESC)
    WHERE status = 'PENDING';

CREATE INDEX idx_trend_signal_pending
    ON trend_signal (product_category, composite_score DESC)
    WHERE action_taken = 'PENDING';

CREATE INDEX idx_operating_brief_engine_recent
    ON operating_brief (engine_type, created_at DESC);

CREATE INDEX idx_wrr_snapshot_date
    ON wrr_snapshot (snapshot_date DESC);
```

---

## Section 05 — API Schemas (Pydantic)

<!-- GENERATE: Generate all files in freshprice/api/v1/schemas/ -->

### common.py — Response envelope

```python
# freshprice/api/v1/schemas/common.py
from pydantic import BaseModel
from typing import TypeVar, Generic
from uuid import UUID

T = TypeVar("T")

class PaginationMeta(BaseModel):
    total: int
    next_cursor: str | None
    has_more: bool

class SuccessResponse(BaseModel, Generic[T]):
    success: bool = True
    data: T
    error: None = None
    meta: PaginationMeta | None = None

class ErrorDetail(BaseModel):
    code: str
    message: str
    fields: dict[str, list[str]] | None = None

class ErrorResponse(BaseModel):
    success: bool = False
    data: None = None
    error: ErrorDetail

class AsyncJobResponse(BaseModel):
    job_id: UUID
    status: str = "PENDING"
    estimated_seconds: int
    poll_url: str
```

### farmer_offers.py — Request/response schemas

```python
# freshprice/api/v1/schemas/farmer_offers.py
from pydantic import BaseModel, Field, field_validator
from uuid import UUID
from decimal import Decimal
from datetime import datetime
from freshprice.domain.enums import FarmerOfferStatus

class SubmitFarmerOfferRequest(BaseModel):
    farmer_name: str = Field(..., min_length=2, max_length=100)
    farmer_phone: str = Field(..., pattern=r"^\+?[0-9]{10,15}$")
    product_category: str = Field(..., min_length=2, max_length=50)
    product_name: str = Field(..., min_length=2, max_length=100)
    quantity_kg: Decimal = Field(..., gt=0, le=10000)
    offered_price_per_kg: Decimal = Field(..., gt=0)
    seller_shelf_life_hrs: int = Field(..., gt=0, le=720)

class AcceptOfferRequest(BaseModel):
    final_intake_price: Decimal = Field(..., gt=0)
    store_id: UUID

class CounterOfferRequest(BaseModel):
    counter_price: Decimal = Field(..., gt=0)

class DeclineOfferRequest(BaseModel):
    reason: str = Field(..., min_length=5, max_length=500)

class ViabilityCheckResponse(BaseModel):
    shelf_life: str
    inventory_conflict: str
    break_even: str
    worst_case_pl: str
    demand_timing: str
    justifications: dict[str, str]
    is_viable: bool
    composite_score: float

class FarmerOfferResponse(BaseModel):
    id: UUID
    farmer_name: str
    product_category: str
    product_name: str
    quantity_kg: str           # Decimal as string
    offered_price_per_kg: str
    seller_shelf_life_hrs: int
    offered_at: datetime
    status: FarmerOfferStatus
    viability_check: ViabilityCheckResponse | None
    operating_brief_id: UUID | None
    expected_profit_min: str | None
    expected_profit_max: str | None

    class Config:
        from_attributes = True
```

---

## Section 06 — Service Layer

<!-- GENERATE: Generate freshprice/services/farmer_service.py, pricing_service.py, trend_service.py, brief_service.py, wrr_service.py -->

### FarmerService — full interface

```python
# freshprice/services/farmer_service.py
# RULES:
# - Never import SQLAlchemy here. Only use repositories.
# - Never raise HTTPException. Only raise domain exceptions from core/exceptions.py.
# - All methods async.
# - Every write operation takes a typed Command object.

from uuid import UUID
from decimal import Decimal
from freshprice.domain.commands.accept_farmer_offer import AcceptFarmerOfferCommand
from freshprice.domain.commands.counter_farmer_offer import CounterFarmerOfferCommand
from freshprice.domain.commands.decline_farmer_offer import DeclineFarmerOfferCommand
from freshprice.domain.entities.farmer_offer import FarmerOffer, FarmerOfferOutcome, ViabilityCheckResult
from freshprice.repositories.farmer_offer_repo import FarmerOfferRepository
from freshprice.repositories.inventory_repo import InventoryBatchRepository
from freshprice.services.brief_service import BriefService
from freshprice.core.exceptions import (
    FarmerOfferNotFoundError, InvalidOfferStatusError,
    ConflictingInventoryError, InsufficientRiskBufferError
)
from freshprice.domain.enums import FarmerOfferStatus, BatchType
import hashlib

class FarmerService:
    def __init__(
        self,
        offer_repo: FarmerOfferRepository,
        batch_repo: InventoryBatchRepository,
        brief_service: BriefService,
    ):
        self._offers = offer_repo
        self._batches = batch_repo
        self._briefs = brief_service

    async def submit_offer(
        self,
        farmer_name: str,
        farmer_phone: str,
        product_category: str,
        product_name: str,
        quantity_kg: Decimal,
        offered_price_per_kg: Decimal,
        seller_shelf_life_hrs: int,
    ) -> tuple[FarmerOffer, str]:
        """
        Submit a new farmer offer.
        Returns (saved_offer, job_id) — brief generation is async.
        """
        phone_hash = hashlib.sha256(farmer_phone.encode()).hexdigest()
        offer = FarmerOffer.create(
            farmer_name=farmer_name,
            farmer_phone_hash=phone_hash,
            product_category=product_category,
            product_name=product_name,
            quantity_kg=quantity_kg,
            offered_price_per_kg=offered_price_per_kg,
            seller_shelf_life_hrs=seller_shelf_life_hrs,
        )
        await self._offers.save(offer)

        # Trigger async brief generation (returns job_id)
        job_id = await self._briefs.enqueue_farmer_brief(offer)
        return offer, job_id

    async def accept_offer(self, cmd: AcceptFarmerOfferCommand) -> FarmerOffer:
        offer = await self._offers.get_by_id(cmd.offer_id)
        if offer is None:
            raise FarmerOfferNotFoundError(cmd.offer_id)
        if offer.status != FarmerOfferStatus.PENDING:
            raise InvalidOfferStatusError(offer.status, FarmerOfferStatus.PENDING)

        # Business rule: cannot accept if same category has conflicting URGENT batch
        conflict = await self._batches.find_urgent_by_category(
            category=offer.product_category, store_id=cmd.store_id
        )
        if conflict:
            raise ConflictingInventoryError(conflict.id)

        accepted = offer.accept(final_price=cmd.final_intake_price)
        await self._offers.save(accepted)

        # Create inventory batch from the accepted offer
        from freshprice.domain.entities.inventory_batch import InventoryBatch
        from datetime import datetime, timedelta
        expiry_at = datetime.utcnow() + timedelta(hours=offer.seller_shelf_life_hrs)
        batch = InventoryBatch.create(
            sku_id=await self._batches.find_or_create_sku_id(
                offer.product_name, offer.product_category
            ),
            store_id=cmd.store_id,
            quantity=int(offer.quantity_kg),
            unit_cost=cmd.final_intake_price,
            original_price=await self._batches.get_market_price(offer.product_category),
            expiry_at=expiry_at,
            batch_type=BatchType.FARMER_SURPLUS,
            source_id=offer.id,
        )
        await self._batches.save(batch)

        # Update offer with batch link
        linked = FarmerOffer(**{**accepted.__dict__, "inventory_batch_id": batch.id})
        await self._offers.save(linked)

        return linked

    async def calculate_viability(self, offer: FarmerOffer, store_id: UUID) -> ViabilityCheckResult:
        """Run the 5-factor viability check. Used by brief generation."""
        from freshprice.domain.value_objects.viability_result import ViabilityCalculator
        calculator = ViabilityCalculator(
            offer_repo=self._offers,
            batch_repo=self._batches,
        )
        return await calculator.calculate(offer, store_id)
```

### PricingService — interface spec

```python
# freshprice/services/pricing_service.py
# Strategy pattern: pluggable pricing curve algorithms

from decimal import Decimal
from uuid import UUID
from abc import ABC, abstractmethod
from freshprice.domain.entities.inventory_batch import InventoryBatch
from freshprice.domain.enums import ExpiryUrgency

# STRATEGY INTERFACE
class PricingCurveStrategy(ABC):
    @abstractmethod
    def calculate_price(self, batch: InventoryBatch, current_velocity: float) -> Decimal:
        pass

# CONCRETE STRATEGY 1: Velocity-Adaptive (default)
class VelocityAdaptiveStrategy(PricingCurveStrategy):
    """
    Adjusts price based on gap between current velocity and required velocity.
    required_velocity = quantity_remaining / hours_remaining
    If current_velocity < required_velocity: discount more
    If current_velocity > required_velocity: hold or nudge up
    """
    def calculate_price(self, batch: InventoryBatch, current_velocity: float) -> Decimal:
        if batch.hours_remaining <= 0:
            return batch.floor_price
        required_velocity = batch.quantity_remaining / batch.hours_remaining
        if required_velocity == 0:
            return batch.original_price
        velocity_ratio = current_velocity / required_velocity
        # velocity_ratio > 1: selling fast enough, hold price
        # velocity_ratio < 1: selling too slow, discount
        urgency_factor = self._urgency_multiplier(batch.urgency)
        discount = max(0, (1 - velocity_ratio) * urgency_factor)
        new_price = batch.original_price * Decimal(str(1 - min(discount, 0.75)))
        return max(batch.floor_price, new_price.quantize(Decimal("0.01")))

    def _urgency_multiplier(self, urgency: ExpiryUrgency) -> float:
        return {
            ExpiryUrgency.FRESH:    0.10,
            ExpiryUrgency.WATCH:    0.25,
            ExpiryUrgency.URGENT:   0.50,
            ExpiryUrgency.CRITICAL: 0.75,
        }[urgency]

# CONCRETE STRATEGY 2: Step Decay
class StepDecayStrategy(PricingCurveStrategy):
    """Fixed discount steps at each urgency tier."""
    DISCOUNTS = {
        ExpiryUrgency.FRESH:    0.00,
        ExpiryUrgency.WATCH:    0.10,
        ExpiryUrgency.URGENT:   0.30,
        ExpiryUrgency.CRITICAL: 0.55,
    }
    def calculate_price(self, batch: InventoryBatch, current_velocity: float) -> Decimal:
        discount = self.DISCOUNTS[batch.urgency]
        new_price = batch.original_price * Decimal(str(1 - discount))
        return max(batch.floor_price, new_price.quantize(Decimal("0.01")))


class AntiHackGuard:
    """
    Checks every proposed price for hacking patterns before applying.
    Raises AntiHackViolationError on detection.
    """
    def check(self, batch: InventoryBatch, proposed_price: Decimal) -> None:
        from freshprice.core.exceptions import AntiHackViolationError
        # Guard 1: Deep discount on fresh stock
        discount_pct = float((batch.original_price - proposed_price) / batch.original_price)
        if discount_pct > 0.65 and batch.hours_remaining > 48:
            raise AntiHackViolationError(
                "EARLY_DEEP_DISCOUNT",
                f"Cannot apply {discount_pct:.0%} discount with {batch.hours_remaining:.0f}hrs remaining"
            )
        # Guard 2: Below floor price
        if proposed_price < batch.floor_price:
            raise AntiHackViolationError(
                "BELOW_FLOOR_PRICE",
                f"Price {proposed_price} is below floor {batch.floor_price}"
            )
```

---

## Section 07 — Repository Layer

<!-- GENERATE: Generate all files in freshprice/repositories/ -->

### Repository interface contract

```python
# freshprice/repositories/inventory_repo.py
# RULES:
# - Only file that imports SQLAlchemy tables.
# - Returns domain entities, never ORM objects.
# - All queries use explicit column selection — no SELECT *.
# - All methods async.

from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update
from freshprice.domain.entities.inventory_batch import InventoryBatch
from freshprice.domain.enums import BatchStatus, ExpiryUrgency
from datetime import datetime, timedelta

class InventoryBatchRepository:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def get_by_id(self, batch_id: UUID) -> InventoryBatch | None:
        # SELECT explicit columns WHERE id = batch_id
        # Return InventoryBatch.from_row(row) or None
        pass

    async def find_active_by_store(
        self, store_id: UUID, urgency: ExpiryUrgency | None = None
    ) -> list[InventoryBatch]:
        # SELECT ... WHERE store_id = ? AND status = 'ACTIVE'
        # Optional filter: AND urgency tier computed from expiry_at
        pass

    async def find_urgent_by_category(
        self, category: str, store_id: UUID
    ) -> InventoryBatch | None:
        # SELECT first batch WHERE store_id = ? AND category = ? AND status = 'ACTIVE'
        # AND expiry_at <= NOW() + 24hrs
        pass

    async def save(self, batch: InventoryBatch) -> None:
        # UPSERT on id
        pass

    async def find_or_create_sku_id(self, product_name: str, category: str) -> UUID:
        # Find existing SKU or create new one
        pass

    async def get_market_price(self, category: str) -> "Decimal":
        # Return average listed_price for this category over last 7 days
        pass
```

---

## Section 08 — Celery Workers

<!-- GENERATE: Generate freshprice/workers/celery_app.py, brief_worker.py, price_worker.py, trend_worker.py -->

### celery_app.py

```python
# freshprice/workers/celery_app.py
from celery import Celery
from celery.schedules import crontab
from freshprice.core.config import settings

celery_app = Celery(
    "freshprice",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_routes={
        "freshprice.workers.brief_worker.*":  {"queue": "briefs"},
        "freshprice.workers.price_worker.*":  {"queue": "pricing"},
        "freshprice.workers.trend_worker.*":  {"queue": "trends"},
        "freshprice.workers.wrr_worker.*":    {"queue": "analytics"},
    },
    beat_schedule={
        "recalculate-prices-every-15-min": {
            "task": "freshprice.workers.price_worker.recalculate_all_batch_prices",
            "schedule": crontab(minute="*/15"),
        },
        "poll-social-trends-hourly": {
            "task": "freshprice.workers.trend_worker.poll_all_sources",
            "schedule": crontab(minute=0),
        },
        "calculate-daily-wrr": {
            "task": "freshprice.workers.wrr_worker.calculate_daily_wrr",
            "schedule": crontab(hour=23, minute=59),
        },
        "resolve-farmer-outcomes": {
            "task": "freshprice.workers.brief_worker.resolve_pending_farmer_outcomes",
            "schedule": crontab(minute="*/30"),
        },
    }
)
```

### brief_worker.py — Operating Brief generation

```python
# freshprice/workers/brief_worker.py
# RULES:
# - Never block. All LLM calls are in Celery tasks.
# - Retry on LLMTimeoutError with exponential backoff.
# - Never swallow exceptions silently — always log structured data.
# - After every successful brief: broadcast via Supabase Realtime.

from celery import shared_task
from freshprice.workers.celery_app import celery_app
from freshprice.core.exceptions import LLMTimeoutError, BriefValidationError

@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    queue="briefs",
)
def generate_operating_brief(self, trigger_payload: dict) -> dict:
    """
    trigger_payload schema:
    {
        "engine_type": "FARMER" | "PRICING" | "TREND",
        "context_id": "uuid-string",
        "state_snapshot": {
            # For FARMER: offer fields + viability result
            # For PRICING: batch fields + current velocity
            # For TREND: signal fields + suggested order
        }
    }

    Returns: {"brief_id": "uuid", "quality_score": float}

    On LLMTimeoutError: retries up to 3 times with backoff.
    On BriefValidationError: does NOT retry — logs for SFT improvement.
    """
    # Implementation:
    # 1. Build prompt via OperatingBriefFactory.for_engine(engine_type).build_prompt(state_snapshot)
    # 2. Call llm_client.generate(prompt, max_new_tokens=800, temperature=0.3, timeout=40)
    # 3. Parse with BriefParser.parse(raw, engine_type)
    # 4. Validate with BriefValidator.validate(brief)
    # 5. Score with BriefQualityScorer.score(brief, state_snapshot)
    # 6. Save to DB via brief_repo.save(brief, quality_score)
    # 7. Cache: redis.set(f"freshprice:latest_brief:{engine_type}", brief.to_json(), ex=86400)
    # 8. Broadcast: supabase.realtime.broadcast("briefs", brief.to_dict())
    # 9. Return {"brief_id": str(saved.id), "quality_score": quality_score}
    pass
```

---

## Section 09 — API Routers

<!-- GENERATE: Generate all files in freshprice/api/v1/routers/ -->

### Router contract (farmer_offers.py as example)

```python
# freshprice/api/v1/routers/farmer_offers.py
# RULES:
# - Routers are THIN. Only: validate request, call service, return response.
# - No business logic in routers. Ever.
# - Convert domain exceptions to HTTPException here (and only here).
# - All endpoints require Depends(get_current_seller).

from fastapi import APIRouter, Depends, HTTPException, status
from uuid import UUID
from freshprice.api.v1.schemas.farmer_offers import (
    SubmitFarmerOfferRequest, AcceptOfferRequest,
    CounterOfferRequest, DeclineOfferRequest, FarmerOfferResponse
)
from freshprice.api.v1.schemas.common import SuccessResponse, AsyncJobResponse
from freshprice.domain.commands.accept_farmer_offer import AcceptFarmerOfferCommand
from freshprice.core.exceptions import (
    FarmerOfferNotFoundError, InvalidOfferStatusError, ConflictingInventoryError
)

router = APIRouter(prefix="/farmer-offers", tags=["Farmer Offers"])

@router.get("", response_model=SuccessResponse[list[FarmerOfferResponse]])
async def list_farmer_offers(
    status_filter: str | None = None,
    limit: int = 20,
    cursor: str | None = None,
    # seller: Seller = Depends(get_current_seller),
    # farmer_service: FarmerService = Depends(get_farmer_service),
):
    # offers = await farmer_service.list_offers(status=status_filter, limit=limit, cursor=cursor)
    # return SuccessResponse(data=offers)
    pass

@router.post("", response_model=SuccessResponse[AsyncJobResponse], status_code=201)
async def submit_farmer_offer(body: SubmitFarmerOfferRequest):
    # offer, job_id = await farmer_service.submit_offer(...)
    # return SuccessResponse(data=AsyncJobResponse(job_id=..., estimated_seconds=30, poll_url=f"/api/v1/jobs/{job_id}"))
    pass

@router.post("/{offer_id}/accept", response_model=SuccessResponse[FarmerOfferResponse])
async def accept_offer(offer_id: UUID, body: AcceptOfferRequest):
    try:
        # cmd = AcceptFarmerOfferCommand(offer_id=offer_id, final_intake_price=body.final_intake_price, ...)
        # offer = await farmer_service.accept_offer(cmd)
        # return SuccessResponse(data=FarmerOfferResponse.model_validate(offer))
        pass
    except FarmerOfferNotFoundError:
        raise HTTPException(status_code=404, detail="Farmer offer not found")
    except InvalidOfferStatusError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ConflictingInventoryError as e:
        raise HTTPException(status_code=422, detail=f"Conflicting urgent batch: {e.batch_id}")

@router.post("/{offer_id}/counter", response_model=SuccessResponse[FarmerOfferResponse])
async def counter_offer(offer_id: UUID, body: CounterOfferRequest):
    pass

@router.post("/{offer_id}/decline", response_model=SuccessResponse[FarmerOfferResponse])
async def decline_offer(offer_id: UUID, body: DeclineOfferRequest):
    pass

@router.get("/{offer_id}/outcome")
async def get_offer_outcome(offer_id: UUID):
    pass

@router.get("/risk-buffer")
async def get_risk_buffer():
    pass
```

---

## Section 10 — Frontend (Next.js 14)

<!-- GENERATE: Generate the hooks, key components, and API client. -->

### API Client — typed fetch wrapper

```typescript
// freshprice-web/lib/api-client.ts
// RULES:
// - All API calls go through this client. Never raw fetch in components.
// - All responses validated with zod at runtime.
// - Automatic token refresh on 401.

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${API_BASE}/api/v1${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
      // Auth token injected here from Supabase session
    },
  })
  const json = await res.json()
  if (!json.success) {
    throw new APIError(json.error.code, json.error.message, json.error.fields)
  }
  return json.data as T
}

export class APIError extends Error {
  constructor(
    public code: string,
    message: string,
    public fields?: Record<string, string[]>
  ) {
    super(message)
    this.name = "APIError"
  }
}

export const api = {
  inventory: {
    listBatches: (storeId: string) =>
      apiFetch<InventoryBatch[]>(`/inventory/batches?store_id=${storeId}`),
    updatePrice: (batchId: string, price: string) =>
      apiFetch(`/inventory/batches/${batchId}/price`, {
        method: "PATCH",
        body: JSON.stringify({ new_price: price }),
      }),
  },
  farmerOffers: {
    list: () => apiFetch<FarmerOfferResponse[]>("/farmer-offers"),
    submit: (body: SubmitFarmerOfferRequest) =>
      apiFetch("/farmer-offers", { method: "POST", body: JSON.stringify(body) }),
    accept: (offerId: string, body: AcceptOfferRequest) =>
      apiFetch(`/farmer-offers/${offerId}/accept`, {
        method: "POST", body: JSON.stringify(body),
      }),
    counter: (offerId: string, price: string) =>
      apiFetch(`/farmer-offers/${offerId}/counter`, {
        method: "POST", body: JSON.stringify({ counter_price: price }),
      }),
    decline: (offerId: string, reason: string) =>
      apiFetch(`/farmer-offers/${offerId}/decline`, {
        method: "POST", body: JSON.stringify({ reason }),
      }),
  },
  trends: {
    listActive: () => apiFetch<TrendSignalResponse[]>("/trends/active"),
    approve: (trendId: string, quantityKg: number) =>
      apiFetch(`/trends/${trendId}/approve`, {
        method: "POST", body: JSON.stringify({ final_quantity_kg: quantityKg }),
      }),
    decline: (trendId: string, reason: string) =>
      apiFetch(`/trends/${trendId}/decline`, {
        method: "POST", body: JSON.stringify({ reason }),
      }),
  },
  analytics: {
    wrr: (days: number = 30) => apiFetch(`/analytics/wrr?days=${days}`),
  },
}
```

### Real-time hooks

```typescript
// freshprice-web/hooks/useRealtimeInventory.ts
"use client"
import { useEffect, useState } from "react"
import { createClientComponentClient } from "@supabase/auth-helpers-nextjs"
import { api } from "@/lib/api-client"
import type { InventoryBatch } from "@/lib/types/api"

export function useRealtimeInventory(storeId: string) {
  const [batches, setBatches] = useState<InventoryBatch[]>([])
  const [loading, setLoading] = useState(true)
  const supabase = createClientComponentClient()

  useEffect(() => {
    api.inventory.listBatches(storeId)
      .then(setBatches)
      .finally(() => setLoading(false))

    const channel = supabase
      .channel(`inventory-${storeId}`)
      .on("broadcast", { event: "price_updated" }, ({ payload }) => {
        setBatches(prev => prev.map(b =>
          b.id === payload.batch_id
            ? { ...b, listed_price: payload.new_price, urgency: payload.urgency }
            : b
        ))
      })
      .on("broadcast", { event: "batch_status_changed" }, ({ payload }) => {
        if (["EXPIRED", "CLEARED"].includes(payload.new_status)) {
          setBatches(prev => prev.filter(b => b.id !== payload.batch_id))
        }
      })
      .subscribe()

    return () => { supabase.removeChannel(channel) }
  }, [storeId])

  return { batches, loading }
}

// freshprice-web/hooks/useRealtimeBriefs.ts
"use client"
import { useEffect, useState } from "react"
import { createClientComponentClient } from "@supabase/auth-helpers-nextjs"
import type { OperatingBriefSummary } from "@/lib/types/api"

export function useRealtimeBriefs() {
  const [briefs, setBriefs] = useState<OperatingBriefSummary[]>([])
  const supabase = createClientComponentClient()

  useEffect(() => {
    const channel = supabase
      .channel("briefs-feed")
      .on("broadcast", { event: "brief_generated" }, ({ payload }) => {
        setBriefs(prev => [payload as OperatingBriefSummary, ...prev.slice(0, 49)])
      })
      .subscribe()
    return () => { supabase.removeChannel(channel) }
  }, [])

  return briefs
}
```

### ExpiryBar component

```typescript
// freshprice-web/components/inventory/ExpiryBar.tsx
"use client"
import { useEffect, useState } from "react"
import type { InventoryBatch } from "@/lib/types/api"

interface Props {
  batch: InventoryBatch
}

const URGENCY_COLORS: Record<string, string> = {
  FRESH:    "bg-green-500",
  WATCH:    "bg-yellow-400",
  URGENT:   "bg-orange-500",
  CRITICAL: "bg-red-600 animate-pulse",
}

export function ExpiryBar({ batch }: Props) {
  const [hoursLeft, setHoursLeft] = useState(0)

  useEffect(() => {
    const update = () => {
      const ms = new Date(batch.expiry_at).getTime() - Date.now()
      setHoursLeft(Math.max(0, ms / 3_600_000))
    }
    update()
    const interval = setInterval(update, 60_000)  // Update every minute
    return () => clearInterval(interval)
  }, [batch.expiry_at])

  const pct = Math.min(100, (hoursLeft / (batch.original_shelf_life_hrs ?? 72)) * 100)
  const color = URGENCY_COLORS[batch.urgency] ?? "bg-gray-400"

  return (
    <div className="w-full">
      <div className="flex justify-between text-xs text-gray-500 mb-1">
        <span>{batch.urgency}</span>
        <span>{hoursLeft.toFixed(1)}h left</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-1000 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
```

---

## Section 11 — Testing

<!-- GENERATE: Generate tests/conftest.py and test fixtures. Generate test files for farmer_service and pricing guard. -->

```python
# tests/conftest.py
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from uuid import uuid4
from decimal import Decimal
from datetime import datetime, timedelta
from freshprice.domain.entities.inventory_batch import InventoryBatch
from freshprice.domain.entities.farmer_offer import FarmerOffer
from freshprice.domain.enums import BatchStatus, BatchType, FarmerOfferStatus

TEST_DATABASE_URL = "postgresql+asyncpg://test:test@localhost:5432/freshprice_test"

@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine(TEST_DATABASE_URL)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()  # Always rollback test data

# FIXTURE BUILDERS — use keyword arguments, all have sensible defaults

def build_active_batch(
    hours_remaining: float = 48.0,
    quantity_remaining: int = 20,
    unit_cost: Decimal = Decimal("30.00"),
    original_price: Decimal = Decimal("80.00"),
    **kwargs
) -> InventoryBatch:
    return InventoryBatch(
        id=kwargs.get("id", uuid4()),
        sku_id=kwargs.get("sku_id", uuid4()),
        store_id=kwargs.get("store_id", uuid4()),
        quantity=kwargs.get("quantity", 20),
        quantity_remaining=quantity_remaining,
        unit_cost=unit_cost,
        listed_price=kwargs.get("listed_price", original_price),
        original_price=original_price,
        expiry_at=datetime.utcnow() + timedelta(hours=hours_remaining),
        batch_type=BatchType.REGULAR,
        status=BatchStatus.ACTIVE,
    )

def build_pending_offer(
    quantity_kg: Decimal = Decimal("50.0"),
    offered_price_per_kg: Decimal = Decimal("35.00"),
    shelf_life_hrs: int = 48,
    category: str = "fruits",
) -> FarmerOffer:
    return FarmerOffer(
        id=uuid4(),
        farmer_name="Test Farmer",
        farmer_phone_hash="abc123hash",
        product_category=category,
        product_name="Test Mango",
        quantity_kg=quantity_kg,
        offered_price_per_kg=offered_price_per_kg,
        seller_shelf_life_hrs=shelf_life_hrs,
        offered_at=datetime.utcnow(),
        status=FarmerOfferStatus.PENDING,
    )

def build_accept_command(
    offer_id=None,
    final_intake_price: Decimal = Decimal("42.00"),
    store_id=None,
):
    from freshprice.domain.commands.accept_farmer_offer import AcceptFarmerOfferCommand
    return AcceptFarmerOfferCommand(
        offer_id=offer_id or uuid4(),
        final_intake_price=final_intake_price,
        seller_id=uuid4(),
        store_id=store_id or uuid4(),
    )
```

---

## Section 12 — Docker & Environment

<!-- GENERATE: Generate docker-compose.yml, Dockerfile for backend, Dockerfile for frontend, .env.example -->

```yaml
# docker-compose.yml
version: "3.9"

x-api-env: &api-env
  DATABASE_URL: ${DATABASE_URL}
  REDIS_URL: redis://redis:6379/0
  SUPABASE_URL: ${SUPABASE_URL}
  SUPABASE_SERVICE_KEY: ${SUPABASE_SERVICE_KEY}
  HF_API_KEY: ${HF_API_KEY}
  HF_MODEL_ID: ${HF_MODEL_ID}
  ENVIRONMENT: ${ENVIRONMENT:-development}

services:
  api:
    build: { context: ./freshprice-api, dockerfile: Dockerfile }
    command: uvicorn freshprice.main:app --host 0.0.0.0 --port 8000 --workers 4
    environment: *api-env
    ports: ["8000:8000"]
    depends_on: [postgres, redis]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  worker:
    build: { context: ./freshprice-api, dockerfile: Dockerfile }
    command: celery -A freshprice.workers.celery_app worker --loglevel=info -Q briefs,pricing,trends,analytics --concurrency=4
    environment: *api-env
    depends_on: [api, redis]
    restart: unless-stopped

  beat:
    build: { context: ./freshprice-api, dockerfile: Dockerfile }
    command: celery -A freshprice.workers.celery_app beat --loglevel=info --scheduler redbeat.RedBeatScheduler
    environment: *api-env
    depends_on: [worker]
    restart: unless-stopped

  frontend:
    build: { context: ./freshprice-web, dockerfile: Dockerfile }
    command: node server.js
    environment:
      NEXT_PUBLIC_SUPABASE_URL: ${SUPABASE_URL}
      NEXT_PUBLIC_SUPABASE_ANON_KEY: ${SUPABASE_ANON_KEY}
      NEXT_PUBLIC_API_URL: http://api:8000
    ports: ["3000:3000"]
    depends_on: [api]
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    volumes: [postgres_data:/var/lib/postgresql/data]
    environment:
      POSTGRES_DB: freshprice
      POSTGRES_USER: ${DB_USER:-freshprice}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports: ["5432:5432"]

  redis:
    image: redis:7-alpine
    volumes: [redis_data:/data]
    command: redis-server --appendonly yes
    ports: ["6379:6379"]

volumes:
  postgres_data:
  redis_data:
```

---

## Section 13 — Claude Code Generation Instructions

> **Read this before generating anything.**

### Order of generation (strict)

```
1. freshprice/core/         → config, db, redis, exceptions, circuit_breaker
2. freshprice/domain/       → enums, all entities, value objects, all commands
3. alembic/                 → initial migration from Section 04 SQL schema
4. freshprice/repositories/ → all repos (skeleton with method signatures + docstrings)
5. freshprice/services/     → all services (full implementation using repos + commands)
6. freshprice/api/v1/       → schemas first, then routers (thin, convert exceptions to HTTP)
7. freshprice/workers/      → celery_app first, then each worker
8. freshprice/integrations/ → llm_client, google_trends, instagram, whatsapp
9. freshprice/main.py       → app factory, router registration, middleware
10. tests/conftest.py       → fixtures
11. tests/                  → unit tests for services (mock repos), integration tests for repos
12. freshprice-web/         → types/api.ts first, then api-client.ts, then hooks, then components
```

### Rules for all generated code

```
✅ Python 3.11+ syntax throughout (match/case, X | Y unions, dataclasses)
✅ All functions and methods have full type annotations
✅ All async where I/O is involved — no sync DB or HTTP calls
✅ Domain exceptions from core/exceptions.py — never HTTPException in services
✅ Repository pattern strict — services never import SQLAlchemy
✅ Command objects for all write operations — never raw **kwargs
✅ Pydantic v2 syntax (model_validate, not from_orm)
✅ TypeScript strict mode — no "any", interface for shapes, const assertions
✅ Named exports for utilities, default only for Next.js pages
✅ "use client" only when useState/useEffect/event handlers needed

❌ No print() statements — use structured logger
❌ No bare except: — always specific exception types
❌ No SELECT * — explicit column selection in all SQL
❌ No magic numbers inline — use constants from core/config.py
❌ No business logic in routers or components
❌ No mutable default arguments in Python
❌ No time.sleep() — asyncio.sleep() only
```

### When you are unsure about a business rule

Check Section 09 (Three Core Engines) first.
If still unclear, add a `# TODO: Confirm business rule — see SDD Section 09` comment and move on.
Do not invent business logic that is not in this spec.
