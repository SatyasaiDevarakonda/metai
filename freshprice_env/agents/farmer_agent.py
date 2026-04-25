"""FarmerAgent — LLM-driven (or rule-based) farmer with persistent reputation.

Replaces the static ``_scheduled_farmer_offer`` injection in
FreshPriceEnv with a real actor:

  - Each farmer has a stable identity (farmer_id) that persists across
    episodes via the ReputationStore.
  - Reserve price is *adjusted* by current trust: a store that has
    burned this farmer (or others, via pool_trust_summary) faces a
    higher reserve.
  - Offers can be ACCEPT / COUNTER / DECLINE. The farmer's reaction to
    a counter depends on its persona + trust + urgency.
  - When an LLM client is provided, the farmer generates its messages
    via the LLM. Otherwise it falls back to a deterministic policy
    (great for tests and offline rollouts).

Personas:

  PRAGMATIC      — accepts most reasonable counters, mild trust impact
  AGGRESSIVE     — walks fast on lowballs, big trust hit
  COOPERATIVE    — gives early discounts to high-trust stores
  RECIPROCAL     — tit-for-tat: matches store's past behaviour
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from freshprice_env.entities import SimulatedFarmerOffer
from freshprice_env.enums import FarmerOfferStatus
from freshprice_env.persistence.reputation_store import (
    FarmerReputation,
    ReputationStore,
)
from freshprice_env.protocol.market_bus import MarketBus, MessageVerb


class FarmerPersona(str, Enum):
    PRAGMATIC = "PRAGMATIC"
    AGGRESSIVE = "AGGRESSIVE"
    COOPERATIVE = "COOPERATIVE"
    RECIPROCAL = "RECIPROCAL"


@dataclass(frozen=True)
class FarmerProfile:
    """Static identity for a farmer in the ecosystem."""

    farmer_id: str
    farmer_name: str
    region: str
    persona: FarmerPersona
    base_reserve_per_kg: float
    typical_quantity_kg: float
    typical_shelf_life_hrs: int
    primary_category: str
    primary_product: str


@dataclass(frozen=True)
class FarmerCounterDecision:
    """Outcome of a farmer responding to the store's counter offer."""

    decision: str    # ACCEPT | COUNTER | WALK
    counter_price_per_kg: float | None
    rationale: str


# ---------------------------------------------------------------------------
# Default farmer roster
# ---------------------------------------------------------------------------

DEFAULT_FARMER_ROSTER: list[FarmerProfile] = [
    FarmerProfile(
        farmer_id="farmer_rajan",
        farmer_name="Rajan Patel",
        region="Ratnagiri",
        persona=FarmerPersona.PRAGMATIC,
        base_reserve_per_kg=32.0,
        typical_quantity_kg=50.0,
        typical_shelf_life_hrs=48,
        primary_category="fruits",
        primary_product="mangoes",
    ),
    FarmerProfile(
        farmer_id="farmer_geeta",
        farmer_name="Geeta Devi",
        region="Hosakote",
        persona=FarmerPersona.AGGRESSIVE,
        base_reserve_per_kg=28.0,
        typical_quantity_kg=30.0,
        typical_shelf_life_hrs=24,
        primary_category="dairy",
        primary_product="curd",
    ),
    FarmerProfile(
        farmer_id="farmer_suresh",
        farmer_name="Suresh Patil",
        region="Pune",
        persona=FarmerPersona.COOPERATIVE,
        base_reserve_per_kg=14.0,
        typical_quantity_kg=12.0,
        typical_shelf_life_hrs=30,
        primary_category="vegetables",
        primary_product="spinach",
    ),
    FarmerProfile(
        farmer_id="farmer_anita",
        farmer_name="Anita Kumari",
        region="Kolar",
        persona=FarmerPersona.RECIPROCAL,
        base_reserve_per_kg=18.0,
        typical_quantity_kg=20.0,
        typical_shelf_life_hrs=36,
        primary_category="vegetables",
        primary_product="tomatoes",
    ),
    FarmerProfile(
        farmer_id="farmer_vijay",
        farmer_name="Vijay Rao",
        region="Belgaum",
        persona=FarmerPersona.PRAGMATIC,
        base_reserve_per_kg=15.0,
        typical_quantity_kg=25.0,
        typical_shelf_life_hrs=20,
        primary_category="vegetables",
        primary_product="capsicum",
    ),
]


# ---------------------------------------------------------------------------
# FarmerAgent
# ---------------------------------------------------------------------------

class FarmerAgent:
    """A farmer that emits offers and reacts to counters with memory.

    This class is invoked by the MarketCommonsEnv / FreshPriceEnv to
    generate a fresh offer at scheduled ticks, and to decide how to
    respond when the store COUNTERs.
    """

    def __init__(
        self,
        profile: FarmerProfile,
        reputation_store: ReputationStore,
        rng: random.Random,
        llm_client=None,
        bus: MarketBus | None = None,
    ) -> None:
        self.profile = profile
        self._store = reputation_store
        self._rng = rng
        self._llm = llm_client
        self._bus = bus

        # Ensure the farmer exists in the reputation store
        self._store.upsert_farmer(
            farmer_id=profile.farmer_id,
            farmer_name=profile.farmer_name,
            base_reserve_per_kg=profile.base_reserve_per_kg,
            initial_trust=0.6,
        )

    # ------------------------------------------------------------------
    # Reputation accessor
    # ------------------------------------------------------------------

    def reputation(self) -> FarmerReputation:
        rep = self._store.get(self.profile.farmer_id)
        assert rep is not None, "farmer must exist in store after __init__"
        return rep

    # ------------------------------------------------------------------
    # Offer generation
    # ------------------------------------------------------------------

    def generate_offer(
        self,
        offer_id: str,
        current_tick: int,
        urgency_factor: float = 1.0,
    ) -> SimulatedFarmerOffer:
        """Produce a new offer. Reserve is modulated by trust.

        ``urgency_factor`` (>= 1.0) shrinks the offer-vs-reserve premium
        when the farmer's product is highly perishable that day. Default
        1.0 keeps a normal premium.
        """
        rep = self.reputation()
        adjusted_reserve = rep.adjusted_reserve()
        # Premium above reserve: pragmatic farmers add 15-25%, aggressive
        # 25-40%, cooperative 5-15% with high-trust stores
        prem = self._premium_for_persona(rep)
        prem /= max(1.0, urgency_factor)
        offered_price = round(adjusted_reserve * (1.0 + prem), 2)
        qty = self.profile.typical_quantity_kg * self._rng.uniform(0.85, 1.15)
        offer = SimulatedFarmerOffer(
            offer_id=offer_id,
            farmer_name=self.profile.farmer_name,
            product_category=self.profile.primary_category,
            product_name=self.profile.primary_product,
            quantity_kg=round(qty, 1),
            offered_price_per_kg=offered_price,
            seller_shelf_life_hrs=self.profile.typical_shelf_life_hrs,
            offered_at_tick=current_tick,
            status=FarmerOfferStatus.PENDING,
        )

        if self._bus is not None:
            self._bus.post(
                tick=current_tick,
                sender_id=self.profile.farmer_id,
                verb=MessageVerb.BID,
                body=(
                    f"{self.profile.region} {self.profile.primary_product}: "
                    f"{offer.quantity_kg}kg at Rs{offered_price}/kg, "
                    f"shelf {offer.seller_shelf_life_hrs}h"
                ),
                receiver_id=None,   # broadcast to all stores
                payload={
                    "offer_id": offer_id,
                    "price_per_kg": offered_price,
                    "quantity_kg": offer.quantity_kg,
                    "shelf_life_hrs": offer.seller_shelf_life_hrs,
                    "trust_score": round(rep.trust_score, 3),
                },
            )
        return offer

    def _premium_for_persona(self, rep: FarmerReputation) -> float:
        """Return premium-over-reserve for offered price."""
        rng = self._rng
        if self.profile.persona == FarmerPersona.PRAGMATIC:
            base = rng.uniform(0.15, 0.25)
        elif self.profile.persona == FarmerPersona.AGGRESSIVE:
            base = rng.uniform(0.25, 0.40)
        elif self.profile.persona == FarmerPersona.COOPERATIVE:
            # Cooperative farmers give breaks to high-trust stores
            base = rng.uniform(0.05, 0.18)
            base *= max(0.5, 1.0 - rep.trust_score * 0.5)
        else:  # RECIPROCAL
            # Mirror the trust score: high trust → low premium, low trust → high
            base = 0.10 + (1.0 - rep.trust_score) * 0.30
        return base

    # ------------------------------------------------------------------
    # Response to a counter offer from the store
    # ------------------------------------------------------------------

    def respond_to_counter(
        self,
        offer: SimulatedFarmerOffer,
        store_counter_price: float,
        current_tick: int,
        store_id: str = "store_001",
    ) -> FarmerCounterDecision:
        """Decide how to respond to a store's COUNTER on this offer.

        Logs the response on the bus (if attached). Trust adjustments
        happen *here* because that's when the farmer learns the store's
        intent.
        """
        rep = self.reputation()
        reserve = rep.adjusted_reserve()
        is_lowball = store_counter_price < reserve
        margin_pct = (store_counter_price - reserve) / max(reserve, 1e-3)

        decision: str
        counter_price: float | None = None
        rationale: str

        if self.profile.persona == FarmerPersona.AGGRESSIVE:
            if is_lowball or margin_pct < 0.05:
                decision = "WALK"
                rationale = "lowball: walking"
            else:
                decision = "ACCEPT"
                rationale = "fair price, accepting"
        elif self.profile.persona == FarmerPersona.COOPERATIVE:
            if is_lowball and rep.trust_score > 0.7:
                decision = "ACCEPT"
                rationale = "loyal store: honoring counter despite reserve"
            elif is_lowball:
                decision = "COUNTER"
                counter_price = round((store_counter_price + reserve) / 2.0, 2)
                rationale = "splitting the difference"
            else:
                decision = "ACCEPT"
                rationale = "reasonable counter, accepting"
        elif self.profile.persona == FarmerPersona.RECIPROCAL:
            if is_lowball and rep.trust_score < 0.4:
                decision = "WALK"
                rationale = "store has lowballed before, walking"
            elif is_lowball:
                decision = "COUNTER"
                counter_price = round(reserve * 1.05, 2)
                rationale = "matching with reserve+5%"
            else:
                decision = "ACCEPT"
                rationale = "fair, accepting"
        else:  # PRAGMATIC
            if is_lowball and margin_pct < -0.10:
                decision = "WALK"
                rationale = "below cost, walking"
            elif is_lowball:
                decision = "COUNTER"
                counter_price = round(reserve * 1.03, 2)
                rationale = "counter near reserve"
            else:
                decision = "ACCEPT"
                rationale = "acceptable, taking it"

        # Trust update happens through the reputation store as an
        # interaction. The actual ACCEPT/DECLINE is recorded by whoever
        # owns the offer flow (FreshPriceEnv); here we record the
        # COUNTER as either ACCEPT (we accept the store's counter) or
        # DECLINE (we walk).
        recorded_decision = "ACCEPT" if decision == "ACCEPT" else "DECLINE"
        self._store.record_interaction(
            episode_id=offer.offer_id,   # cheap stand-in; env can override later
            tick=current_tick,
            farmer_id=self.profile.farmer_id,
            store_id=store_id,
            offer_price_per_kg=offer.offered_price_per_kg,
            decision=recorded_decision,
            counter_price_per_kg=store_counter_price,
            reserve_at_time=reserve,
        )

        if self._bus is not None:
            self._bus.post(
                tick=current_tick,
                sender_id=self.profile.farmer_id,
                verb=MessageVerb.COUNTER if decision == "COUNTER" else (
                    MessageVerb.COMMIT if decision == "ACCEPT" else MessageVerb.CHAT
                ),
                body=f"{decision}: {rationale}",
                receiver_id=store_id,
                payload={
                    "offer_id": offer.offer_id,
                    "decision": decision,
                    "counter_price_per_kg": counter_price,
                    "trust_after": round(self.reputation().trust_score, 3),
                },
            )

        return FarmerCounterDecision(
            decision=decision,
            counter_price_per_kg=counter_price,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Trust adjustment on outcome (called by env after store decision)
    # ------------------------------------------------------------------

    def record_store_decision(
        self,
        offer: SimulatedFarmerOffer,
        store_decision: str,
        counter_price_per_kg: float | None,
        episode_id: str,
        store_id: str,
        current_tick: int,
    ) -> FarmerReputation:
        """Persist the store's decision on this offer to the reputation store."""
        rep = self.reputation()
        self._store.record_interaction(
            episode_id=episode_id,
            tick=current_tick,
            farmer_id=self.profile.farmer_id,
            store_id=store_id,
            offer_price_per_kg=offer.offered_price_per_kg,
            decision=store_decision.upper(),
            counter_price_per_kg=counter_price_per_kg,
            reserve_at_time=rep.adjusted_reserve(),
        )
        return self.reputation()


# ---------------------------------------------------------------------------
# Roster builder
# ---------------------------------------------------------------------------

def build_default_farmer_pool(
    reputation_store: ReputationStore,
    rng: random.Random,
    bus: MarketBus | None = None,
    llm_client=None,
) -> dict[str, FarmerAgent]:
    """Build the canonical farmer roster keyed by farmer_id."""
    return {
        p.farmer_id: FarmerAgent(
            profile=p,
            reputation_store=reputation_store,
            rng=rng,
            bus=bus,
            llm_client=llm_client,
        )
        for p in DEFAULT_FARMER_ROSTER
    }
