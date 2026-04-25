"""InfluencerSignalAgent — emits trend signals, some genuine, some paid promotion.

The TrendEngine in QStorePrice consumes social signals to decide
whether to restock viral categories. The original implementation
treats every signal as ground truth. Real markets aren't like that:
half of "trending" hashtags are paid promotion, ghost-engagement, or
brand-amplified noise. An agent that approves every high-score signal
is exploitable.

This agent emits a mix of GENUINE and PAID signals, each with a
``is_paid_promotion`` flag *hidden* from the StoreAgent. The
StoreAgent must verify by cross-referencing observable corroborants
the env exposes:

  - consumer_search_velocity[category]   (real consumer interest proxy)
  - co_purchase_uptick[category]         (lagging confirmation)
  - bus disclosures from CooperativeCompetitor

A new anti-hack guard penalises trend approvals where:

  - is_paid_promotion=True AND consumer_search_velocity is low

so the policy gradient learns epistemic discipline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

from freshprice_env.entities import SimulatedTrendSignal
from freshprice_env.enums import SignalSource, TrendAction
from freshprice_env.protocol.market_bus import MarketBus, MessageVerb


class TrendVeracity(str, Enum):
    GENUINE = "GENUINE"
    PAID_PROMOTION = "PAID_PROMOTION"
    GHOST_ENGAGEMENT = "GHOST_ENGAGEMENT"


@dataclass
class TaggedTrendSignal:
    """Wraps a SimulatedTrendSignal with auditor-visible veracity.

    The signal is exposed to the StoreAgent as a normal trend signal;
    the ``veracity`` field stays in the env's private state so the
    auditor and the ToM probe can grade later.
    """

    signal: SimulatedTrendSignal
    veracity: TrendVeracity
    consumer_search_velocity: float    # 0..1 — observable to LLM
    co_purchase_uptick: float          # 0..1 — observable to LLM
    promotion_brand: str | None = None
    promotion_disclosure_required: bool = False


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class InfluencerAgent:
    """Emits a mix of trend signals; some are paid.

    Configurable per scenario:
      - paid_promotion_rate: probability a generated signal is paid
      - search_velocity_floor_for_paid: paid signals get suspiciously low
        consumer_search_velocity (the tell)
    """

    rng: random.Random
    bus: MarketBus | None = None
    paid_promotion_rate: float = 0.40
    search_velocity_floor_for_paid: float = 0.25
    search_velocity_floor_for_genuine: float = 0.55

    _emitted: list[TaggedTrendSignal] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit_signal(
        self,
        category: str,
        current_tick: int,
    ) -> TaggedTrendSignal:
        """Generate one tagged signal for the given category."""
        is_paid = self.rng.random() < self.paid_promotion_rate
        veracity = (
            TrendVeracity.PAID_PROMOTION if is_paid
            else TrendVeracity.GENUINE
        )

        # Composite score: paid signals tend to *over*report (75-95)
        # because the brand is buying high engagement metrics; genuine
        # signals are noisier (55-90).
        if is_paid:
            comp = round(self.rng.uniform(75, 95), 1)
        else:
            comp = round(self.rng.uniform(55, 90), 1)

        signal = SimulatedTrendSignal(
            category=category,
            composite_score=comp,
            signal_source=self.rng.choice([
                SignalSource.INSTAGRAM, SignalSource.YOUTUBE,
                SignalSource.GOOGLE_TRENDS, SignalSource.ZOMATO,
            ]),
            detected_at_tick=current_tick,
            action_taken=TrendAction.PENDING,
            suggested_order_kg=round(self.rng.uniform(8, 22), 1),
            recipe_simplicity=round(self.rng.uniform(0.4, 0.95), 2),
            ingredient_rarity=round(self.rng.uniform(0.1, 0.7), 2),
            view_velocity=round(self.rng.uniform(0.5, 1.0), 2),
            local_relevance=round(self.rng.uniform(0.3, 0.95), 2),
            historical_conversion=round(self.rng.uniform(0.2, 0.6), 2),
        )

        # Observable corroborants
        if is_paid:
            consumer_search = round(
                self.rng.uniform(0.05, self.search_velocity_floor_for_paid),
                3,
            )
            co_purchase = round(self.rng.uniform(0.05, 0.30), 3)
        else:
            consumer_search = round(
                self.rng.uniform(self.search_velocity_floor_for_genuine, 0.95),
                3,
            )
            co_purchase = round(self.rng.uniform(0.30, 0.85), 3)

        tagged = TaggedTrendSignal(
            signal=signal,
            veracity=veracity,
            consumer_search_velocity=consumer_search,
            co_purchase_uptick=co_purchase,
            promotion_brand=(
                self.rng.choice(["FarmFresh", "DesiKitchen", "SnackHub", "GoldenHarvest"])
                if is_paid else None
            ),
            promotion_disclosure_required=is_paid,
        )
        self._emitted.append(tagged)

        if self.bus is not None:
            self.bus.post(
                tick=current_tick,
                sender_id="influencer",
                verb=MessageVerb.BROADCAST,
                body=(
                    f"trend signal: {category} @ score {comp:.0f} "
                    f"(search velocity {consumer_search:.2f}, "
                    f"co-purchase uptick {co_purchase:.2f})"
                ),
                receiver_id=None,
                payload={
                    "kind": "TREND_SIGNAL",
                    "category": category,
                    "composite_score": comp,
                    "consumer_search_velocity": consumer_search,
                    "co_purchase_uptick": co_purchase,
                    # veracity stays out of the broadcast payload —
                    # the LLM has to infer it
                },
            )

        return tagged

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def emitted(self) -> list[TaggedTrendSignal]:
        return list(self._emitted)

    def disinformation_summary(self) -> dict:
        n_total = len(self._emitted)
        n_paid = sum(
            1 for s in self._emitted
            if s.veracity == TrendVeracity.PAID_PROMOTION
        )
        return {
            "n_signals": n_total,
            "n_paid": n_paid,
            "paid_share": round(n_paid / n_total, 3) if n_total else 0.0,
        }
