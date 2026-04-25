"""FreshPrice 7-Engine reward computation -> Store Efficiency Score (SES).

This is the canonical reward computation per the FreshPrice strategy
document (Sections 5 + 7). The agent writes a single Operating Brief
which can carry directives for any subset of seven engines:

    Engine 1 - Dynamic Pricing            (always active)        r1
    Engine 2 - Farmer Offer               (when offer pending)   r2
    Engine 3 - Social Trend               (when signal present)  r3
    Engine 4 - Intra-Fleet Rebalancing    (multi-store only)     r4
    Engine 5 - Micro-Manufacturer         (near-expiry routing)  r5
    Engine 6 - Event Pre-Positioning      (event detected)       r6
    Engine 7 - Surplus Box                (Friday assembly)      r7

SES = sum(SES_WEIGHT_R{i} * r{i}) for i in 1..7. Used as the primary
curriculum-promotion metric: an agent must reach SES >= 0.70 over five
consecutive evaluation episodes to advance to the next training
scenario.

The classes here are stateful per-episode (so they can track running
totals like "items routed to processors this week") and reset on
env.reset(). They do NOT replace the existing PricingEngine/
FarmerEngine/TrendEngine -- those still own r1/r2/r3 computation.
This module *aggregates* their output and adds engines 4-7.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field

from freshprice_env.constants import (
    EVENT_PRESTOCK_LEAD_HOURS_MAX,
    EVENT_PRESTOCK_LEAD_HOURS_MIN,
    INTER_STORE_TRANSFER_COST_RS_PER_KG,
    MICROMFG_HOURS_TO_EXPIRY_FLOOR,
    MICROMFG_RECOVERY_RATIO_DEFAULT,
    R4_RECKLESS_TRANSFER_PENALTY,
    R4_TRANSFER_COST_PENALTY_PER_KG,
    R4_TRANSFER_REVENUE_BONUS,
    R5_EARLY_ROUTING_PENALTY,
    R5_LATE_ROUTING_PENALTY,
    R6_EVENT_DEMAND_UPLIFT_DENOMINATOR,
    R6_EVENT_NO_STOCKOUT_BONUS,
    R6_EVENT_OVERSTOCK_PENALTY,
    R7_CANCEL_PENALTY_PER_SUBSCRIBER,
    R7_FIVE_STAR_BONUS_PER_SUBSCRIBER,
    SES_WEIGHT_R1_PRICING,
    SES_WEIGHT_R2_FARMER,
    SES_WEIGHT_R3_TREND,
    SES_WEIGHT_R4_INTRAFLEET,
    SES_WEIGHT_R5_MICROMFG,
    SES_WEIGHT_R6_EVENT,
    SES_WEIGHT_R7_SURPLUSBOX,
    SURPLUS_BOX_DEFAULT_SUBSCRIBERS,
    SURPLUS_BOX_PRICE_PER_KG,
    SURPLUS_BOX_TARGET_WEIGHT_KG_MAX,
    SURPLUS_BOX_TARGET_WEIGHT_KG_MIN,
)
from freshprice_env.entities import SimulatedBatch
from freshprice_env.enums import BatchStatus, ExpiryUrgency, ExternalEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine 4 — Intra-Fleet Rebalancing
# ---------------------------------------------------------------------------


@dataclass
class IntraFleetTransfer:
    source_store: str
    dest_store: str
    batch_id: str
    units: int


@dataclass
class IntraFleetEngine:
    """Computes r4 from TRANSFER directives in the brief.

    The actual transfer execution lives in MultiStoreFreshPriceEnv; this
    class scores transfers for the SES even when running in single-store
    mode (for Stable Week / Farmer Week scenarios that do not enable
    Engine 4, r4 stays at 0).
    """

    transfers_this_brief: list[IntraFleetTransfer] = field(default_factory=list)
    revenue_recovered_rs: float = 0.0
    waste_prevented_units: int = 0
    transfer_cost_rs: float = 0.0
    reckless_count: int = 0

    def execute(
        self,
        transfers: list[IntraFleetTransfer],
        batches_by_id: dict[str, SimulatedBatch],
    ) -> dict:
        """Score the transfers for r4. Returns a snapshot dict."""
        for t in transfers:
            self.transfers_this_brief.append(t)
            batch = batches_by_id.get(t.batch_id)
            if batch is None:
                continue
            # A transfer is "reckless" if the source batch has no urgency
            # AND the destination is hypothetical (we treat any TRANSFER
            # with FRESH source as reckless: no waste was being prevented).
            if batch.urgency == ExpiryUrgency.FRESH:
                self.reckless_count += 1
                continue
            avg_weight_kg = 0.5     # one unit ~= 500g (groceries default)
            self.transfer_cost_rs += (
                t.units * avg_weight_kg * INTER_STORE_TRANSFER_COST_RS_PER_KG
            )
            # Optimistic revenue model: 80% of transferred units sell at the
            # current price at the destination. The env's actual sales
            # engine will report the real number; we use this as a proxy.
            est_revenue = 0.80 * t.units * batch.current_price
            self.revenue_recovered_rs += est_revenue
            self.waste_prevented_units += t.units
        return self.snapshot()

    def compute_brief_reward(self) -> float:
        if not self.transfers_this_brief:
            return 0.0
        gross = (
            self.waste_prevented_units * R4_TRANSFER_REVENUE_BONUS
            - self.transfer_cost_rs * R4_TRANSFER_COST_PENALTY_PER_KG
            - self.reckless_count * R4_RECKLESS_TRANSFER_PENALTY
        )
        return round(gross, 4)

    def snapshot(self) -> dict:
        return {
            "transfers_this_brief": len(self.transfers_this_brief),
            "revenue_recovered_rs": round(self.revenue_recovered_rs, 2),
            "waste_prevented_units": self.waste_prevented_units,
            "transfer_cost_rs": round(self.transfer_cost_rs, 2),
            "reckless_count": self.reckless_count,
        }

    def reset_brief(self) -> None:
        self.__init__()


# ---------------------------------------------------------------------------
# Engine 5 — Micro-Manufacturer Pipeline
# ---------------------------------------------------------------------------


@dataclass
class MicroMfgRouting:
    batch_id: str
    processor: str = "default_processor"
    floor_price_per_unit: float | None = None


@dataclass
class MicroMfgEngine:
    """Routes near-expiry batches to registered processors.

    Strategy (Section 5.5):
      eligible if hours_to_expiry <= MICROMFG_HOURS_TO_EXPIRY_FLOOR
      r5 = recovered / (batch_cost * unsold_fraction); 30% recovery > 0.
    Anti-hack:
      routing FRESH/WATCH stock is "early routing" -> penalty.
      routing past expiry is "late routing" -> penalty (processor wont take it).
    """

    routings_this_brief: list[dict] = field(default_factory=list)
    total_recovered_rs: float = 0.0
    early_routing_count: int = 0
    late_routing_count: int = 0
    accepted_count: int = 0

    def execute(
        self,
        routings: list[MicroMfgRouting],
        batches_by_id: dict[str, SimulatedBatch],
    ) -> dict:
        for r in routings:
            batch = batches_by_id.get(r.batch_id)
            if batch is None:
                self.routings_this_brief.append({
                    "batch_id": r.batch_id, "accepted": False,
                    "reason": "batch_not_found",
                })
                continue
            hrs = float(getattr(batch, "hours_to_expiry", 0.0))
            if hrs > MICROMFG_HOURS_TO_EXPIRY_FLOOR:
                self.early_routing_count += 1
                self.routings_this_brief.append({
                    "batch_id": r.batch_id, "accepted": False,
                    "reason": f"early_routing (hrs={hrs:.1f} > {MICROMFG_HOURS_TO_EXPIRY_FLOOR:.0f})",
                    "early": True,
                })
                continue
            if hrs <= 0 or batch.status != BatchStatus.ACTIVE:
                self.late_routing_count += 1
                self.routings_this_brief.append({
                    "batch_id": r.batch_id, "accepted": False,
                    "reason": "late_routing (already expired)", "late": True,
                })
                continue
            ratio = MICROMFG_RECOVERY_RATIO_DEFAULT
            recovered = round(
                batch.quantity_remaining * batch.original_price * ratio, 2,
            )
            self.total_recovered_rs += recovered
            self.accepted_count += 1
            self.routings_this_brief.append({
                "batch_id": r.batch_id, "accepted": True,
                "units": batch.quantity_remaining, "rs": recovered,
                "processor": r.processor,
            })
            # Mark batch as cleared via the processor route (env can choose
            # to apply this state mutation; we leave it advisory here).
        return self.snapshot()

    def compute_brief_reward(self) -> float:
        if not self.routings_this_brief:
            return 0.0
        # Coarse model: each accepted routing adds 0.18 (matching strategy
        # "even 30% recovery is positive"); penalties subtract.
        return round(
            self.accepted_count * 0.18
            - self.early_routing_count * R5_EARLY_ROUTING_PENALTY
            - self.late_routing_count * R5_LATE_ROUTING_PENALTY,
            4,
        )

    def snapshot(self) -> dict:
        return {
            "routings_this_brief": list(self.routings_this_brief),
            "accepted": self.accepted_count,
            "early_routing": self.early_routing_count,
            "late_routing": self.late_routing_count,
            "total_recovered_rs": round(self.total_recovered_rs, 2),
        }

    def reset_brief(self) -> None:
        self.__init__()


# ---------------------------------------------------------------------------
# Engine 6 — Event Pre-Positioning
# ---------------------------------------------------------------------------


@dataclass
class EventPrestock:
    category: str
    quantity_units: int
    target_event_tick: int


@dataclass
class EventEngine:
    """Pre-positions stock for detected events.

    Strategy (Section 5.6):
      lead time must be 4-48 hours ahead of the event
      r6 = event_day_revenue_uplift / baseline; +0.30 zero-stockout bonus;
      -0.20 if pre-stock spoils unsold post-event.

    The env feeds this engine the upcoming event tick and category-demand
    multipliers; the engine scores the prestock decision.
    """

    prestocks_this_brief: list[EventPrestock] = field(default_factory=list)
    valid_lead_count: int = 0
    too_early_count: int = 0
    too_late_count: int = 0
    overstock_units: int = 0       # units that spoiled post-event
    stockouts: int = 0             # event ticks with stockout

    def execute(
        self,
        prestocks: list[EventPrestock],
        current_tick: int,
        ticks_per_hour: int = 4,    # 15-min ticks
    ) -> dict:
        for p in prestocks:
            self.prestocks_this_brief.append(p)
            hours_ahead = (p.target_event_tick - current_tick) / ticks_per_hour
            if hours_ahead < EVENT_PRESTOCK_LEAD_HOURS_MIN:
                self.too_late_count += 1
            elif hours_ahead > EVENT_PRESTOCK_LEAD_HOURS_MAX:
                self.too_early_count += 1
            else:
                self.valid_lead_count += 1
        return self.snapshot()

    def compute_brief_reward(self) -> float:
        if not self.prestocks_this_brief and self.stockouts == 0 and self.overstock_units == 0:
            return 0.0
        # Reward for valid leads, penalty for stockouts + overstock.
        return round(
            self.valid_lead_count * 0.18
            - self.too_late_count * 0.10
            - self.too_early_count * 0.05
            + (R6_EVENT_NO_STOCKOUT_BONUS if self.valid_lead_count > 0
               and self.stockouts == 0 else 0.0)
            - R6_EVENT_OVERSTOCK_PENALTY * min(1.0, self.overstock_units
                                               / R6_EVENT_DEMAND_UPLIFT_DENOMINATOR),
            4,
        )

    def snapshot(self) -> dict:
        return {
            "prestocks_this_brief": [p.__dict__ for p in self.prestocks_this_brief],
            "valid_lead": self.valid_lead_count,
            "too_late": self.too_late_count,
            "too_early": self.too_early_count,
            "overstock_units": self.overstock_units,
            "stockouts": self.stockouts,
        }

    def reset_brief(self) -> None:
        # NB: episode-running totals (overstock_units, stockouts) are reset
        # at episode boundary, not per brief. The env owns that signal.
        self.prestocks_this_brief = []
        self.valid_lead_count = 0
        self.too_late_count = 0
        self.too_early_count = 0


# ---------------------------------------------------------------------------
# Engine 7 — Surplus Box Subscription
# ---------------------------------------------------------------------------


@dataclass
class SurplusBoxSelection:
    batch_id: str
    units: int


@dataclass
class SurplusBoxEngine:
    """Weekly Friday assembly of near-expiry items into a subscriber box.

    Strategy (Section 5.7):
      target weight 1.5-2.0 kg; subscribers rate the box;
      r7 = retention_rate * (revenue / cost). Bonus per 5-star, penalty
      per cancel.
    """

    subscriber_count: int = SURPLUS_BOX_DEFAULT_SUBSCRIBERS
    boxes_assembled_this_episode: int = 0
    five_star_count: int = 0
    cancel_count: int = 0
    last_box_revenue_rs: float = 0.0
    last_box_cost_rs: float = 0.0
    last_box_weight_kg: float = 0.0

    def assemble(
        self,
        selections: list[SurplusBoxSelection],
        batches_by_id: dict[str, SimulatedBatch],
        rng: random.Random,
        avg_weight_kg_per_unit: float = 0.25,
    ) -> dict:
        """Assemble + dispatch a box. Returns a snapshot dict."""
        if not selections:
            return self.snapshot()
        weight_kg = sum(s.units * avg_weight_kg_per_unit for s in selections)
        cost = 0.0
        revenue = 0.0
        for s in selections:
            batch = batches_by_id.get(s.batch_id)
            if batch is None:
                continue
            cost += s.units * batch.original_price * 0.4   # marginal cost share
            revenue += s.units * SURPLUS_BOX_PRICE_PER_KG * avg_weight_kg_per_unit
        # Subscriber satisfaction model: weight in target range -> 70%
        # 5-star, otherwise 30%. Out-of-range weight produces cancellations.
        if SURPLUS_BOX_TARGET_WEIGHT_KG_MIN <= weight_kg <= SURPLUS_BOX_TARGET_WEIGHT_KG_MAX:
            five_star = int(self.subscriber_count * 0.70)
            cancels   = int(self.subscriber_count * 0.05)
        else:
            five_star = int(self.subscriber_count * 0.30)
            cancels   = int(self.subscriber_count * 0.20)
        # Add small random variance.
        five_star = max(0, five_star + rng.randint(-2, 2))
        cancels   = max(0, cancels + rng.randint(-1, 1))
        self.five_star_count += five_star
        self.cancel_count += cancels
        self.subscriber_count = max(0, self.subscriber_count - cancels)
        self.last_box_revenue_rs = round(revenue, 2)
        self.last_box_cost_rs = round(cost, 2)
        self.last_box_weight_kg = round(weight_kg, 2)
        self.boxes_assembled_this_episode += 1
        return self.snapshot()

    def compute_brief_reward(self) -> float:
        if self.last_box_revenue_rs == 0 and self.boxes_assembled_this_episode == 0:
            return 0.0
        retention = max(0.0, 1.0 - (self.cancel_count
                                    / max(1, self.subscriber_count + self.cancel_count)))
        rev_cost_ratio = (self.last_box_revenue_rs
                          / max(1.0, self.last_box_cost_rs))
        return round(
            retention * min(2.0, rev_cost_ratio) / 2.0   # normalise to [0,1]
            + self.five_star_count * R7_FIVE_STAR_BONUS_PER_SUBSCRIBER / max(1, self.subscriber_count)
            - self.cancel_count * R7_CANCEL_PENALTY_PER_SUBSCRIBER / max(1, self.subscriber_count),
            4,
        )

    def snapshot(self) -> dict:
        return {
            "subscriber_count": self.subscriber_count,
            "boxes_this_episode": self.boxes_assembled_this_episode,
            "five_star_count": self.five_star_count,
            "cancel_count": self.cancel_count,
            "last_box_revenue_rs": self.last_box_revenue_rs,
            "last_box_cost_rs": self.last_box_cost_rs,
            "last_box_weight_kg": self.last_box_weight_kg,
        }

    def reset_brief(self) -> None:
        # Per-brief: clear "last box" so r7 is not double-counted.
        self.last_box_revenue_rs = 0.0
        self.last_box_cost_rs = 0.0
        self.last_box_weight_kg = 0.0


# ---------------------------------------------------------------------------
# Composite SES + directive parsers
# ---------------------------------------------------------------------------


def store_efficiency_score(
    *,
    r1: float, r2: float, r3: float, r4: float, r5: float, r6: float, r7: float,
) -> float:
    """SES = sum(w_i * r_i) per FreshPrice strategy Section 7."""
    return round(
        SES_WEIGHT_R1_PRICING * r1
        + SES_WEIGHT_R2_FARMER * r2
        + SES_WEIGHT_R3_TREND * r3
        + SES_WEIGHT_R4_INTRAFLEET * r4
        + SES_WEIGHT_R5_MICROMFG * r5
        + SES_WEIGHT_R6_EVENT * r6
        + SES_WEIGHT_R7_SURPLUSBOX * r7,
        4,
    )


# Brief directive parsers — tolerant to LLM formatting variation.

_DIRECTIVE_BLOCK_RE = re.compile(
    r"DIRECTIVE\s*:?\s*(?:```(?:json)?)?\s*(\{.*?\})\s*(?:```)?\s*(?:CONFIDENCE|$)",
    re.DOTALL | re.IGNORECASE,
)


def _safe_load_directive(brief: str) -> dict | None:
    """Pull the JSON object inside the DIRECTIVE block, balancing braces.

    The regex above can fail on nested objects; this is the brace-matched
    version used as a fallback. Same logic as BriefParser._extract_directive_json
    but without depending on the parser's full contract (so engines 4-7
    can score side directives without needing a successful parse of the
    brief's primary engine).
    """
    import json
    idx = brief.upper().find("DIRECTIVE")
    if idx == -1:
        return None
    tail = brief[idx:]
    start = tail.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(tail[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(tail[start:i + 1])
                except (ValueError, json.JSONDecodeError):
                    return None
    return None


def parse_intrafleet_transfers(brief: str) -> list[IntraFleetTransfer]:
    directive = _safe_load_directive(brief)
    if not isinstance(directive, dict):
        return []
    actions = directive.get("intrafleet_actions") or directive.get("transfers") or []
    out = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        try:
            out.append(IntraFleetTransfer(
                source_store=str(a.get("source_store", "store_A")),
                dest_store=str(a.get("dest_store", "store_B")),
                batch_id=str(a["batch_id"]),
                units=int(a.get("units", 0)),
            ))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def parse_micromfg_routings(brief: str) -> list[MicroMfgRouting]:
    directive = _safe_load_directive(brief)
    if not isinstance(directive, dict):
        return []
    actions = directive.get("micromfg_actions") or directive.get("micro_manufacturer") or []
    out = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        try:
            out.append(MicroMfgRouting(
                batch_id=str(a["batch_id"]),
                processor=str(a.get("processor", "default_processor")),
                floor_price_per_unit=a.get("floor_price"),
            ))
        except (KeyError, TypeError):
            continue
    return out


def parse_event_prestocks(brief: str) -> list[EventPrestock]:
    directive = _safe_load_directive(brief)
    if not isinstance(directive, dict):
        return []
    actions = directive.get("event_actions") or directive.get("prestock") or []
    out = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        try:
            out.append(EventPrestock(
                category=str(a["category"]),
                quantity_units=int(a.get("quantity_units", a.get("quantity", 0))),
                target_event_tick=int(a.get("target_event_tick", 0)),
            ))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def parse_surplus_box_selections(brief: str) -> list[SurplusBoxSelection]:
    directive = _safe_load_directive(brief)
    if not isinstance(directive, dict):
        return []
    actions = directive.get("surplus_box_actions") or directive.get("surplus_box") or []
    out = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        try:
            out.append(SurplusBoxSelection(
                batch_id=str(a["batch_id"]),
                units=int(a.get("units", 1)),
            ))
        except (KeyError, ValueError, TypeError):
            continue
    return out
