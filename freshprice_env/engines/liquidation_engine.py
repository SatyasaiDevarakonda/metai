"""LiquidationEngine — B2B firesale channel for dead stock.

Real quick-commerce stores have a B2B / smart-bazaar liquidation
channel for batches that won't move via discounts in time. The agent
can issue a ``LIQUIDATE`` action against a CRITICAL batch with under
``LIQUIDATION_MIN_URGENCY_HOURS`` of shelf life remaining and recover
``LIQUIDATION_RECOVERY_RATIO`` of original price.

Anti-hack: liquidating a FRESH or WATCH batch is a
``RECKLESS_LIQUIDATION`` violation that zeroes the brief's reward and
pushes a flag into the env's RuleExecutor pipeline. This is *the*
defense against the LLM dumping inventory to game R1.

Reward (``r7_liquidation``):
  +R7_LIQUIDATION_BONUS  per legitimately-liquidated batch where the
                         expected discount-driven sell-through would
                         have recovered LESS than the liquidation
                         channel did
  -R7_LIQUIDATION_RECKLESS per anti-hack-flagged liquidation

The engine is additive: it stands alone from the PRICING engine. The
env consumes a side-channel directive list and ticks this engine after
the pricing engine completes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from freshprice_env.constants import (
    LIQUIDATION_MIN_URGENCY_HOURS,
    LIQUIDATION_RECKLESS_PENALTY,
    LIQUIDATION_RECOVERY_RATIO,
    R7_LIQUIDATION_BONUS,
    R7_LIQUIDATION_RECKLESS,
)
from freshprice_env.entities import SimulatedBatch
from freshprice_env.enums import BatchStatus


@dataclass
class LiquidationDecision:
    batch_id: str
    channel: str = "B2B"


@dataclass
class LiquidationResult:
    batch_id: str
    accepted: bool
    units_liquidated: int = 0
    rs_recovered: float = 0.0
    reason: str = ""           # accepted / floor_violated / urgency_too_high
    reckless: bool = False     # True if anti-hack flagged


@dataclass
class LiquidationEngine:
    """Stateful B2B liquidation channel.

    Public surface:
        execute(decisions, batches, rng) -> [LiquidationResult]
        compute_brief_reward() -> r7_liquidation (resets stats)
        snapshot() -> dict for the dashboard / prompt
    """

    results_this_brief: list[LiquidationResult] = field(default_factory=list)
    total_recovered_rs: float = 0.0
    total_reckless: int = 0

    # ------------------------------------------------------------------
    # Action API
    # ------------------------------------------------------------------

    def execute(
        self,
        decisions: list[LiquidationDecision],
        batches_by_id: dict[str, SimulatedBatch],
        rng: random.Random,
    ) -> list[LiquidationResult]:
        """Apply liquidation decisions in-place against batches.

        Each accepted decision sets the batch status to LIQUIDATED, marks
        the units as recovered, and records the cash flow. Reckless
        attempts (FRESH/WATCH stock) are *not* applied — they are
        flagged and consume the directive without effect.
        """
        out: list[LiquidationResult] = []
        for d in decisions:
            batch = batches_by_id.get(d.batch_id)
            if batch is None:
                out.append(LiquidationResult(
                    batch_id=d.batch_id, accepted=False,
                    reason="batch_not_found",
                ))
                continue
            if batch.status != BatchStatus.ACTIVE:
                out.append(LiquidationResult(
                    batch_id=d.batch_id, accepted=False,
                    reason=f"batch_status={batch.status.value}",
                ))
                continue

            hrs = float(getattr(batch, "hours_to_expiry", 0.0))
            # Anti-hack: liquidating non-near-expiry stock.
            if hrs > LIQUIDATION_MIN_URGENCY_HOURS:
                self.total_reckless += 1
                out.append(LiquidationResult(
                    batch_id=d.batch_id, accepted=False,
                    reason=f"urgency_too_low (hours_to_expiry={hrs:.1f})",
                    reckless=True,
                ))
                continue

            # Accept: recover LIQUIDATION_RECOVERY_RATIO of original price
            # for the remaining units, retire the batch.
            units = int(batch.quantity_remaining)
            recovered = round(
                units * batch.original_price * LIQUIDATION_RECOVERY_RATIO,
                2,
            )
            batch.quantity_remaining = 0
            batch.status = BatchStatus.LIQUIDATED
            self.total_recovered_rs += recovered
            out.append(LiquidationResult(
                batch_id=d.batch_id, accepted=True,
                units_liquidated=units, rs_recovered=recovered,
                reason="ok",
            ))
        self.results_this_brief.extend(out)
        return out

    # ------------------------------------------------------------------
    # Reward / inspection
    # ------------------------------------------------------------------

    def compute_brief_reward(self) -> float:
        """r7_liquidation contribution for this brief."""
        legit = sum(1 for r in self.results_this_brief if r.accepted)
        reckless = sum(1 for r in self.results_this_brief if r.reckless)
        return round(
            legit * R7_LIQUIDATION_BONUS - reckless * R7_LIQUIDATION_RECKLESS,
            4,
        )

    def reset_brief(self) -> None:
        self.results_this_brief = []

    def snapshot(self) -> dict:
        return {
            "total_recovered_rs": round(self.total_recovered_rs, 2),
            "total_reckless": self.total_reckless,
            "this_brief": [
                {
                    "batch_id": r.batch_id,
                    "accepted": r.accepted,
                    "units": r.units_liquidated,
                    "rs": r.rs_recovered,
                    "reason": r.reason,
                    "reckless": r.reckless,
                }
                for r in self.results_this_brief
            ],
        }


def parse_liquidate_directive(directive: dict) -> list[LiquidationDecision]:
    """Pull LIQUIDATE actions out of a parsed PRICING directive.

    The brief format reuses the existing PRICING action list:
        {"engine": "PRICING", "actions": [
            {"action": "LIQUIDATE", "batch_id": "...", "channel": "B2B"}, ...
        ]}
    """
    actions = directive.get("actions", []) if isinstance(directive, dict) else []
    out: list[LiquidationDecision] = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        if a.get("action") != "LIQUIDATE":
            continue
        bid = a.get("batch_id")
        if not bid:
            continue
        out.append(LiquidationDecision(
            batch_id=str(bid),
            channel=str(a.get("channel", "B2B")),
        ))
    return out
