"""RiderPoolEngine — Blinkit/Zepto-style courier bottleneck.

Quick commerce in 2026 lives on a 10-minute promise. Behind that promise
is a finite pool of riders: when discounts spike demand the rider queue
grows, freshness clocks on queued orders keep ticking, and orders that
wait too long for a rider count as **transit spoilage** — a real cost
that does not show up in any of the existing reward components.

This engine is intentionally **additive**: the existing PRICING /
FARMER / TREND engines do not know about it. ``MarketCommonsEnv`` ticks
this engine after the pricing engine has computed sales, feeds the per-
brief delivery-quality reward into the env's info dict as
``r6_delivery_quality``, and the market bus carries ``RIDER_SATURATED``
events when the queue exceeds capacity.

Why this matters for RL:
  - Over-aggressive discounts now have a globally visible cost
    (rider saturation -> transit spoilage -> r6 penalty).
  - The premium cohort walks when ETAs slip, so r6 indirectly affects
    consumer demand on the next tick (modeled in ``ConsumerCohortAgent``).
  - The agent has to learn throughput-aware pricing, not just
    inventory-clearance pricing.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from freshprice_env.constants import (
    BLINKIT_DEFAULT_RIDER_COUNT,
    BLINKIT_EXPRESS_PROMISE_MINUTES,
    BLINKIT_RIDER_FERRY_MINUTES,
    BLINKIT_STANDARD_PROMISE_MINUTES,
    R6_BONUS_CAP_PER_BRIEF,
    R6_ON_TIME_BONUS,
    R6_TRANSIT_SPOILAGE_PENALTY,
    TICK_DURATION_MINUTES,
)


@dataclass
class Order:
    """One delivery in flight or queued."""

    order_id: str
    batch_id: str
    category: str
    units: int
    eta_tier: str            # "EXPRESS" or "STANDARD"
    placed_tick: int         # tick the order was placed
    promised_minutes: float  # promised end-to-end ETA in minutes
    freshness_at_placed_hrs: float
    rider_assigned_tick: int | None = None
    delivered_tick: int | None = None

    def waiting_minutes(self, current_tick: int) -> float:
        return max(0.0, (current_tick - self.placed_tick) * TICK_DURATION_MINUTES)


@dataclass
class RiderPoolStats:
    """Per-brief accounting consumed by the env / dashboard."""

    orders_delivered: int = 0
    orders_on_time: int = 0
    orders_late: int = 0
    transit_spoiled: int = 0
    queue_peak: int = 0
    saturation_ticks: int = 0     # ticks during which queue > 0
    avg_eta_minutes: float = 0.0
    avg_wait_minutes: float = 0.0


@dataclass
class RiderPoolEngine:
    """Tickable rider pool for one dark store.

    Public surface:
        tick(current_tick, sales_this_tick, batches, rng) -> events
        compute_brief_reward() -> r6_delivery_quality (resets stats)
        snapshot() -> dict for the dashboard / prompt
    """

    rider_count: int = BLINKIT_DEFAULT_RIDER_COUNT
    pending: deque[Order] = field(default_factory=deque)
    active: list[Order] = field(default_factory=list)
    delivered_log: list[Order] = field(default_factory=list)
    stats: RiderPoolStats = field(default_factory=RiderPoolStats)
    _order_seq: int = 0

    # ------------------------------------------------------------------
    # Tick API
    # ------------------------------------------------------------------

    def tick(
        self,
        *,
        current_tick: int,
        sales_this_tick: dict[str, int],
        batches_by_id: dict[str, "object"],
        rng: random.Random,
        eta_tier_by_batch: dict[str, str] | None = None,
    ) -> list[dict]:
        """Advance the rider pool by one tick. Returns bus events."""
        events: list[dict] = []
        eta_tier_by_batch = eta_tier_by_batch or {}

        # 1. Spawn orders for units sold this tick.
        for batch_id, units in sales_this_tick.items():
            if units <= 0:
                continue
            batch = batches_by_id.get(batch_id)
            if batch is None:
                continue
            tier = eta_tier_by_batch.get(batch_id, "STANDARD")
            promised = (
                BLINKIT_EXPRESS_PROMISE_MINUTES
                if tier == "EXPRESS"
                else BLINKIT_STANDARD_PROMISE_MINUTES
            )
            self._order_seq += 1
            order = Order(
                order_id=f"O{self._order_seq:06d}",
                batch_id=batch_id,
                category=getattr(batch, "category", "unknown"),
                units=int(units),
                eta_tier=tier,
                placed_tick=current_tick,
                promised_minutes=promised,
                freshness_at_placed_hrs=float(
                    getattr(batch, "hours_to_expiry", 24.0)
                ),
            )
            self.pending.append(order)

        # 2. Free riders whose orders have ferried.
        ferry_ticks = max(1, int(BLINKIT_RIDER_FERRY_MINUTES // TICK_DURATION_MINUTES))
        still_active: list[Order] = []
        for o in self.active:
            assert o.rider_assigned_tick is not None
            if current_tick - o.rider_assigned_tick >= ferry_ticks:
                o.delivered_tick = current_tick
                self.delivered_log.append(o)
                self.stats.orders_delivered += 1
                eta = (o.delivered_tick - o.placed_tick) * TICK_DURATION_MINUTES
                if eta <= o.promised_minutes:
                    self.stats.orders_on_time += 1
                else:
                    self.stats.orders_late += 1
                self._update_avg(eta, o.waiting_minutes(o.rider_assigned_tick))
            else:
                still_active.append(o)
        self.active = still_active

        # 3. Assign idle riders to queued orders.
        idle = self.rider_count - len(self.active)
        while idle > 0 and self.pending:
            o = self.pending.popleft()
            o.rider_assigned_tick = current_tick
            self.active.append(o)
            idle -= 1

        # 4. Detect transit spoilage on queued orders. An order that has
        #    waited longer than its remaining freshness margin is a
        #    transit-spoiled order — its category gets a customer
        #    complaint, and it leaves the queue without revenue.
        survivors: deque[Order] = deque()
        for o in self.pending:
            wait_hrs = o.waiting_minutes(current_tick) / 60.0
            if wait_hrs >= o.freshness_at_placed_hrs:
                self.stats.transit_spoiled += 1
                events.append({
                    "kind": "transit_spoiled",
                    "order_id": o.order_id, "batch_id": o.batch_id,
                    "category": o.category, "units": o.units,
                    "wait_minutes": round(o.waiting_minutes(current_tick), 1),
                })
            else:
                survivors.append(o)
        self.pending = survivors

        # 5. Saturation event — queue depth exceeds capacity headroom.
        queue_depth = len(self.pending)
        self.stats.queue_peak = max(self.stats.queue_peak, queue_depth)
        if queue_depth > 0:
            self.stats.saturation_ticks += 1
        if queue_depth > self.rider_count:
            events.append({
                "kind": "rider_saturated",
                "queue_depth": queue_depth,
                "capacity": self.rider_count,
                "tick": current_tick,
            })

        return events

    # ------------------------------------------------------------------
    # Reward / inspection
    # ------------------------------------------------------------------

    def compute_brief_reward(self) -> float:
        """r6_delivery_quality for the brief just completed.

        Positive when on-time deliveries dominate; negative when transit
        spoilage occurs. Cap is symmetric.
        """
        bonus = min(
            R6_BONUS_CAP_PER_BRIEF,
            self.stats.orders_on_time * R6_ON_TIME_BONUS,
        )
        penalty = self.stats.transit_spoiled * R6_TRANSIT_SPOILAGE_PENALTY
        return round(bonus - penalty, 4)

    def snapshot(self) -> dict:
        return {
            "rider_count": self.rider_count,
            "queue_depth": len(self.pending),
            "active_orders": len(self.active),
            "orders_delivered": self.stats.orders_delivered,
            "orders_on_time": self.stats.orders_on_time,
            "orders_late": self.stats.orders_late,
            "transit_spoiled": self.stats.transit_spoiled,
            "queue_peak": self.stats.queue_peak,
            "saturation_ticks": self.stats.saturation_ticks,
            "avg_eta_minutes": round(self.stats.avg_eta_minutes, 1),
        }

    def reset_brief_stats(self) -> None:
        """Call at the end of each brief so r6 measures *this* brief only."""
        self.stats = RiderPoolStats()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_avg(self, eta: float, wait: float) -> None:
        n = self.stats.orders_delivered
        if n <= 1:
            self.stats.avg_eta_minutes = eta
            self.stats.avg_wait_minutes = wait
            return
        self.stats.avg_eta_minutes = (
            (self.stats.avg_eta_minutes * (n - 1) + eta) / n
        )
        self.stats.avg_wait_minutes = (
            (self.stats.avg_wait_minutes * (n - 1) + wait) / n
        )
