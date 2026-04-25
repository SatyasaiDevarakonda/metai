"""RegulatorAgent — emits mid-episode policy drift.

The regulator is a procedural agent that watches an episode tick and
fires policy changes at scheduled times. Each policy change:

  1. Mutates the active schema in SchemaRegistry (e.g., PRICING v1 → v2)
  2. Posts a BROADCAST message on the MarketBus describing the change
  3. Optionally tightens / loosens reward thresholds

Drift schedules per scenario:

  STABLE_WEEK     — no drift (control)
  BUSY_WEEKEND    — single drift at tick 192: TREND v1 → v2 (verification)
  FARMER_WEEK     — drift at tick 240: FARMER v1 → v2 (FSSAI traceability)
  TREND_WEEK      — drift at tick 288: TREND v1 → v2 + PRICING v1 → v2
  CRISIS_WEEK     — three drifts: tick 144, 336, 528 (price-cap escalation)
  REGULATORY_WEEK — *new scenario*: drifts every ~96 ticks, full chaos

The agent's verbal style is deliberately bureaucratic so judges
recognise it as the regulator on the dashboard.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from freshprice_env.brief_pipeline.schema_registry import (
    SchemaRegistry,
    default_registry,
)
from freshprice_env.enums import BriefEngineType, CurriculumScenario
from freshprice_env.protocol.market_bus import MarketBus, MessageVerb


@dataclass(frozen=True)
class PolicyChange:
    """A scheduled regulatory event."""

    tick: int
    engine: BriefEngineType
    new_version: str
    headline: str
    body: str


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

_STABLE_DRIFT: list[PolicyChange] = []

_BUSY_DRIFT: list[PolicyChange] = [
    PolicyChange(
        tick=192,
        engine=BriefEngineType.TREND,
        new_version="v2",
        headline="ASCI-IND/2026/04 — Influencer disclosure mandate",
        body=(
            "Effective immediately: TREND directives must include a "
            "`source_verification` field for any orders triggered by social "
            "trend signals. Unverified trend orders will be rejected."
        ),
    ),
]

_FARMER_DRIFT: list[PolicyChange] = [
    PolicyChange(
        tick=240,
        engine=BriefEngineType.FARMER,
        new_version="v2",
        headline="FSSAI/PROD/2026/15 — Farmer traceability requirement",
        body=(
            "Effective immediately: FARMER directives must carry the "
            "originating `farmer_id` for traceability. Acceptances without "
            "farmer_id will be auto-declined."
        ),
    ),
]

_TREND_DRIFT: list[PolicyChange] = [
    PolicyChange(
        tick=288,
        engine=BriefEngineType.TREND,
        new_version="v2",
        headline="ASCI-IND/2026/04 — Influencer disclosure mandate",
        body=(
            "Effective immediately: TREND directives must include a "
            "`source_verification` field."
        ),
    ),
    PolicyChange(
        tick=480,
        engine=BriefEngineType.PRICING,
        new_version="v2",
        headline="FSSAI/COLD/2026/22 — Cold-chain logging required",
        body=(
            "PRICING directives for dairy and leafy greens must record a "
            "`cold_chain_log` value (timestamped boolean). Missing logs "
            "will fail validation."
        ),
    ),
]

_CRISIS_DRIFT: list[PolicyChange] = [
    PolicyChange(
        tick=144,
        engine=BriefEngineType.PRICING,
        new_version="v2",
        headline="FSSAI/COLD/2026/22 — Cold-chain logging required",
        body=(
            "PRICING directives must include `cold_chain_log` for dairy and "
            "leafy greens."
        ),
    ),
    PolicyChange(
        tick=336,
        engine=BriefEngineType.FARMER,
        new_version="v2",
        headline="FSSAI/PROD/2026/15 — Farmer traceability",
        body="FARMER directives must include `farmer_id`.",
    ),
    PolicyChange(
        tick=528,
        engine=BriefEngineType.PRICING,
        new_version="v3",
        headline="DPIIT/CAP/2026/08 — Discount cap (price-cap)",
        body=(
            "Emergency: minimum discount multiplier raised to 0.40. Field "
            "renamed to `clearance_action`. Briefs using `price_multiplier` "
            "will fail validation."
        ),
    ),
]

# REGULATORY_WEEK — chaotic, every ~96 ticks
_REGULATORY_DRIFT: list[PolicyChange] = [
    PolicyChange(
        tick=96,
        engine=BriefEngineType.TREND,
        new_version="v2",
        headline="ASCI-IND/2026/04 — Influencer disclosure",
        body="TREND directives must include `source_verification`.",
    ),
    PolicyChange(
        tick=192,
        engine=BriefEngineType.FARMER,
        new_version="v2",
        headline="FSSAI/PROD/2026/15 — Farmer traceability",
        body="FARMER directives must include `farmer_id`.",
    ),
    PolicyChange(
        tick=336,
        engine=BriefEngineType.PRICING,
        new_version="v2",
        headline="FSSAI/COLD/2026/22 — Cold-chain logging",
        body="PRICING directives need `cold_chain_log`.",
    ),
    PolicyChange(
        tick=432,
        engine=BriefEngineType.PRICING,
        new_version="v3",
        headline="DPIIT/CAP/2026/08 — Discount cap",
        body=(
            "Floor multiplier raised to 0.40. Field renamed to "
            "`clearance_action`."
        ),
    ),
]


_SCHEDULES: dict[CurriculumScenario, list[PolicyChange]] = {
    CurriculumScenario.STABLE_WEEK:     _STABLE_DRIFT,
    CurriculumScenario.BUSY_WEEKEND:    _BUSY_DRIFT,
    CurriculumScenario.FARMER_WEEK:     _FARMER_DRIFT,
    CurriculumScenario.TREND_WEEK:      _TREND_DRIFT,
    CurriculumScenario.CRISIS_WEEK:     _CRISIS_DRIFT,
    CurriculumScenario.REGULATORY_WEEK: _REGULATORY_DRIFT,
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class RegulatoryEvent:
    """An applied policy change. Surfaced in info dicts and bus."""

    tick: int
    engine: str
    version: str
    headline: str
    body: str
    schema_history_index: int


class RegulatorAgent:
    """Stateless schedule executor that mutates the SchemaRegistry."""

    def __init__(
        self,
        scenario: CurriculumScenario,
        bus: MarketBus | None = None,
        registry: SchemaRegistry | None = None,
        custom_schedule: list[PolicyChange] | None = None,
    ) -> None:
        self.scenario = scenario
        self._bus = bus
        self._registry = registry or default_registry()
        self._registry.reset_to_v1()
        self._schedule: list[PolicyChange] = list(
            custom_schedule if custom_schedule is not None
            else _SCHEDULES.get(scenario, [])
        )
        self._fired_indexes: set[int] = set()
        self._applied: list[RegulatoryEvent] = []

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, current_tick: int) -> list[RegulatoryEvent]:
        """Fire any policies whose tick has been reached."""
        fired_now: list[RegulatoryEvent] = []
        for i, pc in enumerate(self._schedule):
            if i in self._fired_indexes:
                continue
            if current_tick >= pc.tick:
                self._registry.set_version(pc.engine, pc.new_version, current_tick)
                self._fired_indexes.add(i)
                ev = RegulatoryEvent(
                    tick=current_tick,
                    engine=pc.engine.value,
                    version=pc.new_version,
                    headline=pc.headline,
                    body=pc.body,
                    schema_history_index=len(self._registry.history()) - 1,
                )
                self._applied.append(ev)
                fired_now.append(ev)
                if self._bus is not None:
                    self._bus.post(
                        tick=current_tick,
                        sender_id="regulator",
                        verb=MessageVerb.BROADCAST,
                        body=f"{pc.headline} — {pc.body}",
                        receiver_id=None,
                        payload={
                            "engine": pc.engine.value,
                            "new_version": pc.new_version,
                            "headline": pc.headline,
                            "kind": "POLICY_CHANGE",
                        },
                    )
        return fired_now

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def applied_events(self) -> list[RegulatoryEvent]:
        return list(self._applied)

    def schedule_summary(self) -> list[dict]:
        return [
            {
                "tick": pc.tick,
                "engine": pc.engine.value,
                "new_version": pc.new_version,
                "headline": pc.headline,
                "fired": i in self._fired_indexes,
            }
            for i, pc in enumerate(self._schedule)
        ]

    @property
    def registry(self) -> SchemaRegistry:
        return self._registry
