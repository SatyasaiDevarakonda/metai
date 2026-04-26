"""ExpertAgent — Snorkel-style simulated expert with preference drift.

Strategy doc Section 9 (Snorkel AI bonus): an expert intervenes on
~15% of major decisions; after episode 10, the expert's preference
shifts from revenue-maximisation to waste-minimisation. The agent
under training has to detect this drift and update its brief
priorities accordingly.

This file gives the project the Snorkel-bonus agent the strategy
mentions. The expert produces an alternative brief whenever it
"intervenes"; the env (or a comparison harness) can override the
hero's brief with the expert's brief on those steps so the trained
model sees the corrected behaviour. The `PreferencePhase` enum lets
callers introspect which preference is currently active.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from freshprice_env.agents.consumer_cohort_agent import ConsumerCohortAgent  # noqa: F401 (re-export shape)


class PreferencePhase(str, Enum):
    """Two preference regimes the expert switches between."""
    REVENUE_FIRST = "REVENUE_FIRST"   # default: maximise revenue per brief
    WASTE_FIRST = "WASTE_FIRST"       # post-drift: minimise expired units


@dataclass
class ExpertIntervention:
    """One intervention record for the comparison harness / oversight log."""
    episode: int
    brief_index: int
    phase: PreferencePhase
    rationale: str
    suggested_brief: str


class ExpertAgent:
    """Stochastic expert that overrides the hero on ~15% of briefs.

    Public surface:
        maybe_intervene(episode, brief_idx, prompt) -> ExpertIntervention | None
        observe_outcome(episode_idx) -> None       # advances the phase clock

    The agent is deterministic given its seed: the same (episode, brief)
    pair always produces the same intervention decision. This is how
    the comparison cell (cell-comparison) reports a stable "expert
    overrides X% of decisions" stat.
    """

    INTERVENTION_RATE: float = 0.15
    DRIFT_AT_EPISODE: int = 10

    def __init__(
        self,
        seed: int = 0,
        intervention_rate: float | None = None,
        drift_at_episode: int | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        if intervention_rate is not None:
            self.INTERVENTION_RATE = intervention_rate
        if drift_at_episode is not None:
            self.DRIFT_AT_EPISODE = drift_at_episode
        self._current_episode = 0
        self._intervention_log: list[ExpertIntervention] = []

    # ------------------------------------------------------------------
    # Phase clock
    # ------------------------------------------------------------------

    def observe_outcome(self, episode_idx: int) -> None:
        """Tell the agent that a new episode has begun."""
        self._current_episode = episode_idx

    @property
    def phase(self) -> PreferencePhase:
        return (
            PreferencePhase.WASTE_FIRST
            if self._current_episode >= self.DRIFT_AT_EPISODE
            else PreferencePhase.REVENUE_FIRST
        )

    # ------------------------------------------------------------------
    # Intervention API
    # ------------------------------------------------------------------

    def maybe_intervene(
        self,
        episode: int,
        brief_idx: int,
        prompt: str,
    ) -> ExpertIntervention | None:
        """With probability INTERVENTION_RATE, produce an alternative brief.

        Returns ``None`` when the expert chooses not to intervene.
        """
        # Deterministic per (episode, brief_idx) so reruns are stable.
        local_rng = random.Random(hash((episode, brief_idx, "expert")) & 0xffffffff)
        if local_rng.random() > self.INTERVENTION_RATE:
            return None

        phase = self.phase
        if phase == PreferencePhase.REVENUE_FIRST:
            brief = self._revenue_brief()
            rationale = "expert: prioritise revenue; hold price on FRESH stock"
        else:
            brief = self._waste_brief()
            rationale = "expert (post-drift): prioritise waste-minimisation; aggressive clearance"

        record = ExpertIntervention(
            episode=episode,
            brief_index=brief_idx,
            phase=phase,
            rationale=rationale,
            suggested_brief=brief,
        )
        self._intervention_log.append(record)
        return record

    def interventions_this_run(self) -> list[ExpertIntervention]:
        return list(self._intervention_log)

    # ------------------------------------------------------------------
    # Brief templates per phase
    # ------------------------------------------------------------------

    @staticmethod
    def _revenue_brief() -> str:
        return (
            "SITUATION: Expert override -- prioritise revenue. Holding "
            "price on FRESH stock; selective clearance only on "
            "near-expiry batches.\n\n"
            "SIGNAL ANALYSIS: N/A\n\n"
            "VIABILITY CHECK: N/A\n\n"
            "RECOMMENDATION: Hold prices to preserve margin; discount only CRITICAL/URGENT.\n\n"
            "DIRECTIVE:\n"
            '{"engine": "PRICING", "actions": []}\n\n'
            "CONFIDENCE: HIGH"
        )

    @staticmethod
    def _waste_brief() -> str:
        return (
            "SITUATION: Expert override (post-drift) -- prioritise waste "
            "minimisation. Aggressive clearance and B2B routing for any "
            "batch under 24 hours.\n\n"
            "SIGNAL ANALYSIS: N/A\n\n"
            "VIABILITY CHECK: N/A\n\n"
            "RECOMMENDATION: Discount CRITICAL/URGENT batches deeply; "
            "route stranded inventory to processors via Engine 5.\n\n"
            "DIRECTIVE:\n"
            '{"engine": "PRICING", "actions": []}\n\n'
            "CONFIDENCE: HIGH"
        )
