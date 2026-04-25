"""Collects episode trajectories, selects the best, and generates DPO preference pairs.

Constitutional audit gates entry into the buffer.
High-regret pairs are oversampled 3x during DPO training.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

from freshprice_env.enums import CurriculumScenario

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """Full record of a single training episode."""

    episode_num: int
    scenario: CurriculumScenario
    wrr: float
    brief_quality_score: float
    constitutional_passed: bool
    episode_valid: bool
    briefs: list[dict]
    # Each brief: {tick, engine_type, prompt, raw_response, quality_score, reward_delta}
    reward_engine_snapshot: dict
    # Output of WRRRewardEngine.to_wandb_log() for this episode


@dataclass
class DPOPair:
    """A preference pair for DPO fine-tuning."""

    prompt: str
    chosen: str           # The better Operating Brief
    rejected: str         # The worse Operating Brief
    regret_score: float   # 0.0-1.0 — how much better chosen is vs rejected
    engine_type: str
    scenario: str


# Default minimum buffer size before DPO is worth running. Lowered from
# the legacy 4 to 2 so short Kaggle runs (3-4 episodes) actually train
# DPO. Tunable via the constructor or run_dpo() callsite.
DEFAULT_DPO_MIN_BUFFER: int = 2


@dataclass
class DPOReadiness:
    """Why DPO can / cannot run right now — used by the report summary."""

    can_run: bool
    buffer_size: int
    min_required: int
    pairs_available: int
    engine_coverage: dict[str, int]
    reason: str

    def to_dict(self) -> dict:
        return {
            "dpo_can_run": self.can_run,
            "buffer_size": self.buffer_size,
            "min_required": self.min_required,
            "pairs_available": self.pairs_available,
            "engine_coverage": self.engine_coverage,
            "reason": self.reason,
        }


class TrajectoryBuffer:
    """Buffer of top episode trajectories for DPO pair generation."""

    def __init__(
        self,
        max_trajectories: int = 50,
        top_n_for_dpo: int = 20,
        rng: random.Random | None = None,
    ) -> None:
        self._buffer: list[Trajectory] = []
        self.max_trajectories = max_trajectories
        self.top_n_for_dpo = top_n_for_dpo
        self.rng = rng or random.Random(42)

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def add(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the buffer.

        Only adds if episode_valid AND constitutional_passed.
        If buffer exceeds max_trajectories: evicts lowest-WRR trajectory.
        """
        if not trajectory.episode_valid or not trajectory.constitutional_passed:
            logger.debug(
                "Trajectory %d rejected: valid=%s, constitutional=%s",
                trajectory.episode_num,
                trajectory.episode_valid,
                trajectory.constitutional_passed,
            )
            return

        self._buffer.append(trajectory)

        if len(self._buffer) > self.max_trajectories:
            # Remove the lowest-WRR trajectory
            self._buffer.sort(key=lambda t: t.wrr)
            evicted = self._buffer.pop(0)
            logger.debug(
                "Evicted trajectory %d (WRR=%.3f) from buffer",
                evicted.episode_num, evicted.wrr,
            )

    def get_top_n(self, n: int | None = None) -> list[Trajectory]:
        """Return top N trajectories sorted by WRR descending."""
        count = n if n is not None else self.top_n_for_dpo
        sorted_buf = sorted(self._buffer, key=lambda t: t.wrr, reverse=True)
        return sorted_buf[:count]

    def dpo_readiness(self, min_buffer: int = DEFAULT_DPO_MIN_BUFFER) -> DPOReadiness:
        """Honest answer to "can DPO actually run?".

        The Kaggle notebook used to print ``DPO enabled : True`` even
        when DPO had been skipped because the buffer was too small.
        Use this method in the summary cell instead — the resulting
        flag is *what actually happened*, not a config value.

        ``engine_coverage`` is the number of trajectories per engine
        type so the caller can warn when DPO would only update
        PRICING behaviour.
        """
        coverage: dict[str, int] = {"PRICING": 0, "FARMER": 0, "TREND": 0}
        for traj in self._buffer:
            engines_in_traj: set[str] = set()
            for brief in traj.briefs:
                e = brief.get("engine_type", "PRICING")
                engines_in_traj.add(e)
            for e in engines_in_traj:
                coverage[e] = coverage.get(e, 0) + 1
        size = len(self._buffer)
        # Estimate pairs without actually generating them
        pairs_est = sum(len(t.briefs) for t in self.get_top_n())
        if size < min_buffer:
            return DPOReadiness(
                can_run=False, buffer_size=size, min_required=min_buffer,
                pairs_available=0, engine_coverage=coverage,
                reason=f"buffer has {size} clean episode(s); need >= {min_buffer}",
            )
        return DPOReadiness(
            can_run=True, buffer_size=size, min_required=min_buffer,
            pairs_available=pairs_est, engine_coverage=coverage,
            reason=(
                f"ready: {size} clean trajectories, ~{pairs_est} pairs"
                if any(v > 0 for v in coverage.values()) else
                "ready but no engine coverage recorded"
            ),
        )

    # ------------------------------------------------------------------
    # DPO pair generation
    # ------------------------------------------------------------------

    def generate_dpo_pairs(
        self,
        counterfactual_engine: object,
    ) -> list[DPOPair]:
        """Generate DPO preference pairs from top N trajectories.

        Args:
            counterfactual_engine: A CounterfactualEngine instance with a
                generate_synthetic_rejected(brief, prompt) method.

        Strategy:
        1. Extract briefs from top N trajectories (chosen pool)
        2. For each chosen brief, find a counterfactual from lower-ranked
           trajectories with matching tick/scenario/engine_type
        3. If no match found, use counterfactual_engine for synthetic rejected
        4. High-regret pairs (> 0.7) duplicated 3x in output
        """
        top_n = self.get_top_n()
        if not top_n:
            return []

        # Build the "rejected" pool from trajectories NOT in top N
        top_episode_nums = {t.episode_num for t in top_n}
        lower_trajectories = [
            t for t in self._buffer if t.episode_num not in top_episode_nums
        ]

        # Index lower briefs by (engine_type, scenario)
        lower_brief_index: dict[tuple[str, str], list[dict]] = {}
        for traj in lower_trajectories:
            for brief in traj.briefs:
                key = (brief.get("engine_type", ""), traj.scenario.name)
                if key not in lower_brief_index:
                    lower_brief_index[key] = []
                lower_brief_index[key].append(brief)

        pairs: list[DPOPair] = []

        for traj in top_n:
            for brief in traj.briefs:
                prompt = brief.get("prompt", "")
                chosen_response = brief.get("raw_response", "")
                chosen_quality = brief.get("quality_score", 0.0)
                chosen_reward = brief.get("reward_delta", 0.0)
                engine_type = brief.get("engine_type", "PRICING")

                if not chosen_response:
                    continue

                # Find counterfactual from lower pool
                key = (engine_type, traj.scenario.name)
                candidates = lower_brief_index.get(key, [])

                rejected_response = None
                rejected_quality = 0.0
                rejected_reward = 0.0

                if candidates:
                    # Pick the candidate with the largest quality gap
                    best_candidate = None
                    best_gap = -1.0
                    for cand in candidates:
                        cand_quality = cand.get("quality_score", 0.0)
                        cand_reward = cand.get("reward_delta", 0.0)
                        gap = (chosen_quality - cand_quality) + (chosen_reward - cand_reward)
                        if gap > best_gap and cand.get("raw_response"):
                            best_gap = gap
                            best_candidate = cand

                    if best_candidate is not None:
                        rejected_response = best_candidate["raw_response"]
                        rejected_quality = best_candidate.get("quality_score", 0.0)
                        rejected_reward = best_candidate.get("reward_delta", 0.0)

                # Synthetic fallback
                if rejected_response is None and counterfactual_engine is not None:
                    synthetic = _call_counterfactual(
                        counterfactual_engine, brief, prompt,
                    )
                    if synthetic is not None:
                        rejected_response = synthetic
                        rejected_quality = max(0.0, chosen_quality - 0.3)
                        rejected_reward = max(0.0, chosen_reward - 0.1)

                if rejected_response is None:
                    continue

                # Compute regret score normalised to [0, 1]
                raw_regret = (
                    (chosen_quality - rejected_quality)
                    + (chosen_reward - rejected_reward)
                )
                regret_score = max(0.0, min(1.0, raw_regret / 2.0))

                pair = DPOPair(
                    prompt=prompt,
                    chosen=chosen_response,
                    rejected=rejected_response,
                    regret_score=round(regret_score, 4),
                    engine_type=engine_type,
                    scenario=traj.scenario.name,
                )

                pairs.append(pair)

                # High-regret oversampling: duplicate 3x
                if regret_score > 0.7:
                    pairs.append(pair)
                    pairs.append(pair)

        return pairs

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return buffer statistics for WandB logging."""
        if not self._buffer:
            return {
                "buffer_size": 0,
                "buffer_wrr_mean": 0.0,
                "buffer_wrr_max": 0.0,
                "buffer_wrr_min": 0.0,
                "valid_trajectories": 0,
            }

        wrrs = [t.wrr for t in self._buffer]
        return {
            "buffer_size": len(self._buffer),
            "buffer_wrr_mean": round(sum(wrrs) / len(wrrs), 4),
            "buffer_wrr_max": round(max(wrrs), 4),
            "buffer_wrr_min": round(min(wrrs), 4),
            "valid_trajectories": len(self._buffer),
        }

    # ------------------------------------------------------------------
    # Level-based cleanup
    # ------------------------------------------------------------------

    def clear_below_level(self, scenario: CurriculumScenario) -> int:
        """Remove all trajectories from scenarios below the given level.

        Called when curriculum promotes — old easy-scenario trajectories
        should not pollute DPO pairs for harder scenarios.
        Returns count of removed trajectories.
        """
        threshold = scenario.value
        before = len(self._buffer)
        self._buffer = [
            t for t in self._buffer if t.scenario.value >= threshold
        ]
        removed = before - len(self._buffer)
        if removed > 0:
            logger.info(
                "Cleared %d trajectories below level %d (%s)",
                removed, threshold, scenario.name,
            )
        return removed


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _call_counterfactual(
    engine: object, brief: dict, prompt: str,
) -> str | None:
    """Safely call counterfactual_engine.generate_synthetic_rejected().

    Returns the synthetic rejected string or None if the engine
    doesn't support it or raises.
    """
    method = getattr(engine, "generate_synthetic_rejected", None)
    if method is None:
        return None
    try:
        return method(brief, prompt)
    except Exception:
        logger.debug("Counterfactual generation failed", exc_info=True)
        return None
