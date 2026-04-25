"""Curriculum progression manager for the 5 training scenarios.

Tracks episode performance, decides when to promote, and provides
the current scenario to the training loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from freshprice_env.constants import (
    CURRICULUM_PROMOTION_WINDOW,
    CURRICULUM_PROMOTION_WRR_THRESHOLD,
)
from freshprice_env.enums import CurriculumScenario

logger = logging.getLogger(__name__)

# Maximum curriculum level — derived from the enum so adding REGULATORY_WEEK
# (level 5) automatically extends the curriculum. Previously hardcoded to
# CRISIS_WEEK.value which capped progression at 4.
_MAX_LEVEL: int = max(s.value for s in CurriculumScenario)

# Minimum constitutional pass rate for "eval above promotion" — a high WRR
# alone is no longer sufficient. Tuned to mirror the reward.py audit floor.
EVAL_PROMOTION_CONSTITUTIONAL_FLOOR: float = 0.60


@dataclass
class EpisodeResult:
    """Record of a single training episode."""

    episode_num: int
    scenario: CurriculumScenario
    wrr: float
    brief_quality_score: float
    anti_hack_violations: int
    constitutional_passed: bool
    episode_valid: bool


class CurriculumManager:
    """Manages curriculum progression across the 5 training scenarios."""

    def __init__(self) -> None:
        self.current_scenario: CurriculumScenario = CurriculumScenario.STABLE_WEEK
        self.current_level: int = 0
        self.episodes_in_level: int = 0
        self.total_episodes: int = 0
        # Sliding window — keep last 10 results for current level only
        self._recent_results: list[EpisodeResult] = []
        # Records each promotion: {from_level, to_level, episode_num, avg_wrr}
        self._promotion_history: list[dict] = []

    # ------------------------------------------------------------------
    # Episode recording and promotion
    # ------------------------------------------------------------------

    def record_episode(self, result: EpisodeResult) -> bool:
        """Record an episode result and check for promotion.

        Returns True if promotion occurred this episode, False otherwise.

        Only episodes that are both episode_valid AND constitutional_passed
        count toward the promotion window. Invalid or constitutionally-failed
        episodes are recorded but do not advance the promotion window.
        """
        self.total_episodes += 1
        self.episodes_in_level += 1
        self._recent_results.append(result)

        # Keep sliding window at max 10 entries
        if len(self._recent_results) > 10:
            self._recent_results = self._recent_results[-10:]

        # Already at max level — no promotion possible
        if self.current_level >= _MAX_LEVEL:
            return False

        # Collect valid episodes from recent results
        valid_results = [
            r for r in self._recent_results
            if r.episode_valid and r.constitutional_passed
        ]

        # Need at least CURRICULUM_PROMOTION_WINDOW valid episodes
        if len(valid_results) < CURRICULUM_PROMOTION_WINDOW:
            return False

        # Check the last CURRICULUM_PROMOTION_WINDOW valid episodes
        window = valid_results[-CURRICULUM_PROMOTION_WINDOW:]
        avg_wrr = sum(r.wrr for r in window) / len(window)

        if avg_wrr >= CURRICULUM_PROMOTION_WRR_THRESHOLD:
            self._promote(avg_wrr)
            return True

        return False

    def _promote(self, avg_wrr: float) -> None:
        """Advance to the next scenario."""
        old_level = self.current_level
        new_level = old_level + 1

        self._promotion_history.append({
            "from_level": old_level,
            "to_level": new_level,
            "episode_num": self.total_episodes,
            "avg_wrr": round(avg_wrr, 4),
        })

        self.current_level = new_level
        self.current_scenario = CurriculumScenario(new_level)
        self.episodes_in_level = 0
        self._recent_results = []

        logger.info(
            "Curriculum promoted: level %d (%s) → level %d (%s) at episode %d (avg WRR: %.3f)",
            old_level, CurriculumScenario(old_level).name,
            new_level, self.current_scenario.name,
            self.total_episodes, avg_wrr,
        )

    # ------------------------------------------------------------------
    # Evaluation scheduling
    # ------------------------------------------------------------------

    def should_run_evaluation(self, eval_interval: int = 10) -> bool:
        """Returns True every eval_interval episodes within the current level.

        Evaluation episodes use fixed seeds and do not count toward
        promotion or trajectory collection.
        """
        if self.episodes_in_level == 0:
            return False
        return self.episodes_in_level % eval_interval == 0

    def get_eval_seeds(self, n: int = 5) -> list[int]:
        """Return a fixed list of evaluation seeds for the current level.

        Level-specific: level 0 → [0,1,2,3,4], level 2 → [2000,2001,2002,2003,2004].
        Same seeds every eval run at this level for reproducibility.
        """
        return [self.current_level * 1000 + i for i in range(n)]

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    @staticmethod
    def is_eval_above_promotion(
        eval_episodes: list,
        wrr_threshold: float = CURRICULUM_PROMOTION_WRR_THRESHOLD,
        constitutional_pass_rate_floor: float = EVAL_PROMOTION_CONSTITUTIONAL_FLOOR,
    ) -> tuple[bool, dict]:
        """Combined WRR + constitutional-pass-rate gate.

        The notebook used to print "ABOVE PROMOTION THRESHOLD" using only
        the mean WRR. That hid the fact that 3 of 4 eval episodes were
        constitutionally failing. This helper makes the gate explicit:

          - mean WRR >= ``wrr_threshold``
          - AND constitutional pass rate >= ``constitutional_pass_rate_floor``

        ``eval_episodes`` accepts either a list of EvalEpisodeResult /
        EpisodeResult objects (with ``.wrr`` and ``.constitutional_passed``)
        or a list of dicts with the same keys. Returns ``(passes, diagnostics)``
        so the caller can print *why* it failed when it does.
        """
        if not eval_episodes:
            return False, {
                "n": 0, "wrr_mean": 0.0,
                "constitutional_pass_rate": 0.0,
                "wrr_ok": False, "constitution_ok": False,
                "reason": "no eval episodes",
            }

        wrrs: list[float] = []
        n_const_pass = 0
        for ep in eval_episodes:
            wrr = (
                ep.get("wrr") if isinstance(ep, dict) else getattr(ep, "wrr", 0.0)
            )
            const = (
                ep.get("constitutional_passed") if isinstance(ep, dict)
                else getattr(ep, "constitutional_passed", True)
            )
            wrrs.append(float(wrr or 0.0))
            n_const_pass += 1 if const else 0

        wrr_mean = sum(wrrs) / len(wrrs)
        const_rate = n_const_pass / len(eval_episodes)
        wrr_ok = wrr_mean >= wrr_threshold
        const_ok = const_rate >= constitutional_pass_rate_floor

        diag = {
            "n": len(eval_episodes),
            "wrr_mean": round(wrr_mean, 4),
            "wrr_threshold": wrr_threshold,
            "wrr_ok": wrr_ok,
            "constitutional_pass_rate": round(const_rate, 3),
            "constitutional_floor": constitutional_pass_rate_floor,
            "constitution_ok": const_ok,
        }
        if not wrr_ok and not const_ok:
            diag["reason"] = (
                f"WRR {wrr_mean:.3f} < {wrr_threshold} AND "
                f"constitutional pass {const_rate:.0%} < "
                f"{constitutional_pass_rate_floor:.0%}"
            )
        elif not wrr_ok:
            diag["reason"] = f"WRR {wrr_mean:.3f} < {wrr_threshold}"
        elif not const_ok:
            diag["reason"] = (
                f"constitutional pass {const_rate:.0%} < "
                f"{constitutional_pass_rate_floor:.0%}"
            )
        else:
            diag["reason"] = "above both thresholds"
        return (wrr_ok and const_ok), diag

    def get_status(self) -> dict:
        """Return a dict suitable for WandB logging and terminal display."""
        valid_results = [
            r for r in self._recent_results
            if r.episode_valid and r.constitutional_passed
        ]

        if valid_results:
            recent_wrr_mean = sum(r.wrr for r in valid_results) / len(valid_results)
        else:
            recent_wrr_mean = 0.0

        wrr_to_promotion = max(0.0, CURRICULUM_PROMOTION_WRR_THRESHOLD - recent_wrr_mean)

        return {
            "curriculum_level": self.current_level,
            "scenario_name": self.current_scenario.name,
            "episodes_in_level": self.episodes_in_level,
            "total_episodes": self.total_episodes,
            "recent_wrr_mean": round(recent_wrr_mean, 4),
            "recent_wrr_window": len(valid_results),
            "wrr_to_promotion": round(wrr_to_promotion, 4),
            "promotions": list(self._promotion_history),
        }
