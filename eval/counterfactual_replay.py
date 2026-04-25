"""Counterfactual replay — swap one decision, re-run the next K ticks.

Lets the demo show "if our agent had ACCEPTed this offer instead of
DECLINING, WRR would have moved from 0.42 to 0.61 over the next 96 ticks."
The dashboard puts a slider on the timeline; dragging it builds and
runs a CounterfactualReplay and shows the divergence in real time.

Usage:

    replay = CounterfactualReplay(scenario, seed=42, brief_log=briefs)
    fork = replay.swap_decision_at_brief(
        brief_index=14,
        new_brief=alternate_brief_text,
    )
    fork.run_until(target_tick=288)
    print(fork.divergence_summary())   # {"wrr_delta": +0.18, ...}

The implementation is *episode-stable*: replaying with the same seed
and same brief log reproduces the original WRR exactly. That stability
is what lets the counterfactual difference be attributable to the
single swapped brief.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from freshprice_env.enums import CurriculumScenario
from freshprice_env.freshprice_env import FreshPriceEnv


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class BriefLogEntry:
    """One brief from a logged episode."""

    tick: int
    engine_type: str
    brief_text: str


@dataclass
class ReplayPoint:
    """Per-brief snapshot during a replay run."""

    brief_index: int
    tick: int
    wrr: float
    reward: float
    engine_type: str
    quality_score: float


@dataclass
class CounterfactualFork:
    """One forked timeline. Holds points and exposes a divergence summary."""

    fork_id: str
    swapped_at_brief: int
    points: list[ReplayPoint] = field(default_factory=list)
    final_wrr: float = 0.0
    notes: str = ""

    def divergence_summary(self, baseline: "CounterfactualFork") -> dict:
        """Diff this fork against a baseline run (same seed, no swap)."""
        return {
            "fork_id": self.fork_id,
            "swapped_at_brief": self.swapped_at_brief,
            "baseline_final_wrr": round(baseline.final_wrr, 4),
            "fork_final_wrr": round(self.final_wrr, 4),
            "wrr_delta": round(self.final_wrr - baseline.final_wrr, 4),
            "n_points": len(self.points),
        }


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

class CounterfactualReplay:
    """Re-run an episode with one decision swapped.

    The caller provides the original brief log. The replay runs a fresh
    FreshPriceEnv with the same seed; deterministic engines + same brief
    inputs reproduce the original WRR. To produce a fork, supply a
    ``new_brief`` for a chosen ``brief_index``.
    """

    def __init__(
        self,
        scenario: CurriculumScenario,
        seed: int,
        brief_log: list[BriefLogEntry],
    ) -> None:
        self.scenario = scenario
        self.seed = int(seed)
        self.brief_log: list[BriefLogEntry] = list(brief_log)

    # ------------------------------------------------------------------
    # Baseline replay
    # ------------------------------------------------------------------

    def run_baseline(
        self,
        max_briefs: int | None = None,
    ) -> CounterfactualFork:
        return self._run(
            swap_brief_index=None,
            replacement_brief=None,
            max_briefs=max_briefs,
            fork_id="baseline",
        )

    # ------------------------------------------------------------------
    # Forked run with one swap
    # ------------------------------------------------------------------

    def swap_decision_at_brief(
        self,
        brief_index: int,
        new_brief: str,
        max_briefs: int | None = None,
        fork_id: str | None = None,
    ) -> CounterfactualFork:
        if not (0 <= brief_index < len(self.brief_log)):
            raise IndexError(f"brief_index {brief_index} out of range")
        return self._run(
            swap_brief_index=brief_index,
            replacement_brief=new_brief,
            max_briefs=max_briefs,
            fork_id=fork_id or f"fork_at_{brief_index}",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(
        self,
        swap_brief_index: int | None,
        replacement_brief: str | None,
        max_briefs: int | None,
        fork_id: str,
    ) -> CounterfactualFork:
        env = FreshPriceEnv(scenario=self.scenario, seed=self.seed)
        obs, info = env.reset()
        points: list[ReplayPoint] = []
        last_wrr = 0.0

        n = len(self.brief_log) if max_briefs is None else min(
            max_briefs, len(self.brief_log)
        )
        for i in range(n):
            entry = self.brief_log[i]
            brief_text = (
                replacement_brief
                if (swap_brief_index is not None and i == swap_brief_index)
                else entry.brief_text
            )
            obs, reward, terminated, truncated, info = env.step(brief_text)
            wrr = env._state.wrr if env._state is not None else last_wrr
            last_wrr = wrr
            points.append(ReplayPoint(
                brief_index=i,
                tick=info.get("tick", i * 8),
                wrr=round(float(wrr), 4),
                reward=round(float(reward), 4),
                engine_type=info.get("engine_type", entry.engine_type),
                quality_score=round(float(info.get("quality_score", 0.0)), 4),
            ))
            if terminated:
                break

        return CounterfactualFork(
            fork_id=fork_id,
            swapped_at_brief=swap_brief_index if swap_brief_index is not None else -1,
            points=points,
            final_wrr=round(last_wrr, 4),
            notes=(
                f"Replayed {len(points)} briefs; swap at "
                f"{swap_brief_index if swap_brief_index is not None else 'none'}"
            ),
        )


# ---------------------------------------------------------------------------
# Loaders / serialization
# ---------------------------------------------------------------------------

def load_brief_log_from_jsonl(path: str | Path) -> list[BriefLogEntry]:
    """Load a brief log written by FreshPriceEnv.get_episode_record() or
    by inference.py's [STEP] log.

    Tolerant: any row missing tick / engine / brief_text is skipped.
    """
    out: list[BriefLogEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tick = obj.get("tick")
            engine = obj.get("engine_type") or obj.get("engine") or "PRICING"
            brief = obj.get("brief_text") or obj.get("raw_response") or obj.get("brief") or ""
            if tick is None or not brief:
                continue
            out.append(BriefLogEntry(
                tick=int(tick),
                engine_type=str(engine),
                brief_text=str(brief),
            ))
    return out


def fork_to_dict(fork: CounterfactualFork) -> dict:
    return {
        "fork_id": fork.fork_id,
        "swapped_at_brief": fork.swapped_at_brief,
        "final_wrr": fork.final_wrr,
        "notes": fork.notes,
        "points": [
            {
                "brief_index": p.brief_index,
                "tick": p.tick,
                "wrr": p.wrr,
                "reward": p.reward,
                "engine_type": p.engine_type,
                "quality_score": p.quality_score,
            }
            for p in fork.points
        ],
    }
