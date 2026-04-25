"""ScenarioComposer — adaptive curriculum that targets the agent's weak spots.

Theme #4 (Self-Improvement). Rather than running the canonical
five-level curriculum end-to-end, the composer keeps a rolling
"failure log" of episodes the agent did poorly on, and uses a
lightweight Bayesian-style search to propose the *next* scenario from
the parameter space:

    - weather: NORMAL / SUNNY / RAINY / HOT / COLD
    - event:   NONE / FESTIVAL / SPORTS_EVENT / LOCAL_HOLIDAY
    - n_farmer_offers: 0..3
    - n_trend_signals: 0..3
    - competitor_persona: AGGRESSIVE / COOPERATIVE / RECIPROCAL / RANDOM
    - schema_drift_count: 0..4
    - difficulty_score: scalar 0..1 (informational)

Algorithm: Thompson-sampling-lite over a Beta posterior per
parameter-value cell. After each episode, the cell associated with the
sampled config gets a (1 - WRR) reward signal — i.e. low WRR raises
the probability that the same cell is sampled again. This focuses the
agent's training on the configurations it currently struggles with.

This file is dependency-free (no scikit-optimize, no GPyTorch). It's
~150 lines of Python and an episode-log JSONL.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from freshprice_env.agents.competitor_store_agent import CompetitorPersona
from freshprice_env.enums import (
    CurriculumScenario,
    ExternalEvent,
    WeatherCondition,
)


@dataclass
class ComposedScenario:
    """A concrete scenario configuration the env can consume."""

    base_scenario: CurriculumScenario
    weather: WeatherCondition
    event: ExternalEvent
    n_farmer_offers: int
    n_trend_signals: int
    competitor_persona: CompetitorPersona
    schema_drift_count: int
    seed: int
    difficulty_score: float = 0.5

    def to_dict(self) -> dict:
        d = asdict(self)
        d["base_scenario"] = self.base_scenario.name
        d["weather"] = self.weather.value
        d["event"] = self.event.value
        d["competitor_persona"] = self.competitor_persona.value
        return d


@dataclass
class _CellPosterior:
    """Beta(α,β) posterior of "failure rate" for one parameter-value cell."""

    alpha: float = 1.0
    beta: float = 1.0

    def sample(self, rng: random.Random) -> float:
        """Sample from Beta(α,β) using Python stdlib."""
        # rng.betavariate calls random.gammavariate internally
        return rng.betavariate(self.alpha, self.beta)

    def update(self, failure_signal: float) -> None:
        """Update with a failure signal in [0,1]."""
        s = max(0.0, min(1.0, float(failure_signal)))
        self.alpha += s
        self.beta += (1.0 - s)


# ---------------------------------------------------------------------------
# ScenarioComposer
# ---------------------------------------------------------------------------

# Parameter spaces
_WEATHERS = list(WeatherCondition)
_EVENTS = list(ExternalEvent)
_PERSONAS = list(CompetitorPersona)
_FARMER_RANGE = list(range(0, 4))
_TREND_RANGE = list(range(0, 4))
_DRIFT_RANGE = list(range(0, 5))


class ScenarioComposer:
    """Adaptive scenario sampler with per-cell Beta posteriors.

    Construct once per training run. Call ``next_scenario()`` to get
    the next config to train on. After the episode, call
    ``record_outcome(config, wrr)`` to update posteriors.
    """

    def __init__(
        self,
        base_scenarios: list[CurriculumScenario] | None = None,
        seed: int = 42,
        log_path: str | Path | None = None,
        wrr_target: float = 0.70,
    ) -> None:
        self._rng = random.Random(seed)
        self._base_scenarios = list(
            base_scenarios or [
                CurriculumScenario.STABLE_WEEK,
                CurriculumScenario.BUSY_WEEKEND,
                CurriculumScenario.FARMER_WEEK,
                CurriculumScenario.TREND_WEEK,
                CurriculumScenario.CRISIS_WEEK,
                CurriculumScenario.REGULATORY_WEEK,
            ]
        )
        self._target = float(wrr_target)
        self._log_path = Path(log_path) if log_path else None

        # One posterior dict per parameter axis
        self._post_weather: dict[str, _CellPosterior] = {
            w.value: _CellPosterior() for w in _WEATHERS
        }
        self._post_event: dict[str, _CellPosterior] = {
            e.value: _CellPosterior() for e in _EVENTS
        }
        self._post_persona: dict[str, _CellPosterior] = {
            p.value: _CellPosterior() for p in _PERSONAS
        }
        self._post_n_farmer: dict[int, _CellPosterior] = {
            n: _CellPosterior() for n in _FARMER_RANGE
        }
        self._post_n_trend: dict[int, _CellPosterior] = {
            n: _CellPosterior() for n in _TREND_RANGE
        }
        self._post_n_drift: dict[int, _CellPosterior] = {
            n: _CellPosterior() for n in _DRIFT_RANGE
        }
        self._post_base: dict[str, _CellPosterior] = {
            s.name: _CellPosterior() for s in self._base_scenarios
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def next_scenario(self) -> ComposedScenario:
        """Sample the next scenario via Thompson sampling per cell."""
        base_name = self._argmax_sample(
            self._post_base,
            list(self._post_base.keys()),
        )
        base = CurriculumScenario[base_name]

        weather_v = self._argmax_sample(
            self._post_weather, [w.value for w in _WEATHERS]
        )
        event_v = self._argmax_sample(
            self._post_event, [e.value for e in _EVENTS]
        )
        persona_v = self._argmax_sample(
            self._post_persona, [p.value for p in _PERSONAS]
        )
        n_farmer = self._argmax_sample(self._post_n_farmer, _FARMER_RANGE)
        n_trend = self._argmax_sample(self._post_n_trend, _TREND_RANGE)
        n_drift = self._argmax_sample(self._post_n_drift, _DRIFT_RANGE)

        # Difficulty: rough sum of "hard" choices
        difficulty = (
            (1.0 if weather_v in ("HOT", "RAINY", "COLD") else 0.0)
            + (1.0 if event_v in ("FESTIVAL", "SPORTS_EVENT") else 0.0)
            + 0.20 * n_farmer
            + 0.20 * n_trend
            + 0.30 * n_drift
            + (0.30 if persona_v == "AGGRESSIVE" else 0.0)
        )
        difficulty = round(min(1.0, difficulty / 4.0), 3)

        return ComposedScenario(
            base_scenario=base,
            weather=WeatherCondition(weather_v),
            event=ExternalEvent(event_v),
            n_farmer_offers=int(n_farmer),
            n_trend_signals=int(n_trend),
            competitor_persona=CompetitorPersona(persona_v),
            schema_drift_count=int(n_drift),
            seed=self._rng.randint(0, 999_999),
            difficulty_score=difficulty,
        )

    def _argmax_sample(self, posteriors: dict, values: list):
        """Sample one value per posterior, return the value with the
        highest sampled failure-probability."""
        sampled = [(v, posteriors[v].sample(self._rng)) for v in values]
        sampled.sort(key=lambda x: x[1], reverse=True)
        return sampled[0][0]

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def record_outcome(self, config: ComposedScenario, wrr: float) -> None:
        """Update posteriors using a normalized failure signal.

        A WRR below ``wrr_target`` becomes a "failure" signal in [0,1].
        WRR at target → 0.5. WRR at 0 → 1.0. WRR at 1 → 0.0.
        """
        failure = max(0.0, min(1.0, 1.0 - (wrr / max(self._target, 1e-3))))
        # Squash above target — already winning, don't focus there
        failure = min(failure, 0.5) if wrr >= self._target else failure

        self._post_base[config.base_scenario.name].update(failure)
        self._post_weather[config.weather.value].update(failure)
        self._post_event[config.event.value].update(failure)
        self._post_persona[config.competitor_persona.value].update(failure)
        self._post_n_farmer[config.n_farmer_offers].update(failure)
        self._post_n_trend[config.n_trend_signals].update(failure)
        self._post_n_drift[config.schema_drift_count].update(failure)

        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "config": config.to_dict(),
                    "wrr": float(wrr),
                    "failure_signal": failure,
                }) + "\n")

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def hardest_cells(self, top_k: int = 5) -> dict:
        """Return the top-K hardest values per axis (for the dashboard)."""
        def _hardness(d: dict) -> list:
            scored = [(k, p.alpha / (p.alpha + p.beta)) for k, p in d.items()]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [{"value": k, "failure_rate": round(s, 3)} for k, s in scored[:top_k]]
        return {
            "base_scenario": _hardness(self._post_base),
            "weather": _hardness(self._post_weather),
            "event": _hardness(self._post_event),
            "competitor_persona": _hardness(self._post_persona),
            "n_farmer_offers": _hardness(self._post_n_farmer),
            "n_trend_signals": _hardness(self._post_n_trend),
            "schema_drift_count": _hardness(self._post_n_drift),
        }
