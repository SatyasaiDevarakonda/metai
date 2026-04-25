"""ExternalShockEngine — weather conditions and local events that shift demand.

Wires into FreshPriceEnv.step() to modify sales_velocity each tick.
This adds Theme #3 (World Modeling) realism: the agent must reason about
external context that it cannot control, only anticipate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from freshprice_env.constants import (
    FESTIVAL_DEMAND_MULTIPLIERS,
    SPORTS_EVENT_DAIRY_MULTIPLIER,
    SPORTS_EVENT_PACKAGED_MULTIPLIER,
    WEATHER_COLD_BAKERY_DAIRY_MULTIPLIER,
    WEATHER_HOT_DAIRY_MULTIPLIER,
    WEATHER_HOT_FRUITS_MULTIPLIER,
    WEATHER_RAIN_DEMAND_MULTIPLIER,
)
from freshprice_env.enums import CurriculumScenario, ExternalEvent, WeatherCondition


@dataclass
class ExternalShock:
    """A weather/event state active from `start_tick` until the next shock."""
    weather: WeatherCondition
    event: ExternalEvent
    start_tick: int

    def describe(self) -> str:
        parts = [f"Weather: {self.weather.value}"]
        if self.event != ExternalEvent.NONE:
            parts.append(f"Event: {self.event.value}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Scenario shock schedules
# ---------------------------------------------------------------------------

_STABLE_WEEK_SHOCKS: list[tuple[int, WeatherCondition, ExternalEvent]] = [
    (0,   WeatherCondition.SUNNY,  ExternalEvent.NONE),
    (288, WeatherCondition.NORMAL, ExternalEvent.NONE),
]

_BUSY_WEEKEND_SHOCKS: list[tuple[int, WeatherCondition, ExternalEvent]] = [
    (0,   WeatherCondition.SUNNY,  ExternalEvent.NONE),
    (192, WeatherCondition.SUNNY,  ExternalEvent.FESTIVAL),   # Day 2 festival
    (384, WeatherCondition.NORMAL, ExternalEvent.NONE),
]

_FARMER_WEEK_SHOCKS: list[tuple[int, WeatherCondition, ExternalEvent]] = [
    (0,   WeatherCondition.NORMAL, ExternalEvent.NONE),
    (192, WeatherCondition.RAINY,  ExternalEvent.NONE),       # Rain mid-week suppresses demand
    (480, WeatherCondition.SUNNY,  ExternalEvent.NONE),
]

_TREND_WEEK_SHOCKS: list[tuple[int, WeatherCondition, ExternalEvent]] = [
    (0,   WeatherCondition.HOT,    ExternalEvent.NONE),       # Heat spikes fruit demand
    (288, WeatherCondition.HOT,    ExternalEvent.SPORTS_EVENT),  # Sports event + heat
    (480, WeatherCondition.NORMAL, ExternalEvent.NONE),
]

_CRISIS_WEEK_SHOCKS: list[tuple[int, WeatherCondition, ExternalEvent]] = [
    (0,   WeatherCondition.RAINY,  ExternalEvent.NONE),       # Rain hits footfall
    (144, WeatherCondition.HOT,    ExternalEvent.FESTIVAL),   # Sudden festival + heat
    (336, WeatherCondition.COLD,   ExternalEvent.NONE),       # Cold snap
    (528, WeatherCondition.NORMAL, ExternalEvent.SPORTS_EVENT),
]

_SCENARIO_SCHEDULES: dict[CurriculumScenario, list[tuple]] = {
    CurriculumScenario.STABLE_WEEK:  _STABLE_WEEK_SHOCKS,
    CurriculumScenario.BUSY_WEEKEND: _BUSY_WEEKEND_SHOCKS,
    CurriculumScenario.FARMER_WEEK:  _FARMER_WEEK_SHOCKS,
    CurriculumScenario.TREND_WEEK:   _TREND_WEEK_SHOCKS,
    CurriculumScenario.CRISIS_WEEK:  _CRISIS_WEEK_SHOCKS,
}


class ExternalShockEngine:
    """Manages the sequence of external shocks for one episode.

    Called each tick by FreshPriceEnv to:
      1. Advance to the correct shock for the current tick
      2. Return per-category demand multipliers the pricing engine applies
    """

    def __init__(self, scenario: CurriculumScenario, rng: random.Random) -> None:
        self._rng = rng
        raw = _SCENARIO_SCHEDULES.get(scenario, _STABLE_WEEK_SHOCKS)
        # Add ±5% noise to demand multipliers for realism
        self._shocks: list[ExternalShock] = [
            ExternalShock(weather=w, event=e, start_tick=t)
            for t, w, e in raw
        ]
        self._current_shock: ExternalShock = self._shocks[0]

    def tick(self, current_tick: int) -> ExternalShock:
        """Advance shock state and return the active shock."""
        for shock in reversed(self._shocks):
            if current_tick >= shock.start_tick:
                self._current_shock = shock
                break
        return self._current_shock

    @property
    def current_shock(self) -> ExternalShock:
        return self._current_shock

    def demand_multiplier(self, category: str) -> float:
        """Return the combined demand multiplier for a category under the current shock.

        Base = 1.0. Weather and event effects are multiplicative.
        A ±5% noise is added so the agent cannot perfectly predict demand.
        """
        shock = self._current_shock
        mult = 1.0

        # --- Weather effects ---
        if shock.weather == WeatherCondition.RAINY:
            mult *= WEATHER_RAIN_DEMAND_MULTIPLIER
        elif shock.weather == WeatherCondition.HOT:
            if category == "fruits":
                mult *= WEATHER_HOT_FRUITS_MULTIPLIER
            elif category == "dairy":
                mult *= WEATHER_HOT_DAIRY_MULTIPLIER
        elif shock.weather == WeatherCondition.COLD:
            if category in ("bakery", "dairy"):
                mult *= WEATHER_COLD_BAKERY_DAIRY_MULTIPLIER

        # --- Event effects ---
        if shock.event == ExternalEvent.FESTIVAL:
            mult *= FESTIVAL_DEMAND_MULTIPLIERS.get(category, 1.3)
        elif shock.event == ExternalEvent.SPORTS_EVENT:
            if category == "packaged":
                mult *= SPORTS_EVENT_PACKAGED_MULTIPLIER
            elif category == "dairy":
                mult *= SPORTS_EVENT_DAIRY_MULTIPLIER
        elif shock.event == ExternalEvent.LOCAL_HOLIDAY:
            mult *= 1.3  # general footfall boost

        # ±5% noise
        mult *= self._rng.uniform(0.95, 1.05)
        return round(mult, 3)

    def describe_current(self) -> str:
        return self._current_shock.describe()
