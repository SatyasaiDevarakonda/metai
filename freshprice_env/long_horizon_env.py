"""LongHorizonFreshPriceEnv — 28-day episodes with sparse weekly rewards.

Theme #2 alignment (Super Long-Horizon Planning & Instruction Following):
  - Standard env: 7 days, 672 ticks, 84 briefs, reward every brief
  - Long-horizon env: 28 days, 2688 ticks, 336 briefs, reward every 7 days ONLY
  - The agent must plan procurement 2+ weeks ahead based on trend signals
  - Sparse reward means early mistakes compound — no quick corrections
  - Tests whether agent can maintain consistent strategy over 336 turns
    (far exceeding typical LLM context windows → forces external state tracking)

Key challenges added:
  1. Sparse reward: WRR delta only given at end of each 7-day week (not per-brief)
  2. Multi-week inventory cycles: farmer offers have 14-day delivery windows
  3. Seasonal demand: demand velocity shifts across the 4 simulated weeks
  4. Context limit stress: 336 briefs requires the agent to reason about
     decisions made 200+ turns ago — tests durable internal representations

Usage:
    env = LongHorizonFreshPriceEnv(scenario=CurriculumScenario.CRISIS_WEEK, seed=42)
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(brief_text)
    # reward is 0.0 most steps; only non-zero at day 7, 14, 21, 28
"""

from __future__ import annotations

from freshprice_env.constants import (
    LONG_HORIZON_TICKS,
    SPARSE_REWARD_INTERVAL_DAYS,
    TICKS_PER_DAY,
    TICKS_PER_BRIEF,
)
from freshprice_env.enums import CurriculumScenario
from freshprice_env.freshprice_env import FreshPriceEnv


# Seasonal demand multipliers per week (week 0 = baseline)
_SEASONAL_MULTIPLIERS: list[float] = [1.0, 1.2, 0.8, 1.5]  # Weeks 0-3


class LongHorizonFreshPriceEnv(FreshPriceEnv):
    """28-day extension of FreshPriceEnv with sparse weekly rewards.

    Inherits all engines, pipeline, and reward logic from FreshPriceEnv.
    Overrides:
      - TOTAL_TICKS → LONG_HORIZON_TICKS (2688)
      - Reward function → 0.0 until end of each 7-day window
      - Seasonal multipliers applied to demand velocity each week
    """

    def __init__(
        self,
        scenario: CurriculumScenario = CurriculumScenario.CRISIS_WEEK,
        seed: int = 42,
        render_mode: str = "none",
        llm_client=None,
    ) -> None:
        super().__init__(
            scenario=scenario,
            seed=seed,
            render_mode=render_mode,
            llm_client=llm_client,
            brief_interval_ticks=TICKS_PER_BRIEF,
        )
        self._long_horizon_ticks = LONG_HORIZON_TICKS
        self._sparse_interval_ticks = SPARSE_REWARD_INTERVAL_DAYS * TICKS_PER_DAY
        self._week_wrr_checkpoints: list[float] = []
        self._last_sparse_wrr: float = 0.0

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._week_wrr_checkpoints = []
        self._last_sparse_wrr = 0.0
        info["mode"] = "long_horizon_28day"
        info["total_ticks"] = self._long_horizon_ticks
        info["reward_schedule"] = f"Sparse — every {SPARSE_REWARD_INTERVAL_DAYS} days"
        return obs, info

    # ------------------------------------------------------------------
    # step — override reward and termination
    # ------------------------------------------------------------------

    def step(self, action: str):
        obs, reward, terminated, truncated, info = super().step(action)

        # Override termination: run 4x longer
        terminated = self._current_tick >= self._long_horizon_ticks

        # Apply seasonal demand multiplier for current week
        week_idx = min(self._current_tick // TICKS_PER_DAY // 7, 3)
        season_mult = _SEASONAL_MULTIPLIERS[week_idx]
        if self._state is not None and season_mult != 1.0:
            for bid in list(self._state.sales_velocity.keys()):
                self._state.sales_velocity[bid] = round(
                    self._state.sales_velocity[bid] * season_mult, 3
                )

        # Sparse reward: only pay out at week boundaries
        is_week_boundary = (
            self._current_tick > 0
            and self._current_tick % self._sparse_interval_ticks == 0
        )
        if is_week_boundary:
            current_wrr = self._state.wrr if self._state else 0.0
            sparse_reward = current_wrr - self._last_sparse_wrr
            self._last_sparse_wrr = current_wrr
            self._week_wrr_checkpoints.append(current_wrr)
            info["week_boundary"] = True
            info["week_wrr"] = current_wrr
            info["week_number"] = len(self._week_wrr_checkpoints)
            reward = sparse_reward
        else:
            reward = 0.0  # Sparse: no reward between week boundaries

        if terminated:
            info["week_wrr_history"] = self._week_wrr_checkpoints
            info["mode"] = "long_horizon_28day"

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> dict:
        s = super().state()
        s["mode"] = "long_horizon_28day"
        s["total_ticks"] = self._long_horizon_ticks
        s["week_number"] = len(self._week_wrr_checkpoints) + 1
        s["week_wrr_history"] = list(self._week_wrr_checkpoints)
        s["next_reward_in_ticks"] = (
            self._sparse_interval_ticks
            - (self._current_tick % self._sparse_interval_ticks)
        )
        return s
