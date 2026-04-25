"""MultiAgentFreshPriceEnv — two-sided market with LLM store-manager + ConsumerAgent.

Theme #1 alignment (Multi-Agent Interactions):
  - The LLM controls the store (pricing, farmer offers, trend restocking)
  - The ConsumerAgent reacts to the LLM's decisions (price elasticity, event response)
  - Each affects the other: lower price → consumer buys more → less waste → higher WRR
  - The agent must model consumer behaviour to make good decisions ("theory of mind")

Architecture:
  ┌─────────────────────────────┐
  │  MultiAgentFreshPriceEnv    │
  │  ┌───────────────────────┐  │
  │  │  FreshPriceEnv (core) │  │
  │  │  + ExternalShockEngine│  │
  │  └───────────────────────┘  │
  │  ┌───────────────────────┐  │
  │  │  ConsumerAgent        │  │
  │  │  (demand feedback)    │  │
  │  └───────────────────────┘  │
  └─────────────────────────────┘

Usage:
    env = MultiAgentFreshPriceEnv(scenario=CurriculumScenario.CRISIS_WEEK)
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(llm_brief_text)
    # info["consumer_observation"] gives what consumers see this tick
"""

from __future__ import annotations

import random

import gymnasium as gym

from freshprice_env.agents.consumer_agent import ConsumerAgent
from freshprice_env.constants import TICKS_PER_BRIEF
from freshprice_env.enums import CurriculumScenario
from freshprice_env.freshprice_env import FreshPriceEnv


class MultiAgentFreshPriceEnv(gym.Env):
    """Two-sided FreshPrice market: LLM store-manager vs reactive ConsumerAgent.

    The consumer agent modifies effective demand velocity each tick,
    making price signals matter more — good discounts genuinely increase
    sell-through, bad discounts leave waste on the shelf.

    The observation string includes a "CONSUMER SIGNALS" section so the LLM
    can reason about buyer behaviour in its Operating Brief.
    """

    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        scenario: CurriculumScenario = CurriculumScenario.STABLE_WEEK,
        seed: int = 42,
        render_mode: str = "none",
        llm_client=None,
        consumer_price_sensitivity: float = 1.5,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self._seed = seed
        self.render_mode = render_mode

        self._core_env = FreshPriceEnv(
            scenario=scenario,
            seed=seed,
            render_mode=render_mode,
            llm_client=llm_client,
        )
        self._consumer = ConsumerAgent(
            rng=random.Random(seed + 1000),
            price_sensitivity=consumer_price_sensitivity,
        )

        self.observation_space = self._core_env.observation_space
        self.action_space = self._core_env.action_space

        self._last_consumer_obs: dict = {}

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = self._core_env.reset(seed=seed, options=options)
        if self._core_env._state is not None:
            self._last_consumer_obs = self._consumer.observe(self._core_env._state)
            info["consumer_observation"] = self._last_consumer_obs

        # Augment observation with consumer signals
        obs = self._augment_obs(obs)
        return obs, info

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: str):
        # 1. Run consumer agent to set demand boosts before the core env ticks
        if self._core_env._state is not None:
            boosts = self._consumer.act(self._core_env._state)
            self._core_env._state.consumer_demand_boost = boosts

        # 2. Step the core environment
        obs, reward, terminated, truncated, info = self._core_env.step(action)

        # 3. Record consumer observation for this step
        if self._core_env._state is not None:
            self._last_consumer_obs = self._consumer.observe(self._core_env._state)
            info["consumer_observation"] = self._last_consumer_obs

        # 4. Augment observation with consumer signals for next brief
        if not terminated:
            obs = self._augment_obs(obs)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # OpenEnv state
    # ------------------------------------------------------------------

    def state(self) -> dict:
        s = self._core_env.state()
        s["consumer_observation"] = self._last_consumer_obs
        s["mode"] = "multi_agent"
        return s

    def render(self):
        result = self._core_env.render()
        if self.render_mode == "human" and self._last_consumer_obs:
            discounts = self._last_consumer_obs.get("visible_discounts", [])
            if discounts:
                print(f"  Consumer sees: {len(discounts)} discounted items")
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _augment_obs(self, obs: str) -> str:
        """Append a CONSUMER SIGNALS section to the LLM prompt."""
        if not self._last_consumer_obs:
            return obs

        weather = self._last_consumer_obs.get("weather", "NORMAL")
        event = self._last_consumer_obs.get("event", "NONE")
        discounts = self._last_consumer_obs.get("visible_discounts", [])
        high_demand = self._last_consumer_obs.get("high_demand_signals", [])

        lines = [
            "\n\n## CONSUMER SIGNALS",
            f"  Weather: {weather} | Active Event: {event}",
        ]
        if discounts:
            lines.append("  Visible discounts driving consumer attention:")
            for d in discounts[:3]:  # cap at 3 for prompt length
                lines.append(
                    f"    {d['category']} {d['batch_id']}: "
                    f"{d['discount_pct']:.0f}% off @ Rs{d['current_price']:.0f} "
                    f"({d['urgency']})"
                )
        if high_demand:
            lines.append(f"  High-demand categories: {', '.join(high_demand)}")

        return obs + "\n".join(lines)
