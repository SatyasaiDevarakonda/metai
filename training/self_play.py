"""Self-play loop for the negotiation env (Theme #4 self-improvement).

The trained policy plays both sides of a NegotiationEnv simultaneously
against a *frozen older checkpoint*. Every K episodes the frozen
opponent is rotated to a more recent past checkpoint, so the agent
faces a steadily-improving adversary — recursive skill amplification
without any external opponent dataset.

The key trick: each frozen checkpoint is wrapped in an LLM-callable
``policy`` that the env calls turn-by-turn. We don't actually load
LLM weights here (heavy deps); instead we expose a callable interface
and let the user plug in a ``HeroPolicy`` (live model) and a
``FrozenPolicy`` (snapshot loader). Both implement the same protocol:

    policy(prompt: str, role: str) -> str   # returns brief text

For tests / dry-runs, ``ScriptedNegotiationPolicy`` is provided.

Outputs:
  - JSONL of (episode_id, hero_role, opponent_id, hero_reward, opp_reward,
    deal/no_deal, rounds_used)
  - On-disk frozen-checkpoint pool (managed elsewhere; this module just
    accepts a list of policy objects).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from freshprice_env.negotiation_env import NegotiationEnv, NegotiationOutcome


class NegotiationPolicy(Protocol):
    """Anything that can take (prompt, role) and return a brief string."""
    policy_id: str

    def __call__(self, prompt: str, role: str) -> str:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Scripted reference policy (always available, no LLM needed)
# ---------------------------------------------------------------------------

@dataclass
class ScriptedNegotiationPolicy:
    """Deterministic scripted policy used as a stand-in opponent.

    Useful for: smoke tests, ablation baselines, and the very first
    self-play episodes before any frozen checkpoint exists.
    """

    policy_id: str = "scripted_v0"
    accept_threshold_pct: float = 0.05    # accept counters within 5% of opening

    def __call__(self, prompt: str, role: str) -> str:
        # Find the current price on table and the agent's reserve/max
        import re
        m_price = re.search(r"Current offer on table: Rs([0-9.]+)", prompt)
        m_reserve = re.search(r"YOUR RESERVE PRICE: Rs([0-9.]+)", prompt)
        m_max = re.search(r"YOUR MAX PRICE: Rs([0-9.]+)", prompt)
        try:
            cur = float(m_price.group(1)) if m_price else 0.0
        except ValueError:
            cur = 0.0
        if role == "STORE":
            try:
                limit = float(m_max.group(1)) if m_max else cur
            except ValueError:
                limit = cur
            if cur <= limit * (1.0 + self.accept_threshold_pct):
                return '{"decision": "ACCEPT", "reason": "within tolerance"}'
            counter = round(limit * 0.95, 2)
            return f'{{"decision": "COUNTER", "price": {counter}, "reason": "below my max"}}'
        else:  # FARMER
            try:
                reserve = float(m_reserve.group(1)) if m_reserve else cur
            except ValueError:
                reserve = cur
            if cur >= reserve * (1.0 + self.accept_threshold_pct):
                return '{"decision": "ACCEPT", "reason": "above my reserve"}'
            counter = round(reserve * 1.10, 2)
            return f'{{"decision": "COUNTER", "price": {counter}, "reason": "need premium"}}'


# ---------------------------------------------------------------------------
# Self-play runner
# ---------------------------------------------------------------------------

@dataclass
class SelfPlayResult:
    episode_id: str
    seed: int
    hero_role: str
    opponent_id: str
    rounds_used: int
    outcome: str                  # DEAL | NO_DEAL
    hero_total_reward: float
    opponent_total_reward: float


class SelfPlayRunner:
    """Pairs a hero policy against rotating frozen opponents."""

    def __init__(
        self,
        hero: NegotiationPolicy,
        opponent_pool: list[NegotiationPolicy],
        log_path: str | Path | None = None,
        seed: int = 42,
        rotate_every_n: int = 8,
    ) -> None:
        if not opponent_pool:
            raise ValueError("opponent_pool must contain at least one policy")
        self._hero = hero
        self._pool = list(opponent_pool)
        self._log_path = Path(log_path) if log_path else None
        self._rng = random.Random(seed)
        self._rotate_every_n = max(1, int(rotate_every_n))
        self._opponent_idx = 0
        self._episodes_with_current = 0

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, n_episodes: int) -> list[SelfPlayResult]:
        results: list[SelfPlayResult] = []
        for i in range(n_episodes):
            res = self._run_one_episode(episode_idx=i)
            results.append(res)
            if self._log_path is not None:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "episode_id": res.episode_id,
                        "seed": res.seed,
                        "hero_role": res.hero_role,
                        "opponent_id": res.opponent_id,
                        "rounds": res.rounds_used,
                        "outcome": res.outcome,
                        "hero_reward": res.hero_total_reward,
                        "opp_reward": res.opponent_total_reward,
                    }) + "\n")
            self._maybe_rotate()
        return results

    def _maybe_rotate(self) -> None:
        self._episodes_with_current += 1
        if self._episodes_with_current >= self._rotate_every_n:
            self._opponent_idx = (self._opponent_idx + 1) % len(self._pool)
            self._episodes_with_current = 0

    def _run_one_episode(self, episode_idx: int) -> SelfPlayResult:
        opponent = self._pool[self._opponent_idx]
        hero_role_first = "STORE" if episode_idx % 2 == 0 else "FARMER"
        seed = self._rng.randint(0, 999999)
        env = NegotiationEnv(seed=seed)
        obs, info = env.reset()
        active_role = info.get("active_role", "FARMER")

        hero_total = 0.0
        opp_total = 0.0
        rounds_used = 0
        outcome = "NO_DEAL"

        # Track which side the hero is *playing* during each turn
        # (in this env, role changes each round).
        for _ in range(8):    # safety bound — env max is MAX_ROUNDS=4
            policy = (
                self._hero if active_role == hero_role_first
                else opponent
            )
            action = policy(obs, active_role)
            obs, reward, done, _, info = env.step(action)
            if active_role == hero_role_first:
                hero_total += float(reward)
            else:
                opp_total += float(reward)
            rounds_used += 1
            if done:
                outcome = info.get("outcome", "NO_DEAL")
                break
            active_role = info.get("active_role", active_role)

        return SelfPlayResult(
            episode_id=f"sp_{episode_idx:05d}",
            seed=seed,
            hero_role=hero_role_first,
            opponent_id=getattr(opponent, "policy_id", "anon"),
            rounds_used=rounds_used,
            outcome=outcome,
            hero_total_reward=round(hero_total, 4),
            opponent_total_reward=round(opp_total, 4),
        )


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def smoke_test(n_episodes: int = 4) -> list[SelfPlayResult]:
    """Run a tiny self-play round with two scripted policies.

    Used by validate_submission.py and as a usage example.
    """
    hero = ScriptedNegotiationPolicy(policy_id="scripted_hero")
    pool = [
        ScriptedNegotiationPolicy(
            policy_id=f"scripted_opp_{p:.2f}",
            accept_threshold_pct=p,
        )
        for p in (0.02, 0.05, 0.08, 0.12)
    ]
    runner = SelfPlayRunner(hero=hero, opponent_pool=pool, seed=11)
    return runner.run(n_episodes)
