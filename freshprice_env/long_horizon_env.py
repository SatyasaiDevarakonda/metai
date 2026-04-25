"""LongHorizonFreshPriceEnv — 30-day episodes that *force* externalized memory.

Theme #2 alignment (Super Long-Horizon Planning & Instruction Following):

The original 7-day env (672 ticks, 84 briefs) easily fits into a 32K
context. So merely making episodes longer doesn't test long-horizon
behaviour — Qwen-2.5-7B can still keep everything in context.

This env makes the long horizon *bite*:

  - 30 days = 2880 ticks = 360 briefs (4× the 7-day baseline)
  - The prompt is forcibly truncated to LONG_HORIZON_PROMPT_TAIL_CHARS
    so only the last ~4 briefs of context survive
  - Reward is sparse (paid weekly, not per-brief) so early mistakes don't
    surface for 96+ briefs
  - The agent gets a durable AgentNotebook (NOTE / RECALL / COMMIT /
    UPDATE_PLAN) — its only way to remember decisions across the gap
  - Commitments made on day N are auto-graded when their due_tick
    arrives. Plan-adherence (r4) is a first-class reward component.
  - Seasonal demand multipliers per week (slow shift)

If the agent doesn't write to the notebook it *cannot* solve this — by
design. That makes "uses the notebook properly" a falsifiable signal of
long-horizon planning skill, distinct from raw WRR.
"""

from __future__ import annotations

from freshprice_env.constants import (
    LONG_HORIZON_BRIEFS_PER_EPISODE,
    LONG_HORIZON_PROMPT_TAIL_CHARS,
    LONG_HORIZON_RECENT_BRIEFS_WINDOW,
    LONG_HORIZON_TICKS,
    R4_BROKEN_COMMITMENT_PENALTY,
    R4_HONORED_COMMITMENT_BONUS,
    SPARSE_REWARD_INTERVAL_DAYS,
    TICKS_PER_BRIEF,
    TICKS_PER_DAY,
)
from freshprice_env.enums import CurriculumScenario
from freshprice_env.freshprice_env import FreshPriceEnv
from freshprice_env.notebook import (
    AgentNotebook,
    NotebookDirectiveExecutor,
    extract_notebook_directives,
)
from freshprice_env.notebook.notebook_directives import evaluate_commitment


# Seasonal demand multipliers across the 4-5 simulated weeks of the month.
# 30 days = 4 full weeks + 2 transition days; bucket by week index.
_SEASONAL_MULTIPLIERS: list[float] = [1.0, 1.2, 0.8, 1.5, 1.1]


class LongHorizonFreshPriceEnv(FreshPriceEnv):
    """30-day extension of FreshPriceEnv with sparse weekly rewards + notebook.

    Inherits all engines, pipeline, and reward logic from FreshPriceEnv.
    Overrides:

      - TOTAL_TICKS → LONG_HORIZON_TICKS (2880)
      - Per-brief reward → 0.0 except at week boundaries (sparse)
      - Plus r4 plan-adherence each brief that resolves commitments
      - Prompt → notebook block prepended, then truncated tail
      - Seasonal multipliers applied to demand velocity each week
    """

    def __init__(
        self,
        scenario: CurriculumScenario = CurriculumScenario.CRISIS_WEEK,
        seed: int = 42,
        render_mode: str = "none",
        llm_client=None,
        prompt_tail_chars: int = LONG_HORIZON_PROMPT_TAIL_CHARS,
        enable_notebook: bool = True,
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
        self._prompt_tail_chars = max(800, int(prompt_tail_chars))
        self._enable_notebook = bool(enable_notebook)

        self._notebook: AgentNotebook = AgentNotebook()
        self._week_wrr_checkpoints: list[float] = []
        self._last_sparse_wrr: float = 0.0
        # r4 reward accumulators
        self._r4_episode_total: float = 0.0
        self._r4_last_brief: float = 0.0
        # Recent brief tail for truncation (newest first)
        self._recent_brief_summaries: list[str] = []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._notebook.reset()
        self._week_wrr_checkpoints = []
        self._last_sparse_wrr = 0.0
        self._r4_episode_total = 0.0
        self._r4_last_brief = 0.0
        self._recent_brief_summaries = []
        info.update({
            "mode": "long_horizon_30day",
            "total_ticks": self._long_horizon_ticks,
            "total_briefs": LONG_HORIZON_BRIEFS_PER_EPISODE,
            "reward_schedule": f"Sparse — every {SPARSE_REWARD_INTERVAL_DAYS} days",
            "notebook_enabled": self._enable_notebook,
            "prompt_tail_chars": self._prompt_tail_chars,
        })
        if self._enable_notebook:
            obs = self._wrap_with_notebook(obs)
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: str):
        # 1. Apply notebook directives BEFORE the core env steps so that
        #    NOTE/COMMIT/UPDATE_PLAN are visible to the agent on the next
        #    prompt build.
        notebook_results: list = []
        if self._enable_notebook:
            directives = extract_notebook_directives(action)
            notebook_results = NotebookDirectiveExecutor.apply(
                directives, self._notebook, self._current_tick,
            )

        # 2. Step the core env
        obs, _per_brief_reward, terminated, truncated, info = super().step(action)

        # 3. Override termination: 4× longer episode
        terminated = self._current_tick >= self._long_horizon_ticks

        # 4. Apply seasonal demand multiplier for current week
        week_idx = min(self._current_tick // TICKS_PER_DAY // 7, len(_SEASONAL_MULTIPLIERS) - 1)
        season_mult = _SEASONAL_MULTIPLIERS[week_idx]
        if self._state is not None and season_mult != 1.0:
            for bid in list(self._state.sales_velocity.keys()):
                self._state.sales_velocity[bid] = round(
                    self._state.sales_velocity[bid] * season_mult, 3
                )

        # 5. Auto-resolve any commitments whose due_tick has passed
        r4_brief = 0.0
        resolved_this_brief: list[dict] = []
        if self._enable_notebook and self._state is not None:
            resolved = self._notebook.auto_resolve_due(
                self._current_tick,
                lambda c: evaluate_commitment(c, self._state),
            )
            for c in resolved:
                resolved_this_brief.append({
                    "commitment_id": c.commitment_id,
                    "kind": c.kind,
                    "target": c.target,
                    "honored": c.honored,
                    "reason": c.resolution_reason,
                })
                if c.honored:
                    r4_brief += R4_HONORED_COMMITMENT_BONUS
                else:
                    r4_brief -= R4_BROKEN_COMMITMENT_PENALTY

        self._r4_last_brief = r4_brief
        self._r4_episode_total += r4_brief

        # 6. Reward shaping: sparse weekly WRR delta + per-brief r4
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
            reward = sparse_reward + r4_brief
        else:
            reward = r4_brief    # only plan-adherence bites between week boundaries

        # 7. Update brief tail (used by prompt truncation)
        self._recent_brief_summaries.append(_summarize_brief(action))
        if len(self._recent_brief_summaries) > LONG_HORIZON_RECENT_BRIEFS_WINDOW:
            self._recent_brief_summaries.pop(0)

        # 8. Surface notebook info
        info["notebook"] = {
            "directives_applied": [
                {"verb": r.directive.verb, "ok": r.ok, "detail": r.detail}
                for r in notebook_results
            ],
            "resolved_this_brief": resolved_this_brief,
            "open_commitments": self._notebook.open_count,
            "honored": self._notebook.honored_count,
            "broken": self._notebook.broken_count,
            "adherence_score": round(self._notebook.adherence_score(), 4),
        }
        info["r4_plan_adherence"] = round(r4_brief, 4)
        info["r4_episode_total"] = round(self._r4_episode_total, 4)

        # 9. Truncate + wrap obs with notebook for next brief
        if not terminated:
            obs = self._truncate_obs(obs)
            if self._enable_notebook:
                obs = self._wrap_with_notebook(obs)

        if terminated:
            info["week_wrr_history"] = self._week_wrr_checkpoints
            info["mode"] = "long_horizon_30day"
            info["notebook_replay"] = self._notebook.to_replay()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def state(self) -> dict:
        s = super().state()
        s.update({
            "mode": "long_horizon_30day",
            "total_ticks": self._long_horizon_ticks,
            "week_number": len(self._week_wrr_checkpoints) + 1,
            "week_wrr_history": list(self._week_wrr_checkpoints),
            "next_reward_in_ticks": (
                self._sparse_interval_ticks
                - (self._current_tick % self._sparse_interval_ticks)
            ),
            "notebook_enabled": self._enable_notebook,
            "open_commitments": self._notebook.open_count,
            "adherence_score": round(self._notebook.adherence_score(), 4),
        })
        return s

    # ------------------------------------------------------------------
    # Notebook accessor (for dashboard / replay)
    # ------------------------------------------------------------------

    @property
    def notebook(self) -> AgentNotebook:
        return self._notebook

    # ------------------------------------------------------------------
    # Prompt shaping
    # ------------------------------------------------------------------

    def _wrap_with_notebook(self, obs: str) -> str:
        """Prepend the notebook block to the observation."""
        block = self._notebook.render_prompt_block(self._current_tick)
        guide = (
            "## NOTEBOOK USAGE\n"
            "  Your context is short and the episode is long. Use these verbs\n"
            "  inside a `## NOTEBOOK` section of your brief to remember things:\n"
            "    NOTE: <key> -> <value>\n"
            "    NOTE_PIN: <key> -> <value>     (always re-injected)\n"
            "    COMMIT: <kind>:<target>@<tick> | <comment>\n"
            "      kinds: inventory_below, inventory_above, wrr_above,\n"
            "             accept_offer, decline_offer, restock_category, custom\n"
            "    UPDATE_PLAN: <free text plan>\n"
            "    RESOLVE: <commitment_id> ok|fail [reason]   (manual mark)\n"
        )
        return f"{block}\n\n{guide}\n{obs}"

    def _truncate_obs(self, obs: str) -> str:
        """Trim the *body* of obs to LONG_HORIZON_PROMPT_TAIL_CHARS.

        Keeps the first 600 chars (the SITUATION header) so the agent still
        sees its task framing, plus the tail of the inventory/signal block.
        """
        if len(obs) <= self._prompt_tail_chars:
            return obs
        head_keep = 600
        tail_keep = max(0, self._prompt_tail_chars - head_keep - 80)
        head = obs[:head_keep]
        tail = obs[-tail_keep:] if tail_keep > 0 else ""
        return (
            head
            + "\n…[context truncated — older inventory rolled off; "
              "rely on your NOTEBOOK above]…\n"
            + tail
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize_brief(brief_text: str) -> str:
    """One-line gist of a brief for the rolling tail."""
    if not brief_text:
        return ""
    head = brief_text.strip().splitlines()
    if not head:
        return ""
    return head[0][:100]
