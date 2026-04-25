"""Formal task graders for hackathon evaluation.

Three tasks at Easy/Medium/Hard difficulty. Each returns TaskGraderResult
with score 0.0-1.0. Judges run these to score the environment.
"""

from __future__ import annotations

from freshprice_env.entities import SimulatedBatch, SimulatedFarmerOffer, SimulatedTrendSignal
from freshprice_env.enums import (
    BatchStatus,
    BatchType,
    BriefEngineType,
    CurriculumScenario,
    ExpiryUrgency,
    FarmerOfferStatus,
    SignalSource,
    TrendAction,
)
from freshprice_env.freshprice_env import FreshPriceEnv
from freshprice_env.models import TaskGraderResult


class TaskGrader:
    """Runs and grades the three formal evaluation tasks."""

    # ------------------------------------------------------------------
    # Task 1 (Easy): Clear the Spinach
    # ------------------------------------------------------------------

    @staticmethod
    def grade_task_1(
        episode_record: list[dict], final_state, env: FreshPriceEnv,
    ) -> TaskGraderResult:
        """Clear the Spinach — clear all spinach batches before expiry."""
        spinach_batches = [
            b for b in final_state.batches if b.category == "leafy_greens"
        ]
        if not spinach_batches:
            return TaskGraderResult(
                task_id="task_1", task_name="Clear the Spinach",
                score=0.0, passed=False,
                details={"error": "No spinach batches found in final state"},
            )

        total = len(spinach_batches)
        cleared = sum(1 for b in spinach_batches if b.status == BatchStatus.CLEARED)
        expired = sum(1 for b in spinach_batches if b.status == BatchStatus.EXPIRED)
        clearance_rate = cleared / total if total > 0 else 0.0

        # Check average sell price vs original for cleared batches
        # Approximate from state: if batch cleared, price was adequate
        # Use the batch's current_price vs original_price as proxy
        price_ratios = []
        for b in spinach_batches:
            if b.status == BatchStatus.CLEARED and b.original_price > 0:
                price_ratios.append(b.current_price / b.original_price)

        avg_price_ratio = sum(price_ratios) / len(price_ratios) if price_ratios else 0.0

        # Score
        if clearance_rate >= 1.0 and avg_price_ratio >= 0.80:
            score = 1.0
        elif clearance_rate >= 1.0:
            score = 0.7  # Cleared but over-discounted
        elif clearance_rate >= 0.70:
            score = 0.4
        elif clearance_rate > 0.0:
            score = 0.1
        else:
            score = 0.0

        return TaskGraderResult(
            task_id="task_1",
            task_name="Clear the Spinach",
            score=score,
            passed=score >= 0.4,
            details={
                "total_spinach": total,
                "cleared": cleared,
                "expired": expired,
                "clearance_rate": round(clearance_rate, 3),
                "avg_price_ratio": round(avg_price_ratio, 3),
            },
        )

    # ------------------------------------------------------------------
    # Task 2 (Medium): Evaluate the Farmer
    # ------------------------------------------------------------------

    @staticmethod
    def grade_task_2(
        episode_record: list[dict], final_state, env: FreshPriceEnv,
    ) -> TaskGraderResult:
        """Evaluate the Farmer — correctly respond to a high-viability offer."""
        # Find farmer actions in episode record
        farmer_actions = []
        for brief in episode_record:
            if brief.get("engine_type") == "FARMER":
                farmer_actions.append(brief)

        if not farmer_actions:
            return TaskGraderResult(
                task_id="task_2", task_name="Evaluate the Farmer",
                score=0.0, passed=False,
                details={"error": "No farmer actions taken — offer expired"},
            )

        # Check what decision was made on the target offer
        decision_made = None
        counter_price = None
        for brief in farmer_actions:
            directive = brief.get("directive", {})
            if not isinstance(directive, dict):
                continue
            for action in directive.get("actions", []):
                decision_made = action.get("decision", "").upper()
                counter_price = action.get("counter_price")
                break
            if decision_made:
                break

        if decision_made is None:
            score = 0.0
        elif decision_made == "ACCEPT":
            score = 1.0
        elif decision_made == "COUNTER" and counter_price is not None:
            # Check if counter is within 15% of offered price (Rs 35/kg)
            offered = 35.0
            if abs(counter_price - offered) / offered <= 0.15:
                score = 1.0
            else:
                score = 0.6  # Accepted but at unprofitable price
        elif decision_made == "DECLINE":
            score = 0.3  # Missed opportunity
        else:
            score = 0.0

        return TaskGraderResult(
            task_id="task_2",
            task_name="Evaluate the Farmer",
            score=score,
            passed=score >= 0.6,
            details={
                "decision": decision_made,
                "counter_price": counter_price,
                "farmer_briefs_count": len(farmer_actions),
            },
        )

    # ------------------------------------------------------------------
    # Task 3 (Hard): Ride the Trend
    # ------------------------------------------------------------------

    @staticmethod
    def grade_task_3(
        episode_record: list[dict], final_state, env: FreshPriceEnv,
    ) -> TaskGraderResult:
        """Ride the Trend — approve trend + manage expiring batch simultaneously."""
        trend_approved = False
        pricing_discounted = False
        brief_qualities = []

        for brief in episode_record:
            quality = brief.get("quality_score", 0.0)
            brief_qualities.append(quality)

            engine = brief.get("engine_type", "")
            directive = brief.get("directive", {})
            if not isinstance(directive, dict):
                continue

            if engine == "TREND":
                for action in directive.get("actions", []):
                    if action.get("decision", "").upper() == "APPROVE":
                        trend_approved = True

            if engine == "PRICING":
                for action in directive.get("actions", []):
                    pm = action.get("price_multiplier", 1.0)
                    if pm < 0.80:
                        pricing_discounted = True

        avg_quality = sum(brief_qualities) / len(brief_qualities) if brief_qualities else 0.0

        # Check if trend was approved within first 16 ticks (2 brief cycles)
        early_trend = False
        for brief in episode_record[:4]:
            if brief.get("engine_type") == "TREND":
                directive = brief.get("directive", {})
                if isinstance(directive, dict):
                    for action in directive.get("actions", []):
                        if action.get("decision", "").upper() == "APPROVE":
                            early_trend = True

        # Score
        if early_trend and pricing_discounted:
            score = 1.0
        elif trend_approved and not pricing_discounted:
            score = 0.7
        elif not trend_approved and pricing_discounted:
            score = 0.5
        elif not trend_approved and not pricing_discounted and avg_quality > 0.5:
            score = 0.2
        else:
            score = 0.0

        return TaskGraderResult(
            task_id="task_3",
            task_name="Ride the Trend",
            score=score,
            passed=score >= 0.5,
            details={
                "trend_approved": trend_approved,
                "early_trend_approval": early_trend,
                "pricing_discounted": pricing_discounted,
                "avg_brief_quality": round(avg_quality, 3),
                "briefs_evaluated": len(episode_record),
            },
        )

    # ------------------------------------------------------------------
    # Run all tasks
    # ------------------------------------------------------------------

    @staticmethod
    def run_all_tasks(
        model_fn: callable,
        seed: int = 42,
    ) -> list[TaskGraderResult]:
        """Run all 3 tasks against a model function.

        model_fn: callable(prompt: str) -> str (returns an Operating Brief)
        """

        class ModelClient:
            def generate(self, prompt: str) -> str:
                return model_fn(prompt)

        client = ModelClient()
        results: list[TaskGraderResult] = []

        # Task 1: Clear the Spinach — Stable Week, short episode
        env1 = FreshPriceEnv(
            scenario=CurriculumScenario.STABLE_WEEK, seed=seed,
            llm_client=client, brief_interval_ticks=8,
        )
        obs, info = env1.reset(seed=seed)
        done = False
        while not done:
            brief = model_fn(obs)
            obs, reward, done, truncated, info = env1.step(brief)
        results.append(TaskGrader.grade_task_1(
            env1.get_episode_record(), env1._state, env1,
        ))

        # Task 2: Evaluate the Farmer — Farmer Week
        env2 = FreshPriceEnv(
            scenario=CurriculumScenario.FARMER_WEEK, seed=seed + 1,
            llm_client=client, brief_interval_ticks=8,
        )
        obs, info = env2.reset(seed=seed + 1)
        done = False
        step = 0
        while not done and step < 20:  # Only need first few briefs for farmer task
            brief = model_fn(obs)
            obs, reward, done, truncated, info = env2.step(brief)
            step += 1
        results.append(TaskGrader.grade_task_2(
            env2.get_episode_record(), env2._state, env2,
        ))

        # Task 3: Ride the Trend — Trend Week
        env3 = FreshPriceEnv(
            scenario=CurriculumScenario.TREND_WEEK, seed=seed + 2,
            llm_client=client, brief_interval_ticks=8,
        )
        obs, info = env3.reset(seed=seed + 2)
        done = False
        step = 0
        while not done and step < 20:  # First few briefs for trend + pricing
            brief = model_fn(obs)
            obs, reward, done, truncated, info = env3.step(brief)
            step += 1
        results.append(TaskGrader.grade_task_3(
            env3.get_episode_record(), env3._state, env3,
        ))

        return results
