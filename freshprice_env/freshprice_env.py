"""FreshPriceEnv — the main OpenEnv RL environment for perishable goods intelligence.

Everything generated so far plugs into this file:
  Engines: PricingEngine, FarmerEngine, TrendEngine
  Reward:  WRRRewardEngine
  Pipeline: PromptBuilder, Parser, Validator, QualityScorer, RuleExecutor
  State:   MarketStateBuilder, SimulatedMarketState
"""

from __future__ import annotations

import logging
import random
from dataclasses import replace

import gymnasium as gym

from freshprice_env.brief_pipeline.parser import BriefParser
from freshprice_env.brief_pipeline.prompt_builder import OperatingBriefPromptBuilder
from freshprice_env.brief_pipeline.quality_scorer import BriefQualityScorer
from freshprice_env.brief_pipeline.rule_executor import RuleExecutor
from freshprice_env.brief_pipeline.validator import BriefValidator
from freshprice_env.constants import (
    MAX_ACTIVE_FARMER_OFFERS,
    TICKS_PER_BRIEF,
    TICKS_PER_DAY,
    TOTAL_TICKS,
    TREND_SCORE_THRESHOLD,
)
from freshprice_env.engines.farmer_engine import FarmerEngine
from freshprice_env.engines.pricing_engine import PricingEngine
from freshprice_env.engines.trend_engine import TrendEngine
from freshprice_env.entities import (
    SimulatedFarmerOffer,
    SimulatedMarketState,
)
from freshprice_env.enums import (
    BriefEngineType,
    CurriculumScenario,
    FarmerOfferStatus,
    SignalSource,
    TrendAction,
)
from freshprice_env.external_shocks import ExternalShockEngine
from freshprice_env.market_state import MarketStateBuilder
from freshprice_env.reward import WRRRewardEngine

logger = logging.getLogger(__name__)


class FreshPriceEnv(gym.Env):
    """OpenEnv RL environment for FreshPrice AI training.

    One step = one Operating Brief cycle (brief_interval_ticks simulation ticks).
    The LLM generates text (Operating Brief), the environment simulates the world.
    """

    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        scenario: CurriculumScenario = CurriculumScenario.STABLE_WEEK,
        seed: int = 42,
        render_mode: str = "none",
        llm_client=None,
        brief_interval_ticks: int = TICKS_PER_BRIEF,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self._seed = seed
        self.render_mode = render_mode
        self.llm_client = llm_client
        self.brief_interval_ticks = brief_interval_ticks

        self.rng = random.Random(seed)

        # Gym spaces — LLM receives/produces text, not numeric arrays.
        # Charset includes all printable ASCII + common Unicode used in prompts
        # (emoji urgency markers, rupee sign, arrows, etc.)
        _charset = frozenset(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " \t\n\r"
            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
            "\U0001f7e2\U0001f7e1\U0001f7e0\U0001f534\U0001f4f1"  # emoji: 🟢🟡🟠🔴📱
            "\u26a0\ufe0f"  # ⚠️
            "\u2014\u2192\u2500\u2550"  # —→─═
        )
        self.observation_space = gym.spaces.Text(min_length=0, max_length=16384, charset=_charset)
        self.action_space = gym.spaces.Text(min_length=0, max_length=4096, charset=_charset)

        # Engine instances — created fresh each reset()
        self._pricing_engine: PricingEngine | None = None
        self._farmer_engine: FarmerEngine | None = None
        self._trend_engine: TrendEngine | None = None
        self._reward_engine: WRRRewardEngine | None = None
        self._shock_engine: ExternalShockEngine | None = None
        self._state: SimulatedMarketState | None = None

        # Episode tracking
        self._current_tick: int = 0
        self._last_brief_tick: int = -1
        self._last_directive: dict | None = None
        self._last_engine_type: BriefEngineType | None = None
        self._previous_wrr: float = 0.0
        self._episode_briefs: list[dict] = []

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[str, dict]:
        """Reset the environment for a new episode.

        Returns (initial_observation_string, info_dict).
        """
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed
            self.rng = random.Random(seed)

        # Build initial state
        self._state = MarketStateBuilder.build(self.scenario, self.rng)

        # Create fresh engine instances
        self._pricing_engine = PricingEngine(self.rng)
        self._farmer_engine = FarmerEngine(self.rng)
        self._trend_engine = TrendEngine(self.rng)
        self._reward_engine = WRRRewardEngine()
        self._shock_engine = ExternalShockEngine(self.scenario, self.rng)

        # Reset tick counters
        self._current_tick = 0
        self._last_brief_tick = -1
        self._last_directive = None
        self._last_engine_type = None
        self._previous_wrr = 0.0
        self._episode_briefs = []

        # Score any pre-loaded farmer offers
        scored_offers: list[SimulatedFarmerOffer] = []
        for offer in self._state.pending_offers:
            if offer.viability_score is None:
                scored_offers.append(
                    self._farmer_engine.score_offer(offer, self._state)
                )
            else:
                scored_offers.append(offer)
        self._state.pending_offers = scored_offers

        # Determine first engine type
        engine_type = self._initial_engine_type()
        self._last_engine_type = engine_type

        # Build initial prompt
        prompt = OperatingBriefPromptBuilder.build(self._state, engine_type)

        info = {
            "tick": 0,
            "scenario": self.scenario.name,
            "engine_type": engine_type.value,
        }

        return prompt, info

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: str,
    ) -> tuple[str, float, bool, bool, dict]:
        """Process one Operating Brief cycle.

        Args:
            action: Raw LLM text output (the Operating Brief).

        Returns:
            (next_observation, reward, terminated, truncated, info)
        """
        # Keep state.tick synchronised so engines see correct hour_of_day / day_of_week
        # and validators see correct expiry windows.
        self._state.tick = self._current_tick

        info: dict = {
            "tick": self._current_tick,
            "engine_type": self._last_engine_type.value if self._last_engine_type else "PRICING",
            "parse_success": True,
            "validation_success": True,
            "used_fallback": False,
        }

        engine_type = self._last_engine_type or BriefEngineType.PRICING
        brief_dict: dict | None = None
        quality_score = 0.0

        # ----------------------------------------------------------
        # 1. PARSE
        # ----------------------------------------------------------
        parse_result = BriefParser.parse(action, engine_type)

        if not parse_result.success:
            info["parse_success"] = False
            info["parse_failure_reason"] = parse_result.failure_reason
            info["used_fallback"] = True
            brief_dict = self._make_fallback_brief(engine_type)
        else:
            brief_dict = parse_result.brief

        # ----------------------------------------------------------
        # 2. VALIDATE
        # ----------------------------------------------------------
        if info["parse_success"] and brief_dict is not None:
            val_result = BriefValidator.validate(brief_dict, self._state)
            if not val_result.valid:
                info["validation_success"] = False
                info["validation_errors"] = val_result.errors
                info["used_fallback"] = True
                brief_dict = self._make_fallback_brief(engine_type)
            else:
                if val_result.warnings:
                    info["validation_warnings"] = val_result.warnings

        # ----------------------------------------------------------
        # 3. SCORE brief quality
        # ----------------------------------------------------------
        if not info["used_fallback"] and brief_dict is not None:
            quality_score = BriefQualityScorer.score(brief_dict, self._state)
            self._reward_engine.record_brief_quality(quality_score)
            self._last_directive = brief_dict["directive"]
            self._last_engine_type = brief_dict["engine_type"]

        info["quality_score"] = quality_score

        # ----------------------------------------------------------
        # 4. EXECUTE directive via RuleExecutor
        # ----------------------------------------------------------
        execution = RuleExecutor.execute(brief_dict, self._state)

        # Wire anti-hack violations to reward engine
        for pa in execution.pricing_actions:
            if pa.was_antihack_blocked:
                self._reward_engine.record_antihack_violation(
                    self._current_tick, "PRICING", "EARLY_DISCOUNT",
                    f"batch={pa.batch_id} price={pa.new_price:.2f}",
                )
            if pa.was_below_floor:
                self._reward_engine.record_antihack_violation(
                    self._current_tick, "PRICING", "BELOW_FLOOR",
                    f"batch={pa.batch_id} clamped_to={pa.new_price:.2f}",
                )

        for fa in execution.farmer_actions:
            if fa.was_antihack_blocked:
                self._reward_engine.record_antihack_violation(
                    self._current_tick, "FARMER", "RECKLESS_ACCEPT",
                    f"offer={fa.offer_id}",
                )

        for ta in execution.trend_actions:
            if ta.was_capped:
                self._reward_engine.record_antihack_violation(
                    self._current_tick, "TREND", "ORDER_CAP",
                    f"{ta.category}:{ta.order_quantity_kg:.0f}kg",
                )

        if execution.execution_warnings:
            info["execution_warnings"] = execution.execution_warnings

        # ----------------------------------------------------------
        # 5. RUN brief_interval_ticks simulation ticks
        # ----------------------------------------------------------
        # Build the pricing directive dict for PricingEngine consumption
        pricing_directive = self._build_pricing_directive(execution) if engine_type == BriefEngineType.PRICING else None
        farmer_directive = self._build_farmer_directive(execution) if engine_type == BriefEngineType.FARMER else None
        trend_directive = self._build_trend_directive(execution) if engine_type == BriefEngineType.TREND else None

        for i in range(self.brief_interval_ticks):
            tick = self._current_tick
            is_first_tick = (i == 0)

            # Advance simulated time so engines see the right hour/day this tick
            self._state.tick = tick

            r2_action = 0.0
            r3_action = 0.0

            # a0. Update external shock state and write to market state
            shock = self._shock_engine.tick(tick)
            self._state.weather_condition = shock.weather
            self._state.active_event = shock.event

            # a. Apply trend demand boost (every tick)
            self._state = self._trend_engine.apply_trend_demand_boost(self._state)

            # b. Run pricing tick — directive applied on first tick only
            directive_for_tick = pricing_directive if is_first_tick else None
            self._state, r1 = self._pricing_engine.tick(self._state, directive_for_tick)

            # c. Farmer outcome resolution (every tick)
            r2_delta = self._farmer_engine.resolve_outcomes(self._state)

            # d. Trend outcome resolution (every tick)
            r3_delta = self._trend_engine.resolve_trend_outcomes(self._state)

            # e. Process FARMER directive (first tick only)
            if is_first_tick and farmer_directive is not None:
                self._state, r2_action = self._farmer_engine.process_directive(
                    self._state, farmer_directive,
                )

            # f. Process TREND directive (first tick only)
            if is_first_tick and trend_directive is not None:
                self._state, r3_action = self._trend_engine.process_directive(
                    self._state, trend_directive, tick,
                )

            # g. Record tick rewards
            self._reward_engine.record_tick(
                r1, r2_delta + r2_action, r3_delta + r3_action, tick,
            )

            # h. Inject scheduled farmer offers
            self._maybe_inject_farmer_offer(tick)

            # i. Inject scheduled trend signals
            self._maybe_inject_trend_signal(tick)

            # j. Increment tick
            self._current_tick += 1

        self._last_brief_tick = self._current_tick

        # ----------------------------------------------------------
        # 6. COMPUTE reward for this brief cycle
        # ----------------------------------------------------------
        current_wrr = self._state.wrr
        reward = current_wrr - self._previous_wrr
        self._previous_wrr = current_wrr

        # ----------------------------------------------------------
        # 7. CHECK termination
        # ----------------------------------------------------------
        terminated = self._current_tick >= TOTAL_TICKS
        truncated = False

        # ----------------------------------------------------------
        # 8. Final episode reward (if terminated)
        # ----------------------------------------------------------
        if terminated:
            final_reward = self._reward_engine.compute_episode_reward(self._state)
            info["final_reward"] = final_reward
            info["constitutional_audit"] = self._reward_engine.constitutional_audit()

        # ----------------------------------------------------------
        # Record this brief in episode log
        # ----------------------------------------------------------
        self._episode_briefs.append({
            "tick": self._current_tick - self.brief_interval_ticks,
            "engine_type": engine_type.value,
            "prompt": "",  # Omitted for memory — can be reconstructed from state
            "raw_response": action,
            "parse_success": info["parse_success"],
            "quality_score": quality_score,
            "reward_delta": reward,
        })

        # ----------------------------------------------------------
        # 9. DETERMINE next engine and build prompt
        # ----------------------------------------------------------
        if not terminated:
            next_engine = self._determine_next_engine()
            self._last_engine_type = next_engine
            next_prompt = OperatingBriefPromptBuilder.build(self._state, next_engine)
            info["next_engine_type"] = next_engine.value
        else:
            next_prompt = ""

        # ----------------------------------------------------------
        # 10. Return
        # ----------------------------------------------------------
        return next_prompt, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Engine selection
    # ------------------------------------------------------------------

    def _initial_engine_type(self) -> BriefEngineType:
        """Determine the engine type for tick 0."""
        if self.scenario == CurriculumScenario.FARMER_WEEK:
            return BriefEngineType.FARMER
        if self.scenario == CurriculumScenario.TREND_WEEK:
            return BriefEngineType.TREND
        # STABLE_WEEK, BUSY_WEEKEND, CRISIS_WEEK: start with PRICING
        return BriefEngineType.PRICING

    def _determine_next_engine(self) -> BriefEngineType:
        """Decide which engine the next Operating Brief targets.

        Priority:
        1. FARMER: pending offer expires within 16 ticks (4 hours)
        2. TREND:  actionable signal with composite_score >= 80.0
        3. FARMER: any pending offer exists
        4. TREND:  any actionable signal above threshold
        5. PRICING: default
        """
        # Scenario constraints
        farmer_eligible = self.scenario not in (
            CurriculumScenario.STABLE_WEEK,
            CurriculumScenario.BUSY_WEEKEND,
        )
        trend_eligible = self.scenario not in (
            CurriculumScenario.STABLE_WEEK,
            CurriculumScenario.FARMER_WEEK,
        )

        # Priority 1: imminent farmer offer expiry
        if farmer_eligible:
            for offer in self._state.pending_offers:
                if (
                    offer.status == FarmerOfferStatus.PENDING
                    and offer.is_expired(self._current_tick + 16)
                ):
                    return BriefEngineType.FARMER

        # Priority 2: high-confidence trend signal
        if trend_eligible:
            for sig in self._state.trend_signals.values():
                if (
                    sig.action_taken == TrendAction.PENDING
                    and sig.is_actionable(self._current_tick)
                    and sig.composite_score >= 80.0
                ):
                    return BriefEngineType.TREND

        # Priority 3: any pending farmer offer
        if farmer_eligible:
            for offer in self._state.pending_offers:
                if offer.status == FarmerOfferStatus.PENDING:
                    return BriefEngineType.FARMER

        # Priority 4: any actionable trend signal
        if trend_eligible:
            for sig in self._state.trend_signals.values():
                if (
                    sig.action_taken == TrendAction.PENDING
                    and sig.is_actionable(self._current_tick)
                ):
                    return BriefEngineType.TREND

        # Priority 5: default
        return BriefEngineType.PRICING

    # ------------------------------------------------------------------
    # Scheduled injections
    # ------------------------------------------------------------------

    def _maybe_inject_farmer_offer(self, tick: int) -> None:
        """Inject farmer offers at scheduled ticks based on scenario."""
        if len(self._state.pending_offers) >= MAX_ACTIVE_FARMER_OFFERS:
            return

        offer = self._scheduled_farmer_offer(tick)
        if offer is None:
            return

        # Score the offer immediately
        scored = self._farmer_engine.score_offer(offer, self._state)
        self._state.pending_offers.append(scored)
        logger.info("Injected farmer offer %s at tick %d", scored.offer_id, tick)

    def _scheduled_farmer_offer(self, tick: int) -> SimulatedFarmerOffer | None:
        """Return a farmer offer if one is scheduled for this tick, else None."""
        schedules: dict[CurriculumScenario, dict[int, tuple]] = {
            CurriculumScenario.FARMER_WEEK: {
                192: ("offer_003", "Suresh Patil", "vegetables", "spinach",
                      12.0, 14.0, 30),
                384: ("offer_004", "Anita Kumari", "vegetables", "tomatoes",
                      20.0, 18.0, 36),
            },
            CurriculumScenario.CRISIS_WEEK: {
                96:  ("offer_003", "Vijay Rao", "fruits", "mangoes",
                      35.0, 32.0, 48),
                288: ("offer_004", "Geeta Devi", "dairy", "curd",
                      25.0, 28.0, 24),
                480: ("offer_005", "Raju Singh", "vegetables", "capsicum",
                      18.0, 15.0, 20),
            },
            CurriculumScenario.TREND_WEEK: {},
            CurriculumScenario.STABLE_WEEK: {},
            CurriculumScenario.BUSY_WEEKEND: {},
        }

        scenario_schedule = schedules.get(self.scenario, {})
        params = scenario_schedule.get(tick)
        if params is None:
            return None

        oid, name, cat, product, qty, price, shelf = params

        # ±10% price variance, ±5hrs shelf life variance
        price_var = price * self.rng.uniform(0.90, 1.10)
        shelf_var = max(6, shelf + self.rng.randint(-5, 5))

        return SimulatedFarmerOffer(
            offer_id=oid,
            farmer_name=name,
            product_category=cat,
            product_name=product,
            quantity_kg=round(qty, 1),
            offered_price_per_kg=round(price_var, 2),
            seller_shelf_life_hrs=shelf_var,
            offered_at_tick=tick,
            status=FarmerOfferStatus.PENDING,
        )

    def _maybe_inject_trend_signal(self, tick: int) -> None:
        """Inject trend signals at scheduled ticks based on scenario."""
        if self.scenario == CurriculumScenario.BUSY_WEEKEND and tick == 192:
            # Boost existing fruits signal from 72 to 81
            existing = self._state.trend_signals.get("fruits")
            if existing is not None and existing.action_taken == TrendAction.PENDING:
                self._state.trend_signals["fruits"] = replace(
                    existing,
                    composite_score=81.0,
                    ingredient_rarity=existing.ingredient_rarity + 0.03 * self.rng.uniform(0.8, 1.2),
                    view_velocity=existing.view_velocity + 0.02 * self.rng.uniform(0.8, 1.2),
                )

        elif self.scenario == CurriculumScenario.TREND_WEEK and tick == 288:
            base_scores = {
                "recipe_simplicity": 0.72, "ingredient_rarity": 0.68,
                "view_velocity": 0.70, "local_relevance": 0.60,
                "historical_conversion": 0.58,
            }
            factor_scores = {
                k: max(0.0, min(1.0, v + self.rng.uniform(-0.05, 0.05)))
                for k, v in base_scores.items()
            }
            self._trend_engine.inject_trend_signal(
                self._state, "herbs", 69.0, SignalSource.ZOMATO,
                suggested_order_kg=5.0, current_tick=tick,
                factor_scores=factor_scores,
            )

        elif self.scenario == CurriculumScenario.CRISIS_WEEK and tick == 144:
            base_scores = {
                "recipe_simplicity": 0.88, "ingredient_rarity": 0.85,
                "view_velocity": 0.86, "local_relevance": 0.78,
                "historical_conversion": 0.72,
            }
            factor_scores = {
                k: max(0.0, min(1.0, v + self.rng.uniform(-0.05, 0.05)))
                for k, v in base_scores.items()
            }
            self._trend_engine.inject_trend_signal(
                self._state, "vegetables", 84.0, SignalSource.INSTAGRAM,
                suggested_order_kg=15.0, current_tick=tick,
                factor_scores=factor_scores,
            )

    # ------------------------------------------------------------------
    # Directive conversion helpers
    # ------------------------------------------------------------------

    def _build_pricing_directive(self, execution) -> dict | None:
        """Convert ExecutionResult pricing actions back to the dict PricingEngine expects."""
        if not execution.pricing_actions:
            return None
        return {
            "engine": "PRICING",
            "actions": [
                {
                    "batch_id": a.batch_id,
                    "price_multiplier": a.new_price / batch.original_price
                        if (batch := self._find_batch(a.batch_id)) and batch.original_price > 0
                        else 1.0,
                    "flash_sale": a.flash_sale,
                    "bundle_with": a.bundle_with,
                }
                for a in execution.pricing_actions
                if not a.was_antihack_blocked
            ],
        }

    def _build_farmer_directive(self, execution) -> dict | None:
        """Convert ExecutionResult farmer actions back to the dict FarmerEngine expects."""
        if not execution.farmer_actions:
            return None
        return {
            "engine": "FARMER",
            "actions": [
                {
                    "offer_id": a.offer_id,
                    "decision": a.decision,
                    "counter_price": a.counter_price,
                }
                for a in execution.farmer_actions
            ],
        }

    def _build_trend_directive(self, execution) -> dict | None:
        """Convert ExecutionResult trend actions back to the dict TrendEngine expects."""
        if not execution.trend_actions:
            return None
        return {
            "engine": "TREND",
            "actions": [
                {
                    "category": a.category,
                    "decision": a.decision,
                    "order_quantity_kg": a.order_quantity_kg,
                }
                for a in execution.trend_actions
            ],
        }

    def _find_batch(self, batch_id: str):
        """Find a batch by ID in current state."""
        for b in self._state.batches:
            if b.batch_id == batch_id:
                return b
        return None

    def _make_fallback_brief(self, engine_type: BriefEngineType) -> dict:
        """Create a no-op fallback brief when parse/validation fails."""
        if self._last_directive is not None:
            return {
                "engine_type": engine_type,
                "situation": "Fallback — previous brief reused",
                "signal_analysis": None,
                "viability_check": None,
                "recommendation": "Reusing last valid directive",
                "directive": self._last_directive,
                "confidence": "LOW",
            }
        # No previous directive — return an empty-actions directive
        return {
            "engine_type": engine_type,
            "situation": "Fallback — no valid directive available",
            "signal_analysis": None,
            "viability_check": None,
            "recommendation": "No action",
            "directive": {"engine": engine_type.value, "actions": []},
            "confidence": "LOW",
        }

    # ------------------------------------------------------------------
    # state (required by OpenEnv spec)
    # ------------------------------------------------------------------

    def state(self) -> dict:
        """Return current environment state as a plain dict.

        Required by the OpenEnv spec. Called by openenv validate and
        by inference.py for structured logging.
        """
        if self._state is None:
            return {"status": "not_started"}

        active = [b for b in self._state.batches
                  if b.status.value == "ACTIVE"]
        critical = [b for b in active
                    if b.urgency.value == "CRITICAL"]

        return {
            "tick": self._current_tick,
            "day_of_week": self._state.day_of_week,
            "hour_of_day": self._state.hour_of_day,
            "scenario": self.scenario.name,
            "wrr_so_far": self._state.wrr,
            "active_batches": len(active),
            "critical_batches": len(critical),
            "pending_offers": len(self._state.pending_offers),
            "active_trends": len(self._state.trend_signals),
            "risk_buffer_balance": self._state.risk_buffer_balance,
            "engine_type": self._last_engine_type.value if self._last_engine_type else "PRICING",
            "episode_complete": self._current_tick >= TOTAL_TICKS,
            "weather": self._state.weather_condition.value,
            "event": self._state.active_event.value,
        }

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------

    def render(self) -> str | None:
        """Print ASCII summary of current state if render_mode is 'human'."""
        if self.render_mode != "human" or self._state is None:
            return None

        from freshprice_env.enums import BatchStatus, ExpiryUrgency

        active = [b for b in self._state.batches if b.status == BatchStatus.ACTIVE]
        critical = [b for b in active if b.urgency == ExpiryUrgency.CRITICAL]
        pending = [o for o in self._state.pending_offers if o.status == FarmerOfferStatus.PENDING]
        active_trends = [
            (cat, s) for cat, s in self._state.trend_signals.items()
            if s.action_taken in (TrendAction.PENDING, TrendAction.APPROVED)
        ]

        summary = (
            f"--- Tick {self._current_tick}/{TOTAL_TICKS} | "
            f"Day {self._state.day_of_week} {self._state.hour_of_day}:00 | "
            f"WRR: {self._state.wrr:.3f} ---\n"
            f"  Active batches: {len(active)} | CRITICAL: {len(critical)} | "
            f"Pending offers: {len(pending)} | Active trends: {len(active_trends)}\n"
            f"  Risk buffer: Rs {self._state.risk_buffer_balance:.0f}"
        )
        print(summary)
        return summary

    # ------------------------------------------------------------------
    # Episode record
    # ------------------------------------------------------------------

    def get_episode_record(self) -> list[dict]:
        """Return the full record of every brief this episode.

        Used by TrajectoryBuffer for DPO pair generation.
        """
        return list(self._episode_briefs)
