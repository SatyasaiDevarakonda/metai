"""MarketCommonsEnv — partially-observable N-actor marketplace.

This is the headline Theme #1 environment. A hero StoreAgent (the LLM
under training) operates alongside:

  - 1+ CompetitorStoreAgent(s) running their own simulated state
  - A pool of FarmerAgents that emit BIDs into a shared message bus
  - (Wired by the env) a RegulatorAgent emitting policy drift
  - The MarketBus carrying all messages

Information asymmetry:
  - Each store sees its own inventory + cash, plus public messages on
    the bus
  - Farmer reserve prices are private (only inferable from history)
  - Competitor cash and inventory are private

A farmer offer is a *contested resource*: both stores see the BID on
the bus, but only the first ACCEPT message wins the supply. Dragging a
negotiation lets the competitor snap it.

The hero's brief can carry up to 3 sections this env consumes:

  ## NOTEBOOK   — same notebook directives as long_horizon_env
  ## MESSAGES   — verbs the hero broadcasts to other agents
  DIRECTIVE     — the same JSON DIRECTIVE the core env uses

Cooperation Index (returned each step) measures pareto-improving
exchanges: bilateral COMMITs with positive surplus for both sides,
forwarded offers, transfer-bypass arrangements.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import gymnasium as gym

from freshprice_env.agents.competitor_store_agent import (
    CompetitorAction,
    CompetitorPersona,
    CompetitorStoreAgent,
)
from freshprice_env.agents.consumer_cohort_agent import ConsumerCohortAgent
from freshprice_env.agents.farmer_agent import (
    FarmerAgent,
    build_default_farmer_pool,
)
from freshprice_env.agents.regulator_agent import RegulatorAgent
from freshprice_env.engines.liquidation_engine import (
    LiquidationEngine,
    parse_liquidate_directive,
)
from freshprice_env.engines.rider_pool_engine import RiderPoolEngine
from freshprice_env.brief_pipeline.schema_registry import (
    SchemaRegistry,
    default_registry,
)
from freshprice_env.constants import (
    COOPERATION_INDEX_MESSAGE_WEIGHT,
    COOPERATION_INDEX_PARETO_WEIGHT,
    COOPERATION_INDEX_TRANSFER_WEIGHT,
    MAX_ACTIVE_FARMER_OFFERS,
    TICKS_PER_BRIEF,
)
from freshprice_env.enums import BriefEngineType, CurriculumScenario, FarmerOfferStatus
from freshprice_env.freshprice_env import FreshPriceEnv
from freshprice_env.persistence.reputation_store import (
    ReputationStore,
    default_store,
)
from freshprice_env.protocol.market_bus import (
    MarketBus,
    MessageVerb,
    parse_messages_from_brief,
)


@dataclass
class CooperationStats:
    """Per-step accounting for the cooperation index."""

    pareto_commits: int = 0
    forwarded_offers: int = 0
    transfers_executed: int = 0
    bilateral_messages: int = 0

    def index(self) -> float:
        """0..1+ scalar. Capped at ~1.5 for headline display."""
        score = (
            self.transfers_executed * COOPERATION_INDEX_TRANSFER_WEIGHT
            + self.bilateral_messages * COOPERATION_INDEX_MESSAGE_WEIGHT
            + self.pareto_commits * COOPERATION_INDEX_PARETO_WEIGHT
            + self.forwarded_offers * 0.30
        )
        return round(min(1.5, score), 4)


@dataclass
class MarketCommonsStepInfo:
    """Structured per-step diagnostic blob returned by step()."""

    hero_info: dict
    competitor_actions: list[dict] = field(default_factory=list)
    farmer_messages: list[dict] = field(default_factory=list)
    bus_messages_this_step: list[dict] = field(default_factory=list)
    cooperation_stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

class MarketCommonsEnv(gym.Env):
    """Multi-agent market with one hero + competitor stores + farmer pool.

    Drives the headline Theme #1 demo. The hero's gym observation/action
    spaces are inherited from FreshPriceEnv. The env additionally
    surfaces a "MARKET COMMONS" block in the prompt (recent bus messages,
    competitor visible moves, farmer pool trust summary) so the LLM can
    do theory-of-mind reasoning on what other agents will do next.
    """

    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        scenario: CurriculumScenario = CurriculumScenario.CRISIS_WEEK,
        seed: int = 42,
        render_mode: str = "none",
        llm_client=None,
        n_competitors: int = 1,
        competitor_personas: list[CompetitorPersona] | None = None,
        reputation_store: ReputationStore | None = None,
        bus: MarketBus | None = None,
        episode_id: str | None = None,
        enable_regulator: bool = True,
        schema_registry: SchemaRegistry | None = None,
        enable_blinkit: bool = False,
        rider_count: int | None = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self._seed = seed
        self.render_mode = render_mode
        self._episode_id = episode_id or f"mce_{scenario.name}_{seed}"

        self._rng = random.Random(seed)
        self._bus = bus or MarketBus()
        self._reputation = reputation_store or default_store()

        self._registry = schema_registry or default_registry()
        self._enable_regulator = bool(enable_regulator)
        self._regulator = RegulatorAgent(
            scenario=scenario, bus=self._bus, registry=self._registry,
        ) if self._enable_regulator else None
        self._regulatory_events: list[dict] = []

        # --- Hero env -------------------------------------------------
        self._hero = FreshPriceEnv(
            scenario=scenario,
            seed=seed,
            render_mode=render_mode,
            llm_client=llm_client,
        )
        self._hero_id = "store_001"

        # --- Competitor agents ---------------------------------------
        if competitor_personas is None:
            competitor_personas = [CompetitorPersona.RECIPROCAL]
        if len(competitor_personas) < n_competitors:
            # repeat last persona to fill
            while len(competitor_personas) < n_competitors:
                competitor_personas.append(competitor_personas[-1])
        self._competitors: list[CompetitorStoreAgent] = [
            CompetitorStoreAgent(
                store_id=f"store_{i + 2:03d}",
                persona=competitor_personas[i],
                rng=random.Random(seed + 1000 + i),
                bus=self._bus,
                llm_client=None,    # scripted competitor by default
            )
            for i in range(n_competitors)
        ]

        # --- Farmer pool ---------------------------------------------
        self._farmer_pool: dict[str, FarmerAgent] = build_default_farmer_pool(
            reputation_store=self._reputation,
            rng=random.Random(seed + 5000),
            bus=self._bus,
            llm_client=None,
        )

        # --- Spaces ---------------------------------------------------
        self.observation_space = self._hero.observation_space
        self.action_space = self._hero.action_space

        # Per-step accounting
        self._coop_stats = CooperationStats()
        self._coop_episode_total: float = 0.0
        self._last_hero_action_kind: str | None = None
        self._farmer_offers_seen: set[str] = set()
        self._last_seq: int = 0
        self._last_step_messages: list[dict] = []

        # ------------------------------------------------------------------
        # Blinkit layer (Theme #1 quick-commerce realism). Opt-in: when
        # enabled, after the hero env steps we (a) pass the deltas in
        # batch.quantity_remaining to a RiderPoolEngine to compute
        # r6_delivery_quality, (b) parse LIQUIDATE actions out of the
        # PRICING directive and run them through a LiquidationEngine to
        # compute r7_liquidation, and (c) ask a ConsumerCohortAgent for
        # per-cohort retention given the rider's average ETA. Both reward
        # components flow into the env's reward + info dict.
        # ------------------------------------------------------------------
        self._enable_blinkit = bool(enable_blinkit)
        self._rider_pool: RiderPoolEngine | None = None
        self._liquidation: LiquidationEngine | None = None
        self._cohort_agent: ConsumerCohortAgent | None = None
        if self._enable_blinkit:
            from freshprice_env.constants import BLINKIT_DEFAULT_RIDER_COUNT
            self._rider_pool = RiderPoolEngine(
                rider_count=rider_count or BLINKIT_DEFAULT_RIDER_COUNT,
            )
            self._liquidation = LiquidationEngine()
            self._cohort_agent = ConsumerCohortAgent(
                rng=random.Random(seed + 7000),
            )
        self._prev_quantities: dict[str, int] = {}
        self._r6_episode_total: float = 0.0
        self._r7_episode_total: float = 0.0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = seed
            self._rng = random.Random(seed)

        obs, info = self._hero.reset(seed=seed, options=options)
        self._bus.clear()
        self._coop_stats = CooperationStats()
        self._coop_episode_total = 0.0
        self._farmer_offers_seen = set()
        self._last_seq = 0
        self._last_hero_action_kind = None
        self._last_step_messages = []
        self._regulatory_events = []

        # Reset Blinkit layer
        self._prev_quantities = self._snapshot_batch_quantities()
        self._r6_episode_total = 0.0
        self._r7_episode_total = 0.0
        if self._enable_blinkit and self._rider_pool is not None:
            self._rider_pool = RiderPoolEngine(rider_count=self._rider_pool.rider_count)
        if self._enable_blinkit and self._liquidation is not None:
            self._liquidation = LiquidationEngine()
        if self._enable_regulator and self._regulator is not None:
            # Reset registry to v1 and re-build the regulator with the
            # current scenario's schedule
            self._registry.reset_to_v1()
            self._regulator = RegulatorAgent(
                scenario=self.scenario, bus=self._bus, registry=self._registry,
            )

        # Announce episode start on bus
        self._bus.post(
            tick=0, sender_id="env", verb=MessageVerb.BROADCAST,
            body=f"Episode {self._episode_id} | scenario {self.scenario.name}",
            payload={"scenario": self.scenario.name, "n_competitors": len(self._competitors)},
        )

        info["episode_id"] = self._episode_id
        info["mode"] = "market_commons"
        info["n_competitors"] = len(self._competitors)
        info["n_farmers"] = len(self._farmer_pool)

        obs = self._wrap_with_market_block(obs)
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: str):
        """One brief cycle.

        Order of operations:
          0. Regulator fires any pending policy changes for this tick
          1. Hero brief is parsed for ## MESSAGES → posted to bus
          2. Hero env steps (PRICING/FARMER/TREND DIRECTIVE applied)
          3. Competitors observe + act (their actions also post to bus)
          4. Farmer pool emits scheduled BIDs (creates new offers in
             hero's pending_offers if winnable)
          5. Cooperation Index updates, info assembled
        """
        # --- 0. Regulator tick --------------------------------------
        regulatory_fired: list[dict] = []
        if self._enable_regulator and self._regulator is not None:
            for ev in self._regulator.tick(self._hero._current_tick):
                regulatory_fired.append({
                    "tick": ev.tick,
                    "engine": ev.engine,
                    "version": ev.version,
                    "headline": ev.headline,
                    "body": ev.body,
                })
            self._regulatory_events.extend(regulatory_fired)

        # --- 1. Hero messages → bus ---------------------------------
        hero_messages = parse_messages_from_brief(
            action, sender_id=self._hero_id, tick=self._hero._current_tick,
        )
        for verb, receiver, body, payload in hero_messages:
            self._bus.post(
                tick=self._hero._current_tick,
                sender_id=self._hero_id,
                verb=verb,
                body=body,
                receiver_id=receiver,
                payload=payload,
            )
            # Bilateral message?
            if receiver and not receiver.startswith("env"):
                self._coop_stats.bilateral_messages += 1

        # Track hero action kind for competitor reciprocal logic
        self._last_hero_action_kind = self._infer_hero_action_kind(action)

        # --- 2. Hero env steps --------------------------------------
        obs, reward, terminated, truncated, info = self._hero.step(action)

        # --- 3. Competitor turn -------------------------------------
        competitor_action_dicts: list[dict] = []
        if self._hero._state is not None:
            farmer_offers = list(self._hero._state.pending_offers)
            active_trend_cats = list(self._hero._state.trend_signals.keys())
            for comp in self._competitors:
                comp.observe(
                    own_state=None,
                    hero_state=self._hero._state,
                    last_hero_action_kind=self._last_hero_action_kind,
                )
                comp_actions = comp.act(
                    own_state=None,
                    hero_state=self._hero._state,
                    farmer_offers_in_pool=farmer_offers,
                    active_trend_categories=active_trend_cats,
                    current_tick=self._hero._current_tick,
                )
                for a in comp_actions:
                    competitor_action_dicts.append({
                        "store_id": comp.store_id,
                        "kind": a.kind,
                        "target": a.target_id,
                        "payload": a.payload,
                        "body": a.body,
                    })
                # Apply contention: if competitor accepted a farmer offer
                # that is still PENDING in hero's state, mark it gone
                self._apply_competitor_contention(comp_actions)

        # --- 4. Farmer pool emissions -------------------------------
        farmer_msgs = self._maybe_emit_farmer_offers(self._hero._current_tick)

        # --- 5. Cooperation index -----------------------------------
        coop_idx = self._coop_stats.index()
        self._coop_episode_total += coop_idx

        new_messages = self._bus.messages_since(self._last_seq)
        self._last_seq = new_messages[-1].seq if new_messages else self._last_seq
        new_msg_dicts = [m.to_dict() for m in new_messages]
        self._last_step_messages = new_msg_dicts

        # --- 5b. Blinkit layer ----------------------------------------
        # When enabled, compute r6_delivery_quality (rider pool) and
        # r7_liquidation (B2B firesale) on top of the hero reward, and
        # populate the info dict with rider/cohort/liquidation snapshots
        # for the dashboard's Simulation Theater.
        blinkit_info = self._tick_blinkit_layer(action) if self._enable_blinkit else None
        if blinkit_info is not None:
            r6 = blinkit_info["r6_delivery_quality"]
            r7 = blinkit_info["r7_liquidation"]
            self._r6_episode_total += r6
            self._r7_episode_total += r7
            reward = float(reward) + r6 + r7

        info.update({
            "episode_id": self._episode_id,
            "mode": "market_commons",
            "competitor_actions": competitor_action_dicts,
            "farmer_messages": farmer_msgs,
            "bus_messages_this_step": new_msg_dicts,
            "bus_total_messages": len(self._bus.all_messages()),
            "cooperation_index": coop_idx,
            "cooperation_episode_total": round(self._coop_episode_total, 4),
            "pool_trust": self._reputation.pool_trust_summary(),
            "regulatory_events_this_step": regulatory_fired,
            "active_schemas": {
                "PRICING": self._registry.active_version(BriefEngineType.PRICING),
                "FARMER": self._registry.active_version(BriefEngineType.FARMER),
                "TREND": self._registry.active_version(BriefEngineType.TREND),
            } if self._enable_regulator else None,
        })
        if blinkit_info is not None:
            info.update(blinkit_info)

        # Augment obs with the MARKET COMMONS prompt block for next brief
        if not terminated:
            obs = self._wrap_with_market_block(obs)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Bus accessor
    # ------------------------------------------------------------------

    @property
    def bus(self) -> MarketBus:
        return self._bus

    @property
    def hero(self) -> FreshPriceEnv:
        return self._hero

    @property
    def episode_id(self) -> str:
        return self._episode_id

    # ------------------------------------------------------------------
    # State (for OpenEnv adapter)
    # ------------------------------------------------------------------

    def state(self) -> dict:
        s = self._hero.state()
        s.update({
            "mode": "market_commons",
            "episode_id": self._episode_id,
            "n_competitors": len(self._competitors),
            "n_farmers": len(self._farmer_pool),
            "cooperation_index_total": round(self._coop_episode_total, 4),
            "pool_trust": self._reputation.pool_trust_summary(),
            "bus_total_messages": len(self._bus.all_messages()),
            "recent_bus_messages": [
                m.to_dict() for m in self._bus.all_messages()[-20:]
            ],
            "active_schemas": ({
                "PRICING": self._registry.active_version(BriefEngineType.PRICING),
                "FARMER": self._registry.active_version(BriefEngineType.FARMER),
                "TREND": self._registry.active_version(BriefEngineType.TREND),
            } if self._enable_regulator else None),
            "regulatory_events": list(self._regulatory_events),
        })
        return s

    # ------------------------------------------------------------------
    # Blinkit layer helpers
    # ------------------------------------------------------------------

    def _snapshot_batch_quantities(self) -> dict[str, int]:
        """Read current ``quantity_remaining`` for every active batch.

        Used to compute sales deltas across a brief: if a batch had 50
        units before the hero step and 42 after, 8 units sold this brief
        and the rider pool needs to ferry one or more orders for them.
        """
        if self._hero._state is None:
            return {}
        return {
            b.batch_id: int(getattr(b, "quantity_remaining", 0))
            for b in self._hero._state.batches
        }

    def _tick_blinkit_layer(self, action: str) -> dict:
        """Run the Blinkit-style mechanics on top of the hero step.

        Returns a dict slated to be merged into the env's info dict with
        keys: r6_delivery_quality, r7_liquidation, rider_pool, cohorts,
        liquidation, plus the per-episode running totals.
        """
        assert self._rider_pool is not None
        assert self._liquidation is not None
        assert self._cohort_agent is not None

        # 1. Sales deltas per batch over this brief.
        new_quantities = self._snapshot_batch_quantities()
        sales_this_brief: dict[str, int] = {}
        for bid, prev in self._prev_quantities.items():
            now = new_quantities.get(bid, 0)
            sold = max(0, prev - now)
            if sold > 0:
                sales_this_brief[bid] = sold
        self._prev_quantities = new_quantities

        # 2. Rider pool tick. Use the hero's current_tick so freshness
        #    clocks decay correctly relative to env time.
        batches_by_id = {
            b.batch_id: b for b in (self._hero._state.batches if self._hero._state else [])
        }
        rider_events = self._rider_pool.tick(
            current_tick=self._hero._current_tick,
            sales_this_tick=sales_this_brief,
            batches_by_id=batches_by_id,
            rng=self._rng,
        )

        # 3. Liquidation: parse LIQUIDATE actions from the brief and
        #    execute against the hero's batches.
        liq_decisions = self._extract_liquidate_decisions(action)
        liq_results = self._liquidation.execute(
            liq_decisions, batches_by_id, self._rng,
        )

        # 4. Consumer cohort retention given the rider's avg ETA.
        if self._hero._state is not None:
            cohort_boosts = self._cohort_agent.act(
                self._hero._state,
                avg_eta_minutes=self._rider_pool.stats.avg_eta_minutes,
            )
            cohort_obs = self._cohort_agent.observe(
                self._hero._state,
                avg_eta_minutes=self._rider_pool.stats.avg_eta_minutes,
            )
        else:
            cohort_boosts, cohort_obs = {}, {}

        # 5. Compute brief rewards and reset per-brief stats.
        r6 = self._rider_pool.compute_brief_reward()
        r7 = self._liquidation.compute_brief_reward()
        rider_snapshot = self._rider_pool.snapshot()
        liquidation_snapshot = self._liquidation.snapshot()
        self._rider_pool.reset_brief_stats()
        self._liquidation.reset_brief()

        # Surface saturation events on the bus so the dashboard sees them.
        for ev in rider_events:
            if ev.get("kind") == "rider_saturated":
                try:
                    self._bus.post(
                        tick=self._hero._current_tick,
                        sender_id="env",
                        verb=MessageVerb.BROADCAST,
                        body=f"rider pool saturated: queue={ev.get('queue_depth')}",
                        payload=ev,
                    )
                except Exception:  # noqa: BLE001 — bus shouldn't crash the step
                    pass

        return {
            "r6_delivery_quality": round(r6, 4),
            "r7_liquidation": round(r7, 4),
            "r6_episode_total": round(self._r6_episode_total + r6, 4),
            "r7_episode_total": round(self._r7_episode_total + r7, 4),
            "rider_pool": rider_snapshot,
            "liquidation": {
                "this_brief": liquidation_snapshot.get("this_brief", []),
                "total_recovered_rs": liquidation_snapshot.get("total_recovered_rs", 0.0),
                "total_reckless": liquidation_snapshot.get("total_reckless", 0),
            },
            "cohorts": cohort_obs,
            "cohort_demand_boosts": cohort_boosts,
            "rider_events": rider_events,
            "sales_this_brief": sales_this_brief,
        }

    @staticmethod
    def _extract_liquidate_decisions(brief_text: str):
        """Pull LIQUIDATE actions out of the brief's PRICING DIRECTIVE.

        Reuses the same BriefParser the rest of the pipeline uses, so
        the regex for finding the JSON block matches the production
        contract instead of being a one-off here.
        """
        from freshprice_env.brief_pipeline.parser import BriefParser
        from freshprice_env.enums import BriefEngineType
        result = BriefParser.parse(brief_text, BriefEngineType.PRICING)
        if not result.success or not result.brief:
            return []
        directive = result.brief.get("directive_dict") or result.brief.get("directive") or {}
        return parse_liquidate_directive(directive)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wrap_with_market_block(self, obs: str) -> str:
        """Prepend the MARKET COMMONS block to the hero's prompt."""
        recent = self._bus.all_messages()[-12:]
        pool = self._reputation.pool_trust_summary()
        lines = [
            "## MARKET COMMONS (other actors you share a market with)",
            f"  Hero store id: {self._hero_id}",
            f"  Competitors: " + ", ".join(c.store_id for c in self._competitors),
            f"  Farmer pool size: {pool['n_farmers']} | "
            f"mean trust: {pool['mean_trust']:.2f} "
            f"(min {pool['min_trust']:.2f}, max {pool['max_trust']:.2f})",
        ]
        if self._enable_regulator:
            lines.append(
                "  Active directive schemas: "
                f"PRICING {self._registry.active_version(BriefEngineType.PRICING)}, "
                f"FARMER {self._registry.active_version(BriefEngineType.FARMER)}, "
                f"TREND {self._registry.active_version(BriefEngineType.TREND)}"
            )
            if self._regulatory_events:
                lines.append(
                    "  Last regulator broadcast: "
                    + self._regulatory_events[-1]["headline"]
                )
        lines.append("")
        lines.append("## RECENT MESSAGES (bus, last 12)")
        if not recent:
            lines.append("  (no messages yet — bus is quiet)")
        else:
            for m in recent:
                tag = f"@{m.receiver_id}" if m.receiver_id else "*all*"
                lines.append(
                    f"  [t{m.tick} seq{m.seq}] {m.sender_id} → {tag} "
                    f"{m.verb.value}: {m.body[:120]}"
                )
        lines += [
            "",
            "## MESSAGE PROTOCOL (you may emit these in your brief)",
            "  Use a `## MESSAGES` section. One message per line:",
            "    CHAT @<receiver>: <free text>",
            "    BID @<receiver>: <price>/kg, <qty>kg, <comment>",
            "    COUNTER @<receiver>: <new price>/kg",
            "    COMMIT @<receiver>: <terms>",
            "    REVEAL @<receiver>: <private info you choose to share>",
            "  Use `*all*` or omit @<receiver> for broadcast.",
        ]
        return "\n".join(lines) + "\n\n" + obs

    def _infer_hero_action_kind(self, brief_text: str) -> str:
        """Best-effort label for the most-recent hero action.

        Used by RECIPROCAL competitors to mirror behavior.
        """
        text = (brief_text or "").upper()
        if '"DECISION": "ACCEPT"' in text or '"DECISION":"ACCEPT"' in text:
            return "ACCEPT_OFFER"
        if '"DECISION": "DECLINE"' in text or '"DECISION":"DECLINE"' in text:
            return "DECLINE_OFFER"
        if '"DECISION": "COUNTER"' in text or '"DECISION":"COUNTER"' in text:
            return "COUNTER_OFFER"
        if '"PRICE_MULTIPLIER":' in text and any(
            f'"PRICE_MULTIPLIER": {x}' in text or f'"PRICE_MULTIPLIER":{x}' in text
            for x in ("0.5", "0.6", "0.55", "0.45")
        ):
            return "PRICE_DROP"
        if "## MESSAGES" in (brief_text or "").upper():
            return "BILATERAL_MESSAGE"
        return "BRIEF_NORMAL"

    def _apply_competitor_contention(
        self, competitor_actions: list[CompetitorAction],
    ) -> None:
        """If a competitor ACCEPTs a farmer offer first, kill it from hero state.

        Models the supply-chain contention: a farmer can only sell once.
        """
        if self._hero._state is None:
            return
        for ca in competitor_actions:
            if ca.kind != "OFFER_BID":
                continue
            if ca.payload.get("decision", "").upper() != "ACCEPT":
                continue
            target_id = ca.target_id
            if target_id is None:
                continue
            for i, off in enumerate(self._hero._state.pending_offers):
                if off.offer_id == target_id and off.is_pending:
                    # Snip it: status DECLINED from hero's POV (lost to competitor)
                    from dataclasses import replace
                    self._hero._state.pending_offers[i] = replace(
                        off, status=FarmerOfferStatus.DECLINED,
                    )
                    self._bus.post(
                        tick=self._hero._current_tick,
                        sender_id=ca.payload.get("sender", "env"),
                        verb=MessageVerb.BROADCAST,
                        body=(
                            f"offer {target_id} won by competitor — "
                            "no longer available"
                        ),
                        payload={"offer_id": target_id, "lost_to": "competitor"},
                    )
                    break

    def _maybe_emit_farmer_offers(self, current_tick: int) -> list[dict]:
        """Schedule-driven farmer emissions, but now via FarmerAgents.

        The hero env normally calls _maybe_inject_farmer_offer; we
        complement it by occasionally triggering an additional FarmerAgent
        BID into the bus so the multi-agent dynamics are visible. Hero
        env's pending_offers is the canonical inventory of available
        offers; we add a fresh one on roughly a 144-tick (1.5-day) cadence.
        """
        out: list[dict] = []
        if self._hero._state is None:
            return out
        # Throttle: at most every 144 ticks
        if current_tick == 0 or current_tick % 144 != 0:
            return out
        if len(self._hero._state.pending_offers) >= MAX_ACTIVE_FARMER_OFFERS:
            return out

        farmer = self._rng.choice(list(self._farmer_pool.values()))
        offer_id = f"mc_offer_{current_tick:04d}_{farmer.profile.farmer_id[-4:]}"
        if offer_id in self._farmer_offers_seen:
            return out
        self._farmer_offers_seen.add(offer_id)
        offer = farmer.generate_offer(offer_id, current_tick)
        # Score and inject into hero's pending offers
        scored = self._hero._farmer_engine.score_offer(offer, self._hero._state)
        self._hero._state.pending_offers.append(scored)
        out.append({
            "offer_id": offer_id,
            "farmer_id": farmer.profile.farmer_id,
            "farmer_name": farmer.profile.farmer_name,
            "trust": round(farmer.reputation().trust_score, 3),
            "category": offer.product_category,
            "price_per_kg": offer.offered_price_per_kg,
            "quantity_kg": offer.quantity_kg,
        })
        return out
