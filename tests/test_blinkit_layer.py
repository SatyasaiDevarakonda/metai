"""Smoke tests for the Blinkit-style additive layer.

Covers the three new pieces in isolation:

  - RiderPoolEngine ferries orders, fires saturation events, computes r6
  - ConsumerCohortAgent walks premium cohort on slow ETA + URGENT mix
  - LiquidationEngine accepts CRITICAL stock, flags reckless attempts

These are unit-level tests; integration with MarketCommonsEnv is left
for the env-level test file (existing tests must keep passing).
"""

from __future__ import annotations

import random
import unittest
from dataclasses import dataclass

from freshprice_env.agents.consumer_cohort_agent import ConsumerCohortAgent
from freshprice_env.engines.liquidation_engine import (
    LiquidationDecision, LiquidationEngine, parse_liquidate_directive,
)
from freshprice_env.engines.rider_pool_engine import RiderPoolEngine
from freshprice_env.enums import BatchStatus, ExpiryUrgency


@dataclass
class _FakeBatch:
    batch_id: str
    category: str
    urgency: ExpiryUrgency
    hours_to_expiry: float
    original_price: float
    current_price: float
    quantity_remaining: int
    status: BatchStatus = BatchStatus.ACTIVE


class _FakeState:
    def __init__(self, batches):
        self.batches = batches


class TestRiderPoolEngine(unittest.TestCase):
    def test_orders_ferry_and_score(self):
        engine = RiderPoolEngine(rider_count=2)
        b = _FakeBatch("B1", "dairy", ExpiryUrgency.WATCH, 24.0, 80.0, 80.0, 10)
        rng = random.Random(0)

        # Tick 0: place 1 order, rider grabs it.
        engine.tick(current_tick=0,
                    sales_this_tick={"B1": 1},
                    batches_by_id={"B1": b}, rng=rng)
        self.assertEqual(len(engine.active), 1)
        self.assertEqual(len(engine.pending), 0)

        # Ticks 1..n: ferry completes (~14 min / 15 min/tick).
        for t in range(1, 5):
            engine.tick(current_tick=t,
                        sales_this_tick={}, batches_by_id={"B1": b}, rng=rng)
        self.assertGreaterEqual(engine.stats.orders_delivered, 1)
        self.assertGreaterEqual(engine.compute_brief_reward(), 0.0)

    def test_saturation_event_when_queue_overruns_capacity(self):
        # 5 distinct batches sell at once -> 5 orders, 1 rider, queue=4 > capacity=1.
        engine = RiderPoolEngine(rider_count=1)
        batches = {
            f"B{i}": _FakeBatch(
                f"B{i}", "dairy", ExpiryUrgency.WATCH, 48.0, 80.0, 80.0, 10,
            )
            for i in range(5)
        }
        events = engine.tick(
            current_tick=0,
            sales_this_tick={k: 1 for k in batches},
            batches_by_id=batches,
            rng=random.Random(0),
        )
        kinds = [e["kind"] for e in events]
        self.assertIn("rider_saturated", kinds)


class TestConsumerCohortAgent(unittest.TestCase):
    def test_premium_walks_on_critical_stock(self):
        rng = random.Random(0)
        agent = ConsumerCohortAgent(rng)
        batches = [
            _FakeBatch("B1", "dairy", ExpiryUrgency.CRITICAL, 4.0, 80.0, 40.0, 100),
        ]
        state = _FakeState(batches)
        agent.act(state, avg_eta_minutes=8.0)
        self.assertLess(agent.last_retention["PREMIUM"], 1.0)
        # Bargain hunters happily buy CRITICAL.
        self.assertGreaterEqual(agent.last_retention["BARGAIN"], 0.9)

    def test_premium_walks_on_slow_eta(self):
        rng = random.Random(0)
        agent = ConsumerCohortAgent(rng)
        batches = [
            _FakeBatch("B1", "dairy", ExpiryUrgency.FRESH, 96.0, 80.0, 80.0, 100),
        ]
        state = _FakeState(batches)
        agent.act(state, avg_eta_minutes=30.0)  # premium tolerance is 12 min
        self.assertLess(agent.last_retention["PREMIUM"], 0.7)


class TestLiquidationEngine(unittest.TestCase):
    def test_accepts_critical_stock(self):
        engine = LiquidationEngine()
        b = _FakeBatch("B1", "dairy", ExpiryUrgency.CRITICAL, 2.0, 80.0, 35.0, 20)
        out = engine.execute([LiquidationDecision("B1")], {"B1": b}, random.Random(0))
        self.assertTrue(out[0].accepted)
        self.assertGreater(out[0].rs_recovered, 0)
        self.assertEqual(b.status, BatchStatus.LIQUIDATED)
        self.assertEqual(b.quantity_remaining, 0)
        self.assertGreater(engine.compute_brief_reward(), 0)

    def test_flags_reckless_liquidation(self):
        engine = LiquidationEngine()
        b = _FakeBatch("B1", "dairy", ExpiryUrgency.FRESH, 96.0, 80.0, 80.0, 20)
        out = engine.execute([LiquidationDecision("B1")], {"B1": b}, random.Random(0))
        self.assertFalse(out[0].accepted)
        self.assertTrue(out[0].reckless)
        # Anti-hack penalty dominates the reward.
        self.assertLess(engine.compute_brief_reward(), 0)
        # Batch was NOT touched.
        self.assertEqual(b.status, BatchStatus.ACTIVE)
        self.assertEqual(b.quantity_remaining, 20)

    def test_parse_directive(self):
        directive = {
            "engine": "PRICING",
            "actions": [
                {"action": "DISCOUNT", "batch_id": "B0", "discount_pct": 30},
                {"action": "LIQUIDATE", "batch_id": "B1", "channel": "B2B"},
                {"action": "LIQUIDATE", "batch_id": "B2"},
            ],
        }
        decisions = parse_liquidate_directive(directive)
        self.assertEqual([d.batch_id for d in decisions], ["B1", "B2"])
        self.assertEqual(decisions[0].channel, "B2B")


class TestMarketCommonsEnvWiring(unittest.TestCase):
    """Regression tests proving the Blinkit layer is reachable through
    MarketCommonsEnv.step() and not just isolated unit tests."""

    def _make_env(self, *, enable_blinkit: bool):
        from freshprice_env.market_commons_env import MarketCommonsEnv
        from freshprice_env.enums import CurriculumScenario
        from freshprice_env.persistence.reputation_store import ReputationStore
        return MarketCommonsEnv(
            scenario=CurriculumScenario.STABLE_WEEK, seed=42,
            reputation_store=ReputationStore(":memory:"),
            enable_blinkit=enable_blinkit,
        )

    @staticmethod
    def _fallback_brief():
        return (
            "SITUATION: ok.\n\n"
            "SIGNAL ANALYSIS: N/A\n\n"
            "VIABILITY CHECK: N/A\n\n"
            "RECOMMENDATION: hold.\n\n"
            "DIRECTIVE:\n"
            '{"engine": "PRICING", "actions": []}\n\n'
            "CONFIDENCE: MEDIUM"
        )

    def test_default_disabled_no_r6_r7_keys(self):
        env = self._make_env(enable_blinkit=False)
        obs, info = env.reset()
        obs, reward, done, t, info = env.step(self._fallback_brief())
        self.assertNotIn("r6_delivery_quality", info)
        self.assertNotIn("r7_liquidation", info)
        self.assertNotIn("rider_pool", info)

    def test_enabled_populates_r6_r7_and_snapshots(self):
        env = self._make_env(enable_blinkit=True)
        obs, info = env.reset()
        obs, reward, done, t, info = env.step(self._fallback_brief())
        self.assertIn("r6_delivery_quality", info)
        self.assertIn("r7_liquidation", info)
        self.assertIn("rider_pool", info)
        self.assertIn("cohorts", info)
        self.assertEqual(len(info["cohorts"].get("cohorts", [])), 3)
        # No LIQUIDATE -> r7 must be exactly 0 (not flagged, not credited).
        self.assertEqual(info["r7_liquidation"], 0.0)

    def test_reckless_liquidate_negative_r7_through_env(self):
        import json
        env = self._make_env(enable_blinkit=True)
        obs, info = env.reset()
        batch_id = env.hero._state.batches[0].batch_id  # FRESH/WATCH at start
        directive = {"engine": "PRICING",
                     "actions": [{"action": "LIQUIDATE", "batch_id": batch_id}]}
        brief = (
            "SITUATION: reckless liquidate test.\n\n"
            "SIGNAL ANALYSIS: N/A\n\n"
            "VIABILITY CHECK: N/A\n\n"
            "RECOMMENDATION: attempt to dump fresh stock; engine should flag.\n\n"
            f"DIRECTIVE:\n{json.dumps(directive)}\n\n"
            "CONFIDENCE: MEDIUM"
        )
        obs, reward, done, t, info = env.step(brief)
        self.assertTrue(info.get("parse_success"))
        self.assertLess(info["r7_liquidation"], 0.0)
        results = (info.get("liquidation") or {}).get("this_brief", [])
        self.assertTrue(results and results[0]["reckless"])


if __name__ == "__main__":
    unittest.main()
