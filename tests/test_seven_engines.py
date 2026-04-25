"""Tests for the FreshPrice 7-engine SES reward path.

Covers:
  - Engines 4-7 in isolation (parse + execute + compute_brief_reward)
  - SES weights sum to 1.0
  - FreshPriceEnv.step() populates r1..r7 + store_efficiency_score in info
  - active_engines() returns the right set per scenario
  - Reckless / early-routing / late-routing anti-hack paths
"""

from __future__ import annotations

import json
import random
import unittest
from dataclasses import dataclass

from freshprice_env.constants import (
    SES_WEIGHT_R1_PRICING, SES_WEIGHT_R2_FARMER, SES_WEIGHT_R3_TREND,
    SES_WEIGHT_R4_INTRAFLEET, SES_WEIGHT_R5_MICROMFG, SES_WEIGHT_R6_EVENT,
    SES_WEIGHT_R7_SURPLUSBOX,
)
from freshprice_env.engines.seven_engines_reward import (
    EventEngine, EventPrestock,
    IntraFleetEngine, IntraFleetTransfer,
    MicroMfgEngine, MicroMfgRouting,
    SurplusBoxEngine, SurplusBoxSelection,
    parse_event_prestocks, parse_intrafleet_transfers,
    parse_micromfg_routings, parse_surplus_box_selections,
    store_efficiency_score,
)
from freshprice_env.enums import (
    ACTIVE_ENGINES_BY_SCENARIO, BatchStatus, CurriculumScenario,
    ExpiryUrgency, active_engines,
)


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


class TestSESWeights(unittest.TestCase):
    def test_weights_sum_to_one(self):
        total = (
            SES_WEIGHT_R1_PRICING + SES_WEIGHT_R2_FARMER + SES_WEIGHT_R3_TREND
            + SES_WEIGHT_R4_INTRAFLEET + SES_WEIGHT_R5_MICROMFG
            + SES_WEIGHT_R6_EVENT + SES_WEIGHT_R7_SURPLUSBOX
        )
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_ses_of_all_ones(self):
        self.assertAlmostEqual(
            store_efficiency_score(r1=1, r2=1, r3=1, r4=1, r5=1, r6=1, r7=1),
            1.0, places=4,
        )

    def test_ses_of_all_zeros(self):
        self.assertEqual(
            store_efficiency_score(r1=0, r2=0, r3=0, r4=0, r5=0, r6=0, r7=0),
            0.0,
        )


class TestActiveEnginesMap(unittest.TestCase):
    def test_strategy_engine_activations(self):
        # Per FreshPrice strategy Section 8.
        self.assertEqual(active_engines(CurriculumScenario.STABLE_WEEK), frozenset({1}))
        self.assertEqual(active_engines(CurriculumScenario.BUSY_WEEKEND), frozenset({1, 4, 6}))
        self.assertEqual(active_engines(CurriculumScenario.FARMER_WEEK), frozenset({1, 2, 5}))
        self.assertEqual(active_engines(CurriculumScenario.TREND_WEEK), frozenset({1, 3, 7}))
        self.assertEqual(active_engines(CurriculumScenario.CRISIS_WEEK),
                         frozenset({1, 2, 3, 4, 5, 6, 7}))


class TestIntraFleetEngine(unittest.TestCase):
    def test_legitimate_transfer_scores_positive(self):
        eng = IntraFleetEngine()
        b = _FakeBatch("B1", "dairy", ExpiryUrgency.URGENT, 18.0, 80, 50, 10)
        eng.execute([IntraFleetTransfer("A", "B", "B1", 5)], {"B1": b})
        self.assertGreater(eng.compute_brief_reward(), 0.0)
        self.assertEqual(eng.snapshot()["transfers_this_brief"], 1)
        self.assertEqual(eng.snapshot()["reckless_count"], 0)

    def test_fresh_stock_transfer_is_reckless(self):
        eng = IntraFleetEngine()
        b = _FakeBatch("B1", "dairy", ExpiryUrgency.FRESH, 96.0, 80, 80, 10)
        eng.execute([IntraFleetTransfer("A", "B", "B1", 5)], {"B1": b})
        self.assertEqual(eng.snapshot()["reckless_count"], 1)
        self.assertLessEqual(eng.compute_brief_reward(), 0.0)


class TestMicroMfgEngine(unittest.TestCase):
    def test_critical_routing_recovers_revenue(self):
        eng = MicroMfgEngine()
        b = _FakeBatch("B1", "dairy", ExpiryUrgency.CRITICAL, 4.0, 80, 30, 20)
        eng.execute([MicroMfgRouting("B1")], {"B1": b})
        self.assertEqual(eng.snapshot()["accepted"], 1)
        self.assertGreater(eng.snapshot()["total_recovered_rs"], 0)
        self.assertGreater(eng.compute_brief_reward(), 0)

    def test_early_routing_penalised(self):
        eng = MicroMfgEngine()
        b = _FakeBatch("B1", "fruits", ExpiryUrgency.WATCH, 48.0, 100, 100, 10)
        eng.execute([MicroMfgRouting("B1")], {"B1": b})
        self.assertEqual(eng.snapshot()["early_routing"], 1)
        self.assertLess(eng.compute_brief_reward(), 0)


class TestEventEngine(unittest.TestCase):
    def test_valid_lead_scores_positive(self):
        eng = EventEngine()
        # 20 hours ahead of event @ 4 ticks/hour -> target_event_tick = 80 from tick 0
        eng.execute([EventPrestock("dairy", 20, 80)], current_tick=0)
        self.assertEqual(eng.snapshot()["valid_lead"], 1)
        self.assertGreater(eng.compute_brief_reward(), 0)

    def test_too_late_penalty(self):
        eng = EventEngine()
        # Event 1 hour from now -> below the 4-hour minimum lead
        eng.execute([EventPrestock("dairy", 20, 4)], current_tick=0)
        self.assertEqual(eng.snapshot()["too_late"], 1)


class TestSurplusBoxEngine(unittest.TestCase):
    def test_in_target_weight_range_scores_positive(self):
        eng = SurplusBoxEngine()
        b = _FakeBatch("B1", "fruits", ExpiryUrgency.URGENT, 18.0, 100, 70, 10)
        # 7 units × 0.25 kg = 1.75 kg (in target range 1.5-2.0)
        eng.assemble([SurplusBoxSelection("B1", 7)], {"B1": b}, random.Random(0))
        snap = eng.snapshot()
        self.assertGreaterEqual(snap["last_box_weight_kg"], 1.5)
        self.assertLessEqual(snap["last_box_weight_kg"], 2.0)
        self.assertGreater(snap["five_star_count"], snap["cancel_count"])


class TestDirectiveParsers(unittest.TestCase):
    BRIEF_TEMPLATE = (
        "SITUATION: x.\n\nSIGNAL ANALYSIS: N/A\n\nVIABILITY CHECK: N/A\n\n"
        "RECOMMENDATION: x.\n\nDIRECTIVE:\n{directive}\n\nCONFIDENCE: MEDIUM"
    )

    def _wrap(self, directive: dict) -> str:
        return self.BRIEF_TEMPLATE.format(directive=json.dumps(directive))

    def test_parses_all_four_side_directives(self):
        directive = {
            "engine": "PRICING",
            "actions": [],
            "intrafleet_actions": [
                {"source_store": "A", "dest_store": "B", "batch_id": "B1", "units": 3},
            ],
            "micromfg_actions": [{"batch_id": "B2", "processor": "p1"}],
            "event_actions": [
                {"category": "dairy", "quantity_units": 10, "target_event_tick": 80},
            ],
            "surplus_box_actions": [{"batch_id": "B3", "units": 2}],
        }
        brief = self._wrap(directive)
        self.assertEqual(len(parse_intrafleet_transfers(brief)), 1)
        self.assertEqual(len(parse_micromfg_routings(brief)), 1)
        self.assertEqual(len(parse_event_prestocks(brief)), 1)
        self.assertEqual(len(parse_surplus_box_selections(brief)), 1)

    def test_no_directive_returns_empty(self):
        brief = "SITUATION: nothing here.\n\nCONFIDENCE: LOW"
        self.assertEqual(parse_intrafleet_transfers(brief), [])
        self.assertEqual(parse_micromfg_routings(brief), [])


class TestFreshPriceEnvSESPath(unittest.TestCase):
    """End-to-end: env.step() must populate r1..r7 + SES in info dict."""

    def test_info_dict_has_all_seven_components(self):
        from freshprice_env.freshprice_env import FreshPriceEnv
        env = FreshPriceEnv(scenario=CurriculumScenario.STABLE_WEEK, seed=42)
        env.reset()
        brief = (
            "SITUATION: 7-engines wired test.\n\nSIGNAL ANALYSIS: N/A\n\n"
            "VIABILITY CHECK: N/A\n\nRECOMMENDATION: hold + sample side directive.\n\n"
            'DIRECTIVE:\n{"engine": "PRICING", "actions": [], '
            '"intrafleet_actions": [{"source_store":"A","dest_store":"B",'
            '"batch_id":"batch_0001","units":2}]}\n\n'
            "CONFIDENCE: MEDIUM"
        )
        _, _, _, _, info = env.step(brief)
        for key in ("r1_pricing", "r2_farmer", "r3_trend",
                    "r4_intrafleet", "r5_micromfg",
                    "r6_event", "r7_surplusbox", "store_efficiency_score"):
            self.assertIn(key, info, f"missing {key} in info dict")

    def test_no_side_directives_keeps_r4_r7_at_zero(self):
        from freshprice_env.freshprice_env import FreshPriceEnv
        env = FreshPriceEnv(scenario=CurriculumScenario.STABLE_WEEK, seed=42)
        env.reset()
        brief = (
            "SITUATION: vanilla.\n\nSIGNAL ANALYSIS: N/A\n\nVIABILITY CHECK: N/A\n\n"
            "RECOMMENDATION: hold.\n\n"
            'DIRECTIVE:\n{"engine": "PRICING", "actions": []}\n\n'
            "CONFIDENCE: MEDIUM"
        )
        _, _, _, _, info = env.step(brief)
        self.assertEqual(info["r4_intrafleet"], 0.0)
        self.assertEqual(info["r5_micromfg"], 0.0)
        self.assertEqual(info["r6_event"], 0.0)
        self.assertEqual(info["r7_surplusbox"], 0.0)


if __name__ == "__main__":
    unittest.main()
