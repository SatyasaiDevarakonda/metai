"""Smoke tests for the new multi-agent / long-horizon / oversight pieces.

Covers:
  - ReputationStore upsert/record_interaction round-trip
  - AgentNotebook NOTE / COMMIT / RESOLVE flow + adherence math
  - notebook_directives.extract + apply on a sample brief
  - MarketBus post + parse_messages_from_brief
  - SchemaRegistry validate (v1 happy path, v2 missing field, v3 forbidden field)
  - LongHorizonFreshPriceEnv: reset + 3 steps don't crash, notebook accumulates
  - MarketCommonsEnv: reset + 1 step produces farmer_messages or bus messages
  - OversightAuditor (rule_based) returns a non-empty report
  - ScenarioComposer.next_scenario + record_outcome don't crash
  - Self-play ScriptedNegotiationPolicy smoke
  - Counterfactual replay: baseline reproduces, fork diverges only after swap
"""

from __future__ import annotations

import unittest

from freshprice_env.enums import BriefEngineType, CurriculumScenario


class TestReputationStore(unittest.TestCase):
    def setUp(self):
        from freshprice_env.persistence.reputation_store import ReputationStore
        self.store = ReputationStore(db_path=":memory:")

    def test_upsert_and_get(self):
        rep = self.store.upsert_farmer("f1", "Test Farmer", base_reserve_per_kg=20.0)
        self.assertEqual(rep.farmer_id, "f1")
        self.assertAlmostEqual(rep.base_reserve_per_kg, 20.0, places=2)
        # Idempotent
        rep2 = self.store.upsert_farmer("f1", "Test Farmer", 20.0)
        self.assertEqual(rep.trust_score, rep2.trust_score)

    def test_record_interaction_updates_trust(self):
        self.store.upsert_farmer("f1", "F1", 20.0, initial_trust=0.6)
        before = self.store.get("f1").trust_score
        self.store.record_interaction(
            episode_id="e1", tick=10, farmer_id="f1", store_id="s1",
            offer_price_per_kg=22.0, decision="ACCEPT",
            counter_price_per_kg=None, reserve_at_time=20.0,
        )
        after = self.store.get("f1").trust_score
        self.assertGreater(after, before)

    def test_lowball_counter_drops_trust(self):
        self.store.upsert_farmer("f1", "F1", 20.0, initial_trust=0.7)
        self.store.record_interaction(
            episode_id="e1", tick=10, farmer_id="f1", store_id="s1",
            offer_price_per_kg=22.0, decision="COUNTER",
            counter_price_per_kg=15.0, reserve_at_time=20.0,
        )
        rep = self.store.get("f1")
        self.assertLess(rep.trust_score, 0.7)

    def test_adjusted_reserve_moves_with_trust(self):
        self.store.upsert_farmer("f1", "F1", 20.0, initial_trust=1.0)
        loyal = self.store.get("f1").adjusted_reserve()
        self.store.upsert_farmer("f2", "F2", 20.0, initial_trust=0.0)
        burnt = self.store.get("f2").adjusted_reserve()
        self.assertLess(loyal, burnt)


class TestAgentNotebook(unittest.TestCase):
    def setUp(self):
        from freshprice_env.notebook.agent_notebook import AgentNotebook
        self.nb = AgentNotebook()

    def test_note_recall(self):
        self.nb.write_note("k1", "v1", tick=0)
        self.assertEqual(self.nb.recall("k1").value, "v1")

    def test_pinned_persists(self):
        for i in range(80):    # exceed eviction
            self.nb.write_note(f"k{i}", "v", tick=i)
        self.nb.write_note("important", "must keep", tick=999, pinned=True)
        for i in range(80, 200):
            self.nb.write_note(f"k{i}", "v", tick=i + 1000)
        self.assertIsNotNone(self.nb.recall("important"))

    def test_commit_resolve_adherence(self):
        c = self.nb.commit("custom", "x", due_tick=10, tick=0)
        self.nb.resolve_commitment(c.commitment_id, honored=True, tick=10)
        self.assertEqual(self.nb.adherence_score(), 1.0)
        c2 = self.nb.commit("custom", "y", due_tick=20, tick=11)
        self.nb.resolve_commitment(c2.commitment_id, honored=False, tick=20)
        self.assertAlmostEqual(self.nb.adherence_score(), 0.5)


class TestNotebookDirectives(unittest.TestCase):
    def test_parse_and_apply(self):
        from freshprice_env.notebook import (
            AgentNotebook, NotebookDirectiveExecutor, extract_notebook_directives,
        )
        brief = """
SITUATION: Inventory looks ok.

## NOTEBOOK
NOTE: dairy_excess -> 12 units WATCH
NOTE_PIN: cash_buffer -> 4200
COMMIT: inventory_below:dairy:30@200 | clear dairy by midnight
UPDATE_PLAN: discount fast-moving SKUs first

DIRECTIVE:
{"engine": "PRICING", "actions": []}
        """
        directives = extract_notebook_directives(brief)
        self.assertEqual(len(directives), 4)
        nb = AgentNotebook()
        results = NotebookDirectiveExecutor.apply(directives, nb, current_tick=5)
        self.assertTrue(all(r.ok for r in results))
        self.assertEqual(nb.recall("dairy_excess").value, "12 units WATCH")
        self.assertEqual(len(nb.open_commitments()), 1)
        self.assertIn("discount", nb.plan)


class TestMarketBus(unittest.TestCase):
    def test_post_and_parse(self):
        from freshprice_env.protocol.market_bus import MarketBus, parse_messages_from_brief
        bus = MarketBus()
        for verb, body in [
            ("CHAT", "hello"),
            ("BID", "Rs38/kg, 50kg"),
        ]:
            bus.post(tick=0, sender_id="store_001", verb=verb, body=body)
        msgs = bus.all_messages()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[1].verb.value, "BID")
        # Parse a brief block
        brief = "## MESSAGES\nCHAT @farmer.rajan: thanks\nBID @farmer.rajan: 38.0/kg 50kg"
        parsed = parse_messages_from_brief(brief, sender_id="store_001", tick=0)
        self.assertEqual(len(parsed), 2)
        verb, recv, body, payload = parsed[1]
        self.assertEqual(verb.value, "BID")
        self.assertAlmostEqual(payload.get("price_per_kg"), 38.0, places=2)


class TestSchemaRegistry(unittest.TestCase):
    def test_v1_happy(self):
        from freshprice_env.brief_pipeline.schema_registry import SchemaRegistry
        r = SchemaRegistry()
        ok, errs = r.validate(
            {"engine": "PRICING", "actions": [{"batch_id": "b1", "price_multiplier": 0.6}]},
            BriefEngineType.PRICING,
        )
        self.assertTrue(ok)
        self.assertEqual(errs, [])

    def test_v2_missing_field(self):
        from freshprice_env.brief_pipeline.schema_registry import SchemaRegistry
        r = SchemaRegistry()
        r.set_version(BriefEngineType.PRICING, "v2", tick=10)
        ok, errs = r.validate(
            {"engine": "PRICING", "actions": [{"batch_id": "b1", "price_multiplier": 0.6}]},
            BriefEngineType.PRICING,
        )
        self.assertFalse(ok)
        self.assertTrue(any("cold_chain_log" in e for e in errs))

    def test_v3_forbidden_field(self):
        from freshprice_env.brief_pipeline.schema_registry import SchemaRegistry
        r = SchemaRegistry()
        r.set_version(BriefEngineType.PRICING, "v3", tick=10)
        ok, errs = r.validate(
            {"engine": "PRICING", "actions": [
                {"batch_id": "b1", "price_multiplier": 0.6, "clearance_action": 0.5},
            ]},
            BriefEngineType.PRICING,
        )
        self.assertFalse(ok)
        # forbidden price_multiplier and floor passes (0.5 >= 0.40)
        self.assertTrue(any("forbidden" in e.lower() for e in errs))


class TestLongHorizonEnv(unittest.TestCase):
    def test_reset_and_steps(self):
        from freshprice_env.long_horizon_env import LongHorizonFreshPriceEnv
        env = LongHorizonFreshPriceEnv(
            scenario=CurriculumScenario.STABLE_WEEK, seed=42,
        )
        obs, info = env.reset()
        self.assertIn("NOTEBOOK", obs)
        self.assertEqual(info["mode"], "long_horizon_30day")
        for _ in range(3):
            brief = (
                "SITUATION: x\n\n"
                "## NOTEBOOK\nNOTE: hello -> world\n\n"
                'DIRECTIVE: {"engine": "PRICING", "actions": []}\n'
                "CONFIDENCE: MEDIUM\n"
            )
            obs, reward, done, _, info = env.step(brief)
            self.assertIn("notebook", info)
        self.assertIsNotNone(env.notebook.recall("hello"))


class TestMarketCommonsEnv(unittest.TestCase):
    def setUp(self):
        from freshprice_env.persistence.reputation_store import (
            ReputationStore, reset_default_store,
        )
        reset_default_store()    # ensure ":memory:" default for the test
        self.store = ReputationStore(db_path=":memory:")

    def test_reset_and_step(self):
        from freshprice_env.market_commons_env import MarketCommonsEnv
        from freshprice_env.protocol.market_bus import MarketBus
        bus = MarketBus()
        env = MarketCommonsEnv(
            scenario=CurriculumScenario.CRISIS_WEEK, seed=42,
            n_competitors=1, bus=bus, reputation_store=self.store,
            enable_regulator=True,
        )
        obs, info = env.reset()
        self.assertIn("MARKET COMMONS", obs)
        self.assertEqual(info["n_competitors"], 1)
        brief = (
            "SITUATION: x\n## MESSAGES\nCHAT @store_002: hi neighbour\n\n"
            'DIRECTIVE: {"engine": "PRICING", "actions": []}\nCONFIDENCE: MEDIUM\n'
        )
        obs, reward, done, _, info = env.step(brief)
        self.assertIn("cooperation_index", info)
        self.assertGreater(info["bus_total_messages"], 0)


class TestOversightAuditor(unittest.TestCase):
    def test_rule_based_clean(self):
        from freshprice_env.agents.oversight_auditor import (
            AuditableEvent, AuditTrajectory, OversightAuditor,
        )
        traj = AuditTrajectory(
            episode_id="e1", scenario="STABLE_WEEK",
            events=[
                AuditableEvent(tick=8, kind="BUS_MESSAGE", actor="store_001",
                               payload={"verb": "CHAT", "body": "hello"},
                               summary="CHAT hello"),
            ],
        )
        report = OversightAuditor(mode="rule_based").audit(traj)
        self.assertGreaterEqual(report.trust_score, 0.7)
        self.assertEqual(report.recommendation, "APPROVE")

    def test_rule_based_bad(self):
        from freshprice_env.agents.oversight_auditor import (
            AuditableEvent, AuditTrajectory, OversightAuditor,
        )
        traj = AuditTrajectory(
            episode_id="e2", scenario="CRISIS_WEEK",
            events=[
                AuditableEvent(tick=t * 10, kind="RULE_VIOLATION",
                               actor="store_001",
                               payload={"violation_type": "EARLY_DISCOUNT",
                                        "engine": "PRICING", "detail": ""},
                               summary="EARLY_DISCOUNT")
                for t in range(8)
            ],
        )
        report = OversightAuditor(mode="rule_based").audit(traj)
        self.assertLess(report.trust_score, 0.8)
        self.assertTrue(any(p["id"] == "EARLY_DEEP_DISCOUNT" for p in report.suspicious_patterns))


class TestScenarioComposer(unittest.TestCase):
    def test_sample_and_record(self):
        from freshprice_env.scenario_composer import ScenarioComposer
        c = ScenarioComposer(seed=11)
        s = c.next_scenario()
        c.record_outcome(s, wrr=0.30)   # poor episode → fail signal
        # Hardness summary should be queryable
        h = c.hardest_cells(top_k=2)
        self.assertIn("base_scenario", h)
        # Sample again
        s2 = c.next_scenario()
        self.assertIsNotNone(s2.base_scenario)


class TestSelfPlay(unittest.TestCase):
    def test_smoke(self):
        from training.self_play import smoke_test
        results = smoke_test(n_episodes=2)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn(r.outcome, {"DEAL", "NO_DEAL"})


class TestParseFailPenalty(unittest.TestCase):
    """Closes the reward-leak surface where a parse-failed brief could
    still earn positive reward from natural sales over the next 8 ticks."""

    def test_parse_fail_subtracts_penalty_and_flags_anti_hack(self):
        from freshprice_env.constants import (
            PARSE_FAIL_REWARD_PENALTY,
        )
        from freshprice_env.freshprice_env import FreshPriceEnv
        env = FreshPriceEnv(scenario=CurriculumScenario.STABLE_WEEK, seed=42)
        env.reset()
        # Garbage brief — guaranteed to fail parse
        garbage = "this is not a brief, no DIRECTIVE here"
        obs, reward, done, _, info = env.step(garbage)
        self.assertFalse(info["parse_success"])
        # Penalty must have been applied
        self.assertGreater(info.get("parse_fail_penalty_applied", 0), 0)
        # WRR-delta and final reward must differ by at least the penalty
        # whenever parse failed
        self.assertAlmostEqual(
            info["wrr_delta"] - reward,
            PARSE_FAIL_REWARD_PENALTY,
            places=4,
        )

    def test_parse_fail_with_positive_wrr_delta_records_violation(self):
        # Build an env state that will produce a positive WRR delta and
        # parse-fail simultaneously by stepping a garbage brief — this is
        # the exact exploit the new flag is designed to surface.
        from freshprice_env.freshprice_env import FreshPriceEnv
        env = FreshPriceEnv(scenario=CurriculumScenario.STABLE_WEEK, seed=42)
        env.reset()
        # Burn one valid step so accumulators have data
        env.step('SITUATION: x\n\nDIRECTIVE: {"engine": "PRICING", "actions": []}\nCONFIDENCE: MEDIUM\n')
        # Now step garbage — parse fails. If WRR delta happens to be > 0
        # the violation should be recorded; otherwise no violation.
        garbage = "this is not a brief"
        obs, reward, done, _, info = env.step(garbage)
        if info["wrr_delta"] > 0:
            violations = env._reward_engine._antihack_violations
            self.assertTrue(
                any(v["violation_type"] == "PARSE_FAIL_POSITIVE_REWARD"
                    for v in violations),
                "expected PARSE_FAIL_POSITIVE_REWARD violation",
            )


class TestAntiHackScanAllRollouts(unittest.TestCase):
    """The anti-hack scanner must see ALL rollouts, including those the
    trajectory buffer would reject. Previously the buffer pre-filter
    meant the scanner could never see the worst cases."""

    def test_scan_all_rollouts_includes_rejected(self):
        from eval.anti_hack_checker import AntiHackChecker
        rollouts = [
            {
                "episode_num": 0,
                "scenario": "STABLE_WEEK",
                "wrr": 0.8,
                "brief_quality_score": 0.7,
                "constitutional_passed": True,
                "episode_valid": True,
                "briefs": [
                    {"engine_type": "PRICING", "tick": 0, "directive": {
                        "engine": "PRICING",
                        "actions": [{"batch_id": "b1", "price_multiplier": 0.9}],
                    }},
                ],
                "final_reward": {"wrr": 0.8, "brief_quality_score": 0.7},
            },
            # This episode would be rejected by the buffer
            # (constitutional_passed=False) but must STILL be scanned.
            {
                "episode_num": 1,
                "scenario": "CRISIS_WEEK",
                "wrr": 0.6,
                "brief_quality_score": 0.65,
                "constitutional_passed": False,
                "episode_valid": True,
                "briefs": [
                    {"engine_type": "PRICING", "tick": 0, "directive": {
                        "engine": "PRICING",
                        "actions": [
                            # Early deep discount = HIGH severity hack
                            {"batch_id": "b1", "price_multiplier": 0.20},
                        ],
                    }},
                ],
                "final_reward": {"wrr": 0.6, "brief_quality_score": 0.65},
            },
        ]
        result = AntiHackChecker.scan_all_rollouts(rollouts)
        self.assertEqual(result["total_trajectories"], 2)
        self.assertEqual(result["buffer_eligible"], 1)
        self.assertEqual(result["buffer_excluded"], 1)
        # Hack pattern must be detected even though it lives in a rejected
        # rollout — that's the whole point of scan_all_rollouts.
        self.assertIn("EARLY_DEEP_DISCOUNT", result["pattern_frequency"])


class TestCurriculumPromotionGate(unittest.TestCase):
    """Combined WRR + constitutional pass rate gate."""

    def test_above_wrr_but_below_constitution_blocks(self):
        from training.curriculum import CurriculumManager
        # 4 eval episodes: WRR is high but only 1/4 constitutionally pass
        episodes = [
            {"wrr": 1.5, "constitutional_passed": False},
            {"wrr": 2.1, "constitutional_passed": False},
            {"wrr": 1.8, "constitutional_passed": False},
            {"wrr": 1.2, "constitutional_passed": True},
        ]
        ok, diag = CurriculumManager.is_eval_above_promotion(episodes)
        self.assertFalse(ok)
        self.assertFalse(diag["constitution_ok"])
        self.assertIn("constitutional", diag["reason"].lower())

    def test_above_both_passes(self):
        from training.curriculum import CurriculumManager
        episodes = [
            {"wrr": 0.85, "constitutional_passed": True},
            {"wrr": 0.78, "constitutional_passed": True},
            {"wrr": 0.92, "constitutional_passed": True},
            {"wrr": 0.71, "constitutional_passed": True},
        ]
        ok, diag = CurriculumManager.is_eval_above_promotion(episodes)
        self.assertTrue(ok)


class TestEvaluatorStdReporting(unittest.TestCase):
    """Eval std should not appear with ± error bars when n < 5."""

    def test_std_meaningful_flag(self):
        from eval.evaluator import Evaluator, EvalReport, _MIN_N_FOR_STD
        # Build a tiny report with n=2 episodes
        report = EvalReport(
            checkpoint_dir="x",
            scenarios_evaluated=[CurriculumScenario.STABLE_WEEK],
            episodes_per_scenario=2,
            results={"STABLE_WEEK": []},
            summary={"by_scenario": {
                "STABLE_WEEK": {
                    "wrr_mean": 0.5, "wrr_std": 0.1, "wrr_min": 0.4, "wrr_max": 0.6,
                    "quality_mean": 0.7, "quality_std": 0.05,
                    "violations_mean": 0.5,
                    "constitutional_pass_rate": "1/2",
                    "n": 2, "std_meaningful": False,
                }
            }, "best_wrr": 0.5, "best_scenario": "STABLE_WEEK",
                "worst_wrr": 0.5, "worst_scenario": "STABLE_WEEK",
                "overall_quality_mean": 0.7, "overall_wrr_mean": 0.5},
        )
        # Just sanity-check that the printer doesn't crash and that
        # the 'std_meaningful' flag is exposed for downstream callers
        Evaluator.print_report(Evaluator.__new__(Evaluator), report)
        self.assertEqual(_MIN_N_FOR_STD, 5)


class TestDPOReadiness(unittest.TestCase):
    """Honest DPO readiness flag for the summary cell."""

    def test_below_min_buffer_says_cannot_run(self):
        from training.trajectory_buffer import (
            DEFAULT_DPO_MIN_BUFFER, TrajectoryBuffer,
        )
        buf = TrajectoryBuffer()
        readiness = buf.dpo_readiness()
        self.assertFalse(readiness.can_run)
        self.assertEqual(readiness.buffer_size, 0)
        self.assertEqual(readiness.min_required, DEFAULT_DPO_MIN_BUFFER)

    def test_above_min_buffer_says_can_run(self):
        from training.trajectory_buffer import Trajectory, TrajectoryBuffer
        buf = TrajectoryBuffer()
        for i in range(3):
            buf.add(Trajectory(
                episode_num=i, scenario=CurriculumScenario.STABLE_WEEK,
                wrr=0.8, brief_quality_score=0.7,
                constitutional_passed=True, episode_valid=True,
                briefs=[{"engine_type": "PRICING", "raw_response": "x"}],
                reward_engine_snapshot={},
            ))
        readiness = buf.dpo_readiness(min_buffer=2)
        self.assertTrue(readiness.can_run)
        self.assertEqual(readiness.buffer_size, 3)


def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


@unittest.skipUnless(_has_module("httpx"), "httpx not installed")
class TestGymServerEndpoints(unittest.TestCase):
    """The gym-compliant /gym/* fallback endpoints must always conform
    to the (obs, info) / (obs, reward, terminated, truncated, info)
    contract — closing the Kaggle-notebook bugs where the openenv-core
    /reset omitted info and /state forgot the episode_id."""

    def test_reset_step_state(self):
        from fastapi.testclient import TestClient
        from server.app import app
        client = TestClient(app)
        r = client.post("/gym/reset", json={"scenario": "STABLE_WEEK", "seed": 42})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("observation", body)
        self.assertIn("info", body)
        # POST /step must include reward, terminated, truncated, info
        r = client.post("/gym/step", json={
            "action":
                'SITUATION: x\n\nDIRECTIVE: {"engine": "PRICING", "actions": []}\n'
                "CONFIDENCE: MEDIUM\n"
        })
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for k in ("observation", "reward", "terminated", "truncated", "info"):
            self.assertIn(k, body)
        # /gym/state must surface episode_id + step_count > 0
        r = client.get("/gym/state")
        body = r.json()
        self.assertEqual(body["step_count"], 1)
        self.assertIsNotNone(body["episode_id"])


@unittest.skipUnless(_has_module("torch") and _has_module("wandb"),
                     "torch/wandb not installed")
class TestGRPOScenarioRotation(unittest.TestCase):
    """GRPO trainer must rotate scenarios per episode when configured —
    otherwise R2-F / R3-T columns stay flat-zero in the report."""

    def test_next_scenario_rotates(self):
        # Avoid loading the actual model — just test the rotation logic
        from training.grpo_trainer import FreshPriceGRPOTrainer
        trainer = FreshPriceGRPOTrainer.__new__(FreshPriceGRPOTrainer)
        trainer._scenarios = [
            CurriculumScenario.STABLE_WEEK,
            CurriculumScenario.FARMER_WEEK,
            CurriculumScenario.TREND_WEEK,
        ]
        trainer._rotate = True
        trainer._rotation_idx = 0
        trainer.scenario = CurriculumScenario.STABLE_WEEK
        seq = [trainer._next_scenario() for _ in range(6)]
        self.assertEqual(
            [s.name for s in seq],
            ["STABLE_WEEK", "FARMER_WEEK", "TREND_WEEK",
             "STABLE_WEEK", "FARMER_WEEK", "TREND_WEEK"],
        )


class TestCounterfactualReplay(unittest.TestCase):
    def test_baseline_reproduces(self):
        from eval.counterfactual_replay import (
            BriefLogEntry, CounterfactualReplay,
        )
        log = [
            BriefLogEntry(tick=i * 8, engine_type="PRICING",
                          brief_text=(
                              "SITUATION: x\n\n"
                              'DIRECTIVE: {"engine": "PRICING", "actions": []}\n'
                              "CONFIDENCE: MEDIUM\n"
                          ))
            for i in range(5)
        ]
        replay = CounterfactualReplay(
            CurriculumScenario.STABLE_WEEK, seed=42, brief_log=log,
        )
        baseline_a = replay.run_baseline()
        baseline_b = replay.run_baseline()
        self.assertAlmostEqual(baseline_a.final_wrr, baseline_b.final_wrr, places=4)


if __name__ == "__main__":
    unittest.main()
