"""Tests for the four new dashboard panels' backend feeds:

  - /atlas/inventory: 6 envs, 8 agents, 7 reward components + 3
    global penalties + 6 scenarios.
  - tick frame schema: every frame must carry r1..r7 + SES + the
    penalty_events list (otherwise the Live Reward Signal panel
    silently shows zeros).
  - /training/status: idle by default; POST to /training/event must
    push a step into reinforce_history.
"""

from __future__ import annotations

import unittest


class TestAtlasInventory(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from server.app import app
        self.client = TestClient(app)

    def test_inventory_shape(self):
        r = self.client.get("/atlas/inventory")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(len(data["environments"]), 6)
        self.assertEqual(len(data["agents"]), 8)        # 7 + ExpertAgent
        self.assertEqual(len(data["reward_components"]), 7)
        self.assertEqual(len(data["global_penalties"]), 3)
        self.assertEqual(len(data["scenarios"]), 6)

    def test_reward_weights_sum_to_one(self):
        r = self.client.get("/atlas/inventory")
        weights = [c["weight"] for c in r.json()["reward_components"]]
        self.assertAlmostEqual(sum(weights), 1.0, places=4)

    def test_scenario_engine_map_matches_strategy(self):
        r = self.client.get("/atlas/inventory")
        scenarios = {s["name"]: s["active_engines"] for s in r.json()["scenarios"]}
        self.assertEqual(scenarios["STABLE_WEEK"], [1])
        self.assertEqual(scenarios["BUSY_WEEKEND"], [1, 4, 6])
        self.assertEqual(scenarios["FARMER_WEEK"], [1, 2, 5])
        self.assertEqual(scenarios["TREND_WEEK"], [1, 3, 7])
        self.assertEqual(scenarios["CRISIS_WEEK"], [1, 2, 3, 4, 5, 6, 7])


class TestTrainingStatus(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from server.app import app, _training_state
        self.client = TestClient(app)
        # Reset shared state between tests
        _training_state["current_stage"] = "idle"
        _training_state["reinforce_history"] = []
        _training_state["last_event"] = None

    def test_default_idle(self):
        r = self.client.get("/training/status")
        data = r.json()
        self.assertEqual(data["current_stage"], "idle")
        self.assertEqual(data["reinforce_history"], [])
        self.assertFalse(data["has_history"])

    def test_post_event_advances_stage_and_history(self):
        r = self.client.post("/training/event", json={
            "current_stage": "reinforce",
            "reinforce_step": {"step": 1, "loss": 1.48, "policy_loss": 1.48,
                               "kl": 0.04, "mean_advantage": -0.34},
            "last_event": "REINFORCE step 1: loss=+1.48",
        })
        self.assertEqual(r.status_code, 200)
        r2 = self.client.get("/training/status")
        data = r2.json()
        self.assertEqual(data["current_stage"], "reinforce")
        self.assertEqual(len(data["reinforce_history"]), 1)
        self.assertTrue(data["has_history"])
        self.assertEqual(data["last_event"], "REINFORCE step 1: loss=+1.48")

    def test_history_capped_at_200(self):
        from server.app import _training_state
        _training_state["reinforce_history"] = []
        for i in range(220):
            self.client.post("/training/event", json={
                "reinforce_step": {"step": i, "loss": 1.0, "policy_loss": 1.0,
                                   "kl": 0.0, "mean_advantage": 0.0},
            })
        r = self.client.get("/training/status")
        self.assertLessEqual(len(r.json()["reinforce_history"]), 200)


class TestExtendedTickFrame(unittest.TestCase):
    """Live Reward Signal + Decision Flow panels both consume tick frames.
    Every frame must carry r1..r7, SES, and a (possibly empty) penalty list."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from server.app import app
        self.client = TestClient(app)

    def test_frame_after_episode_run_has_seven_engine_keys(self):
        # Drive an episode through /agent/run_episode (uses scripted backend
        # by default) and inspect the resulting frame.
        r = self.client.post("/agent/run_episode", json={
            "scenario": "STABLE_WEEK", "seed": 42, "max_briefs": 1,
        })
        self.assertEqual(r.status_code, 200)
        frames_resp = self.client.get("/commons/sim_frames?limit=5")
        frames = frames_resp.json()["frames"]
        self.assertGreater(len(frames), 0)
        # Pick the most recent post-step frame (skip the reset frame at index 0)
        post_step_frames = [f for f in frames if f.get("latest_brief")]
        self.assertGreater(len(post_step_frames), 0)
        f = post_step_frames[-1]
        for k in ("r1_pricing", "r2_farmer", "r3_trend",
                  "r4_intrafleet", "r5_micromfg",
                  "r6_event", "r7_surplusbox",
                  "store_efficiency_score", "penalty_events"):
            self.assertIn(k, f, f"tick frame missing {k}")
        self.assertIsInstance(f["penalty_events"], list)


if __name__ == "__main__":
    unittest.main()
