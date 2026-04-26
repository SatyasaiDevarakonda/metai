"""Tests for the before-vs-after-RL comparison rig.

Covers:
  - ExpertAgent: deterministic intervention rate + preference drift after episode 10
  - MultiAgentRuntime: fan-out generate_all() returns one brief per runtime
  - build_comparison_runtime: scripted-only path (no model checkpoints needed)
  - /agent/compare/* endpoints respond + populate the SES delta dict
  - inference_comparison.py CLI smoke (scripted baseline only)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestExpertAgent(unittest.TestCase):
    def test_intervention_is_deterministic_per_seed(self):
        from freshprice_env.agents.expert_agent import ExpertAgent
        a = ExpertAgent(seed=0)
        b = ExpertAgent(seed=0)
        # Same (episode, brief_idx) should give the same intervention decision
        for ep in range(3):
            for bi in range(5):
                ra = a.maybe_intervene(ep, bi, prompt="x")
                rb = b.maybe_intervene(ep, bi, prompt="x")
                self.assertEqual(ra is None, rb is None)

    def test_intervention_rate_matches_target(self):
        from freshprice_env.agents.expert_agent import ExpertAgent
        a = ExpertAgent(seed=0, intervention_rate=0.15)
        n_attempts, n_interventions = 0, 0
        for ep in range(100):
            for bi in range(10):
                if a.maybe_intervene(ep, bi, prompt="x") is not None:
                    n_interventions += 1
                n_attempts += 1
        rate = n_interventions / n_attempts
        # Should be roughly 0.15 (large-N law). Wide tolerance because
        # the deterministic per-(ep,bi) hash is not perfectly uniform.
        self.assertGreater(rate, 0.08)
        self.assertLess(rate, 0.25)

    def test_preference_drift_after_episode_10(self):
        from freshprice_env.agents.expert_agent import (
            ExpertAgent, PreferencePhase,
        )
        a = ExpertAgent(seed=0, drift_at_episode=10)
        a.observe_outcome(5)
        self.assertEqual(a.phase, PreferencePhase.REVENUE_FIRST)
        a.observe_outcome(11)
        self.assertEqual(a.phase, PreferencePhase.WASTE_FIRST)


class TestMultiAgentRuntime(unittest.TestCase):
    def test_generate_all_fans_out(self):
        from server.agent_runtime import (
            MultiAgentRuntime, ScriptedAgentRuntime,
        )
        # Two scripted runtimes with different seeds (deterministically
        # produce identical briefs, but the keys must show up).
        rt = MultiAgentRuntime({
            "baseline": ScriptedAgentRuntime(seed=0),
            "secondary": ScriptedAgentRuntime(seed=1),
        })
        out = rt.generate_all("=== TASK ===\nWrite a PRICING brief.")
        self.assertEqual(set(out.keys()), {"baseline", "secondary"})
        for brief in out.values():
            self.assertIn("DIRECTIVE:", brief)
            self.assertIn("CONFIDENCE:", brief)

    def test_build_comparison_runtime_scripted_only(self):
        from server.agent_runtime import build_comparison_runtime
        rt = build_comparison_runtime(baseline=True)
        self.assertEqual(rt.names, ["baseline"])
        self.assertIn("baseline", rt.info())


class TestCompareEndpoints(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from server.app import app
        self.client = TestClient(app)
        # Drop any cached comparison runtime so this test sees a clean state
        from server.agent_runtime import reset_comparison_runtime
        reset_comparison_runtime()

    def test_info_returns_baseline_runtime(self):
        r = self.client.get("/agent/compare/info")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ready")
        self.assertIn("baseline", data["names"])

    def test_brief_endpoint_returns_brief_per_runtime(self):
        r = self.client.post(
            "/agent/compare/brief",
            json={"prompt": "=== TASK ===\nWrite a PRICING brief."},
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("baseline", data["briefs"])
        for brief in data["briefs"].values():
            self.assertIn("DIRECTIVE:", brief)

    def test_episode_endpoint_returns_per_runtime_results(self):
        r = self.client.post(
            "/agent/compare/episode",
            json={"scenario": "STABLE_WEEK", "seed": 42, "max_briefs": 2},
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["scenario"], "STABLE_WEEK")
        self.assertIn("baseline", data["results"])
        self.assertIn("mean_ses", data["results"]["baseline"])
        self.assertIn("anti_hack_violations", data["results"]["baseline"])

    def test_snapshot_endpoint_serves_data_dir_json(self):
        # Write a tiny snapshot, hit the endpoint, confirm shape.
        snap_path = REPO_ROOT / "data" / "comparison_results.json"
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        snap_path.write_text(json.dumps({
            "produced_at": "2026-01-01T00:00:00+00:00",
            "runtimes": ["baseline"],
            "scenarios": ["STABLE_WEEK"],
            "per_scenario": {"STABLE_WEEK": {"baseline": {"mean_ses": 0.05}}},
            "improvement": {},
        }), encoding="utf-8")
        r = self.client.get("/agent/compare/snapshot")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")
        self.assertEqual(r.json()["results"]["scenarios"], ["STABLE_WEEK"])


class TestInferenceComparisonCLI(unittest.TestCase):
    def test_cli_runs_scripted_only_path(self):
        with tempfile.TemporaryDirectory() as d:
            out_path = Path(d) / "comparison.json"
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "inference_comparison.py"),
                 "--scenarios", "STABLE_WEEK",
                 "--episodes-per-scenario", "1",
                 "--max-briefs", "2",
                 "--out", str(out_path)],
                capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
                timeout=120,
            )
            self.assertEqual(result.returncode, 0,
                             msg=f"CLI failed: stderr={result.stderr}")
            self.assertTrue(out_path.is_file())
            data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("baseline", data["runtimes"])
            self.assertIn("STABLE_WEEK", data["per_scenario"])
            self.assertIn("mean_ses",
                          data["per_scenario"]["STABLE_WEEK"]["baseline"])


if __name__ == "__main__":
    unittest.main()
