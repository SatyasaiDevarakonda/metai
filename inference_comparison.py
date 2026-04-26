"""inference_comparison.py — before vs. after RL evaluation harness.

Runs the same set of scenarios with multiple agent runtimes (typically
baseline + SFT + RL) and writes a single JSON to
``data/comparison_results.json`` that the dashboard consumes for the
"Before vs After RL" panel.

This is the artefact the hackathon's judging criterion #3 explicitly
asks for: "comparison against a baseline -- anything that proves the
agent learned something" (20% of total score).

Usage::

    # After training on Kaggle, point the script at your two checkpoints:
    python inference_comparison.py \\
        --sft-path  /kaggle/working/checkpoints/sft_v1 \\
        --rl-path   /kaggle/working/checkpoints/dpo_round1 \\
        --scenarios STABLE_WEEK FARMER_WEEK CRISIS_WEEK \\
        --max-briefs 8 \\
        --episodes-per-scenario 3

    # If only HF Hub credentials are set, RL falls back to the inference API:
    HF_REPO_ID=your-user/qstoreprice-sft  HF_TOKEN=hf_xxx \\
        python inference_comparison.py --sft-path /kaggle/working/checkpoints/sft_v1

Output JSON shape (consumed by /agent/compare/snapshot)::

    {
      "produced_at": "2026-04-26T12:34:56Z",
      "runtimes":    {"baseline": {...}, "sft": {...}, "rl": {...}},
      "scenarios":   ["STABLE_WEEK", "FARMER_WEEK", ...],
      "per_scenario": {
        "STABLE_WEEK": {
            "baseline": {mean_ses, final_wrr_mean, anti_hack_total, sample_briefs[3]},
            "sft":      {...},
            "rl":       {...}
        },
        ...
      },
      "improvement": {
        "rl_over_baseline_ses_delta": +0.42,
        "rl_over_sft_ses_delta":      +0.18,
        ...
      }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("inference_comparison")


def run_one_episode(runtime, scenario, seed: int, max_briefs: int) -> dict:
    """Run a single episode through one runtime; return per-episode stats."""
    from freshprice_env.freshprice_env import FreshPriceEnv

    env = FreshPriceEnv(scenario=scenario, seed=seed)
    obs, _info = env.reset()
    ses_total, total_reward = 0.0, 0.0
    n_briefs, anti_hack = 0, 0
    sample_briefs: list[str] = []
    for step in range(max_briefs):
        try:
            brief = runtime.generate(obs)
        except Exception as e:  # noqa: BLE001
            brief = f"[runtime failed: {e}]"
        obs, reward, done, _t, info = env.step(brief)
        total_reward += float(reward)
        ses_total += float(info.get("store_efficiency_score", 0.0))
        n_briefs += 1
        if step < 2:
            sample_briefs.append(brief)
        if not info.get("parse_success", True):
            anti_hack += 1
        if done:
            break
    return {
        "steps_completed": n_briefs,
        "total_reward":    round(total_reward, 4),
        "final_wrr":       round(env.state().get("wrr_so_far", 0.0), 4),
        "mean_ses":        round(ses_total / max(1, n_briefs), 4),
        "anti_hack_violations": anti_hack,
        "sample_briefs":   sample_briefs,
    }


def aggregate(per_episode_runs: list[dict]) -> dict:
    """Average a list of episode results into a single per-(runtime, scenario) row."""
    if not per_episode_runs:
        return {}
    n = len(per_episode_runs)
    samples = []
    for r in per_episode_runs:
        samples.extend(r.get("sample_briefs", []))
    return {
        "episodes_run":     n,
        "mean_ses":         round(sum(r["mean_ses"] for r in per_episode_runs) / n, 4),
        "final_wrr_mean":   round(sum(r["final_wrr"] for r in per_episode_runs) / n, 4),
        "total_reward_mean":round(sum(r["total_reward"] for r in per_episode_runs) / n, 4),
        "anti_hack_total":  sum(r["anti_hack_violations"] for r in per_episode_runs),
        "sample_briefs":    samples[:3],   # keep a few for the UI
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the same scenarios through baseline / SFT / RL "
                    "agents and write data/comparison_results.json."
    )
    parser.add_argument("--sft-path", default=os.environ.get("SFT_MODEL_PATH"),
                        help="Local path to the SFT checkpoint (FreshPrice cell-sft-train output).")
    parser.add_argument("--rl-path", default=os.environ.get("RL_MODEL_PATH"),
                        help="Local path to the RL checkpoint (REINFORCE+DPO output).")
    parser.add_argument("--hf-repo-id", default=os.environ.get("HF_REPO_ID"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--scenarios", nargs="+",
                        default=["STABLE_WEEK", "FARMER_WEEK", "TREND_WEEK", "CRISIS_WEEK"],
                        help="CurriculumScenario names to evaluate on.")
    parser.add_argument("--episodes-per-scenario", type=int, default=2,
                        help="Number of seeds (episodes) to average per (runtime, scenario).")
    parser.add_argument("--max-briefs", type=int, default=8,
                        help="Briefs per episode (8 = ~one simulated day).")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Explicit seed list; overrides --episodes-per-scenario.")
    parser.add_argument("--out", default="data/comparison_results.json")
    args = parser.parse_args(argv)

    # Build the multi-agent rig.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from freshprice_env.enums import CurriculumScenario
    from server.agent_runtime import build_comparison_runtime

    rig = build_comparison_runtime(
        baseline=True,
        sft_path=args.sft_path,
        rl_path=args.rl_path,
        hf_repo_id=args.hf_repo_id,
        hf_token=args.hf_token,
    )
    if not rig.names:
        logger.error("No runtimes loaded. Pass --sft-path or --rl-path or set HF_REPO_ID + HF_TOKEN.")
        return 1
    logger.info("Loaded runtimes: %s", rig.names)

    seeds = args.seeds or list(range(42, 42 + args.episodes_per_scenario))

    per_scenario: dict[str, dict[str, dict]] = {}
    started = time.time()
    for scenario_name in args.scenarios:
        try:
            scenario = CurriculumScenario[scenario_name]
        except KeyError:
            logger.warning("Skipping unknown scenario: %s", scenario_name)
            continue
        per_scenario[scenario_name] = {}
        for runtime_name in rig.names:
            runtime = rig.get(runtime_name)
            episode_results = []
            for s in seeds:
                logger.info("  %-12s | %-15s | seed %d", runtime_name, scenario_name, s)
                episode_results.append(
                    run_one_episode(runtime, scenario, seed=s,
                                    max_briefs=args.max_briefs)
                )
            per_scenario[scenario_name][runtime_name] = aggregate(episode_results)

    # Improvement summary across all scenarios.
    def _scenario_means(metric: str, runtime: str) -> float:
        vals = [
            per_scenario[s].get(runtime, {}).get(metric, 0.0)
            for s in args.scenarios if s in per_scenario
        ]
        return round(sum(vals) / max(1, len(vals)), 4)

    runtimes_present = list(rig.names)
    baseline_ses = _scenario_means("mean_ses", "baseline") if "baseline" in runtimes_present else None
    sft_ses      = _scenario_means("mean_ses", "sft")      if "sft" in runtimes_present else None
    rl_ses       = _scenario_means("mean_ses", "rl")       if "rl" in runtimes_present else None
    improvement: dict[str, float] = {}
    if baseline_ses is not None and sft_ses is not None:
        improvement["sft_over_baseline_ses_delta"] = round(sft_ses - baseline_ses, 4)
    if sft_ses is not None and rl_ses is not None:
        improvement["rl_over_sft_ses_delta"] = round(rl_ses - sft_ses, 4)
    if baseline_ses is not None and rl_ses is not None:
        improvement["rl_over_baseline_ses_delta"] = round(rl_ses - baseline_ses, 4)

    out = {
        "produced_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_info": rig.info(),
        "runtimes":     runtimes_present,
        "scenarios":    list(per_scenario.keys()),
        "seeds":        seeds,
        "max_briefs":   args.max_briefs,
        "per_scenario": per_scenario,
        "improvement":  improvement,
        "wall_clock_seconds": round(time.time() - started, 1),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    logger.info("Wrote %s (%d scenarios, %d runtimes)",
                out_path, len(per_scenario), len(runtimes_present))
    if improvement:
        logger.info("Improvement summary: %s", improvement)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
