"""Run all baselines and produce the comparison table.

Usage:
    python -m eval.baselines.run_baselines
    python -m eval.baselines.run_baselines --episodes 5 --output eval/baseline_results.json

Output (stdout):
    ┌─────────────────────┬────────┬────────────┬────────────┬─────────────────┐
    │ Agent               │  WRR   │ r1_pricing │ r2_farmer  │ CO2 saved (kg)  │
    ├─────────────────────┼────────┼────────────┼────────────┼─────────────────┤
    │ RandomAgent         │  0.05  │   0.03     │   0.00     │    0.8 kg       │
    │ RuleBasedAgent      │  0.34  │   0.28     │   0.21     │   18.4 kg       │
    │ [Trained model]     │  --    │   --       │   --       │   --            │
    └─────────────────────┴────────┴────────────┴────────────┴─────────────────┘
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from eval.baselines.random_agent import RandomAgent
from eval.baselines.rule_based_agent import RuleBasedAgent
from freshprice_env.enums import CurriculumScenario
from freshprice_env.freshprice_env import FreshPriceEnv


def run_agent_episodes(
    agent,
    scenario: CurriculumScenario,
    n_episodes: int,
    base_seed: int = 0,
) -> list[dict]:
    """Run an agent for n_episodes and return per-episode metrics."""
    results = []

    for ep in range(n_episodes):
        env = FreshPriceEnv(scenario=scenario, seed=base_seed + ep * 100)
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        final = info.get("final_reward", {})
        results.append({
            "episode": ep,
            "wrr": final.get("wrr", env._state.wrr if env._state else 0.0),
            "r1_pricing": final.get("r1_pricing", 0.0),
            "r2_farmer": final.get("r2_farmer", 0.0),
            "r3_trend": final.get("r3_trend", 0.0),
            "brief_quality_score": final.get("brief_quality_score", 0.0),
            "anti_hack_violations": final.get("anti_hack_violations", 0),
            "co2_saved_kg": final.get("co2_saved_kg", 0.0),
            "kg_food_saved": final.get("kg_food_saved", 0.0),
            "episode_valid": final.get("episode_valid", True),
        })

    return results


def aggregate(results: list[dict]) -> dict:
    def mean(key):
        vals = [r[key] for r in results]
        return round(statistics.mean(vals), 3) if vals else 0.0

    def std(key):
        vals = [r[key] for r in results]
        return round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0

    return {
        "wrr_mean": mean("wrr"),
        "wrr_std": std("wrr"),
        "r1_pricing": mean("r1_pricing"),
        "r2_farmer": mean("r2_farmer"),
        "r3_trend": mean("r3_trend"),
        "brief_quality_score": mean("brief_quality_score"),
        "anti_hack_violations": mean("anti_hack_violations"),
        "co2_saved_kg": mean("co2_saved_kg"),
        "kg_food_saved": mean("kg_food_saved"),
        "valid_episode_pct": round(
            sum(1 for r in results if r["episode_valid"]) / len(results) * 100, 1
        ) if results else 0.0,
    }


def print_table(rows: list[dict]) -> None:
    header = f"{'Agent':<22} {'WRR':>6} {'r1':>6} {'r2':>6} {'r3':>6} {'CO2 saved':>12}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        agg = row["agg"]
        print(
            f"{row['name']:<22} "
            f"{agg['wrr_mean']:>6.3f} "
            f"{agg['r1_pricing']:>6.3f} "
            f"{agg['r2_farmer']:>6.3f} "
            f"{agg['r3_trend']:>6.3f} "
            f"{agg['co2_saved_kg']:>10.1f} kg"
        )
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FreshPrice baselines")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per agent per scenario")
    parser.add_argument("--scenarios", nargs="+",
                        default=["STABLE_WEEK", "CRISIS_WEEK"],
                        help="Curriculum scenarios to evaluate")
    parser.add_argument("--output", type=str, default="eval/baseline_results.json")
    args = parser.parse_args()

    agents = [
        ("RandomAgent",    RandomAgent(seed=42)),
        ("RuleBasedAgent", RuleBasedAgent()),
    ]

    all_results: dict = {}

    for scenario_name in args.scenarios:
        scenario = CurriculumScenario[scenario_name]
        print(f"\n=== Scenario: {scenario_name} ({args.episodes} episodes each) ===")

        table_rows = []
        for agent_name, agent in agents:
            print(f"  Running {agent_name}...", end=" ", flush=True)
            results = run_agent_episodes(agent, scenario, args.episodes)
            agg = aggregate(results)
            print(f"WRR={agg['wrr_mean']:.3f}")
            table_rows.append({"name": agent_name, "agg": agg, "episodes": results})
            all_results[f"{scenario_name}/{agent_name}"] = agg

        print_table(table_rows)

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
