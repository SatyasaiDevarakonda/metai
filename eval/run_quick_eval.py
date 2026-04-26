"""Quick evaluation script — get real metrics in ~5 minutes without a trained model.

Run this to replace projected numbers in README with actual measurements.

Usage:
    # Run with rule-based agent (no GPU needed, ~2 min)
    python eval/run_quick_eval.py --agent rule_based

    # Run with a trained checkpoint (requires GPU)
    python eval/run_quick_eval.py --agent llm --checkpoint checkpoints/sft_v1

Output:
    Prints a results table and saves eval/quick_eval_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from eval.baselines.rule_based_agent import RuleBasedAgent
from eval.baselines.random_agent import RandomAgent
from freshprice_env.enums import CurriculumScenario
from freshprice_env.freshprice_env import FreshPriceEnv

SCENARIOS = [
    CurriculumScenario.STABLE_WEEK,
    CurriculumScenario.BUSY_WEEKEND,
    CurriculumScenario.FARMER_WEEK,
    CurriculumScenario.TREND_WEEK,
    CurriculumScenario.CRISIS_WEEK,
]


def _load_llm_agent(checkpoint_path: str):
    """Load a trained Unsloth/HF checkpoint as the agent."""
    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    from freshprice_env._gen_utils import quiet_generation_config
    quiet_generation_config(model)

    class _LLMAgent:
        def act(self, obs: str, info: dict) -> str:
            inputs = tokenizer(obs, return_tensors="pt").to("cuda")
            with __import__("torch").no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=0.0,
                    do_sample=False,
                )
            return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return _LLMAgent()


def run_scenario(agent, scenario: CurriculumScenario, n_episodes: int, seed_offset: int = 0) -> dict:
    """Run n_episodes for one scenario and return aggregated metrics."""
    wrrs, r1s, r2s, r3s, co2s, kg_saves, violations = [], [], [], [], [], [], []

    for ep in range(n_episodes):
        env = FreshPriceEnv(scenario=scenario, seed=seed_offset + ep * 100)
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        final = info.get("final_reward", {})
        wrrs.append(final.get("wrr", env._state.wrr if env._state else 0.0))
        r1s.append(final.get("r1_pricing", 0.0))
        r2s.append(final.get("r2_farmer", 0.0))
        r3s.append(final.get("r3_trend", 0.0))
        co2s.append(final.get("co2_saved_kg", 0.0))
        kg_saves.append(final.get("kg_food_saved", 0.0))
        violations.append(final.get("anti_hack_violations", 0))

    def _fmt(lst):
        mean = statistics.mean(lst)
        std = statistics.stdev(lst) if len(lst) > 1 else 0.0
        return {"mean": round(mean, 3), "std": round(std, 3)}

    return {
        "scenario": scenario.name,
        "n_episodes": n_episodes,
        "wrr": _fmt(wrrs),
        "r1_pricing": _fmt(r1s),
        "r2_farmer": _fmt(r2s),
        "r3_trend": _fmt(r3s),
        "co2_saved_kg": _fmt(co2s),
        "kg_food_saved": _fmt(kg_saves),
        "avg_violations": round(statistics.mean(violations), 1),
    }


def print_results(results: list[dict], agent_name: str) -> None:
    print(f"\n{'='*70}")
    print(f"  FreshPrice Quick Eval — {agent_name}")
    print(f"{'='*70}")
    header = f"{'Scenario':<20} {'WRR':>8} {'±std':>6} {'r1':>6} {'r2':>6} {'r3':>6} {'CO2 kg':>8}"
    print(header)
    print("-" * 70)
    for r in results:
        print(
            f"{r['scenario']:<20} "
            f"{r['wrr']['mean']:>8.3f} "
            f"{r['wrr']['std']:>6.3f} "
            f"{r['r1_pricing']['mean']:>6.3f} "
            f"{r['r2_farmer']['mean']:>6.3f} "
            f"{r['r3_trend']['mean']:>6.3f} "
            f"{r['co2_saved_kg']['mean']:>8.1f}"
        )
    print("-" * 70)

    # Overall
    all_wrrs = [r["wrr"]["mean"] for r in results]
    all_co2 = [r["co2_saved_kg"]["mean"] for r in results]
    print(f"{'OVERALL':.<20} {statistics.mean(all_wrrs):>8.3f} {'':>6} {'':>6} {'':>6} {'':>6} {statistics.mean(all_co2):>8.1f}")
    print(f"\nInterpretation:")
    print(f"  WRR = Weekly Waste Recovery Rate (higher is better, target ≥ 0.70)")
    print(f"  CO2 saved = kg CO2 prevented per episode (7-day simulation)")
    print(f"{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FreshPrice quick evaluation")
    parser.add_argument(
        "--agent", choices=["random", "rule_based", "llm"], default="rule_based",
        help="Which agent to evaluate"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint (required for --agent llm)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per scenario")
    parser.add_argument("--output", type=str, default="eval/quick_eval_results.json")
    args = parser.parse_args()

    # Load agent
    if args.agent == "random":
        agent = RandomAgent(seed=42)
        agent_name = "RandomAgent"
    elif args.agent == "rule_based":
        agent = RuleBasedAgent()
        agent_name = "RuleBasedAgent"
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint required for --agent llm")
        agent = _load_llm_agent(args.checkpoint)
        agent_name = f"LLM ({Path(args.checkpoint).name})"

    # Run evaluation
    print(f"Running {args.episodes} episodes × {len(SCENARIOS)} scenarios...")
    t0 = time.time()

    results = []
    for scenario in SCENARIOS:
        print(f"  {scenario.name}...", end=" ", flush=True)
        result = run_scenario(agent, scenario, args.episodes)
        results.append(result)
        print(f"WRR={result['wrr']['mean']:.3f}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    print_results(results, agent_name)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"agent": agent_name, "results": results}
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
