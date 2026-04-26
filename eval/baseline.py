"""Zero-shot base model evaluation — the "before" in before/after training.

Runs the unmodified base model on fixed seeds and records WRR + brief quality.
Judges need this to verify that training actually improved something.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path

from freshprice_env.enums import CurriculumScenario
from freshprice_env.freshprice_env import FreshPriceEnv

logger = logging.getLogger(__name__)


def run_baseline(
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    scenarios: list[str] | None = None,
    episodes_per_scenario: int = 5,
    output_file: str = "eval/baseline_results.json",
    seed: int = 42,
) -> dict:
    """Run zero-shot base model evaluation.

    No fine-tuning. No LoRA. Pure base model inference.
    Uses the same fixed seeds as Evaluator.get_eval_seeds().
    """
    import torch
    from unsloth import FastLanguageModel

    if scenarios is None:
        scenarios = ["STABLE_WEEK", "BUSY_WEEKEND", "FARMER_WEEK", "TREND_WEEK", "CRISIS_WEEK"]

    # 1. Load base model (4-bit, inference only)
    logger.info("Loading base model: %s", model_id)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    from freshprice_env._gen_utils import quiet_generation_config
    quiet_generation_config(model)

    # 2. Run episodes
    results: dict[str, dict] = {}

    for scenario_name in scenarios:
        scenario = CurriculumScenario[scenario_name]
        level = scenario.value
        ep_seeds = [level * 1000 + i for i in range(episodes_per_scenario)]

        logger.info("Evaluating %s (%d episodes)...", scenario_name, len(ep_seeds))

        wrrs: list[float] = []
        qualities: list[float] = []
        violations: list[int] = []

        for i, ep_seed in enumerate(ep_seeds):
            env = FreshPriceEnv(scenario=scenario, seed=ep_seed)
            prompt, info = env.reset(seed=ep_seed)
            done = False

            while not done:
                # Generate brief from base model
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=4096,
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=800,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                response = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )

                prompt, reward, done, truncated, info = env.step(response)

            final = info.get("final_reward", {})
            ep_wrr = final.get("wrr", 0.0)
            ep_quality = final.get("brief_quality_score", 0.0)
            ep_violations = final.get("anti_hack_violations", 0)

            wrrs.append(ep_wrr)
            qualities.append(ep_quality)
            violations.append(ep_violations)

            logger.info(
                "  Episode %d/%d: WRR=%.4f Quality=%.3f Violations=%d",
                i + 1, len(ep_seeds), ep_wrr, ep_quality, ep_violations,
            )

        results[scenario_name] = {
            "wrr_mean": round(_mean(wrrs), 4),
            "wrr_std": round(_std(wrrs), 4),
            "quality_mean": round(_mean(qualities), 4),
            "quality_std": round(_std(qualities), 4),
            "violations_mean": round(_mean([float(v) for v in violations]), 1),
            "episodes": episodes_per_scenario,
        }

    # 3. Save results
    output = {
        "model_id": model_id,
        "evaluation_date": datetime.now().isoformat(),
        "episodes_per_scenario": episodes_per_scenario,
        "seed": seed,
        "results": results,
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # 4. Log summary table
    logger.info("=" * 60)
    logger.info("BASELINE RESULTS (Zero-Shot) -- Model: %s", model_id)
    logger.info("=" * 60)
    logger.info("  %-18s %7s %9s %11s",
                "Scenario", "WRR", "Quality", "Violations")
    for scenario_name, stats in results.items():
        logger.info(
            "  %-18s %7.3f %9.3f %11.1f",
            scenario_name, stats["wrr_mean"],
            stats["quality_mean"], stats["violations_mean"],
        )
    logger.info("Saved to %s", output_file)

    # 5. Return
    return output


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run zero-shot baseline evaluation")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        choices=[s.name for s in CurriculumScenario])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", default="eval/baseline_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_baseline(
        model_id=args.model_id,
        scenarios=args.scenarios,
        episodes_per_scenario=args.episodes,
        output_file=args.output,
        seed=args.seed,
    )
