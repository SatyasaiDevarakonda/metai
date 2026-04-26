"""Held-out evaluation runner — inference only, no training.

Runs episodes with greedy decoding and produces structured reports
comparing model performance across curriculum levels.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from freshprice_env.enums import CurriculumScenario
from freshprice_env.freshprice_env import FreshPriceEnv

logger = logging.getLogger(__name__)


@dataclass
class EvalEpisodeResult:
    """Result of a single evaluation episode."""

    seed: int
    scenario: CurriculumScenario
    wrr: float
    r1_pricing: float
    r2_farmer: float
    r3_trend: float
    brief_quality_score: float
    anti_hack_violations: int
    constitutional_passed: bool
    briefs_written: int
    ticks_completed: int


@dataclass
class EvalReport:
    """Aggregated evaluation report across scenarios."""

    checkpoint_dir: str
    scenarios_evaluated: list[CurriculumScenario]
    episodes_per_scenario: int
    results: dict[str, list[EvalEpisodeResult]]
    summary: dict


# Minimum sample size before reporting std-dev with ± error bars.
# Below this threshold std is computed but the printer omits the ± to
# avoid false rigor (a std from n=2 is meaningless).
_MIN_N_FOR_STD: int = 5


class Evaluator:
    """Runs held-out evaluation episodes with greedy decoding."""

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load checkpoint for inference only. No LoRA training adapters."""
        if self._model is not None:
            return

        from unsloth import FastLanguageModel

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.checkpoint_dir,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self._model)
        from freshprice_env._gen_utils import quiet_generation_config
        quiet_generation_config(self._model)
        logger.info("Loaded model from %s for evaluation", self.checkpoint_dir)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate a brief using greedy decoding.

        Deterministic: temperature not used with do_sample=False.
        """
        self._load_model()

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self._model.device)

        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=800,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def run_evaluation(
        self,
        scenarios: list[CurriculumScenario] | None = None,
        episodes_per_scenario: int = 10,
        seeds: list[int] | None = None,
    ) -> EvalReport:
        """Run evaluation across one or more scenarios.

        Args:
            scenarios: Defaults to all 5 if None.
            episodes_per_scenario: Number of episodes per scenario.
            seeds: If None, deterministic: [level × 1000 + i for i in range(n)].
        """
        self._load_model()

        if scenarios is None:
            scenarios = list(CurriculumScenario)

        results: dict[str, list[EvalEpisodeResult]] = {}

        for scenario in scenarios:
            level = scenario.value
            if seeds is not None:
                ep_seeds = seeds
            else:
                ep_seeds = [level * 1000 + i for i in range(episodes_per_scenario)]

            print(f"\nEvaluating {scenario.name} ({len(ep_seeds)} episodes)...")
            scenario_results: list[EvalEpisodeResult] = []

            for i, seed in enumerate(ep_seeds):
                env = FreshPriceEnv(scenario=scenario, seed=seed, llm_client=self)
                result = self._run_one_episode(env, seed, scenario)
                scenario_results.append(result)

                if (i + 1) % 5 == 0 or i == len(ep_seeds) - 1:
                    avg_wrr = sum(r.wrr for r in scenario_results) / len(scenario_results)
                    print(f"  {i + 1}/{len(ep_seeds)} — avg WRR: {avg_wrr:.3f}")

            results[scenario.name] = scenario_results

        summary = self._build_summary(results)

        return EvalReport(
            checkpoint_dir=self.checkpoint_dir,
            scenarios_evaluated=scenarios,
            episodes_per_scenario=episodes_per_scenario,
            results=results,
            summary=summary,
        )

    def _run_one_episode(
        self,
        env: FreshPriceEnv,
        seed: int,
        scenario: CurriculumScenario,
    ) -> EvalEpisodeResult:
        """Run one episode and return the result."""
        prompt, info = env.reset(seed=seed)
        done = False
        briefs_written = 0

        while not done:
            raw_brief = self.generate(prompt)
            prompt, reward, done, truncated, info = env.step(raw_brief)
            briefs_written += 1

        final_reward = info.get("final_reward", {})
        audit = info.get("constitutional_audit", {})

        return EvalEpisodeResult(
            seed=seed,
            scenario=scenario,
            wrr=final_reward.get("wrr", 0.0),
            r1_pricing=final_reward.get("r1_pricing", 0.0),
            r2_farmer=final_reward.get("r2_farmer", 0.0),
            r3_trend=final_reward.get("r3_trend", 0.0),
            brief_quality_score=final_reward.get("brief_quality_score", 0.0),
            anti_hack_violations=final_reward.get("anti_hack_violations", 0),
            constitutional_passed=audit.get("passed", True),
            briefs_written=briefs_written,
            ticks_completed=final_reward.get("ticks_completed", 0),
        )

    # ------------------------------------------------------------------
    # Report building
    # ------------------------------------------------------------------

    def _build_summary(
        self, results: dict[str, list[EvalEpisodeResult]],
    ) -> dict:
        """Build aggregated summary across all scenarios."""
        all_wrrs: list[float] = []
        all_quality: list[float] = []
        by_scenario: dict[str, dict] = {}
        best_wrr = -1.0
        best_scenario = ""
        worst_wrr = 2.0
        worst_scenario = ""

        for scenario_name, episodes in results.items():
            wrrs = [e.wrr for e in episodes]
            qualities = [e.brief_quality_score for e in episodes]
            violations = [e.anti_hack_violations for e in episodes]
            const_passed = sum(1 for e in episodes if e.constitutional_passed)

            mean_wrr = sum(wrrs) / len(wrrs) if wrrs else 0.0
            std_wrr = _std(wrrs)

            n_eps = len(episodes)
            by_scenario[scenario_name] = {
                "wrr_mean": round(mean_wrr, 4),
                "wrr_std": round(std_wrr, 4),
                # Sample size is preserved so the printer / consumers can
                # decide whether ± is meaningful (n>=5).
                "n": n_eps,
                "std_meaningful": n_eps >= _MIN_N_FOR_STD,
                "wrr_min": round(min(wrrs), 4) if wrrs else 0.0,
                "wrr_max": round(max(wrrs), 4) if wrrs else 0.0,
                "quality_mean": round(sum(qualities) / len(qualities), 4) if qualities else 0.0,
                "quality_std": round(_std(qualities), 4),
                "violations_mean": round(sum(violations) / len(violations), 2) if violations else 0.0,
                "constitutional_pass_rate": f"{const_passed}/{len(episodes)}",
                "constitutional_pass_fraction": (
                    round(const_passed / n_eps, 3) if n_eps else 0.0
                ),
            }

            all_wrrs.extend(wrrs)
            all_quality.extend(qualities)

            if mean_wrr > best_wrr:
                best_wrr = mean_wrr
                best_scenario = scenario_name
            if mean_wrr < worst_wrr:
                worst_wrr = mean_wrr
                worst_scenario = scenario_name

        return {
            "overall_wrr_mean": round(sum(all_wrrs) / len(all_wrrs), 4) if all_wrrs else 0.0,
            "overall_quality_mean": round(sum(all_quality) / len(all_quality), 4) if all_quality else 0.0,
            "best_wrr": round(best_wrr, 4),
            "best_scenario": best_scenario,
            "worst_wrr": round(worst_wrr, 4),
            "worst_scenario": worst_scenario,
            "by_scenario": by_scenario,
        }

    # ------------------------------------------------------------------
    # Report printing
    # ------------------------------------------------------------------

    def print_report(self, report: EvalReport) -> None:
        """Print a formatted eval report to stdout."""
        print("\n" + "=" * 50)
        print("EVALUATION REPORT")
        print(f"Checkpoint: {report.checkpoint_dir}")
        print("=" * 50)

        for scenario_name, stats in report.summary.get("by_scenario", {}).items():
            episodes = report.results.get(scenario_name, [])
            n = len(episodes)
            print(f"\n-- {scenario_name} ({n} episodes) --")
            # Suppress ± when sample is too small (n<5). Reporting std on
            # 2 samples implied false rigor; now the printer prints "(std
            # not meaningful at n=N)" instead of a misleading ±.
            if stats.get("std_meaningful", n >= _MIN_N_FOR_STD):
                print(
                    f"  WRR:           {stats['wrr_mean']:.4f} +/- {stats['wrr_std']:.4f}  "
                    f"[{stats['wrr_min']:.4f} -> {stats['wrr_max']:.4f}]"
                )
                print(f"  Brief Quality: {stats['quality_mean']:.4f} +/- {stats['quality_std']:.4f}")
            else:
                print(
                    f"  WRR:           {stats['wrr_mean']:.4f}  "
                    f"[{stats['wrr_min']:.4f} -> {stats['wrr_max']:.4f}]  "
                    f"(std not meaningful at n={n}; need n>={_MIN_N_FOR_STD})"
                )
                print(f"  Brief Quality: {stats['quality_mean']:.4f}")
            print(f"  Violations:    {stats['violations_mean']:.1f} per episode")
            print(f"  Constitutional Pass Rate: {stats['constitutional_pass_rate']}")

        print(f"\nOVERALL SUMMARY")
        print(f"  Best WRR:      {report.summary['best_wrr']:.4f} ({report.summary['best_scenario']})")
        print(f"  Worst WRR:     {report.summary['worst_wrr']:.4f} ({report.summary['worst_scenario']})")
        print(f"  Mean Quality:  {report.summary['overall_quality_mean']:.4f}")

    # ------------------------------------------------------------------
    # Checkpoint comparison
    # ------------------------------------------------------------------

    def compare_checkpoints(
        self,
        checkpoint_a: str,
        checkpoint_b: str,
        scenarios: list[CurriculumScenario],
        episodes_per_scenario: int = 5,
    ) -> dict:
        """Compare two checkpoints side by side on the same seeds."""
        # Evaluate checkpoint A
        print(f"\n--- Evaluating checkpoint A: {checkpoint_a} ---")
        eval_a = Evaluator(checkpoint_dir=checkpoint_a, device=self.device)
        report_a = eval_a.run_evaluation(scenarios, episodes_per_scenario)

        # Evaluate checkpoint B
        print(f"\n--- Evaluating checkpoint B: {checkpoint_b} ---")
        eval_b = Evaluator(checkpoint_dir=checkpoint_b, device=self.device)
        report_b = eval_b.run_evaluation(scenarios, episodes_per_scenario)

        wrr_a = report_a.summary["overall_wrr_mean"]
        wrr_b = report_b.summary["overall_wrr_mean"]
        quality_a = report_a.summary["overall_quality_mean"]
        quality_b = report_b.summary["overall_quality_mean"]

        by_scenario: dict[str, dict] = {}
        for scenario_name in report_a.summary.get("by_scenario", {}):
            a_stats = report_a.summary["by_scenario"].get(scenario_name, {})
            b_stats = report_b.summary["by_scenario"].get(scenario_name, {})
            by_scenario[scenario_name] = {
                "wrr_a": a_stats.get("wrr_mean", 0.0),
                "wrr_b": b_stats.get("wrr_mean", 0.0),
                "wrr_delta": round(b_stats.get("wrr_mean", 0.0) - a_stats.get("wrr_mean", 0.0), 4),
            }

        comparison = {
            "checkpoint_a": checkpoint_a,
            "checkpoint_b": checkpoint_b,
            "wrr_delta": round(wrr_b - wrr_a, 4),
            "quality_delta": round(quality_b - quality_a, 4),
            "by_scenario": by_scenario,
        }

        # Print comparison table
        print("\n" + "=" * 60)
        print("CHECKPOINT COMPARISON")
        print("=" * 60)
        print(f"  A: {checkpoint_a}")
        print(f"  B: {checkpoint_b}")
        print(f"\n  {'Scenario':<20} {'WRR A':>8} {'WRR B':>8} {'Delta':>8}")
        print(f"  {'─' * 46}")
        for name, stats in by_scenario.items():
            delta_str = f"{stats['wrr_delta']:+.4f}"
            print(f"  {name:<20} {stats['wrr_a']:>8.4f} {stats['wrr_b']:>8.4f} {delta_str:>8}")
        print(f"\n  Overall WRR delta:     {comparison['wrr_delta']:+.4f}")
        print(f"  Overall Quality delta: {comparison['quality_delta']:+.4f}")
        winner = "B" if comparison["wrr_delta"] > 0 else "A" if comparison["wrr_delta"] < 0 else "TIE"
        print(f"  Winner: Checkpoint {winner}")

        return comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _std(values: list[float]) -> float:
    """Standard deviation of a list of floats."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluate a FreshPrice checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--scenarios", nargs="+",
        choices=[s.name for s in CurriculumScenario],
        default=None,
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--compare-with", default=None, help="Second checkpoint to compare")
    args = parser.parse_args()

    evaluator = Evaluator(checkpoint_dir=args.checkpoint)
    scenarios = [CurriculumScenario[s] for s in args.scenarios] if args.scenarios else None

    if args.compare_with:
        evaluator.compare_checkpoints(
            args.checkpoint, args.compare_with,
            scenarios or list(CurriculumScenario), args.episodes,
        )
    else:
        report = evaluator.run_evaluation(
            scenarios=scenarios, episodes_per_scenario=args.episodes,
        )
        evaluator.print_report(report)
