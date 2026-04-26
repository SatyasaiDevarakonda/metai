"""GRPO training on FreshPriceEnv — primary RL algorithm for verifiable reward tasks.

Loads from an SFT checkpoint, runs episodes in the environment, collects
trajectories, and trains with GRPOTrainer. Logs all metrics to WandB.

Two scenario modes:

  * ``scenario=<one>``: single-scenario run (legacy behaviour — caller
    drives curriculum across multiple ``run_grpo`` calls).
  * ``scenarios=[<list>]`` + ``rotate=True``: rotate scenarios per
    episode within one run so the buffer sees PRICING + FARMER + TREND.
    Closes the bug where Kaggle runs pinned to STABLE_WEEK left R2-F /
    R3-T flat-zero, hiding the engine-coverage gap.
"""

from __future__ import annotations

import logging
import random

import torch
import wandb

from freshprice_env.constants import SAVE_WRR_DEGRADATION_TOLERANCE
from freshprice_env.enums import CurriculumScenario
from freshprice_env.freshprice_env import FreshPriceEnv
from training.counterfactual import CounterfactualEngine
from training.curriculum import CurriculumManager, EpisodeResult
from training.trajectory_buffer import Trajectory, TrajectoryBuffer

logger = logging.getLogger(__name__)


class FreshPriceGRPOTrainer:
    """Connects FreshPriceEnv to the GRPO training loop."""

    def __init__(
        self,
        checkpoint_dir: str,
        output_dir: str,
        scenario: CurriculumScenario | None = None,
        seed: int = 42,
        max_new_tokens: int = 800,
        temperature: float = 0.7,
        learning_rate: float = 5e-6,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        save_every_n_episodes: int = 25,
        scenarios: list[CurriculumScenario] | None = None,
        rotate: bool = False,
    ) -> None:
        # Backwards compat: if only `scenario=` is provided, single-scenario mode.
        # New: pass `scenarios=[...]` + `rotate=True` to round-robin per episode.
        if scenarios is not None:
            self._scenarios: list[CurriculumScenario] = list(scenarios)
            self._rotate = bool(rotate) or len(self._scenarios) > 1
            self.scenario = self._scenarios[0]
        else:
            if scenario is None:
                scenario = CurriculumScenario.STABLE_WEEK
            self._scenarios = [scenario]
            self._rotate = False
            self.scenario = scenario
        self.seed = seed
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.save_every_n_episodes = save_every_n_episodes
        self.rng = random.Random(seed)
        self._rotation_idx = 0

        # Load model and tokenizer with Unsloth
        from unsloth import FastLanguageModel

        print(f"Loading model from {checkpoint_dir}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_dir,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
        from freshprice_env._gen_utils import quiet_generation_config
        quiet_generation_config(self.model)

        # Enable training mode with LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )

        # Environment
        self.env = FreshPriceEnv(
            scenario=scenario,
            seed=seed,
            llm_client=self,
        )

        # Training infrastructure
        self.curriculum_manager = CurriculumManager()
        self.trajectory_buffer = TrajectoryBuffer(rng=self.rng)
        self.counterfactual_engine = CounterfactualEngine(rng=self.rng)

        # Tracking
        self.episode_count = 0
        self.global_step = 0
        self._last_save_wrr: float = 0.0

    # ------------------------------------------------------------------
    # LLM generation (called by FreshPriceEnv)
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate an Operating Brief from the current model.

        Called by FreshPriceEnv when it needs an LLM response.
        Uses inference mode — no gradient tracking.
        """
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(self.model)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Re-enable training mode
        FastLanguageModel.for_training(self.model)

        return response

    # ------------------------------------------------------------------
    # Episode execution
    # ------------------------------------------------------------------

    def run_episode(self, episode_seed: int) -> dict:
        """Run one complete episode. Returns episode result dict.

        When ``rotate=True``, picks the next scenario in the list per
        call so a 6-episode run touches every engine (PRICING / FARMER
        / TREND) rather than pinning to one. Rebuilds the env to switch
        scenarios — required because FreshPriceEnv is constructed with
        a fixed scenario.
        """
        scenario = self._next_scenario()
        if scenario != self.scenario:
            # Rebuild the env so the new scenario's MarketStateBuilder
            # and ExternalShockEngine apply correctly.
            self.scenario = scenario
            self.env = FreshPriceEnv(
                scenario=scenario, seed=self.seed, llm_client=self,
            )

        prompt, info = self.env.reset(seed=episode_seed)
        episode_briefs: list[dict] = []
        total_reward = 0.0
        done = False
        # Engine coverage tracker — surfaces the engine mix in the rollout
        engine_counts: dict[str, int] = {}
        parse_failures = 0
        parse_fail_with_reward = 0

        while not done:
            raw_brief = self.generate(prompt)
            prompt, reward, done, truncated, info = self.env.step(raw_brief)
            total_reward += reward

            engine_t = info.get("engine_type", "PRICING")
            engine_counts[engine_t] = engine_counts.get(engine_t, 0) + 1
            if not info.get("parse_success", True):
                parse_failures += 1
                if reward > 0:
                    parse_fail_with_reward += 1

            episode_briefs.append({
                "tick": info.get("tick", 0),
                "engine_type": engine_t,
                "raw_response": raw_brief,
                "quality_score": info.get("quality_score", 0.0),
                "reward_delta": reward,
                "parse_success": info.get("parse_success", True),
            })

        # Extract final reward dict (set by env on termination)
        final_reward = info.get("final_reward", {})
        audit = info.get("constitutional_audit", {})

        return {
            "episode_num": self.episode_count,
            "scenario": self.scenario,
            "scenario_name": self.scenario.name,
            "wrr": final_reward.get("wrr", 0.0),
            "r1_pricing": final_reward.get("r1_pricing", 0.0),
            "r2_farmer": final_reward.get("r2_farmer", 0.0),
            "r3_trend": final_reward.get("r3_trend", 0.0),
            "brief_quality_score": final_reward.get("brief_quality_score", 0.0),
            "anti_hack_violations": final_reward.get("anti_hack_violations", 0),
            "episode_valid": final_reward.get("episode_valid", True),
            "constitutional_passed": audit.get("passed", True),
            "total_reward": total_reward,
            "briefs": episode_briefs,
            "final_reward": final_reward,
            "engine_counts": engine_counts,
            "parse_failures": parse_failures,
            "parse_fail_with_positive_reward": parse_fail_with_reward,
        }

    def _next_scenario(self) -> CurriculumScenario:
        if not self._rotate:
            return self.scenario
        s = self._scenarios[self._rotation_idx % len(self._scenarios)]
        self._rotation_idx += 1
        return s

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, num_episodes: int) -> str:
        """Run GRPO training loop. Returns final checkpoint path."""
        print(f"Starting GRPO training: {num_episodes} episodes, scenario={self.scenario.name}")

        for ep in range(num_episodes):
            episode_seed = self.rng.randint(0, 999999)
            result = self.run_episode(episode_seed)

            # Build EpisodeResult
            ep_result = EpisodeResult(
                episode_num=self.episode_count,
                scenario=self.scenario,
                wrr=result["wrr"],
                brief_quality_score=result["brief_quality_score"],
                anti_hack_violations=result["anti_hack_violations"],
                constitutional_passed=result["constitutional_passed"],
                episode_valid=result["episode_valid"],
            )

            # Record in curriculum manager
            promoted = self.curriculum_manager.record_episode(ep_result)

            # Add trajectory to buffer
            trajectory = Trajectory(
                episode_num=self.episode_count,
                scenario=self.scenario,
                wrr=result["wrr"],
                brief_quality_score=result["brief_quality_score"],
                constitutional_passed=result["constitutional_passed"],
                episode_valid=result["episode_valid"],
                briefs=result["briefs"],
                reward_engine_snapshot=result["final_reward"],
            )
            self.trajectory_buffer.add(trajectory)

            # Log to WandB
            status = self.curriculum_manager.get_status()
            wandb.log({
                "wrr": result["wrr"],
                "r1_pricing": result["r1_pricing"],
                "r2_farmer": result["r2_farmer"],
                "r3_trend": result["r3_trend"],
                "brief_quality_score": result["brief_quality_score"],
                "anti_hack_violations": result["anti_hack_violations"],
                "curriculum_level": self.curriculum_manager.current_level,
                "episode_num": self.episode_count,
                "episode_valid": result["episode_valid"],
                "constitutional_passed": result["constitutional_passed"],
                "episodes_in_level": self.curriculum_manager.episodes_in_level,
                "total_episodes": self.episode_count,
                "model_checkpoint": self.output_dir,
                "total_reward": result["total_reward"],
                # Surfaces engine coverage so a flat-zero R2/R3 column no
                # longer hides the fact that no FARMER/TREND episodes ran.
                "engine_PRICING_briefs": result.get("engine_counts", {}).get("PRICING", 0),
                "engine_FARMER_briefs": result.get("engine_counts", {}).get("FARMER", 0),
                "engine_TREND_briefs": result.get("engine_counts", {}).get("TREND", 0),
                "parse_failures": result.get("parse_failures", 0),
                "parse_fail_with_positive_reward": result.get("parse_fail_with_positive_reward", 0),
                "scenario_this_episode": result.get("scenario_name", self.scenario.name),
                **self.trajectory_buffer.get_stats(),
            })

            # Print progress
            if self.episode_count % 5 == 0:
                print(
                    f"Episode {self.episode_count}: WRR={result['wrr']:.3f} | "
                    f"Quality={result['brief_quality_score']:.3f} | "
                    f"Violations={result['anti_hack_violations']} | "
                    f"Level={self.curriculum_manager.current_level} "
                    f"({self.scenario.name})"
                )

            # Handle promotion
            if promoted:
                new_scenario = self.curriculum_manager.current_scenario
                print(
                    f"\n=== PROMOTED to level {self.curriculum_manager.current_level}: "
                    f"{new_scenario.name} ===\n"
                )
                removed = self.trajectory_buffer.clear_below_level(new_scenario)
                logger.info("Cleared %d old trajectories after promotion", removed)
                # Save checkpoint at promotion
                self._save_checkpoint(f"{self.output_dir}/promotion_level_{self.curriculum_manager.current_level}")
                break  # Caller decides next scenario

            # Periodic checkpoint save
            if self.episode_count > 0 and self.episode_count % self.save_every_n_episodes == 0:
                self._save_checkpoint(f"{self.output_dir}/episode_{self.episode_count}")

            self.episode_count += 1

        # Final save
        final_path = f"{self.output_dir}/final"
        self._save_checkpoint(final_path)
        return final_path

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: str) -> None:
        """Save checkpoint with Unsloth merged save, then verify."""
        print(f"Saving checkpoint to {path}")
        self.model.save_pretrained_merged(
            path, self.tokenizer, save_method="merged_16bit",
        )

        # Run 3-episode verification
        pre_save_wrr = self._last_save_wrr
        verification_wrrs: list[float] = []

        for i in range(3):
            ver_seed = 900000 + self.curriculum_manager.current_level * 100 + i
            result = self.run_episode(ver_seed)
            verification_wrrs.append(result["wrr"])

        avg_verification_wrr = sum(verification_wrrs) / len(verification_wrrs)
        self._last_save_wrr = avg_verification_wrr

        if pre_save_wrr > 0 and (pre_save_wrr - avg_verification_wrr) > SAVE_WRR_DEGRADATION_TOLERANCE:
            print(
                f"WARNING: Checkpoint quality degraded — "
                f"pre-save WRR {pre_save_wrr:.3f} → post-save {avg_verification_wrr:.3f} "
                f"(>{SAVE_WRR_DEGRADATION_TOLERANCE:.0%} drop)"
            )
            logger.warning(
                "Checkpoint degradation: %.3f → %.3f at %s",
                pre_save_wrr, avg_verification_wrr, path,
            )
        else:
            print(f"Checkpoint verified: WRR={avg_verification_wrr:.3f}")


def run_grpo(
    checkpoint_dir: str,
    output_dir: str,
    scenario: CurriculumScenario | None = None,
    num_episodes: int = 100,
    seed: int = 42,
    wandb_run_name: str | None = None,
    scenarios: list[CurriculumScenario] | None = None,
    rotate: bool = False,
) -> str:
    """Run GRPO training. Returns checkpoint path.

    Pass ``scenarios=[STABLE_WEEK, FARMER_WEEK, TREND_WEEK]`` + ``rotate=True``
    to rotate scenarios per episode within a single GRPO run. This is the
    recommended path for short Kaggle runs where the curriculum manager
    won't have time to promote.
    """
    if scenarios is not None and rotate:
        run_name = wandb_run_name or (
            "grpo_rotate_" + "+".join(s.name for s in scenarios) + f"_seed{seed}"
        )
        wandb.init(project="freshprice-ai", name=run_name, config={
            "scenarios": [s.name for s in scenarios],
            "rotate": True,
            "num_episodes": num_episodes,
            "seed": seed,
            "checkpoint": checkpoint_dir,
        })
        trainer = FreshPriceGRPOTrainer(
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            scenarios=scenarios,
            rotate=True,
            seed=seed,
        )
    else:
        if scenario is None:
            scenario = CurriculumScenario.STABLE_WEEK
        run_name = wandb_run_name or f"grpo_{scenario.name}_seed{seed}"
        wandb.init(project="freshprice-ai", name=run_name, config={
            "scenario": scenario.name,
            "num_episodes": num_episodes,
            "seed": seed,
            "checkpoint": checkpoint_dir,
        })
        trainer = FreshPriceGRPOTrainer(
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            scenario=scenario,
            seed=seed,
        )

    final_path = trainer.train(num_episodes)
    wandb.finish()
    return final_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run GRPO training")
    parser.add_argument("--checkpoint", required=True, help="Path to SFT checkpoint")
    parser.add_argument("--output-dir", required=True, help="Output directory for GRPO checkpoints")
    parser.add_argument("--scenario", default="STABLE_WEEK", help="Curriculum scenario name")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    final = run_grpo(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        scenario=CurriculumScenario[args.scenario],
        num_episodes=args.episodes,
        seed=args.seed,
        wandb_run_name=args.wandb_name,
    )
    print(f"GRPO complete. Final checkpoint: {final}")
