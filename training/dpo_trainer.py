"""DPO fine-tuning on preference pairs from TrajectoryBuffer.

Self-improvement phase — runs after GRPO has collected enough trajectories.
High-regret pairs are already duplicated 3x in the pair list by TrajectoryBuffer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from datasets import Dataset

from training.trajectory_buffer import DPOPair

logger = logging.getLogger(__name__)

# Minimum number of DPO pairs required to run training
_MIN_DPO_PAIRS: int = 10

# Post-DPO quality must be within this fraction of pre-DPO
_DPO_QUALITY_TOLERANCE: float = 0.97


def run_dpo(
    checkpoint_dir: str,
    output_dir: str,
    dpo_pairs: list[DPOPair],
    beta: float = 0.1,
    learning_rate: float = 5e-6,
    num_epochs: int = 1,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 4096,
    seed: int = 42,
    report_to: str = "wandb",
    skip_verification: bool = False,
) -> str:
    """Run one DPO fine-tuning cycle. Returns path to saved checkpoint."""
    from unsloth import FastLanguageModel
    from trl import DPOTrainer, DPOConfig

    # 1. Validate input
    if len(dpo_pairs) < _MIN_DPO_PAIRS:
        print(
            f"WARNING: Too few DPO pairs ({len(dpo_pairs)}). "
            f"Minimum {_MIN_DPO_PAIRS} required. Skipping DPO."
        )
        return checkpoint_dir

    # 2. Load model and tokenizer with Unsloth
    print(f"Loading model from {checkpoint_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_dir,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    from freshprice_env._gen_utils import quiet_generation_config
    quiet_generation_config(model)

    # 3. Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
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

    # 4. Build HuggingFace Dataset from DPO pairs
    dataset = _build_dpo_dataset(dpo_pairs)

    # Print statistics
    total = len(dpo_pairs)
    high_regret = sum(1 for p in dpo_pairs if p.regret_score > 0.7)
    engine_counts: dict[str, int] = {}
    for p in dpo_pairs:
        engine_counts[p.engine_type] = engine_counts.get(p.engine_type, 0) + 1

    print(f"DPO dataset: {total} pairs (includes 3x duplication for high-regret)")
    print(f"  High-regret pairs (>0.7): {high_regret}")
    print(f"  By engine: {engine_counts}")

    # 5. Configure DPOTrainer
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        beta=beta,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        seed=seed,
        report_to=report_to,
        remove_unused_columns=False,
        max_length=max_seq_length,
        max_prompt_length=max_seq_length // 2,
    )

    trainer = DPOTrainer(
        model=model,
        # ref_model=None → TRL uses the initial model weights as the implicit
        # reference. More memory-efficient on a single GPU than loading a
        # separate copy. DPOTrainer supports this natively.
        ref_model=None,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # 6. Train
    print("Starting DPO training...")
    trainer_stats = trainer.train()
    print(f"DPO training complete.")
    print(f"  Loss: {trainer_stats.training_loss:.4f}")
    print(f"  Runtime: {trainer_stats.metrics.get('train_runtime', 0):.1f}s")

    # 7. Save with Unsloth merged save — NEVER naive save_pretrained
    print(f"Saving merged checkpoint to {output_dir}")
    model.save_pretrained_merged(
        output_dir,
        tokenizer,
        save_method="merged_16bit",
    )

    # 8. Post-save verification (optional — slow, skips on small GPUs)
    if skip_verification:
        print("Skipping post-DPO verification (skip_verification=True).")
    else:
        try:
            verified = _verify_dpo_checkpoint(
                checkpoint_dir, output_dir, tokenizer, seed,
            )
            if not verified:
                print("WARNING: Post-DPO quality dropped below tolerance — caller should consider reverting")
                logger.warning("DPO checkpoint verification failed for %s", output_dir)
        except Exception as ve:
            print(f"WARNING: Post-DPO verification crashed ({type(ve).__name__}: {ve}). Continuing.")

    # 9. Return checkpoint path
    return output_dir


def _build_dpo_dataset(dpo_pairs: list[DPOPair]) -> Dataset:
    """Convert DPOPair list to a HuggingFace Dataset for DPOTrainer."""
    rows: list[dict] = []
    for pair in dpo_pairs:
        rows.append({
            "prompt": pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
        })
    return Dataset.from_list(rows)


def _verify_dpo_checkpoint(
    pre_dpo_dir: str,
    post_dpo_dir: str,
    tokenizer,
    seed: int,
    n_verification_episodes: int = 3,
) -> bool:
    """Compare pre-DPO and post-DPO checkpoint quality.

    Returns True if post_dpo_mean_wrr >= pre_dpo_mean_wrr × 0.97.
    """
    from freshprice_env.enums import CurriculumScenario
    from freshprice_env.freshprice_env import FreshPriceEnv
    from unsloth import FastLanguageModel

    # Check for cached baseline
    baseline_path = Path(pre_dpo_dir) / "verification_baseline.json"
    pre_dpo_wrrs: list[float] = []

    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        pre_dpo_wrrs = baseline.get("wrrs", [])
        logger.info("Loaded pre-DPO baseline from %s", baseline_path)
    else:
        # Run episodes against pre-DPO model
        pre_dpo_wrrs = _run_verification_episodes(
            pre_dpo_dir, n_verification_episodes, seed,
        )

    # Run episodes against post-DPO model
    post_dpo_wrrs = _run_verification_episodes(
        post_dpo_dir, n_verification_episodes, seed,
    )

    pre_mean = sum(pre_dpo_wrrs) / len(pre_dpo_wrrs) if pre_dpo_wrrs else 0.0
    post_mean = sum(post_dpo_wrrs) / len(post_dpo_wrrs) if post_dpo_wrrs else 0.0
    delta = post_mean - pre_mean
    pct = (delta / pre_mean * 100) if pre_mean > 0 else 0.0

    passed = post_mean >= pre_mean * _DPO_QUALITY_TOLERANCE

    print(f"\n  DPO Verification:")
    print(f"    Pre-DPO WRR:  {pre_mean:.4f}")
    print(f"    Post-DPO WRR: {post_mean:.4f}")
    print(f"    Delta:        {delta:+.4f} ({pct:+.1f}%)")
    print(f"    Result:       {'PASS' if passed else 'FAIL'}\n")

    # Cache post-DPO results for future runs
    baseline_out = Path(post_dpo_dir) / "verification_baseline.json"
    baseline_out.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_out, "w") as f:
        json.dump({"wrrs": post_dpo_wrrs, "mean_wrr": post_mean}, f)

    return passed


def _run_verification_episodes(
    model_dir: str,
    n_episodes: int,
    seed: int,
) -> list[float]:
    """Run n_episodes with a model and return WRR list."""
    from unsloth import FastLanguageModel
    from freshprice_env.enums import CurriculumScenario
    from freshprice_env.freshprice_env import FreshPriceEnv

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    from freshprice_env._gen_utils import quiet_generation_config
    quiet_generation_config(model)

    wrrs: list[float] = []

    for i in range(n_episodes):
        ep_seed = seed + 800000 + i
        env = FreshPriceEnv(
            scenario=CurriculumScenario.STABLE_WEEK,
            seed=ep_seed,
        )

        prompt, _ = env.reset(seed=ep_seed)
        done = False

        while not done:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=4096,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=1.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            prompt, reward, done, truncated, info = env.step(response)

        final_reward = info.get("final_reward", {})
        wrrs.append(final_reward.get("wrr", 0.0))

    return wrrs


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="DPO fine-tuning")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("DPO trainer requires dpo_pairs from TrajectoryBuffer.")
    print("Run via training/train.py which manages the full pipeline.")
    print("Standalone DPO not supported — pairs must come from live trajectories.")
