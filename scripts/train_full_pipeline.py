"""End-to-end training orchestrator (VS Code / HF GPU box).

Replaces the Kaggle notebook for users running locally or on a Hugging
Face dev box. Reads HF_TOKEN + HF_REPO_ID from environment / .env and
walks through:

    1. (opt) generate SFT data
    2. SFT       -> checkpoints/sft_v1
    3. rollouts  -> populates trajectory buffer in memory
    4. REINFORCE -> checkpoints/reinforce_v1
    5. DPO       -> checkpoints/dpo_v1
    6. inference_comparison  -> data/comparison_results.json
    7. (opt) push checkpoints/dpo_v1 to HF Hub

Usage:

    # 1. Put HF creds into .env (or export them)
    cp .env.example .env  &&  $EDITOR .env

    # 2. Install deps (GPU box)
    pip install -r requirements.txt -r requirements_training.txt

    # 3. Run the pipeline
    python scripts/train_full_pipeline.py
       --model-id Qwen/Qwen2.5-7B-Instruct
       --output-root checkpoints
       --push-to-hub                  # optional, uses $HF_REPO_ID

Each stage is idempotent: re-run the script with --skip-sft if you've
already trained SFT and just want to redo RL.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_full_pipeline")


def _load_env_file(path: str | Path = ".env") -> None:
    """Tiny .env loader (no python-dotenv dep). KEY=VALUE per line."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


def _require_hf_token() -> str:
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok or tok == "hf_your_token_here":
        raise SystemExit(
            "HF_TOKEN is not set. Either:\n"
            "   export HF_TOKEN=hf_xxx\n"
            "or fill it into .env (see .env.example)."
        )
    return tok


def stage_sft_generate(args) -> None:
    if args.skip_sft_generate:
        logger.info("[stage 1/7] SFT-generate skipped via flag")
        return
    logger.info("[stage 1/7] generating SFT data ...")
    from training.generate_sft_data import main as gen_main
    sys.argv = ["generate_sft_data", "--output-dir", "training/sft_data"]
    gen_main()


def stage_sft_train(args) -> str:
    out_dir = f"{args.output_root}/sft_v1"
    if args.skip_sft:
        logger.info("[stage 2/7] SFT skipped, using existing %s", out_dir)
        return out_dir
    logger.info("[stage 2/7] running SFT -> %s", out_dir)
    from training.sft_trainer import run_sft_training
    run_sft_training(
        model_id=args.model_id,
        data_dir="training/sft_data",
        output_dir=out_dir,
        num_epochs=args.sft_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.sft_lr,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )
    return out_dir


def stage_rollouts(args, sft_dir: str):
    logger.info("[stage 3/7] collecting rollouts from %s", sft_dir)
    from training.grpo_trainer import GRPOTrainer
    from freshprice_env.enums import CurriculumScenario

    trainer = GRPOTrainer(
        checkpoint_dir=sft_dir,
        scenarios=[CurriculumScenario.STABLE_WEEK,
                   CurriculumScenario.BUSY_WEEKEND,
                   CurriculumScenario.FARMER_WEEK,
                   CurriculumScenario.TREND_WEEK,
                   CurriculumScenario.CRISIS_WEEK],
    )
    buffer = trainer.collect_rollouts(
        n_episodes=args.rollout_episodes,
        max_briefs_per_episode=args.rollout_max_briefs,
    )
    logger.info("rollout buffer size: %d trajectories", len(buffer))
    return trainer.model, trainer.tokenizer, buffer


def stage_reinforce(args, model, tokenizer, buffer) -> str:
    if args.skip_reinforce:
        logger.info("[stage 4/7] REINFORCE skipped via flag")
        return f"{args.output_root}/reinforce_v1"
    logger.info("[stage 4/7] REINFORCE+KL ...")
    from training.reinforce_trainer import run_reinforce_kl
    model, history = run_reinforce_kl(
        model, tokenizer, buffer,
        n_epochs=args.rl_epochs,
        lr=args.rl_lr,
        kl_beta=args.kl_beta,
        grad_accum=args.grad_accum,
        max_seq_len=args.rl_max_seq,
        max_samples=args.rl_max_samples,
        batch_size=args.rl_batch_size,
        use_value_baseline=args.use_value_baseline,
        progress_callback=lambda s: logger.info(
            "step=%d loss=%+.4f kl=%+.4f", s.step, s.loss, s.kl,
        ),
    )
    out_dir = f"{args.output_root}/reinforce_v1"
    model.save_pretrained_merged(out_dir, tokenizer, save_method="merged_16bit")
    logger.info("saved RL-updated model -> %s (%d steps)", out_dir, len(history))
    return out_dir


def stage_dpo(args, sft_dir: str, rl_dir: str) -> str:
    if args.skip_dpo:
        logger.info("[stage 5/7] DPO skipped via flag")
        return rl_dir
    logger.info("[stage 5/7] DPO from %s -> %s_v1", rl_dir, "dpo")
    out_dir = f"{args.output_root}/dpo_v1"
    from training.dpo_trainer import run_dpo_training
    run_dpo_training(
        checkpoint_dir=rl_dir,
        output_dir=out_dir,
        num_epochs=args.dpo_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.dpo_lr,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )
    return out_dir


def stage_comparison(args, sft_dir: str, rl_dir: str) -> None:
    logger.info("[stage 6/7] inference_comparison ...")
    from inference_comparison import main as cmp_main
    cmp_argv = [
        "--sft-path", sft_dir,
        "--rl-path", rl_dir,
        "--scenarios", "STABLE_WEEK", "BUSY_WEEKEND", "CRISIS_WEEK",
        "--episodes-per-scenario", "3",
    ]
    if args.use_external_grader:
        cmp_argv.append("--use-external-grader")
    cmp_main(cmp_argv)


def stage_push_to_hub(args, final_dir: str) -> None:
    if not args.push_to_hub:
        logger.info("[stage 7/7] push to HF Hub skipped (no --push-to-hub)")
        return
    repo_id = os.environ.get("HF_REPO_ID", "").strip()
    if not repo_id:
        raise SystemExit("HF_REPO_ID not set; cannot push.")
    token = _require_hf_token()
    logger.info("[stage 7/7] pushing %s -> %s", final_dir, repo_id)
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, private=args.private_repo)
    api.upload_folder(folder_path=final_dir, repo_id=repo_id,
                      commit_message="QStorePrice trained checkpoint")
    logger.info("pushed -> https://huggingface.co/%s", repo_id)


def main():
    _load_env_file()  # load .env into os.environ

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output-root", default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)

    # SFT
    p.add_argument("--sft-epochs", type=int, default=2)
    p.add_argument("--sft-lr", type=float, default=2e-4)
    p.add_argument("--skip-sft-generate", action="store_true")
    p.add_argument("--skip-sft", action="store_true")

    # Rollouts
    p.add_argument("--rollout-episodes", type=int, default=12)
    p.add_argument("--rollout-max-briefs", type=int, default=8)

    # RL
    p.add_argument("--skip-reinforce", action="store_true")
    p.add_argument("--rl-epochs", type=int, default=1)
    p.add_argument("--rl-lr", type=float, default=5e-6)
    p.add_argument("--kl-beta", type=float, default=0.05)
    p.add_argument("--rl-max-seq", type=int, default=1024)
    p.add_argument("--rl-max-samples", type=int, default=64)
    p.add_argument("--rl-batch-size", type=int, default=2)
    p.add_argument("--use-value-baseline", action="store_true", default=True)

    # DPO
    p.add_argument("--skip-dpo", action="store_true")
    p.add_argument("--dpo-epochs", type=int, default=1)
    p.add_argument("--dpo-lr", type=float, default=5e-7)

    # Comparison + push
    p.add_argument("--use-external-grader", action="store_true")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--private-repo", action="store_true")

    args = p.parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    _require_hf_token()  # fail fast if missing

    stage_sft_generate(args)
    sft_dir = stage_sft_train(args)
    model, tokenizer, buffer = stage_rollouts(args, sft_dir)
    rl_dir = stage_reinforce(args, model, tokenizer, buffer)
    final_dir = stage_dpo(args, sft_dir, rl_dir)
    stage_comparison(args, sft_dir, final_dir)
    stage_push_to_hub(args, final_dir)

    logger.info("=" * 70)
    logger.info("Pipeline complete.")
    logger.info("  SFT checkpoint   : %s", sft_dir)
    logger.info("  RL checkpoint    : %s", rl_dir)
    logger.info("  Final checkpoint : %s", final_dir)
    if args.push_to_hub:
        logger.info("  HF Hub           : https://huggingface.co/%s",
                    os.environ.get("HF_REPO_ID", ""))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
