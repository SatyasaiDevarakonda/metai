"""PPO with clipped surrogate objective (Roadmap #6).

A drop-in alternative to ``training.reinforce_trainer.run_reinforce_kl``
with the standard PPO clipped-surrogate update plus a KL anchor.

Why PPO over REINFORCE:

  - REINFORCE updates with ``-advantage * log pi``, which can take
    arbitrarily large policy steps when the advantage is large -- the
    classic "policy collapse" failure mode.
  - PPO clips the importance ratio ``r = pi_new / pi_old`` to
    ``[1-eps, 1+eps]`` so a single update can never move probability
    mass beyond ``eps`` for any sample. Combined with multiple epochs
    over the same rollout buffer, that gives both stability AND sample
    efficiency.

The math, per (prompt, completion, advantage) tuple:

    ratio        = exp(log pi_new(c|p) - log pi_old(c|p))
    surr1        = ratio * advantage
    surr2        = clip(ratio, 1-eps, 1+eps) * advantage
    L_clip       = -min(surr1, surr2).mean()
    L_value      = MSE(V(p), reward)         # if value head enabled
    L_kl         = beta * KL(pi_new || pi_ref)
    loss         = L_clip + 0.5 * L_value + L_kl

``pi_old`` is the policy at rollout time (the current state of the
LoRA adapter when this function is first called) -- we snapshot its
log-probs once, then reuse them across all PPO epochs.

``pi_ref`` is the SFT reference (LoRA disabled), same trick as
REINFORCE+KL: no second model in VRAM.

This module reuses the batched encoder + value head from
``training.reinforce_trainer``, so the speed knobs and variance
reduction are inherited.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PPOStats:
    """Per-update accounting surfaced to the notebook for plotting."""

    step: int
    loss: float
    policy_loss: float
    kl: float
    clipped_frac: float
    mean_advantage: float
    n_examples: int
    value_loss: float = 0.0


def run_ppo(
    model,
    tokenizer,
    trajectory_buffer,
    *,
    n_epochs: int = 2,
    lr: float = 5e-6,
    kl_beta: float = 0.05,
    clip_eps: float = 0.2,
    grad_accum: int = 4,
    max_seq_len: int = 1024,
    max_samples: int | None = None,
    batch_size: int = 1,
    use_value_baseline: bool = True,
    value_lr: float = 1e-4,
    value_loss_coef: float = 0.5,
    progress_callback=None,
) -> tuple[object, list[PPOStats]]:
    """Run PPO with clipping + KL anchor on the existing rollout buffer.

    Args:
        n_epochs: PPO traditionally runs multiple epochs per rollout
            batch; 2-4 is typical. REINFORCE uses 1.
        clip_eps: PPO clipping range. 0.2 is the canonical value.
        value_loss_coef: weight on the V-head MSE term.
        Other args mirror ``run_reinforce_kl``.
    """
    import numpy as np
    import torch
    from torch.optim import AdamW

    from training.reinforce_trainer import (
        _ValueHead, _encode_batch, _completion_logprobs_batched,
        _pooled_hidden, _set_lora_enabled, _have_trainable_params,
        collect_samples,
    )

    if not _have_trainable_params(model):
        raise RuntimeError(
            "Model has no trainable parameters. Attach a LoRA adapter "
            "before calling run_ppo()."
        )

    samples = collect_samples(trajectory_buffer)
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise RuntimeError(
            "No (prompt, completion, reward) tuples in the trajectory "
            "buffer. Run the rollout cell first."
        )

    rewards = np.array([s[2] for s in samples], dtype=np.float32)
    mean_r = float(rewards.mean())
    std_r = float(rewards.std() + 1e-8)

    trainables = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainables, lr=lr)
    device = next(model.parameters()).device

    value_head = _ValueHead() if use_value_baseline else None
    if use_value_baseline:
        hidden_size = int(getattr(model.config, "hidden_size",
                                  getattr(model.config, "n_embd", 4096)))
        value_head.build(hidden_size, device, lr=value_lr)
        logger.info("PPO: value-head baseline ON (hidden_size=%d)", hidden_size)

    bs = max(1, int(batch_size))

    # ---------------------------------------------------------------
    # Stage 1: snapshot pi_old log-probs and reference log-probs ONCE
    # over the full buffer. They don't move during PPO epochs.
    # ---------------------------------------------------------------
    logger.info("PPO: snapshotting pi_old + pi_ref log-probs over %d samples",
                len(samples))
    old_logps_all = []
    ref_logps_all = []

    model.eval()
    for batch_start in range(0, len(samples), bs):
        chunk = samples[batch_start:batch_start + bs]
        prompts = [s[0] for s in chunk]
        completions = [s[1] for s in chunk]
        full_ids, attn, prompt_lens, _ = _encode_batch(
            tokenizer, prompts, completions, device, max_seq_len,
        )

        # pi_old (LoRA enabled, current state)
        _set_lora_enabled(model, True)
        with torch.inference_mode():
            old_logp = _completion_logprobs_batched(
                model, full_ids, attn, prompt_lens,
            )
        old_logps_all.append(old_logp.detach().cpu())

        # pi_ref (LoRA disabled, SFT base)
        _set_lora_enabled(model, False)
        with torch.inference_mode():
            ref_logp = _completion_logprobs_batched(
                model, full_ids, attn, prompt_lens,
            )
        ref_logps_all.append(ref_logp.detach().cpu())
        _set_lora_enabled(model, True)

    old_logps = torch.cat(old_logps_all)  # (N,)
    ref_logps = torch.cat(ref_logps_all)  # (N,)

    history: list[PPOStats] = []
    step = 0
    grad_accum_count = 0
    accum = {"loss": 0.0, "policy": 0.0, "kl": 0.0, "adv": 0.0,
             "n": 0, "vloss": 0.0, "clipped": 0.0, "clipped_n": 0}

    model.train()

    # ---------------------------------------------------------------
    # Stage 2: PPO epochs over the same buffer with clipped surrogate.
    # ---------------------------------------------------------------
    for epoch in range(n_epochs):
        order = list(np.random.permutation(len(samples)))

        for batch_start in range(0, len(order), bs):
            batch_idx = order[batch_start:batch_start + bs]
            prompts = [samples[i][0] for i in batch_idx]
            completions = [samples[i][1] for i in batch_idx]
            batch_rewards = torch.as_tensor(
                [rewards[i] for i in batch_idx],
                device=device, dtype=torch.float32,
            )
            batch_old_logp = old_logps[batch_idx].to(device)
            batch_ref_logp = ref_logps[batch_idx].to(device)

            full_ids, attn, prompt_lens, K = _encode_batch(
                tokenizer, prompts, completions, device, max_seq_len,
            )

            # Current logprobs (with grad).
            cur_logp = _completion_logprobs_batched(
                model, full_ids, attn, prompt_lens,
            )

            # Advantage.
            if use_value_baseline:
                with torch.no_grad():
                    pooled = _pooled_hidden(model, full_ids, attn)
                values = value_head.predict(pooled.detach())
                adv = (batch_rewards - values.detach()) / std_r
                vloss = value_head.step(pooled, batch_rewards)
            else:
                adv = (batch_rewards - mean_r) / std_r
                vloss = 0.0

            # PPO clipped surrogate.
            ratio = torch.exp(cur_logp - batch_old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL anchor against pi_ref (not pi_old) so we don't drift
            # away from the SFT prior across many PPO epochs.
            kl_loss = (cur_logp - batch_ref_logp).mean()

            loss = policy_loss + kl_beta * kl_loss
            if use_value_baseline:
                # The V-head was already updated with its own optimizer
                # step inside .step(). We log vloss separately; not added
                # to the LM loss to keep gradients separated.
                pass

            # Defensive: skip zero-grad batches (empty completions after
            # truncation). Same edge case REINFORCE handles.
            if not loss.requires_grad:
                logger.warning(
                    "Skipping zero-grad PPO batch (all %d completions empty).",
                    K,
                )
                continue

            (loss / grad_accum).backward()

            # Track clipped fraction for logging.
            with torch.no_grad():
                clipped = ((ratio < 1.0 - clip_eps) |
                           (ratio > 1.0 + clip_eps)).float().sum().item()

            accum["loss"] += float(loss.detach())
            accum["policy"] += float(policy_loss.detach())
            accum["kl"] += float(kl_loss.detach())
            accum["adv"] += float(adv.mean().detach())
            accum["vloss"] += vloss
            accum["n"] += K
            accum["clipped"] += clipped
            accum["clipped_n"] += K
            grad_accum_count += 1

            if grad_accum_count >= grad_accum:
                torch.nn.utils.clip_grad_norm_(trainables, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                stats = PPOStats(
                    step=step,
                    loss=accum["loss"] / grad_accum,
                    policy_loss=accum["policy"] / grad_accum,
                    kl=accum["kl"] / grad_accum,
                    clipped_frac=(accum["clipped"] / max(1, accum["clipped_n"])),
                    mean_advantage=accum["adv"] / grad_accum,
                    n_examples=accum["n"],
                    value_loss=accum["vloss"] / grad_accum,
                )
                history.append(stats)
                if progress_callback is not None:
                    try:
                        progress_callback(stats)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("progress_callback raised: %s", e)
                accum = {"loss": 0.0, "policy": 0.0, "kl": 0.0, "adv": 0.0,
                         "n": 0, "vloss": 0.0, "clipped": 0.0, "clipped_n": 0}
                grad_accum_count = 0

    # Final partial batch.
    if grad_accum_count > 0:
        import torch as _t
        _t.nn.utils.clip_grad_norm_(trainables, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        history.append(PPOStats(
            step=step,
            loss=accum["loss"] / grad_accum_count,
            policy_loss=accum["policy"] / grad_accum_count,
            kl=accum["kl"] / grad_accum_count,
            clipped_frac=(accum["clipped"] / max(1, accum["clipped_n"])),
            mean_advantage=accum["adv"] / grad_accum_count,
            n_examples=accum["n"],
            value_loss=accum["vloss"] / grad_accum_count,
        ))

    return model, history


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import argparse
    parser = argparse.ArgumentParser(description="Run PPO on a saved buffer")
    parser.add_argument("--buffer", required=True,
                        help="path to a pickled TrajectoryBuffer")
    parser.add_argument("--checkpoint", required=True,
                        help="SFT-warmed model dir to fine-tune")
    parser.add_argument("--output", required=True,
                        help="where to save the post-PPO checkpoint")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    import pickle
    from unsloth import FastLanguageModel
    from freshprice_env._gen_utils import quiet_generation_config

    with open(args.buffer, "rb") as f:
        buf = pickle.load(f)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint, max_seq_length=4096,
        dtype=None, load_in_4bit=True,
    )
    quiet_generation_config(model)

    model, history = run_ppo(
        model, tokenizer, buf,
        n_epochs=args.epochs,
        clip_eps=args.clip_eps,
        batch_size=args.batch_size,
        progress_callback=lambda s: print(
            f"step={s.step:3d} loss={s.loss:+.4f} clipped={s.clipped_frac:.2%} "
            f"kl={s.kl:+.4f}"
        ),
    )

    model.save_pretrained_merged(args.output, tokenizer,
                                 save_method="merged_16bit")
    print(f"saved {args.output} ({len(history)} steps)")
