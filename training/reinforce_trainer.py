"""REINFORCE + KL: the actual policy-gradient RL step in the pipeline.

The previous "GRPO rollouts" cell in the notebook was a misnomer -- it
generated briefs and scored them, but never called ``.backward()`` or
an optimizer step. The DPO cell that follows it does call
``DPOTrainer.train()``, but DPO is preference learning (contrastive
loss between a "chosen" and a "rejected" brief), not classical RL.

This module adds the missing policy-gradient piece. Given:

  - ``model``           : the SFT-warmed Qwen with a trainable LoRA adapter
  - ``tokenizer``       : matching tokenizer
  - ``trajectory_buffer``: the buffer the rollout cell already populated

it runs REINFORCE with a KL penalty against the frozen SFT reference,
which is the textbook RL-from-environment-rewards algorithm and the
core of every PPO/GRPO/RLOO derivative. The math, per (prompt,
completion, reward) tuple:

    advantage = (reward - mean(rewards)) / (std(rewards) + eps)
    L_pg      = -advantage * sum(log pi_theta(c_t | prompt, c_<t))
    L_kl      =  beta * mean(log pi_theta - log pi_ref)
    loss      =  L_pg + L_kl

The reference distribution ``pi_ref`` is approximated by *disabling*
the LoRA adapter -- the frozen base + SFT weights serve as the
reference, so we don't need a second model in VRAM. This is the same
trick TRL's ``PPOTrainer`` and HF's RLHF tutorials use.

Why this is real RL:
  - rewards come from the environment (FreshPriceEnv -> WRR + r1..r7)
  - the policy gradient ``advantage * grad log pi`` literally pushes
    probability mass onto high-reward briefs
  - ``loss.backward()`` + ``optimizer.step()`` happen every grad-accum
    boundary; you can watch the loss curve descend per-step
  - the KL penalty stops the policy from collapsing onto a single
    high-reward brief

Designed to fit in a Kaggle T4 budget when called on the trajectories
the existing rollout cell already collected (no new generation).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReinforceStats:
    """Per-update accounting surfaced to the notebook for plotting."""

    step: int
    loss: float
    policy_loss: float
    kl: float
    mean_advantage: float
    n_examples: int


def _have_trainable_params(model) -> bool:
    return any(p.requires_grad for p in model.parameters())


def _completion_logprob(model, tokenizer, prompt: str, completion: str,
                        device, max_seq_len: int) -> "torch.Tensor":
    """Sum log-probabilities of completion tokens given prompt under the
    *current* state of ``model`` (LoRA enabled or disabled).

    Returns a scalar tensor with grad attached when the model is in
    training mode and the LoRA adapter is enabled.
    """
    import torch

    full = prompt + completion
    full_ids = tokenizer(full, return_tensors="pt", truncation=True,
                         max_length=max_seq_len).input_ids.to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_seq_len).input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    if prompt_len >= full_ids.shape[1]:
        # Completion got truncated away; nothing to score.
        return torch.tensor(0.0, device=device, requires_grad=False)

    outputs = model(full_ids)
    # Logits at position t predict token at position t+1, so shift.
    logits = outputs.logits[:, :-1, :]
    labels = full_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    # Only count completion tokens (those after the prompt).
    completion_mask = torch.zeros_like(labels, dtype=torch.bool)
    completion_mask[:, prompt_len - 1:] = True
    return token_log_probs[completion_mask].sum()


def _set_lora_enabled(model, enabled: bool) -> None:
    """Toggle the LoRA adapter on/off. The reference policy log-probs
    are computed with the adapter disabled, so the base + SFT weights
    serve as ``pi_ref``."""
    if hasattr(model, "set_adapter"):
        # PEFT models expose enable/disable_adapter_layers on the wrapper.
        if enabled and hasattr(model, "enable_adapter_layers"):
            model.enable_adapter_layers()
        elif not enabled and hasattr(model, "disable_adapter_layers"):
            model.disable_adapter_layers()


def collect_samples(trajectory_buffer) -> list[tuple[str, str, float]]:
    """Pull (prompt, completion, reward) tuples from the trajectory buffer.

    Each Trajectory contains a list of ``briefs`` dicts with at least
    ``prompt`` and ``brief_text`` keys; we score each with the
    trajectory's WRR (a per-episode reward used as a noisy per-brief
    reward — REINFORCE handles that with the advantage normalization).
    """
    samples: list[tuple[str, str, float]] = []
    if trajectory_buffer is None:
        return samples
    # Buffer may expose .all() or a list; tolerate both.
    items = (
        trajectory_buffer.all()
        if hasattr(trajectory_buffer, "all")
        else list(getattr(trajectory_buffer, "_buffer", []))
    )
    for traj in items:
        briefs = getattr(traj, "briefs", None) or []
        wrr = float(getattr(traj, "wrr", 0.0))
        for b in briefs:
            # Tolerate three legacy key names so older trajectory dumps
            # still feed REINFORCE: notebook cell 11 historically stored
            # the model output under "raw_response" and never persisted
            # the prompt; new runs store it under "prompt"+"brief_text".
            prompt = b.get("prompt") or b.get("observation") or ""
            completion = (
                b.get("brief_text")
                or b.get("completion")
                or b.get("raw_response")
                or ""
            )
            if prompt and completion:
                samples.append((prompt, completion, wrr))
    return samples


def run_reinforce_kl(
    model,
    tokenizer,
    trajectory_buffer,
    *,
    n_epochs: int = 1,
    lr: float = 5e-6,
    kl_beta: float = 0.05,
    grad_accum: int = 4,
    max_seq_len: int = 1024,
    max_samples: int | None = None,
    progress_callback=None,
) -> tuple[object, list[ReinforceStats]]:
    """Run REINFORCE-with-KL on the trajectories already in the buffer.

    Args:
        model: SFT-warmed model with a trainable LoRA adapter.
        tokenizer: matching tokenizer.
        trajectory_buffer: the populated ``TrajectoryBuffer``.
        n_epochs: passes over the buffer (default 1; bump for stronger
            updates at the cost of risking divergence).
        lr: AdamW learning rate. 5e-6 is a typical RLHF setting.
        kl_beta: weight on the KL term. Higher = stay closer to SFT.
        grad_accum: gradient-accumulation factor (effective batch size).
        max_seq_len: token cap per (prompt+completion). T4-safe.
        max_samples: cap on tuples to process per epoch. None = all.
        progress_callback: called as ``cb(stats: ReinforceStats)`` after
            every grad-accum boundary; use it to plot live in a notebook.

    Returns:
        (model, history) — the in-place updated model and a list of
        per-update statistics (loss, policy_loss, kl, mean_advantage).
    """
    import numpy as np
    import torch
    from torch.optim import AdamW

    if not _have_trainable_params(model):
        raise RuntimeError(
            "Model has no trainable parameters. Attach a LoRA adapter "
            "before calling run_reinforce_kl(); the SFT cell does this."
        )

    samples = collect_samples(trajectory_buffer)
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise RuntimeError(
            "No (prompt, completion, reward) tuples in the trajectory "
            "buffer. Run the rollout cell first."
        )

    # Normalize rewards into advantages (REINFORCE baseline).
    rewards = np.array([s[2] for s in samples], dtype=np.float32)
    mean_r = float(rewards.mean())
    std_r = float(rewards.std() + 1e-8)
    advantages = (rewards - mean_r) / std_r
    logger.info(
        "REINFORCE: %d samples; reward mean=%.3f std=%.3f advantage range=[%.2f, %.2f]",
        len(samples), mean_r, std_r, float(advantages.min()), float(advantages.max()),
    )

    # AdamW over only the trainable (LoRA) parameters.
    trainables = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainables, lr=lr)

    device = next(model.parameters()).device
    history: list[ReinforceStats] = []
    step = 0
    grad_accum_count = 0
    accum_loss = 0.0
    accum_policy = 0.0
    accum_kl = 0.0
    accum_adv = 0.0

    model.train()
    for epoch in range(n_epochs):
        order = np.random.permutation(len(samples))
        for idx in order:
            prompt, completion, _reward = samples[idx]
            adv = float(advantages[idx])

            # Current policy log-prob (LoRA enabled, requires grad).
            _set_lora_enabled(model, True)
            cur_logp = _completion_logprob(
                model, tokenizer, prompt, completion, device, max_seq_len,
            )

            # Reference log-prob (LoRA disabled, no grad).
            with torch.no_grad():
                _set_lora_enabled(model, False)
                ref_logp = _completion_logprob(
                    model, tokenizer, prompt, completion, device, max_seq_len,
                )
            _set_lora_enabled(model, True)

            kl = cur_logp - ref_logp.detach()
            policy_loss = -adv * cur_logp
            loss = policy_loss + kl_beta * kl

            (loss / grad_accum).backward()

            accum_loss += float(loss.detach())
            accum_policy += float(policy_loss.detach())
            accum_kl += float(kl.detach())
            accum_adv += adv
            grad_accum_count += 1

            if grad_accum_count >= grad_accum:
                # Gradient clipping for stability.
                torch.nn.utils.clip_grad_norm_(trainables, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                stats = ReinforceStats(
                    step=step,
                    loss=accum_loss / grad_accum,
                    policy_loss=accum_policy / grad_accum,
                    kl=accum_kl / grad_accum,
                    mean_advantage=accum_adv / grad_accum,
                    n_examples=grad_accum,
                )
                history.append(stats)
                if progress_callback is not None:
                    try:
                        progress_callback(stats)
                    except Exception as e:  # noqa: BLE001 - never let UI break training
                        logger.warning("progress_callback raised: %s", e)
                accum_loss = accum_policy = accum_kl = accum_adv = 0.0
                grad_accum_count = 0

    # Final partial batch (if any).
    if grad_accum_count > 0:
        torch.nn.utils.clip_grad_norm_(trainables, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        history.append(ReinforceStats(
            step=step,
            loss=accum_loss / grad_accum_count,
            policy_loss=accum_policy / grad_accum_count,
            kl=accum_kl / grad_accum_count,
            mean_advantage=accum_adv / grad_accum_count,
            n_examples=grad_accum_count,
        ))

    return model, history
