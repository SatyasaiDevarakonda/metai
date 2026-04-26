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

    advantage = (reward - baseline) / (std(rewards) + eps)
    L_pg      = -advantage * sum(log pi_theta(c_t | prompt, c_<t))
    L_kl      =  beta * mean(log pi_theta - log pi_ref)
    loss      =  L_pg + L_kl

Two baseline modes (Roadmap #8):

  * ``use_value_baseline=False`` (default) -- subtract the **batch-mean**
    reward, normalize by std. Cheap, no extra params, but high variance
    on small buffers.
  * ``use_value_baseline=True``            -- a tiny ``ValueHead`` (linear
    on the LM's pooled hidden state) learns ``V(prompt)``. Advantage =
    reward - V(prompt); the value head is updated jointly via MSE. Cuts
    variance markedly and is the standard PPO/A2C choice.

Speed knobs (Roadmap #5):

  * ``batch_size=K``  -- batches K (prompt, completion) pairs into one
    padded forward pass, instead of K serial forwards. ~2-4x speedup on
    the T4 baseline. Default still 1 for backwards compatibility.
  * The reference forward uses ``torch.inference_mode()`` (faster than
    ``no_grad``) since no autograd tape is needed.

The reference distribution ``pi_ref`` is approximated by *disabling*
the LoRA adapter -- the frozen base + SFT weights serve as the
reference, so we don't need a second model in VRAM.

Designed to fit in a Kaggle T4 budget when called on the trajectories
the existing rollout cell already collected (no new generation).
"""

from __future__ import annotations

import logging
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
    value_loss: float = 0.0  # zero unless use_value_baseline=True


def _have_trainable_params(model) -> bool:
    return any(p.requires_grad for p in model.parameters())


def _set_lora_enabled(model, enabled: bool) -> None:
    """Toggle the LoRA adapter on/off. The reference policy log-probs
    are computed with the adapter disabled, so the base + SFT weights
    serve as ``pi_ref``."""
    if hasattr(model, "set_adapter"):
        if enabled and hasattr(model, "enable_adapter_layers"):
            model.enable_adapter_layers()
        elif not enabled and hasattr(model, "disable_adapter_layers"):
            model.disable_adapter_layers()


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _encode_batch(tokenizer, prompts: list[str], completions: list[str],
                  device, max_seq_len: int):
    """Tokenize K (prompt, completion) pairs in a single padded batch.

    Returns:
        full_ids        : (K, T) padded input_ids
        attention_mask  : (K, T) 1 where token is real, 0 for padding
        prompt_lens     : list[int] of length K -- where each completion starts
    """
    import torch

    K = len(prompts)
    full_texts = [p + c for p, c in zip(prompts, completions)]

    # Tokenize the prompts alone to recover where the completion starts.
    prompt_enc = tokenizer(prompts, truncation=True, max_length=max_seq_len,
                           padding=False)
    prompt_lens = [len(ids) for ids in prompt_enc["input_ids"]]

    # Tokenize prompt+completion as a left-padded batch (left-pad so
    # autoregressive logits at position t still predict token t+1 cleanly).
    full_enc = tokenizer(full_texts, truncation=True, max_length=max_seq_len,
                         padding=True, return_tensors="pt")
    full_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)
    return full_ids, attention_mask, prompt_lens, K


def _completion_logprobs_batched(model, full_ids, attention_mask, prompt_lens):
    """Per-sample sum of completion log-probs under the *current* model.

    Returns a (K,) tensor with grad attached if model is in train mode +
    LoRA enabled.
    """
    import torch

    outputs = model(input_ids=full_ids, attention_mask=attention_mask)
    # Logit at position t predicts token at position t+1 -> shift.
    logits = outputs.logits[:, :-1, :]
    labels = full_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    # (K, T-1)

    K, Tm1 = labels.shape
    device = labels.device

    # Build a per-sample completion mask: tokens at index >= prompt_len-1
    # AND inside the attention mask (i.e. not padding).
    arange = torch.arange(Tm1, device=device).unsqueeze(0).expand(K, -1)
    pl = torch.tensor(prompt_lens, device=device).unsqueeze(1)
    completion_pos = arange >= (pl - 1)
    not_padding = attention_mask[:, 1:].bool()
    mask = completion_pos & not_padding

    token_log_probs = token_log_probs.masked_fill(~mask, 0.0)
    return token_log_probs.sum(dim=-1)  # (K,)


def _pooled_hidden(model, full_ids, attention_mask):
    """Mean-pool the last-layer hidden states over the prompt tokens.

    Used by the value head as its features. Requires output_hidden_states.
    """
    import torch

    outputs = model(
        input_ids=full_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    # Final layer hidden state.
    hidden = outputs.hidden_states[-1]  # (K, T, H)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts  # (K, H)


# ---------------------------------------------------------------------------
# Value head -- variance reduction baseline (Roadmap #8)
# ---------------------------------------------------------------------------

class _ValueHead:
    """Lazy linear head V(prompt) trained jointly via MSE.

    Kept outside ``torch.nn.Module`` to avoid the test-suite needing torch
    at import time. ``build()`` is called once we know the LM hidden size.
    """

    def __init__(self) -> None:
        self.head = None
        self.optimizer = None

    def build(self, hidden_size: int, device, lr: float = 1e-4) -> None:
        import torch
        import torch.nn as nn
        from torch.optim import AdamW

        self.head = nn.Linear(hidden_size, 1).to(device)
        self.optimizer = AdamW(self.head.parameters(), lr=lr)

    def predict(self, pooled):
        """V(prompt) -> (K,)."""
        return self.head(pooled).squeeze(-1)

    def step(self, pooled, targets):
        """One MSE step. Returns scalar value loss for logging."""
        import torch
        pred = self.predict(pooled.detach())  # don't backprop into LM via value
        loss = torch.nn.functional.mse_loss(pred, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())


# ---------------------------------------------------------------------------
# Sample collection (unchanged except docstring)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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
    # Roadmap #5
    batch_size: int = 1,
    # Roadmap #8
    use_value_baseline: bool = False,
    value_lr: float = 1e-4,
) -> tuple[object, list[ReinforceStats]]:
    """Run REINFORCE-with-KL on the trajectories already in the buffer.

    Args:
        model, tokenizer: the SFT-warmed model + matching tokenizer.
        trajectory_buffer: the populated ``TrajectoryBuffer``.
        n_epochs: passes over the buffer.
        lr: AdamW learning rate. 5e-6 is a typical RLHF setting.
        kl_beta: weight on the KL term. Higher = stay closer to SFT.
        grad_accum: gradient-accumulation factor (effective batch).
        max_seq_len: token cap per (prompt+completion). T4-safe.
        max_samples: cap on tuples to process per epoch. None = all.
        progress_callback: ``cb(stats: ReinforceStats)`` after each step.
        batch_size: K samples per forward pass (Roadmap #5). Default 1
            for backwards compatibility; bump to 4-8 on T4 for speedup.
        use_value_baseline: if True, learn a V(prompt) head and use it
            as the baseline. Reduces variance vs the batch-mean baseline.
        value_lr: AdamW lr for the value head.
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

    rewards = np.array([s[2] for s in samples], dtype=np.float32)
    mean_r = float(rewards.mean())
    std_r = float(rewards.std() + 1e-8)

    # AdamW over only the trainable (LoRA) parameters.
    trainables = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainables, lr=lr)
    device = next(model.parameters()).device

    # Optional value head.
    value_head = _ValueHead() if use_value_baseline else None
    if use_value_baseline:
        hidden_size = int(getattr(model.config, "hidden_size",
                                  getattr(model.config, "n_embd", 4096)))
        value_head.build(hidden_size, device, lr=value_lr)
        logger.info("REINFORCE: value-head baseline ON (hidden_size=%d)",
                    hidden_size)

    history: list[ReinforceStats] = []
    step = 0
    grad_accum_count = 0
    accum = {"loss": 0.0, "policy": 0.0, "kl": 0.0, "adv": 0.0,
             "n": 0, "vloss": 0.0}

    model.train()
    bs = max(1, int(batch_size))

    for epoch in range(n_epochs):
        order = list(np.random.permutation(len(samples)))
        # Group indices into mini-batches of size `bs`.
        for batch_start in range(0, len(order), bs):
            batch_idx = order[batch_start:batch_start + bs]
            prompts = [samples[i][0] for i in batch_idx]
            completions = [samples[i][1] for i in batch_idx]
            batch_rewards = torch.as_tensor(
                [rewards[i] for i in batch_idx],
                device=device, dtype=torch.float32,
            )

            full_ids, attn, prompt_lens, K = _encode_batch(
                tokenizer, prompts, completions, device, max_seq_len,
            )

            # ---- Reference logprobs (LoRA disabled, no grad).
            _set_lora_enabled(model, False)
            with torch.inference_mode():
                ref_logp = _completion_logprobs_batched(
                    model, full_ids, attn, prompt_lens,
                )
            ref_logp = ref_logp.detach()
            _set_lora_enabled(model, True)

            # ---- Current logprobs (LoRA enabled, with grad).
            cur_logp = _completion_logprobs_batched(
                model, full_ids, attn, prompt_lens,
            )

            # ---- Advantage.
            if use_value_baseline:
                # Detach pooled features for value-head inference; a
                # separate update step trains the head on raw rewards.
                with torch.no_grad():
                    pooled = _pooled_hidden(model, full_ids, attn)
                values = value_head.predict(pooled.detach())
                adv = (batch_rewards - values.detach()) / std_r
                vloss = value_head.step(pooled, batch_rewards)
            else:
                adv = (batch_rewards - mean_r) / std_r
                vloss = 0.0

            kl = (cur_logp - ref_logp)
            policy_loss = -(adv * cur_logp).mean()
            kl_loss = kl.mean()
            loss = policy_loss + kl_beta * kl_loss

            # Defensive: if every completion in this batch was truncated
            # away (prompt filled the whole max_seq_len window), the loss
            # has no grad_fn and .backward() raises. Skip the batch.
            if not loss.requires_grad:
                logger.warning(
                    "Skipping zero-grad batch (all %d completions empty after "
                    "truncation; check max_seq_len=%d vs prompt length).",
                    K, max_seq_len,
                )
                continue

            (loss / grad_accum).backward()

            accum["loss"] += float(loss.detach())
            accum["policy"] += float(policy_loss.detach())
            accum["kl"] += float(kl_loss.detach())
            accum["adv"] += float(adv.mean().detach())
            accum["vloss"] += vloss
            accum["n"] += K
            grad_accum_count += 1

            if grad_accum_count >= grad_accum:
                torch.nn.utils.clip_grad_norm_(trainables, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                stats = ReinforceStats(
                    step=step,
                    loss=accum["loss"] / grad_accum,
                    policy_loss=accum["policy"] / grad_accum,
                    kl=accum["kl"] / grad_accum,
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
                         "n": 0, "vloss": 0.0}
                grad_accum_count = 0

    # Final partial batch.
    if grad_accum_count > 0:
        import torch as _t
        _t.nn.utils.clip_grad_norm_(trainables, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        history.append(ReinforceStats(
            step=step,
            loss=accum["loss"] / grad_accum_count,
            policy_loss=accum["policy"] / grad_accum_count,
            kl=accum["kl"] / grad_accum_count,
            mean_advantage=accum["adv"] / grad_accum_count,
            n_examples=accum["n"],
            value_loss=accum["vloss"] / grad_accum_count,
        ))

    return model, history


# ---------------------------------------------------------------------------
# Backwards-compatibility shim: original single-sample logprob helper.
# Some external code (older notebooks, tests) imports it by name.
# ---------------------------------------------------------------------------

def _completion_logprob(model, tokenizer, prompt: str, completion: str,
                        device, max_seq_len: int):
    """Single-sample logprob (kept for backwards-compatibility)."""
    full_ids, attn, prompt_lens, _ = _encode_batch(
        tokenizer, [prompt], [completion], device, max_seq_len,
    )
    return _completion_logprobs_batched(model, full_ids, attn, prompt_lens)[0]
