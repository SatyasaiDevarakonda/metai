"""Generalized data-budget formula for the SFT warm-start.

The old auto-tune ladder ("if 1.5b then 30, elif 7b then 20, elif 14b
then 15, else 25") was a stack of magic numbers. None of them said
*why*. This module replaces it with one defensible formula derived
from three things that actually matter for SFT-on-format-following:

  1. Format complexity   — how many sections the model has to learn,
                           and how long each section is.
  2. Model capacity      — bigger models pick up rigid formats with
                           fewer demonstrations (well-known scaling
                           result; e.g. Lambda Labs' SFT studies).
  3. Compute budget      — VRAM caps the batch size and therefore the
                           number of examples we can pass through in
                           a reasonable wall-clock.

The formula is calibrated *once* against the Qwen-2.5-1.5B / T4 run
that we know works (270 examples, format recall ≥ 99 %). After
calibration, the same constant carries to bigger models, smaller
models, and bigger / smaller GPUs without re-tuning.

Calibration anchor (do not move without re-deriving ALPHA):
    model:               Qwen/Qwen2.5-1.5B-Instruct  (≈ 1.54 B params)
    format_sections:     6
    avg_completion_chars: ≈ 900
    difficulty_levels:   3
    engine_count:        3
    observed n_per_diff: 30 → 270 examples → ≥ 99 % format recall

That anchor produces ALPHA ≈ 2.48 (re-derive: at 1.5 B params,
6 sections × ln(900) ≈ 40.8 format-complexity, recall 0.99 → 270
examples × ~306 tokens / example = ~82.6 k tokens → ALPHA = 82.6 ×
√1.5 / 40.8 ≈ 2.48). Any model + format + compute combination is then
derived from ALPHA, no further knobs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Calibration constant (see module docstring — derived, not invented)
# ---------------------------------------------------------------------------

# Token-budget coefficient (in kilotokens). Larger -> more examples for
# the same model. Derived from the 1.5B / 270-example anchor; do not
# edit casually. See module docstring for the derivation.
ALPHA: float = 2.48

# Average tokens per training example (Qwen tokenizer measurement on the
# existing 270-example dataset). 1 char ≈ 0.34 tokens for English+JSON.
TOKENS_PER_CHAR: float = 0.34

# Coverage floor: even on a giant model with a tiny format, you still
# need at least this many demonstrations per difficulty bucket to cover
# the difficulty distribution (otherwise easy/medium/hard each have
# fewer than 5 examples and the dataloader thrashes).
MIN_PER_DIFFICULTY: int = 8

# A T4 (16 GB VRAM) safely trains ≤ ~450 examples in our SFT loop in
# under 30 min wall-clock. Scale linearly with VRAM above that.
VRAM_TO_MAX_EXAMPLES_PER_GB: float = 28.0

# Epoch schedule by dataset size (kept small — bigger batches = fewer
# epochs needed, which is the standard SFT heuristic).
def _epochs_for(n_examples: int) -> int:
    if n_examples < 150:
        return 8
    if n_examples < 350:
        return 5
    if n_examples < 600:
        return 4
    return 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataBudget:
    """The decision and the reasoning behind it."""

    n_per_difficulty: int
    total_examples: int
    epochs: int
    rationale: str
    model_params_b: float
    format_complexity: float
    sft_token_target: int
    vram_cap_examples: int


def estimate_model_params_b(model_id: str) -> float:
    """Best-effort param count (in billions) from a HF model ID.

    The string is the public source of truth for HF model sizing; if a
    better signal is available (e.g. ``model.num_parameters()``) prefer
    that and pass it via ``model_params_b_override``.
    """
    mid = model_id.lower()
    # Order matters: longer suffixes first.
    candidates = [
        ("70b", 70.0), ("32b", 32.0), ("14b", 14.0), ("13b", 13.0),
        ("8b", 8.0),   ("7b", 7.0),   ("3b", 3.0),
        ("1.5b", 1.5), ("1b", 1.0),   ("0.5b", 0.5),
    ]
    for needle, value in candidates:
        if needle in mid:
            return value
    return 1.5  # safe default — matches the anchor model


def vram_gb_to_max_examples(vram_gb: float) -> int:
    """Cap the dataset size so a small GPU never gets asked to swallow 600 examples."""
    if vram_gb <= 0:
        return 90        # CPU smoke-test floor (1 example / engine / difficulty cluster)
    return max(90, int(VRAM_TO_MAX_EXAMPLES_PER_GB * vram_gb))


def compute_data_budget(
    *,
    model_id: str,
    vram_gb: float,
    target_format_recall: float = 0.99,
    format_sections: int = 6,
    avg_completion_chars: int = 900,
    difficulty_levels: int = 3,
    engine_count: int = 3,
    model_params_b_override: float | None = None,
) -> DataBudget:
    """Return the SFT data budget plus the reasoning.

    The formula:
        format_complexity = format_sections * log(avg_completion_chars)
        sft_token_target  = ALPHA * format_complexity / sqrt(model_params_b)
                            * recall_pressure(target_format_recall)
        n_per_diff        = ceil(sft_token_target /
                                 (avg_tokens_per_example *
                                  engine_count *
                                  difficulty_levels))

    Args:
        model_id: HF model ID (used to estimate parameter count).
        vram_gb:  Available VRAM in GiB. Pass 0 for CPU.
        target_format_recall: Fraction of generations expected to contain
            all required sections after SFT. 0.99 is the default; raising
            it scales the data budget super-linearly.
        format_sections: Number of required brief sections (default 6
            for the SITUATION/SIGNAL/VIABILITY/RECOMMENDATION/DIRECTIVE/
            CONFIDENCE format).
        avg_completion_chars: Average characters per completion. Measure
            against your generated dataset; default matches QStorePrice.
        difficulty_levels:   Number of difficulty buckets per engine
            (we use easy / medium / hard).
        engine_count:        Number of brief engines (PRICING / FARMER /
            TREND).
        model_params_b_override: Pass an exact param count (in billions)
            from ``model.num_parameters() / 1e9`` if you have it; the
            HF-ID estimator is a fallback.

    Returns:
        DataBudget with the recommendation and a human-readable
        rationale explaining each multiplier.
    """
    if not 0.5 <= target_format_recall < 1.0:
        raise ValueError("target_format_recall must lie in [0.5, 1.0)")

    params_b = (
        model_params_b_override
        if model_params_b_override is not None
        else estimate_model_params_b(model_id)
    )

    format_complexity = format_sections * math.log(max(avg_completion_chars, 2.0))
    avg_tokens_per_example = avg_completion_chars * TOKENS_PER_CHAR

    # Recall pressure: log(1/(1-r)) — at r=0.99 this is ~4.6, at r=0.95 it
    # is ~3.0. Mirrors the "cover the long tail" cost in classification.
    recall_pressure = math.log(1.0 / (1.0 - target_format_recall))
    # Anchor at recall=0.99 (the calibration setting) so that ALPHA
    # carries through unchanged for the default; deviations scale.
    recall_factor = recall_pressure / math.log(1.0 / (1.0 - 0.99))

    sft_token_target = (
        ALPHA
        * format_complexity
        * recall_factor
        / math.sqrt(max(params_b, 0.1))
        * 1000.0  # ALPHA was calibrated against thousands of tokens
    )

    raw_n = sft_token_target / (
        avg_tokens_per_example * engine_count * difficulty_levels
    )
    n_per_difficulty = max(MIN_PER_DIFFICULTY, math.ceil(raw_n))

    # VRAM cap.
    vram_cap = vram_gb_to_max_examples(vram_gb)
    cap_n = max(MIN_PER_DIFFICULTY,
                vram_cap // (engine_count * difficulty_levels))
    capped = n_per_difficulty > cap_n
    if capped:
        n_per_difficulty = cap_n

    total = n_per_difficulty * engine_count * difficulty_levels
    epochs = _epochs_for(total)

    rationale = "\n".join([
        f"  Model params         : {params_b:.2f} B  (from '{model_id}')",
        f"  Format complexity    : {format_complexity:.2f}  "
        f"({format_sections} sections * log({avg_completion_chars} chars))",
        f"  Target format recall : {target_format_recall:.2%}  "
        f"(recall_factor = {recall_factor:.3f})",
        f"  SFT token target     : {sft_token_target:,.0f}  "
        f"(ALPHA={ALPHA} * complexity * recall / sqrt(params))",
        f"  Tokens / example     : {avg_tokens_per_example:.0f}",
        f"  Raw n_per_difficulty : {raw_n:.1f}  -> ceil "
        f"= {math.ceil(raw_n)}",
        f"  VRAM cap             : {vram_cap} examples  "
        f"({vram_gb:.1f} GB * {VRAM_TO_MAX_EXAMPLES_PER_GB:.0f}/GB)"
        + ("  <- APPLIED" if capped else ""),
        f"  Floor                : {MIN_PER_DIFFICULTY} per difficulty",
        "",
        f"  -> n_per_difficulty   = {n_per_difficulty}",
        f"  -> total examples     = {total}  "
        f"({n_per_difficulty} * {engine_count} engines * {difficulty_levels} difficulties)",
        f"  -> epochs             = {epochs}  (size-tier schedule)",
    ])

    return DataBudget(
        n_per_difficulty=n_per_difficulty,
        total_examples=total,
        epochs=epochs,
        rationale=rationale,
        model_params_b=params_b,
        format_complexity=format_complexity,
        sft_token_target=int(sft_token_target),
        vram_cap_examples=vram_cap,
    )


def compute_grpo_episode_budget(
    *,
    vram_gb: float,
    avg_episode_seconds: float = 240.0,
    wall_clock_target_minutes: float = 30.0,
    buffer_admission_rate: float = 0.55,
    min_admitted_pairs: int = 4,
) -> int:
    """How many GRPO episodes to run before DPO.

    The constraint isn't memory (GRPO uses the same checkpoint that
    SFT produced); it's wall-clock and the DPO buffer needs at least
    ``min_admitted_pairs`` valid trajectories to compute preferences.
    On a T4, ~3 episodes fits the 30-minute Kaggle session; on an A100
    we can amortize and push to 12. Scale with VRAM as a proxy for
    overall throughput (faster GPU → faster generation in GRPO too).

    Returns:
        episodes: at least enough for ``min_admitted_pairs`` clean pairs
        given the empirical buffer-admission rate, capped by the
        wall-clock target.
    """
    # Throughput proxy: VRAM scales with GPU class (T4=16, A10/V100~24,
    # A100=40-80). Assume linear improvement.
    throughput_factor = max(1.0, vram_gb / 16.0)
    eps_episode_seconds = avg_episode_seconds / throughput_factor
    wall_clock_eps = max(1, int(
        wall_clock_target_minutes * 60.0 / max(eps_episode_seconds, 30.0)
    ))

    # Floor: enough rollouts to land ``min_admitted_pairs`` clean ones.
    needed_for_dpo = math.ceil(min_admitted_pairs / max(buffer_admission_rate, 0.1))

    return max(needed_for_dpo, min(wall_clock_eps, 24))
