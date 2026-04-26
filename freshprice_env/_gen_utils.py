"""Tiny utility to suppress HF's `max_new_tokens` + `max_length` warning.

Qwen2.5 (and several other models) ship a `generation_config.max_length`
of 32768 in `config.json`. Whenever we call `model.generate(max_new_tokens=N)`
HF sees both values set and emits:

    Both `max_new_tokens` (=600) and `max_length`(=32768) seem to have been
    set. `max_new_tokens` will take precedence...

The fix is to clear `generation_config.max_length` once, after model load,
so only `max_new_tokens` is in play. Call `quiet_generation_config(model)`
right after `from_pretrained(...)`.
"""

from __future__ import annotations


def quiet_generation_config(model) -> None:
    """Silence HF's max_new_tokens/max_length conflict warning.

    Idempotent and safe on models that don't have a generation_config.
    """
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return
    if getattr(cfg, "max_length", None) is not None:
        cfg.max_length = None
    if getattr(cfg, "max_new_tokens", None) is not None:
        cfg.max_new_tokens = None
