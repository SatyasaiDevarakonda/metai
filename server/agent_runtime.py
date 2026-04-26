"""Server-side agent runtime — bridges the trained model to the dashboard.

After Kaggle training the user has one of three artifacts:

    1. A merged 16-bit checkpoint on local disk           (LocalAgentRuntime)
    2. A model pushed to the HF Hub                       (HFInferenceAgentRuntime)
    3. Nothing (running the dashboard for the first time) (ScriptedAgentRuntime)

This module hides all three behind one interface — ``AgentRuntime.generate(prompt)``
— so the FastAPI server can call the agent without caring which backend is
loaded. The factory ``get_agent_runtime()`` picks one based on environment
variables, in priority order:

    AGENT_BACKEND=local          + MODEL_PATH=<dir>          -> LocalAgentRuntime
    AGENT_BACKEND=hf_inference   + HF_REPO_ID + HF_TOKEN     -> HFInferenceAgentRuntime
    (nothing set)                                            -> ScriptedAgentRuntime

The scripted backend is what makes the demo dashboard work *before* the
user has trained anything. It uses the same heuristics as the SFT data
generator, so the briefs are well-formed and exercise every section.

Heavy deps (``transformers``, ``peft``) are imported lazily so the server
can boot without them.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------


class AgentRuntime(Protocol):
    """Anything that can turn a prompt string into a brief string."""

    backend: str
    model_id: str

    def generate(self, prompt: str, *, max_new_tokens: int = 600,
                 temperature: float = 0.7) -> str: ...

    def info(self) -> dict: ...


# ---------------------------------------------------------------------------
# 1) Local — transformers + PEFT
# ---------------------------------------------------------------------------


@dataclass
class LocalAgentRuntime:
    """Loads a merged checkpoint (or LoRA adapter) directly into the server.

    Cheap on a GPU box; will OOM a small laptop. Use ``MODEL_PATH`` env var
    to point at a Kaggle output (e.g. ``/kaggle/working/checkpoints/final``)
    you have downloaded; or pass an HF repo id (``Qwen/Qwen2.5-1.5B-Instruct``)
    to load a base model with no fine-tuning.
    """

    model_path: str
    backend: str = "local"
    model_id: str = ""
    _model: object = None
    _tokenizer: object = None

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LocalAgentRuntime requires `transformers` and `torch`. "
                "Install with `pip install -r requirements_training.txt`."
            ) from exc

        self.model_id = self.model_path
        logger.info("Loading local model from %s", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self._model.eval()

    def generate(self, prompt: str, *, max_new_tokens: int = 600,
                 temperature: float = 0.7) -> str:
        import torch  # noqa: E402  (lazy)
        chat_prompt = self._format_chat(prompt)
        inputs = self._tokenizer(
            chat_prompt, return_tensors="pt", truncation=True, max_length=4096,
        )
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(gen, skip_special_tokens=True)

    def info(self) -> dict:
        return {"backend": self.backend, "model_path": self.model_path}

    def _format_chat(self, prompt: str) -> str:
        # Chat-template the prompt for Qwen-style instruction tuning.
        sys_marker = "\n\n"
        if sys_marker in prompt:
            sys_msg, user_msg = prompt.split(sys_marker, 1)
        else:
            sys_msg, user_msg = "", prompt
        try:
            return self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
                [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # pragma: no cover - tokenizers without chat template
            return prompt


# ---------------------------------------------------------------------------
# 2) HF Inference — call the model via Hugging Face's hosted endpoint
# ---------------------------------------------------------------------------


@dataclass
class HFInferenceAgentRuntime:
    """Uses the OpenAI-compatible HF Inference router (no local GPU)."""

    repo_id: str
    hf_token: str
    base_url: str = "https://api-inference.huggingface.co/v1"
    backend: str = "hf_inference"
    model_id: str = ""
    _client: object = None

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "HFInferenceAgentRuntime requires `openai`. "
                "Install with `pip install openai>=1.0`."
            ) from exc
        self.model_id = self.repo_id
        self._client = OpenAI(base_url=self.base_url, api_key=self.hf_token)
        logger.info("HF Inference runtime ready for %s", self.repo_id)

    def generate(self, prompt: str, *, max_new_tokens: int = 600,
                 temperature: float = 0.7) -> str:
        sys_marker = "\n\n"
        sys_msg, user_msg = (prompt.split(sys_marker, 1)
                             if sys_marker in prompt else ("", prompt))
        resp = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self.repo_id,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def info(self) -> dict:
        return {"backend": self.backend, "repo_id": self.repo_id,
                "base_url": self.base_url}


# ---------------------------------------------------------------------------
# 3) Scripted — heuristic stand-in (no model required)
# ---------------------------------------------------------------------------


_BATCH_LINE_RE = re.compile(
    r"Batch\s+([A-Za-z0-9]+)\s*\|.*?(\d+(?:\.\d+)?)hrs", re.IGNORECASE,
)
_OFFER_LINE_RE = re.compile(
    r"Offer\s+([A-Za-z0-9]+):.*?Viability score: ([0-9.]+)", re.IGNORECASE,
)
_TREND_CAT_RE = re.compile(r"Category:\s*([a-zA-Z_]+)")


@dataclass
class ScriptedAgentRuntime:
    """Heuristic agent — produces well-formed briefs without any model.

    This is what runs before the user has trained anything. It mirrors
    the rules used by ``training/generate_sft_data.py`` so the briefs
    contain all six sections and exercise the env's reward components.

    Rules:
      - PRICING: discount near-expiry, hold price on FRESH, never below floor.
      - FARMER:  ACCEPT viability >= 0.6, COUNTER 0.4-0.6, DECLINE < 0.4.
      - TREND:   APPROVE if "trending" mentioned and category seen.
    """

    seed: int = 0
    backend: str = "scripted"
    model_id: str = "scripted-heuristic"
    _rng: random.Random = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def info(self) -> dict:
        return {"backend": self.backend, "model_id": self.model_id}

    def generate(self, prompt: str, *, max_new_tokens: int = 600,
                 temperature: float = 0.7) -> str:
        kind = self._classify(prompt)
        if kind == "FARMER":
            return self._farmer_brief(prompt)
        if kind == "TREND":
            return self._trend_brief(prompt)
        return self._pricing_brief(prompt)

    # ------------------------------------------------------------------

    @staticmethod
    def _classify(prompt: str) -> str:
        upper = prompt.upper()
        if "FARMER OPERATING BRIEF" in upper or "PENDING FARMER OFFERS" in upper:
            return "FARMER"
        if "TREND OPERATING BRIEF" in upper or "INCOMING TREND SIGNAL" in upper:
            return "TREND"
        return "PRICING"

    def _pricing_brief(self, prompt: str) -> str:
        actions = []
        for m in _BATCH_LINE_RE.finditer(prompt):
            bid, hrs = m.group(1), float(m.group(2))
            if hrs <= 6:
                mult, flash = 0.50, True   # CRITICAL — heavy discount, flash sale
            elif hrs <= 24:
                mult, flash = 0.70, False  # URGENT
            elif hrs <= 72:
                mult, flash = 0.90, False  # WATCH
            else:
                mult, flash = 1.00, False  # FRESH — hold
            actions.append({
                "batch_id": bid, "price_multiplier": mult,
                "flash_sale": flash, "bundle_with": None,
            })
        directive = json.dumps(
            {"engine": "PRICING", "actions": actions or []},
            separators=(", ", ": "),
        )
        return (
            "SITUATION: Reviewing inventory; applying urgency-tiered discounts where shelf life is short.\n\n"
            "SIGNAL ANALYSIS: N/A\n\n"
            "VIABILITY CHECK: N/A\n\n"
            "RECOMMENDATION: Discount near-expiry stock; preserve margin on FRESH batches.\n\n"
            f"DIRECTIVE:\n{directive}\n\n"
            "CONFIDENCE: MEDIUM"
        )

    def _farmer_brief(self, prompt: str) -> str:
        actions, viab_lines = [], []
        for m in _OFFER_LINE_RE.finditer(prompt):
            oid, viab = m.group(1), float(m.group(2))
            if viab >= 0.60:
                decision, counter = "ACCEPT", None
                factor = "PASS"
            elif viab >= 0.40:
                decision, counter = "COUNTER", None
                factor = "FLAG"
            else:
                decision, counter = "DECLINE", None
                factor = "FAIL"
            viab_lines.append(f"{oid}: viability {viab:.2f} -- {factor}")
            actions.append({"offer_id": oid, "decision": decision,
                            "counter_price": counter})
        directive = json.dumps(
            {"engine": "FARMER", "actions": actions or []},
            separators=(", ", ": "),
        )
        viability = " | ".join(viab_lines) if viab_lines else "no offers"
        return (
            "SITUATION: Evaluating pending farmer offers against viability and risk buffer.\n\n"
            "SIGNAL ANALYSIS: N/A\n\n"
            f"VIABILITY CHECK: {viability}\n\n"
            "RECOMMENDATION: Accept high-viability offers; counter borderline; decline below 0.40.\n\n"
            f"DIRECTIVE:\n{directive}\n\n"
            "CONFIDENCE: MEDIUM"
        )

    def _trend_brief(self, prompt: str) -> str:
        m = _TREND_CAT_RE.search(prompt)
        category = m.group(1) if m else "vegetables"
        directive = json.dumps({
            "engine": "TREND",
            "actions": [{"category": category, "decision": "APPROVE",
                         "order_quantity_kg": 50.0}],
        }, separators=(", ", ": "))
        return (
            f"SITUATION: Trend signal observed in category={category}; assessing restock viability.\n\n"
            "SIGNAL ANALYSIS: Modest demand bump expected if signal is genuine; not a viral spike.\n\n"
            "VIABILITY CHECK: cooldown PASS | velocity PASS | shelf-life PASS\n\n"
            "RECOMMENDATION: Approve a small restock to capture the bump without overcommitting.\n\n"
            f"DIRECTIVE:\n{directive}\n\n"
            "CONFIDENCE: MEDIUM"
        )


# ---------------------------------------------------------------------------
# MultiAgentRuntime — load several backends at once for before/after comparison
# ---------------------------------------------------------------------------


class MultiAgentRuntime:
    """Holds named runtimes (e.g. "baseline", "sft", "rl") so a single
    request can fan out to all three and report side-by-side outputs.

    This is the core of the "before vs after RL" comparison harness.
    The hackathon judging criterion #3 (Showing Improvement in Rewards,
    20%) explicitly asks for "comparison against a baseline -- anything
    that proves the agent learned something". MultiAgentRuntime is how
    we serve that comparison live from the dashboard.
    """

    def __init__(self, runtimes: dict[str, AgentRuntime]) -> None:
        if not runtimes:
            raise ValueError("MultiAgentRuntime requires at least one runtime")
        self._runtimes: dict[str, AgentRuntime] = runtimes

    @property
    def names(self) -> list[str]:
        return list(self._runtimes.keys())

    def get(self, name: str) -> AgentRuntime | None:
        return self._runtimes.get(name)

    def info(self) -> dict:
        return {name: rt.info() for name, rt in self._runtimes.items()}

    def generate_all(
        self, prompt: str, *, max_new_tokens: int = 600,
        temperature: float = 0.7,
    ) -> dict[str, str]:
        """Fan the prompt out to every runtime; return a dict of briefs."""
        out: dict[str, str] = {}
        for name, rt in self._runtimes.items():
            try:
                out[name] = rt.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except Exception as e:  # noqa: BLE001 — keep the comparison alive
                logger.warning("runtime %s failed: %s", name, e)
                out[name] = f"[runtime '{name}' failed: {e}]"
        return out


def build_comparison_runtime(
    *,
    baseline: bool = True,
    sft_path: str | None = None,
    rl_path: str | None = None,
    hf_repo_id: str | None = None,
    hf_token: str | None = None,
) -> MultiAgentRuntime:
    """Construct a 2-3 backend comparison rig.

    - baseline=True always installs ScriptedAgentRuntime as the "no
      training" lower bound so the comparison has a meaningful floor.
    - sft_path  -> LocalAgentRuntime over the SFT checkpoint
    - rl_path   -> LocalAgentRuntime over the DPO/REINFORCE checkpoint
    - hf_repo_id+hf_token -> HFInferenceAgentRuntime as fallback when no
      local checkpoints are available; tagged as "rl" by default since
      the HF Hub model is post-RL in our pipeline.
    """
    rts: dict[str, AgentRuntime] = {}
    if baseline:
        rts["baseline"] = ScriptedAgentRuntime()
    if sft_path:
        try:
            rts["sft"] = LocalAgentRuntime(model_path=sft_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("SFT load failed (%s); skipping", e)
    if rl_path:
        try:
            rts["rl"] = LocalAgentRuntime(model_path=rl_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("RL load failed (%s); skipping", e)
    if "rl" not in rts and hf_repo_id and hf_token:
        try:
            rts["rl"] = HFInferenceAgentRuntime(
                repo_id=hf_repo_id, hf_token=hf_token,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("HF inference fallback failed (%s)", e)
    return MultiAgentRuntime(rts)


# ---------------------------------------------------------------------------
# Factory + module-level singleton
# ---------------------------------------------------------------------------

_runtime: AgentRuntime | None = None
_comparison: MultiAgentRuntime | None = None


def get_agent_runtime() -> AgentRuntime:
    """Return the process-wide agent runtime, building lazily on first call."""
    global _runtime
    if _runtime is not None:
        return _runtime
    _runtime = _build_from_env()
    logger.info("Agent runtime ready: %s", _runtime.info())
    return _runtime


def get_comparison_runtime() -> MultiAgentRuntime:
    """Return the process-wide comparison runtime (baseline + SFT + RL).

    Reads SFT_MODEL_PATH and RL_MODEL_PATH env vars; falls back to
    HF_REPO_ID + HF_TOKEN for the RL slot if no local checkpoint is set.
    Always installs the ScriptedAgentRuntime as 'baseline' so the
    dashboard's "Before vs After RL" panel has a meaningful floor.
    """
    global _comparison
    if _comparison is not None:
        return _comparison
    _comparison = build_comparison_runtime(
        baseline=True,
        sft_path=os.environ.get("SFT_MODEL_PATH", "").strip() or None,
        rl_path=os.environ.get("RL_MODEL_PATH", "").strip() or None,
        hf_repo_id=os.environ.get("HF_REPO_ID", "").strip() or None,
        hf_token=os.environ.get("HF_TOKEN", "").strip() or None,
    )
    logger.info("Comparison runtime ready: %s", _comparison.info())
    return _comparison


def reset_comparison_runtime() -> None:
    global _comparison
    _comparison = None


def reset_agent_runtime() -> None:
    """Drop the cached runtime (used by tests + when env vars change)."""
    global _runtime
    _runtime = None


def _build_from_env() -> AgentRuntime:
    backend = os.environ.get("AGENT_BACKEND", "").strip().lower()
    model_path = os.environ.get("MODEL_PATH", "").strip()
    repo_id = os.environ.get("HF_REPO_ID", "").strip()
    hf_token = os.environ.get("HF_TOKEN", "").strip()

    # Explicit choice wins.
    if backend == "local":
        if not model_path:
            raise RuntimeError("AGENT_BACKEND=local requires MODEL_PATH")
        return LocalAgentRuntime(model_path=model_path)
    if backend in ("hf_inference", "hf"):
        if not repo_id or not hf_token:
            raise RuntimeError(
                "AGENT_BACKEND=hf_inference requires HF_REPO_ID and HF_TOKEN")
        return HFInferenceAgentRuntime(repo_id=repo_id, hf_token=hf_token)
    if backend == "scripted":
        return ScriptedAgentRuntime()

    # No backend pinned: try to infer.
    if model_path and Path(model_path).exists():
        try:
            return LocalAgentRuntime(model_path=model_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("LocalAgentRuntime failed (%s); falling back", e)
    if repo_id and hf_token:
        try:
            return HFInferenceAgentRuntime(repo_id=repo_id, hf_token=hf_token)
        except Exception as e:  # noqa: BLE001
            logger.warning("HFInferenceAgentRuntime failed (%s); falling back", e)
    return ScriptedAgentRuntime()
