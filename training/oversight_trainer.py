"""SFT trainer for the OversightAuditor.

Generates synthetic (trajectory_excerpt → audit_report) pairs by
running the rule-based auditor against logged episodes (good and bad
behaviour) and treating those reports as gold targets. A small LLM
(default Qwen-2.5-0.5B) is then SFT-ed to reproduce them.

The synthesis pipeline:

    1. Load a dump of recorded episodes (JSONL emitted by
       eval/counterfactual_replay.py or live runs)
    2. For each episode, build an AuditTrajectory and call the
       rule-based auditor
    3. Emit (input_messages_text, target_audit_text) into a
       HuggingFace ``datasets.Dataset``
    4. Run TRL SFTTrainer (or fall back to a transformers loop)

This file is intentionally light on dependencies — the heavy training
imports happen inside functions so importing this module on a CPU box
without unsloth/TRL still works for offline data prep.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from freshprice_env.agents.oversight_auditor import (
    AuditTrajectory,
    OversightAuditor,
    trajectory_from_market_commons,
)


logger = logging.getLogger(__name__)


@dataclass
class OversightExample:
    """One SFT example."""

    episode_id: str
    scenario: str
    input_text: str         # serialised trajectory excerpt (prompt)
    target_text: str        # rule-based auditor's report (target)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def build_examples_from_episodes_jsonl(
    jsonl_path: str | Path,
    max_examples: int | None = None,
) -> list[OversightExample]:
    """Load logged episodes and synthesise rule-based audit targets.

    The expected JSONL schema (one line per episode):

        {
          "episode_id": "...",
          "scenario": "CRISIS_WEEK",
          "bus_messages": [...],
          "rule_violations": [...],
          "notebook_actions": [...],
          "contention_events": [...]
        }

    Anything missing defaults to []. Episodes with zero events are
    skipped (nothing to audit).
    """
    examples: list[OversightExample] = []
    auditor = OversightAuditor(mode="rule_based")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            episode_id = row.get("episode_id") or "ep_unknown"
            scenario = row.get("scenario") or "STABLE_WEEK"

            traj = trajectory_from_market_commons(
                episode_id=episode_id,
                scenario=scenario,
                bus_messages=row.get("bus_messages", []) or [],
                rule_violations=row.get("rule_violations", []) or [],
                notebook_actions=row.get("notebook_actions", []) or [],
                contention_events=row.get("contention_events", []) or [],
            )
            if not traj.events:
                continue
            report = auditor.audit(traj)
            ex = OversightExample(
                episode_id=episode_id,
                scenario=scenario,
                input_text=_format_input(traj),
                target_text=_format_target(report),
            )
            examples.append(ex)
            if max_examples is not None and len(examples) >= max_examples:
                break

    logger.info("Built %d oversight SFT examples from %s", len(examples), jsonl_path)
    return examples


def write_examples_jsonl(
    examples: list[OversightExample],
    out_path: str | Path,
) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({
                "episode_id": ex.episode_id,
                "scenario": ex.scenario,
                "input": ex.input_text,
                "target": ex.target_text,
            }, ensure_ascii=False) + "\n")
    return str(out)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = OversightAuditor.SYSTEM_PROMPT


def _format_input(traj: AuditTrajectory) -> str:
    lines = [_SYSTEM_PROMPT, ""]
    lines.append(f"# Episode {traj.episode_id} ({traj.scenario})")
    lines.append(f"# Total events: {len(traj.events)}")
    lines.append("")
    for ev in traj.events[-160:]:    # last 160 events fits comfortably
        lines.append(f"[t{ev.tick}] {ev.kind} {ev.actor}: {ev.summary[:160]}")
    lines.append("\n# Write the audit report now.")
    return "\n".join(lines)


def _format_target(report) -> str:
    pat_lines = []
    for p in report.suspicious_patterns:
        tick = p.get("tick") or (p.get("ticks") or [None])[0]
        prefix = f"  - {p.get('id')} @ tick {tick}: " if tick else f"  - {p.get('id')}: "
        pat_lines.append(prefix + (p.get("description", "") or ""))
    return (
        f"TRUST_SCORE: {report.trust_score:.2f}\n"
        f"SUSPICIOUS_PATTERNS:\n" +
        ("\n".join(pat_lines) if pat_lines else "  - NONE\n") + "\n"
        f"NARRATIVE: {report.narrative}\n"
        f"RECOMMENDATION: {report.recommendation}\n"
    )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def run_sft(
    examples: list[OversightExample],
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "checkpoints/oversight_v1",
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_seq_length: int = 4096,
) -> str:
    """SFT the auditor model on examples. Imports heavy deps lazily.

    Returns the output checkpoint path.
    """
    if not examples:
        raise ValueError("no examples to train on")

    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "oversight_trainer.run_sft needs `transformers` and `datasets`. "
            "Install with: pip install transformers datasets"
        ) from exc

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = [{"text": ex.input_text + "\n\n" + ex.target_text} for ex in examples]
    ds = Dataset.from_list(rows)

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            max_length=max_seq_length,
            truncation=True,
            padding=False,
        )
        return out

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=False,
        fp16=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved oversight auditor checkpoint to %s", output_dir)
    return output_dir
