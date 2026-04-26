# RUN.md — End-to-end reproduction guide

Last verified: 2026-04-26 against commit on `main`. **84 tests must pass** before
any training step.

---

## 0. Environment

| Component | Version |
|---|---|
| Python | 3.10–3.14 (3.10 for Kaggle/Unsloth; 3.14 OK for local CPU work) |
| GPU | T4 16 GB (Kaggle free) is the calibration target. A100 / H100 work without changes. |
| OS | Tested on Windows 11 + Ubuntu 22.04 |

```bash
# CPU-only deps for env work + tests
pip install -r requirements.txt

# GPU training extras (Unsloth + bitsandbytes + TRL)
pip install -r requirements_training.txt
```

---

## 1. Smoke + inventory tests (~30s, CPU)

```bash
python -m unittest discover tests -v
```

Expected: **84 passed, 0 failures**. Covers `FreshPriceEnv`, `MarketCommonsEnv`,
the 7-engine reward, the Blinkit layer, the comparison harness, and all four
dashboard panel feeds.

If any fail before training begins, **stop here**. The bug is in the env layer,
not the model.

---

## 2. Generate SFT data (~10s, CPU)

```bash
python -m training.generate_sft_data --output-dir training/sft_data
```

Outputs (per-engine examples + difficulty mix):

| File | Examples | Difficulty split |
|---|---:|---|
| `training/sft_data/pricing_examples.json` | 93 | easy 31 / med 31 / hard 31 |
| `training/sft_data/farmer_examples.json` | 93 | easy 31 / med 31 / hard 31 |
| `training/sft_data/trend_examples.json` | 93 | easy 31 / med 31 / hard 31 |

Total: **279** SFT examples. Number is **derived**, not magic — see
`training/data_budget.py:compute_data_budget()` for the formula and the
calibration anchor (Qwen-1.5B / 270 examples / T4).

Sanity: every example must contain all 6 brief sections (`SITUATION`,
`SIGNAL ANALYSIS`, `VIABILITY CHECK`, `RECOMMENDATION`, `DIRECTIVE`, `CONFIDENCE`).
The generator already verifies this; if it prints `All N examples contain all 6
required sections.`, you're good.

---

## 3. SFT training (Kaggle T4, ~30 min on Qwen-2.5-7B + LoRA)

The Kaggle path is the canonical one. Open `kaggle_qstoreprice.ipynb` in a
Kaggle notebook with **Internet ON** and **GPU = T4 x2** (1 is fine; 2 lets you
parallelise eval).

Run cells top-to-bottom. The cells that actually train:

| Cell ID | What it does |
|---|---|
| `cell-config` | Sets `REPO_DIR`, `CHECKPOINTS_DIR`, etc. |
| `cell-install-unsloth` / `cell-install-deps` | Pip installs |
| `cell-clone` | Clones the repo |
| `cell-hf-auth` | Logs into HF Hub (needed for Qwen-2.5-7B gated download) |
| `cell-smoke-test` ... `cell-blinkit-wired` | Smoke tests, all should print PASSED |
| `cell-sft-generate` | Generates SFT data (same as step 2) |
| `cell-sft-train` | **SFT** — produces `checkpoints/sft_v1/` |
| `cell-grpo-rollouts` | Collects rollouts into `trajectory_buffer` |
| `cell-rl-reinforce` | **REINFORCE+KL** — produces `checkpoints/reinforce_v1/` |
| `cell-dpo` | **DPO** — produces `checkpoints/dpo_v1/` |
| `cell-eval` | Per-scenario eval table |
| `cell-comparison` | Baseline vs SFT vs RL comparison |

### Live dashboard during training (optional but a strong demo)

Set `DASHBOARD_URL` before running `cell-rl-reinforce`:

```python
import os
os.environ["DASHBOARD_URL"] = "https://<your-tunnel>.example.com"
```

Use ngrok or Cloudflare Tunnel from the box running `uvicorn server.app:app` to
expose port 8000. Cell 9b best-effort POSTs each optimizer step to
`/training/event`; failures are silently swallowed so training never blocks.

Expected artifacts after step 3:

```
/kaggle/working/checkpoints/
├── sft_v1/                 # ~50 MB LoRA adapter (or ~14 GB merged 16-bit)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer*
├── reinforce_v1/           # post-RL merged 16-bit
└── dpo_v1/                 # post-DPO merged 16-bit
```

---

## 4. Download checkpoints from Kaggle

Pick one path:

```bash
# (a) Kaggle dataset commit (recommended)
# Inside the notebook:
!cd /kaggle/working/checkpoints && tar -czf /kaggle/working/checkpoints.tar.gz dpo_v1
# Then "Save Version" the notebook -> output appears as kernel output.

# (b) Push to HF Hub from notebook:
model.push_to_hub("you/freshprice-rl-v1", token=HF_TOKEN)
tokenizer.push_to_hub("you/freshprice-rl-v1", token=HF_TOKEN)

# (c) Manual: Notebook -> Output tab -> Download checkpoints/
```

Local extract:

```bash
mkdir -p checkpoints
tar -xzf checkpoints.tar.gz -C checkpoints/
ls checkpoints/dpo_v1/      # adapter_*.safetensors + tokenizer files
```

---

## 5. Inference comparison (CLI)

```bash
python inference_comparison.py \
   --baseline scripted \
   --sft  ./checkpoints/sft_v1 \
   --rl   ./checkpoints/dpo_v1 \
   --scenarios STABLE_WEEK BUSY_WEEKEND CRISIS_WEEK \
   --episodes-per-scenario 3
```

Writes `data/comparison_results.json` with per-scenario WRR / SES / parse-success
for each backend.

Verify:
- `parse_success` should be ≥ 0.9 for SFT and RL, < 0.5 for scripted on hard scenarios.
- `store_efficiency_score` for `rl` should be > `sft` > `baseline` on average.
- If they're all equal, the checkpoints loaded the **base** model, not the LoRA
  adapter. Check `LocalAgentRuntime.__post_init__` log line.

---

## 6. Dashboard

```bash
export SFT_MODEL_PATH=./checkpoints/sft_v1   # optional
export RL_MODEL_PATH=./checkpoints/dpo_v1    # optional (otherwise scripted only)
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Visit <http://localhost:8000/>. Panels:

- **Project Atlas** — 6 envs / 8 agents / 7 reward components / 3 global penalties / 6 scenarios
- **Live Reward Signal** — SES + r1..r7 gauges, red flash on penalty events
- **RL Training Telemetry** — 4-stage indicator + loss/KL canvas + buffer feed
- **Decision Flow** — observation → brief → reward chain → learn step
- **Before vs After RL** — baseline / SFT / RL side-by-side briefs

If `DASHBOARD_URL` was set during Kaggle training, the RL Training Telemetry
panel will have already collected the live training trace.

---

## 7. Optional — PPO + critic + external grader

The repo also ships:

- `training/ppo_trainer.py` — clipped-surrogate PPO with KL anchor (drop-in
  alternative to REINFORCE; same `progress_callback` contract)
- `training/reinforce_trainer.py` value-head baseline — pass
  `use_value_baseline=True` to cut variance
- `eval/external_grader.py` — Patronus / Halluminate adapter; falls back to a
  local heuristic when no API keys are set

```bash
# PPO instead of REINFORCE
python -m training.ppo_trainer  # entry point in __main__

# Add external grader to the comparison
PATRONUS_API_KEY=... python inference_comparison.py --use-external-grader
```

---

## Expected artifacts checklist

After a full pipeline run you should have:

```
training/sft_data/{pricing,farmer,trend}_examples.json   # 3 × 93 examples
checkpoints/sft_v1/                                       # SFT
checkpoints/reinforce_v1/                                 # post-RL
checkpoints/dpo_v1/                                       # post-DPO
data/comparison_results.json                              # 3-backend metrics
plots/reinforce_curve.png                                 # loss + KL trace
training_log.jsonl                                        # one row per opt step
episode_log.jsonl                                         # one row per episode
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Both max_new_tokens (=600) and max_length(=32768) seem to have been set` | Qwen2.5 default `generation_config.max_length` | Already silenced via `freshprice_env._gen_utils.quiet_generation_config()` and a notebook-wide `warnings.filterwarnings(...)` cell |
| REINFORCE prints `Collected 0 (prompt, brief, reward) tuples` | Rollout cell stored only `raw_response` | Already fixed — `collect_samples` accepts `prompt│observation` and `brief_text│completion│raw_response` |
| All scenarios FAIL the quality gate | Brief grader rejecting the SFT output | Inspect `cell-eval` per-section flags; usually a missing `DIRECTIVE` JSON. Re-run `cell-sft-train` with more epochs. |
| `LocalAgentRuntime` loads but generates gibberish | Adapter loaded onto wrong base model | Make sure `MODEL_PATH` points at a *merged 16-bit* checkpoint, not a raw LoRA adapter |
| HF Hub 401 | Qwen-2.5-7B is gated | Accept the licence on the model page, then `huggingface-cli login` |
