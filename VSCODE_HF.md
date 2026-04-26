# VS Code + Hugging Face GPU — full guide (no Kaggle)

This is the alternative to the Kaggle notebook for users who:

* prefer to develop everything in VS Code,
* have purchased Hugging Face GPU credit (Pro / Spaces / Inference Endpoints),
* want one bash command to run SFT → REINFORCE → DPO → comparison → HF push.

---

## 0. Set up `.env`

```bash
cp .env.example .env
$EDITOR .env       # paste your *new* HF_TOKEN (the old one was leaked, revoke it)
```

Required keys:

```
HF_TOKEN=hf_NEW_TOKEN_HERE
HF_REPO_ID=<your-username>/qstoreprice-sft
```

Optional keys (everything else has sensible defaults; see `.env.example`).

> **Security:** `.env` is gitignored. Never paste the token into a chat,
> Slack, code comment, or `git commit` body. Rotate it any time it leaves
> your machine.

---

## 1. Install dependencies

```bash
# In the repo root
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_training.txt             # GPU box only
pip install python-dotenv                            # optional, the script has a tiny built-in loader
```

`requirements_training.txt` pulls Unsloth + bitsandbytes + TRL. It needs
a CUDA GPU at `import` time, so install on the box that will run training.

---

## 2. Choose where the GPU lives

You have three viable HF-backed paths. Pick whichever matches your account:

### Path A — HF Spaces with GPU (Pro account, ZeroGPU credit)

Best when you only have Pro credit and no dedicated VM.

```bash
# 1. Push the repo to a Space:
huggingface-cli login                           # uses HF_TOKEN
huggingface-cli repo create qstoreprice-train --type=space --space_sdk=docker
git remote add space https://huggingface.co/spaces/<you>/qstoreprice-train
git push space main

# 2. In the Space settings (huggingface.co/spaces/<you>/qstoreprice-train/settings):
#    - Hardware: select T4 small / A10G small (cheapest) or A100 (faster)
#    - Variables: set HF_TOKEN, HF_REPO_ID as Secrets
#    - Sleep time: never (so it doesn't pause mid-training)

# 3. Trigger training by SSH'ing into the running Space:
huggingface-cli login
huggingface-cli space-build run \
    --space-id <you>/qstoreprice-train \
    -- python scripts/train_full_pipeline.py --push-to-hub
```

### Path B — HF Jobs (newer, simplest one-shot)

If your account has access to `hf jobs`:

```bash
huggingface-cli jobs run \
    --image python:3.10 \
    --hardware a10g-large \
    --secret HF_TOKEN \
    --env HF_REPO_ID=<you>/qstoreprice-sft \
    -- bash -c "git clone https://github.com/SatyasaiDevarakonda/metai.git && \
                cd metai && pip install -r requirements.txt -r requirements_training.txt && \
                python scripts/train_full_pipeline.py --push-to-hub"
```

### Path C — A real GPU box reachable over SSH from VS Code

Most reliable; uses VS Code Remote-SSH.

```bash
# 1. On your laptop:
code --install-extension ms-vscode-remote.remote-ssh
# 2. F1 -> "Remote-SSH: Connect to Host" -> add your GPU box IP
# 3. Open the metai folder on the remote
# 4. In the remote VS Code terminal:
git clone https://github.com/SatyasaiDevarakonda/metai.git
cd metai
cp .env.example .env && $EDITOR .env
pip install -r requirements.txt -r requirements_training.txt
python scripts/train_full_pipeline.py --push-to-hub
```

---

## 3. The single command that does everything

Once you're on the GPU box (Path A / B / C):

```bash
python scripts/train_full_pipeline.py --push-to-hub
```

What it does, in order:

| Stage | Output |
|---|---|
| 1. SFT data generation (`training/generate_sft_data.py`) | `training/sft_data/{pricing,farmer,trend}_examples.json` |
| 2. SFT (`training/sft_trainer.py`) | `checkpoints/sft_v1/` |
| 3. Rollouts (`training/grpo_trainer.py`) | trajectory buffer (in memory) |
| 4. REINFORCE+KL (`training/reinforce_trainer.py`) | `checkpoints/reinforce_v1/` |
| 5. DPO (`training/dpo_trainer.py`) | `checkpoints/dpo_v1/` |
| 6. Comparison (`inference_comparison.py`) | `data/comparison_results.json` |
| 7. Push to HF Hub | `https://huggingface.co/<your-repo>` |

Useful flags (all optional):

```bash
# Skip stages you've already run
python scripts/train_full_pipeline.py --skip-sft --skip-reinforce
# Run with the cheaper 1.5B model first to validate the pipeline
python scripts/train_full_pipeline.py --model-id Qwen/Qwen2.5-1.5B-Instruct
# Don't push (just write to local disk)
python scripts/train_full_pipeline.py
# Add Patronus / Halluminate scoring to the comparison
EXTERNAL_GRADER=patronus PATRONUS_API_KEY=... \
    python scripts/train_full_pipeline.py --use-external-grader
```

Total wall-clock for Qwen-2.5-7B on an A10G:
- SFT : ~25 min
- Rollouts : ~10 min
- REINFORCE : ~15 min
- DPO : ~12 min
- Comparison : ~5 min
- **Total : ~70 min**

On a T4 (small), expect ~2x.

---

## 4. After training — pull the weights back to your laptop (VS Code)

```bash
# On your laptop (no GPU needed):
git pull
cp .env.example .env && $EDITOR .env       # paste new HF_TOKEN
huggingface-cli login                       # uses HF_TOKEN

# Pull the merged checkpoint
huggingface-cli download <you>/qstoreprice-sft \
    --local-dir checkpoints/dpo_v1
```

Run the dashboard against the trained model:

```bash
export AGENT_BACKEND=local
export MODEL_PATH=$PWD/checkpoints/dpo_v1
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
# http://localhost:8000  -> all 14 panels including Before vs After RL
```

If your laptop has no GPU, skip the local download — point the agent
runtime at HF Inference instead:

```bash
export AGENT_BACKEND=hf_inference
# HF_REPO_ID + HF_TOKEN already in .env
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

The dashboard now serves the trained model through HF's inference router
(uses your HF credit; no local GPU).

---

## 5. Live training dashboard (optional, very impressive)

If you want the dashboard to update in real time while training runs on
the remote GPU:

```bash
# Laptop (terminal 1):
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Laptop (terminal 2):
ngrok http 8000
# -> copy the https URL, e.g. https://abc123.ngrok.io

# Remote GPU box, BEFORE running the pipeline:
export DASHBOARD_URL=https://abc123.ngrok.io
python scripts/train_full_pipeline.py --push-to-hub
```

Every REINFORCE optimizer step now POSTs to `/training/event` on your
laptop — the RL Telemetry panel updates live.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `HF_TOKEN is not set` | `cat .env` — make sure the file is in the project root and has no quotes around the value |
| `401 / gated model` from Qwen | Visit https://huggingface.co/Qwen/Qwen2.5-7B-Instruct, click "Accept license", then re-run |
| `OSError: CUDA out of memory` | Drop to `--model-id Qwen/Qwen2.5-1.5B-Instruct` first, or reduce `--batch-size` to 1 and `--grad-accum` to 8 |
| `cannot import name '_get_template_variables'` | `pip install "transformers>=4.46.0,<4.55.0"` then re-run |
| `Unsloth should be imported before transformers` | Run `python scripts/train_full_pipeline.py` directly — the script imports unsloth first; don't do `import transformers` in any cell preceding it |
| All scenarios show `parse_success=False` | The model is producing free-form text, not the 6-section schema. Re-run SFT with more epochs (`--sft-epochs 3`). |
| Comparison shows RL ≈ baseline | Verify the checkpoint actually loaded — search the log for `Loading local model from <path>` and confirm the path matches `checkpoints/dpo_v1` |

---

## 7. Module map (every Python module has a runnable entry)

| Module | How to exercise from VS Code |
|---|---|
| `freshprice_env/*` | `python -m unittest discover tests -v` (84 tests) |
| `training/sft_trainer.py` | stage 2 of `train_full_pipeline.py`, or `python -m training.sft_trainer ...` |
| `training/grpo_trainer.py` | stage 3 |
| `training/reinforce_trainer.py` | stage 4 (with value-head baseline) |
| `training/ppo_trainer.py` | `python -m training.ppo_trainer --buffer ... --checkpoint ...` |
| `training/dpo_trainer.py` | stage 5 |
| `training/oversight_trainer.py` | `python -c "from training.oversight_trainer import run_sft; ..."` |
| `training/self_play.py` | `python -c "from training.self_play import smoke_test; print(smoke_test(4))"` |
| `eval/evaluator.py` | `python -m eval.evaluator --checkpoint checkpoints/dpo_v1` |
| `eval/baseline.py` | `python -m eval.baseline --model-id Qwen/Qwen2.5-7B-Instruct` |
| `eval/baselines/*` | `python -m eval.baselines.run_baselines` |
| `eval/anti_hack_checker.py` | `python -m eval.anti_hack_checker --buffer trajectory_buffer.jsonl` |
| `eval/counterfactual_replay.py` | `python -c "from eval.counterfactual_replay import ...; ..."` |
| `eval/theory_of_mind_probe.py` | imported by `inference_comparison` when `--probe-tom` is added |
| `eval/external_grader.py` | `EXTERNAL_GRADER=patronus python inference_comparison.py --use-external-grader` |
| `inference_comparison.py` | stage 6, or `python inference_comparison.py ...` |
| `server/app.py` | `uvicorn server.app:app --reload` |
| `server/agent_runtime.py` | imported by `server/app.py`; tested via `python -m unittest tests.test_inference_comparison` |
| `validate_submission.py` | `python validate_submission.py` (final gate) |

---

**TL;DR.** Put your HF token in `.env`, run `python scripts/train_full_pipeline.py --push-to-hub` on whatever HF GPU you have credit for, then `git pull` + `uvicorn server.app:app` on your laptop. No Kaggle needed.
