---
title: QStorePrice Train
emoji: 🥬
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
suggested_hardware: a10g-small
suggested_storage: small
short_description: SFT + REINFORCE + DPO trainer for the QStorePrice agent
---

# QStorePrice training Space

This Space builds the GPU training image (`Dockerfile.train`) and runs
`scripts/train_full_pipeline.py --push-to-hub` once on its allocated
GPU. Trained weights end up on the HF Hub repo named in the
`HF_REPO_ID` Secret.

## One-time setup

1. Hardware: A10G small (recommended). Cost ≈ $1.05 / hour, full
   pipeline ≈ 70 min ≈ $1.20 per run.
2. Settings -> Variables and secrets:
   - `HF_TOKEN`     — read+write token (Secret, NOT a Variable)
   - `HF_REPO_ID`   — `<you>/qstoreprice-sft`
3. Settings -> Sleep: never (so it doesn't pause mid-training)

## What you'll see in the build log

```
[train] starting full pipeline ...
[stage 1/7] generating SFT data ...
[stage 2/7] running SFT -> checkpoints/sft_v1
... (~25 min) ...
[stage 3/7] collecting rollouts ...
[stage 4/7] REINFORCE+KL ...
[stage 5/7] DPO ...
[stage 6/7] inference_comparison ...
[stage 7/7] pushing checkpoints/dpo_v1 -> nandeshjeyalakshmi/qstoreprice-sft
[train] done. weights pushed to nandeshjeyalakshmi/qstoreprice-sft
[idle] training complete -- heartbeat is keeping the Space alive.
```

## After it finishes

On your laptop:

```bash
huggingface-cli download <you>/qstoreprice-sft --local-dir checkpoints/dpo_v1
# OR run dashboard against HF Inference (no GPU needed):
export AGENT_BACKEND=hf_inference
uvicorn server.app:app --reload
```

You can also pause / delete the Space — your trained weights live on HF
Hub, not on the Space's storage.

## Re-running training

Restart the Space (Settings -> Restart this space). It will rebuild the
image, run the pipeline again, and push to the same HF Hub repo
(overwriting the prior version unless you change `HF_REPO_ID`).
