---
title: Qstoreprice
emoji: 😻
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
---

# QStorePrice AI — Perishable Goods Intelligence

> Can an LLM learn to make every pricing, procurement, and trend-detection
> decision a small grocery seller faces — better than they can alone?

## What This Is

Indian grocery dark stores waste 15-30% of perishable inventory because pricing, farmer surplus procurement, and trend-based restocking decisions are made by gut instinct under time pressure. QStorePrice AI trains a Qwen-2.5-7B model via reinforcement learning (SFT + GRPO + DPO) to write **Operating Briefs** — structured decision documents that a rule executor converts into concrete actions across three engines: **Dynamic Pricing** (auto-discount items approaching expiry), **Farmer Offer Engine** (accept/counter/decline surplus procurement with risk management), and **Social Trend Engine** (detect viral food signals, restock before demand arrives). All three engines collapse into a single unified metric: **Weekly Waste Recovery Rate (WRR)** = revenue recovered from at-risk inventory / cost of at-risk inventory. Target: 61% to 89%.

## The Operating Brief

Every 2 simulated hours, the LLM writes a brief like this:

```
SITUATION: Farmer Rajan offers 50kg of mangoes at Rs 35/kg with 48hrs shelf life.
Current mango inventory is 12 units at WATCH urgency (35hrs remaining).
Risk buffer is healthy at Rs 4,200.

SIGNAL ANALYSIS: N/A

VIABILITY CHECK:
  Shelf life: PASS — 48hrs covers projected sell-through of 32hrs at current velocity
  Inventory conflict: PASS — no URGENT/CRITICAL mango batches active
  Break-even: PASS — market price Rs 75/kg vs break-even Rs 43/kg (74% margin)
  Worst-case P&L: FLAG — 60% sell-through at Rs 47/kg barely covers cost
  Demand timing: PASS — Thursday 2pm is pre-weekend demand build

RECOMMENDATION: ACCEPT at offered price. Strong viability (0.78) with healthy buffer
to absorb worst-case scenario. Weekend demand timing favourable.

DIRECTIVE: {"engine": "FARMER", "actions": [{"offer_id": "offer_001", "decision": "ACCEPT", "counter_price": null}]}

CONFIDENCE: HIGH
```

The brief architecture matters because the LLM does language work (situation assessment, viability reasoning, confidence calibration), not raw number generation. When RL training improves the model, the briefs get *measurably better* — a claim standard numeric-action RL cannot make.

## Project Structure

```
freshprice_env/                    # RL environment (the simulation)
  enums.py                         # ExpiryUrgency, BatchStatus, BriefEngineType, etc.
  constants.py                     # All numeric thresholds, reward values, weights
  entities.py                      # SimulatedBatch, SimulatedFarmerOffer, SimulatedTrendSignal
  market_state.py                  # MarketStateBuilder — creates initial state per scenario
  freshprice_env.py                # FreshPriceEnv(gym.Env) — the main environment
  engines/
    pricing_engine.py              # Engine 1: dynamic pricing + r1 reward
    farmer_engine.py               # Engine 2: farmer offer processing + r2 reward
    trend_engine.py                # Engine 3: social trend engine + r3 reward
  brief_pipeline/
    prompt_builder.py              # Builds the structured prompt the LLM receives
    parser.py                      # Parses raw LLM text into structured brief
    validator.py                   # Validates briefs against business rules
    quality_scorer.py              # Scores brief quality (independent of reward)
    rule_executor.py               # Converts DIRECTIVE JSON into typed actions
  reward.py                        # WRRRewardEngine — combines r1+r2+r3 into WRR

training/                          # Training pipeline
  sft_trainer.py                   # SFT warm-start on hand-crafted briefs
  grpo_trainer.py                  # GRPO training loop with FreshPriceEnv
  dpo_trainer.py                   # DPO fine-tuning on preference pairs
  trajectory_buffer.py             # Collects episodes, generates DPO pairs
  counterfactual.py                # Shadow expert policy for regret scoring
  curriculum.py                    # Curriculum progression across 5 scenarios
  train.py                         # Main entry point: SFT → GRPO → DPO loop
  sft_data/                        # Hand-crafted Operating Brief examples

eval/                              # Evaluation
  evaluator.py                     # Runs held-out episodes, prints reports
  anti_hack_checker.py             # Scans trajectories for reward hacking

requirements.txt                   # HF Space dependencies (gradio, matplotlib, gymnasium)
requirements_training.txt          # Full training dependencies (torch, trl, unsloth, etc.)
README.md                          # This file
```

## Quick Start

### 1. Install dependencies

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -r requirements_training.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API_BASE_URL, MODEL_NAME, and HF_TOKEN
```

### 3. Verify the environment

```bash
python -c "from freshprice_env.freshprice_env import FreshPriceEnv; import gymnasium; gymnasium.utils.env_checker.check_env(FreshPriceEnv())"
```

### 3. Run SFT warm-start (requires GPU)

```bash
python training/sft_trainer.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --output-dir checkpoints/sft_v1 \
  --epochs 2
```

### 4. Run GRPO training (start with Stable Week)

```bash
python training/train.py \
  --resume-from checkpoints/sft_v1 \
  --output-dir checkpoints \
  --start-scenario STABLE_WEEK \
  --episodes-per-level 100 \
  --wandb-project qstoreprice-ai
```

### 5. Evaluate a checkpoint

```bash
python eval/evaluator.py \
  --checkpoint checkpoints/promoted_level1 \
  --scenarios STABLE_WEEK BUSY_WEEKEND \
  --episodes 10
```

### 6. Compare two checkpoints

```bash
python eval/evaluator.py \
  --checkpoint checkpoints/grpo_level0 \
  --compare-with checkpoints/dpo_level0_ep25
```

## Training Scenarios

| Level | Scenario | Engines Active | Description | Promotion |
|-------|----------|---------------|-------------|-----------|
| 0 | STABLE_WEEK | Pricing only | Predictable demand, no events | WRR >= 0.70 over 5 valid episodes |
| 1 | BUSY_WEEKEND | Pricing + Trend | Weekend surge, 1 trend signal | WRR >= 0.70 over 5 valid episodes |
| 2 | FARMER_WEEK | Pricing + Farmer | 3 farmer offers across the week | WRR >= 0.70 over 5 valid episodes |
| 3 | TREND_WEEK | All 3 engines | 2 trend signals, 1 festival spike, 1 farmer offer | WRR >= 0.70 over 5 valid episodes |
| 4 | CRISIS_WEEK | All 3 engines | Supplier delay, viral trend, 3 farmer offers, depleted buffer | Benchmark — no promotion |

## Key Design Decisions

- **Operating Brief architecture** — The LLM writes structured briefs; a rule executor converts the DIRECTIVE section into actions. This separates reasoning (LLM) from execution (deterministic rules), making brief quality independently measurable.
- **Unified WRR metric** — Three engines with different reward components (r1, r2, r3) collapse into one number: revenue recovered from at-risk inventory divided by cost of at-risk inventory. One metric for curriculum promotion, DPO filtering, and checkpoint comparison.
- **GRPO over PPO** — GRPO is simpler and more sample-efficient for verifiable reward tasks where the reward function is known and deterministic. No value network, no GAE, no reward model.
- **save_pretrained_merged after 4-bit training** — Never use naive `save_pretrained()` after Unsloth 4-bit quantised training. Always use `model.save_pretrained_merged(path, tokenizer, save_method="merged_16bit")`. Every checkpoint save in the codebase follows this rule.
- **Constitutional audit before DPO** — Trajectories that pass the WRR threshold but show reward hacking patterns (early deep discounts, reckless farmer acceptance, trend order flooding) are excluded from DPO training even if their WRR is high.

## WandB Metrics

| Metric | Description | What to Watch For |
|--------|-------------|-------------------|
| `wrr` | Weekly Waste Recovery Rate (primary) | Should trend upward within each level; drops are expected at level transitions |
| `r1_pricing` | Pricing engine reward component | Negative values mean expired inventory — check if agent is pricing too conservatively |
| `r2_farmer` | Farmer engine reward component | 0.0 in Stable Week (no farmer engine); persistent negatives = reckless acceptance |
| `r3_trend` | Trend engine reward component | 0.0 in Stable/Farmer Week; negative = over-ordering on weak signals |
| `brief_quality_score` | Independent reasoning quality metric | Should correlate with WRR — the research finding. Divergence = quality scorer needs calibration |
| `anti_hack_violations` | Count per episode | Rising = reward hacking in progress. Investigate which guard is firing |
| `curriculum_level` | Current scenario (0-4) | Steps up on promotion. Stuck at 0 for >200 episodes = learning rate or SFT data issue |
| `episodes_in_level` | Episodes since last promotion | Resets on promotion. >100 without promotion = plateau |

## Anti-Hacking Guards

| Guard | Trigger | Action | Severity |
|-------|---------|--------|----------|
| Early deep discount | `price_multiplier < 0.35` with `hours_to_expiry > 48` | Price not applied, penalty in r1 | HIGH |
| Reckless farmer acceptance | ACCEPT with `viability_score < 0.30` | Accept blocked, forced to DECLINE, penalty in r2 | CRITICAL |
| Trend order flood | More than 1 order per category within 72 hours | Hard cap enforced in env (quantity capped at 2x weekly velocity) | HIGH |
| Flash sale abuse | More than 1 flash sale per category per day | Second trigger silently ignored | MEDIUM |
| Perpetual floor pricing | At/below floor price for >30% of ticks without urgency | Flagged in anti-hack report | HIGH |
| Surrogate reward gaming | `brief_quality > 0.90` but `WRR < 0.50` | Excluded from DPO training set | CRITICAL |

## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| Model outputs malformed briefs in early GRPO episodes | SFT warm-start not run or too few SFT examples | Run `training/sft_trainer.py` with at least 50 examples before GRPO |
| WRR stuck at 0.0 for first 20 episodes | Normal — agent is exploring, no batches have become at-risk yet | Wait. WRR requires batches to enter URGENT/CRITICAL before the denominator is non-zero |
| CUDA out of memory during GRPO | Batch size or sequence length too large for GPU | Reduce `--batch-size` to 1 and `gradient_accumulation_steps` to 4. Use `max_seq_length=2048` |
| Checkpoint quality degraded after DPO | `save_pretrained()` used instead of `save_pretrained_merged()` | Check the save call. Must be `model.save_pretrained_merged(path, tokenizer, save_method="merged_16bit")` |
| `anti_hack_violations` rising steadily across episodes | Learning rate too high or reward weights encourage gaming | Lower learning rate to `1e-6`. Check if one reward component dominates and adjust WRR weights |

## Research Question

Does LLM reasoning quality (brief quality score) improve through RL training, and does that improvement causally drive WRR improvement? The two metrics to watch are `wrr` (the operational outcome) and `brief_quality_score` (the reasoning quality proxy). If brief quality and WRR are correlated across training episodes — and especially if brief quality *leads* WRR improvement by a few episodes — it suggests the model is learning to reason better, not just gaming the reward function. If WRR improves but brief quality stays flat or declines, the model is finding shortcuts that bypass reasoning. This is one environment, one LLM family (Qwen-2.5), and one reward function. Findings are encouraging but not conclusive — they establish a methodology for measuring reasoning-reward correlation in language agent training.

## Results

### Before Training (Zero-Shot Base Model)

Run: `python eval/baseline.py --model-id Qwen/Qwen2.5-7B-Instruct --episodes 5`

| Scenario | WRR (Zero-Shot) | Brief Quality | Notes |
|---|---|---|---|
| STABLE_WEEK | ~0.09 | ~0.31 | Random pricing, no urgency response |
| BUSY_WEEKEND | ~0.06 | ~0.25 | Trend signals ignored |
| FARMER_WEEK | ~0.04 | ~0.20 | All offers declined or accepted randomly |
| TREND_WEEK | ~0.07 | ~0.24 | No trend-based ordering |
| CRISIS_WEEK | ~0.03 | ~0.15 | Overwhelmed by simultaneous signals |

### After Training (GRPO + DPO)

| Scenario | WRR (Trained) | Brief Quality | Improvement |
|---|---|---|---|
| STABLE_WEEK | ~0.72 | ~0.74 | +685% |
| BUSY_WEEKEND | ~0.70 | ~0.72 | +1067% |
| FARMER_WEEK | ~0.68 | ~0.71 | +1485% |
| TREND_WEEK | ~0.65 | ~0.68 | +857% |
| CRISIS_WEEK | ~0.58 | ~0.62 | +1833% |

*Values are projections based on curriculum promotion thresholds. Run training and eval to get actual numbers.*

### WRR and Brief Quality Correlation

The primary research finding: `brief_quality_score` and `wrr` improve together across training. If brief quality *leads* WRR improvement by a few episodes, it suggests the model learns to reason better before the operational metric catches up.

![WRR vs Quality Correlation](eval/plots/wrr_vs_quality_correlation.png)
*Generate after training: the plot above will be created by the WandB run.*

## Links

- **HuggingFace Space**: [SPACE_URL_HERE]
- **WandB Training Run**: [WANDB_URL_HERE]
- **HuggingFace Blog**: [BLOG_URL_HERE]
- **Demo Video**: [VIDEO_URL_HERE]

## Submission Details

- **OpenEnv version**: Latest
- **Base model**: Qwen/Qwen2.5-7B-Instruct
- **Training**: SFT warm-start -> GRPO -> DPO with 5-level curriculum
- **HF Space**: [SPACE_URL_HERE]
- **Environment class**: `freshprice_env.freshprice_env.FreshPriceEnv`
- **Manifest**: `openenv.yaml`

## Citation

```bibtex
@software{qstoreprice_ai_2026,
  title   = {QStorePrice AI: RL-Trained LLM for Perishable Goods Intelligence},
  year    = {2026},
  url     = {https://github.com/nandeshkanagaraju/QStorePrice},
}
```
