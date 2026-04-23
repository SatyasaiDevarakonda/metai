# QStorePrice AI — Complete Developer Guide

A new developer joining this project should be able to read this file and understand the entire codebase without asking anyone anything.

---

## 1. What This Project Is

Indian grocery dark stores and small online sellers waste 15-30% of their perishable inventory every week. Tomatoes expire unsold while demand surges for trending recipes. Farmer surplus arrives at the wrong time. Flash sales fire too late. The root cause is the same everywhere: pricing, procurement, and trend-detection decisions are made by gut instinct under time pressure, with no systematic way to learn from outcomes.

QStorePrice AI trains a Qwen-2.5-7B language model via reinforcement learning to write **Operating Briefs** — structured 6-section decision documents that a deterministic rule executor converts into concrete actions. The LLM does language work (situation assessment, viability reasoning, confidence calibration). A rule executor reads the DIRECTIVE section and applies the numeric values. When RL training improves the model, the briefs get measurably better — a claim about language understanding improving that standard numeric-action RL cannot make. The seller can read every decision the AI makes in plain language.

The system has three engines in v1: **Dynamic Pricing** (auto-discount items as they approach expiry), **Farmer Offer Engine** (accept/counter/decline surplus procurement with risk management), and **Social Trend Engine** (detect viral food signals on Instagram/YouTube, restock before demand arrives). All three engines contribute to a single unified metric: **Weekly Waste Recovery Rate (WRR)** = revenue recovered from at-risk inventory / cost of at-risk inventory. The training target is WRR from ~0.09 (zero-shot base model) to >= 0.70 (promotion threshold).

---

## 2. The Complete File Map

### freshprice_env/ — The RL Environment

```
freshprice_env/
  __init__.py               — Package init
  enums.py                  — All enumerations: ExpiryUrgency, BatchStatus, BriefEngineType,
                              FarmerOfferStatus, TrendAction, BriefConfidence, SellerAction,
                              ViabilityOutcome, CompensationPolicy, SignalSource, CurriculumScenario
  constants.py              — Every numeric constant organised by subsystem: episode structure,
                              urgency thresholds, pricing limits, reward values (stored positive),
                              anti-hack thresholds, WRR weights (0.40/0.30/0.30), curriculum
                              promotion thresholds, simulation defaults (6 categories)
  entities.py               — SimulatedBatch (frozen), SimulatedFarmerOffer (frozen),
                              SimulatedTrendSignal (frozen), SimulatedMarketState (mutable —
                              mutated every tick by the environment)
  market_state.py           — MarketStateBuilder with 5 scenario factories (_build_stable_week
                              through _build_crisis_week) + get_base_demand_velocity() with
                              time-of-day and day-of-week multipliers
  reward.py                 — WRRRewardEngine: accumulates r1+r2+r3 per tick, computes
                              episode reward, runs constitutional audit (4 checks), produces
                              WandB-ready log dicts
  freshprice_env.py         — FreshPriceEnv(gym.Env): the main environment. reset() builds
                              state and engines, step() runs the 10-step pipeline (parse →
                              validate → score → execute → simulate → reward → terminate →
                              next engine → next prompt → return)
```

### freshprice_env/engines/ — Simulation Engines

```
freshprice_env/engines/
  __init__.py               — Package init
  pricing_engine.py         — Engine 1: _apply_directive (price multipliers, flash sales,
                              anti-hack blocking), _compute_sales (elasticity model with
                              ±15% noise), _age_batches (decrement hours_to_expiry by 0.25),
                              _update_at_risk_accumulators (deduped via _at_risk_seen set),
                              _compute_r1 (base ratio + near-expiry bonus - expired penalty
                              - antihack penalty)
  farmer_engine.py          — Engine 2: score_offer (5-factor viability: shelf life, inventory
                              conflict, break-even, worst-case P&L, demand timing),
                              process_directive (ACCEPT/COUNTER/DECLINE with simulated farmer
                              response via rng), resolve_outcomes (risk buffer update),
                              _compute_r2
  trend_engine.py           — Engine 3: apply_trend_demand_boost (every tick, recalculated
                              from composite_score), process_directive (APPROVE/DECLINE with
                              72hr cooldown and 2x velocity hard cap), resolve_trend_outcomes
                              (PERFECT/OVERTRADE/NORMAL classification), inject_trend_signal
```

### freshprice_env/brief_pipeline/ — LLM Interface

```
freshprice_env/brief_pipeline/
  __init__.py               — Package init
  prompt_builder.py         — OperatingBriefPromptBuilder: SYSTEM_PROMPT constant + 3
                              engine-specific user prompt builders. Shows full batch IDs
                              (not truncated) so the LLM can write them in DIRECTIVE JSON.
  parser.py                 — BriefParser: regex extraction of 6 sections, JSON parsing with
                              4-step fallback (direct, brace extraction, single→double quotes,
                              trailing comma removal). Never raises — returns ParseResult
                              with success=False and failure_reason.
  validator.py              — BriefValidator: engine-specific validation. Errors = brief
                              rejected (unknown batch_id, multiplier out of range, counter
                              price > 150% of ask). Warnings = brief proceeds (all CRITICALs
                              held at 1.0, flash sale with no credits, low-confidence approval).
  quality_scorer.py         — BriefQualityScorer: 0.0-1.0 score from three sub-scores
                              (situation 0.33, reasoning 0.34, directive 0.33). Keyword-based
                              pattern matching — not semantic understanding. This is the
                              independent research metric tracked alongside WRR.
  rule_executor.py          — RuleExecutor: converts directive dict to typed action objects
                              (PricingAction, FarmerAction, TrendActionResult). Sets
                              was_antihack_blocked / was_capped flags. Never raises — returns
                              ExecutionResult with warnings.
```

### training/ — Training Pipeline

```
training/
  __init__.py               — Package init
  curriculum.py             — CurriculumManager: tracks EpisodeResult, checks promotion
                              (WRR >= 0.70 over 5 consecutive valid+constitutional episodes),
                              provides eval seeds per level, reports status for WandB.
  trajectory_buffer.py      — TrajectoryBuffer: stores top-50 episodes (gated by valid AND
                              constitutional), generates DPO pairs (counterfactual matching by
                              engine_type + scenario, 3x duplication for regret > 0.7),
                              clear_below_level on promotion.
  counterfactual.py         — CounterfactualEngine: deterministic expert policy (rules-based
                              oracle), regret scoring (normalised to [0,1]), synthetic
                              rejected brief generation (Mode 1: template degradation, Mode 2:
                              20% chance LLM-based), episode-level regret analysis with 5
                              regret type classifications.
  sft_trainer.py            — load_sft_dataset (merges 3 JSON files, Qwen chat template),
                              run_sft (Unsloth 4-bit + LoRA r=16 + SFTTrainer + cosine LR),
                              _verify_checkpoint (load, generate, check 6 sections).
  grpo_trainer.py           — FreshPriceGRPOTrainer: generate() for inference (strips prompt
                              tokens), run_episode() (full episode loop), train() (per-episode:
                              run → record → buffer → curriculum → WandB → save),
                              _save_checkpoint (merged save + 3-episode verification).
  dpo_trainer.py            — run_dpo: validates >= 10 pairs, builds HF Dataset, DPOTrainer
                              with ref_model=None (implicit reference for memory efficiency),
                              merged save, pre/post quality comparison (>= 97% of pre-DPO).
  train.py                  — Main orchestrator: Phase 0 (setup + WandB), Phase 1 (SFT or
                              resume), Phase 2 (GRPO + DPO curriculum loop with eval and
                              periodic DPO), KeyboardInterrupt handler saves emergency
                              checkpoint, Phase 3 (summary).
```

### training/sft_data/ — Hand-Crafted Examples

```
training/sft_data/
  pricing_briefs.json       — 20+ PRICING Operating Brief examples covering FRESH/WATCH/
                              URGENT/CRITICAL urgency tiers including 5 where the correct
                              answer overrides the naive choice
  farmer_briefs.json        — 20+ FARMER examples covering ACCEPT/COUNTER/DECLINE with
                              correct viability reasoning
  trend_briefs.json         — 15+ TREND examples covering APPROVE/DECLINE with composite
                              score reasoning
```

### eval/ — Evaluation

```
eval/
  __init__.py               — Package init
  evaluator.py              — Evaluator: loads model for inference only (greedy decoding,
                              do_sample=False), run_evaluation across scenarios with fixed
                              seeds, print_report (WRR mean+/-std, quality, violations,
                              constitutional pass rate), compare_checkpoints (side-by-side
                              table with deltas and winner).
  anti_hack_checker.py      — AntiHackChecker: 8 pattern detectors (EARLY_DEEP_DISCOUNT,
                              PERPETUAL_FLOOR_PRICING, FLASH_SALE_FLOOD, RECKLESS_ACCEPTANCE,
                              SYSTEMATIC_AVOIDANCE, TREND_ORDER_FLOOD, TREND_OVERCONFIDENCE,
                              SURROGATE_REWARD_GAMING). Recommendation: Include/Flag/Exclude.
                              scan_trajectory_buffer for batch analysis.
  baseline.py               — run_baseline: zero-shot base model evaluation on fixed seeds,
                              saves baseline_results.json, prints comparison-ready table.
```

### Root Files

```
app.py                      — Gradio demo for HuggingFace Spaces: scenario selector, state
                              observation display, editable brief template, submit button,
                              live WRR chart with 0.70 threshold line
openenv.yaml                — OpenEnv manifest: environment class path, space descriptions,
                              engine definitions, curriculum config, tags
requirements.txt            — Space dependencies only: gradio, matplotlib, gymnasium, numpy
requirements_training.txt   — Full training stack: torch, transformers, trl, peft, unsloth,
                              accelerate, bitsandbytes, wandb, etc.
README.md                   — Project overview with HF Space frontmatter, quick start,
                              training scenarios, design decisions, WandB metrics, results
CLAUDE.md                   — Guidance for Claude Code: build commands, architecture,
                              critical rules, naming conventions
DEVELOPER_GUIDE.md          — This file
.gitignore                  — Excludes __pycache__, checkpoints, wandb, .env, baseline results
```

---

## 3. Key Design Decisions and Why

### Decision 1: Operating Brief Architecture

The LLM writes a 6-section structured document (SITUATION, SIGNAL ANALYSIS, VIABILITY CHECK, RECOMMENDATION, DIRECTIVE, CONFIDENCE). A deterministic rule executor reads the DIRECTIVE section and converts it to typed action values that the simulation engines consume. The LLM never outputs raw numbers — it writes language. When RL training improves the model's reward, the briefs get measurably better: more accurate situation descriptions, correct viability assessments, better-justified recommendations. This is a claim about language understanding improving through reward optimisation — something standard numeric-action RL cannot demonstrate. As a bonus, every decision is human-readable. The seller can read a brief and understand why the AI discounted their tomatoes.

### Decision 2: Unified WRR Metric

The original design had 7 separate action spaces and objectives. Judge feedback was direct: "7 action spaces is messy." All three engines now contribute to one number: WRR = revenue recovered from at-risk inventory / cost of at-risk inventory. "At-risk" means the batch was URGENT or CRITICAL at any point during the episode. One metric means the agent cannot hack one component (say, over-ordering trend stock) at the expense of another (say, letting farmer batches expire). The pricing engine carries the most weight (0.40 vs 0.30/0.30) because it runs every episode regardless of curriculum level — farmer and trend engines are inactive in early scenarios.

### Decision 3: Cost Locked at At-Risk Transition

When a batch transitions into URGENT or CRITICAL for the first time in an episode, its cost (unit_cost x quantity_remaining at that moment) is added to the WRR denominator and the batch ID is recorded in `PricingEngine._at_risk_seen`. The cost stays in the denominator even if the batch is later cleared at full price. Without this, an agent could game WRR by pre-emptively clearing fresh stock before it reaches at-risk thresholds — the denominator would be artificially small, making any recovered revenue look like a high ratio. The lock ensures the denominator reflects true risk exposure.

### Decision 4: Three Engines in v1

The SDD documents four additional engines for v2: Intra-Fleet Rebalancing, Micro-Manufacturer Pipeline, Event Pre-Positioning, and Surplus Box. These are designed but not implemented in the training environment. The decision was driven by judge feedback: scope to the core research question (does reasoning quality improve through RL?) before adding complexity. Three engines already create a rich enough decision space — 84 briefs per episode across pricing, procurement, and trend detection — to test the hypothesis.

### Decision 5: GRPO over PPO

We use GRPOTrainer from HuggingFace TRL instead of PPO. GRPO (Group Relative Policy Optimisation) is more sample-efficient for verifiable reward tasks where the reward function is known and deterministic. It does not require a value network, GAE computation, or a separate reward model. The official hackathon guide recommends GRPO for exactly this class of problem. The training loop is simpler: environment rollout, reward computation, gradient update. No critic, no KL penalty tuning.

### Decision 6: save_pretrained_merged, Never save_pretrained

Every checkpoint save in the codebase calls `model.save_pretrained_merged(path, tokenizer, save_method="merged_16bit")`. Never `model.save_pretrained()`. After Unsloth 4-bit quantised training with LoRA adapters, naive `save_pretrained()` attempts to merge LoRA weights into the quantised base weights incorrectly. The result is silent quality degradation: the checkpoint loads fine, generates text, but WRR drops 10-30% compared to the in-memory model. This is the most common cause of "the model got worse after saving" and the single most important implementation detail in the training pipeline.

### Decision 7: rng Passed Everywhere, Never Called Globally

Every engine, builder, and factory accepts `rng: random.Random` as a parameter. No file anywhere calls `random.random()` or `random.randint()` from the global module. The environment seeds `rng` with a specific seed before calling any builder. This means episodes can be replayed exactly by seed — critical for debugging (reproduce a specific failure), for the counterfactual engine (replay the expert decision on the same state), and for evaluation (fixed seeds across checkpoints for fair comparison).

### Decision 8: Parser Never Raises

`BriefParser.parse()` returns `ParseResult(success=False, failure_reason="...")` on every failure path. It never raises exceptions. The training loop calls `env.step()` thousands of times per training run. If the parser raised on malformed LLM output, a single bad generation would crash an entire episode. Instead, the environment falls back to the last valid directive and logs the parse failure. The LLM learns from the lower reward of the fallback, not from a crash.

---

## 4. The Operating Brief Format

Every generated brief must have exactly these 6 sections in this order:

```
SITUATION: [2-3 sentences describing current state]
SIGNAL ANALYSIS: [TREND briefs only — N/A for PRICING and FARMER]
VIABILITY CHECK: [FARMER and TREND only — N/A for PRICING]
RECOMMENDATION: [Decision with one-sentence justification]
DIRECTIVE: [Valid JSON that the rule executor parses]
CONFIDENCE: [HIGH | MEDIUM | LOW — one word only]
```

### Complete Example: Farmer Rajan's Mango Offer

**Context**: Farmer Rajan offers 50kg of Alphonso mangoes at Rs 35/kg. Thursday 2pm. Shelf life: 48 hours. No existing mango inventory. Weekend demand expected +35%. Risk buffer healthy at Rs 4,200.

**The brief the LLM should write:**

```
SITUATION: Farmer Rajan offers 50kg of Alphonso mangoes at Rs 35/kg with 48hrs shelf life.
No existing mango inventory in store. Thursday 2pm timing is pre-weekend demand build with
expected +35% velocity uplift. Risk buffer at Rs 4,200 is healthy.

SIGNAL ANALYSIS: N/A

VIABILITY CHECK:
  Shelf life: PASS — 48hrs covers projected sell-through of 28hrs at weekend velocity
  Inventory conflict: PASS — no URGENT/CRITICAL mango batches active
  Break-even: PASS — market price Rs 75/kg vs break-even Rs 43/kg (74% margin)
  Worst-case P&L: FLAG — 60% sell-through at Rs 47/kg barely covers cost at Rs 1,750
  Demand timing: PASS — Thursday pre-weekend positioning is optimal

RECOMMENDATION: ACCEPT at offered price. Viability score 0.82 is strong. Healthy risk buffer
can absorb worst-case scenario. Weekend demand timing makes full sell-through likely.

DIRECTIVE: {"engine": "FARMER", "actions": [{"offer_id": "offer_001", "decision": "ACCEPT", "counter_price": null}]}

CONFIDENCE: HIGH
```

**What the rule executor produces from this DIRECTIVE:**

```python
FarmerAction(
    offer_id="offer_001",
    decision="ACCEPT",
    counter_price=None,
    was_antihack_blocked=False,   # viability 0.82 > 0.30 threshold
)
```

**What the farmer engine does with this action:**
1. Checks viability score (0.82) against anti-hack threshold (0.30) — passes
2. Creates a new `SimulatedBatch(category="fruits", quantity_remaining=50, unit_cost=35.0, original_price=75.0, hours_to_expiry=48.0, batch_type=FARMER_SURPLUS)`
3. Registers the batch in `_active_batch_outcomes` for later outcome resolution
4. Removes the offer from `state.pending_offers`
5. Computes r2 = R2_CLEARED_BATCH_BONUS (0.20) x viability (0.82) = 0.164

---

## 5. The Five Training Scenarios

| Level | Scenario | Engines Active | What It Tests | Promotion Condition |
|-------|----------|---------------|---------------|---------------------|
| 0 | STABLE_WEEK | Pricing only | Basic urgency response: discount CRITICAL, hold FRESH | WRR >= 0.70 over 5 consecutive valid episodes |
| 1 | BUSY_WEEKEND | Pricing + Trend | Weekend demand surge with 1 trend signal. Can the agent approve restocking? | WRR >= 0.70 over 5 consecutive valid episodes |
| 2 | FARMER_WEEK | Pricing + Farmer | 3 farmer offers across the week. Viability assessment, counter-offer strategy | WRR >= 0.70 over 5 consecutive valid episodes |
| 3 | TREND_WEEK | All 3 engines | 2 trend signals + 1 festival demand spike + 1 farmer offer. Multi-engine coordination | WRR >= 0.70 over 5 consecutive valid episodes |
| 4 | CRISIS_WEEK | All 3 engines | Supplier delay, viral trend, 3 farmer offers, depleted risk buffer (60% of seed). The benchmark | No promotion — this is the final test |

**Promotion logic**: An episode counts toward promotion only if both conditions are true: `episode_valid == True` (fewer than 5 anti-hack violations in the episode) AND `constitutional_audit().passed == True` (all 4 constitutional checks pass). Invalid or constitutionally-failed episodes are recorded in the curriculum manager but do not advance the promotion window. The window is the last 5 valid+constitutional episodes — their mean WRR must be >= 0.70. On promotion, `episodes_in_level` resets to 0 and `_recent_results` is cleared.

---

## 6. The Reward System

### WRR Formula

```
WRR = revenue_recovered_accumulator / at_risk_cost_accumulator
```

- **at_risk_cost_accumulator**: When a batch transitions to URGENT or CRITICAL for the first time in the episode, `unit_cost x quantity_remaining` at that moment is added. Tracked via `PricingEngine._at_risk_seen` set — each batch ID is added exactly once.
- **revenue_recovered_accumulator**: When units from an at-risk batch are sold, `units_sold x current_price` is added.
- If `at_risk_cost_accumulator == 0` (no batches became at-risk yet), WRR returns 0.0 — no division by zero.

### Per-Engine Reward Components

**Engine 1 — Pricing (r1)**

| Component | Formula | Value |
|-----------|---------|-------|
| Base ratio | `revenue_this_tick / max_possible_revenue_this_tick` | 0.0–1.0 |
| Near-expiry bonus | Per unit sold with `hours_to_expiry <= 4.0` | +0.15 per unit |
| Expired penalty | Per unit that expired unsold | -0.80 per unit |
| Anti-hack penalty | `price_multiplier < 0.35` with `hours > 48` | -0.40 per violation |

**Engine 2 — Farmer (r2)**

| Component | Formula | Value |
|-----------|---------|-------|
| Cleared batch bonus | Per farmer batch fully cleared before expiry | +0.20 x viability_score |
| Missed opportunity | Declined offer with `viability >= 0.70` | -0.50 |
| Reckless accept penalty | Accepted with `viability < 0.30` | -0.60 |

**Engine 3 — Trend (r3)**

| Component | Formula | Value |
|-----------|---------|-------|
| High-confidence bonus | Approved signal with `composite_score >= 80` | +0.10 |
| Perfect timing bonus | Trend stock sold at >= 90% of original price | +0.25 |
| Overtrade penalty | Trend stock required > 40% discount to clear | -0.30 |
| Order cap penalty | Order quantity was hard-capped at 2x weekly velocity | -0.10 |

### Constitutional Audit (4 Checks)

Run before DPO pair generation. Trajectory excluded if ANY check fails.

1. `antihack_violation_count > 5` — too many guard triggers in one episode
2. `r1 negative for > 30% of ticks` — systematic below-floor pricing
3. `r2_mean < -1.0` (non-zero ticks only) — systematically accepting reckless offers
4. `> 3 trend orders in same category within 200 ticks` — trend order flooding

### WandB Metrics

| Metric | What It Measures | What to Watch For |
|--------|-----------------|-------------------|
| `wrr` | Weekly Waste Recovery Rate (primary) | Should trend upward; drops expected at level transitions |
| `r1_pricing` | Pricing engine mean reward | Negative = expired inventory; check urgency response |
| `r2_farmer` | Farmer engine mean reward | 0.0 in Stable Week; persistent negatives = reckless acceptance |
| `r3_trend` | Trend engine mean reward | 0.0 in Stable/Farmer Week; negative = over-ordering |
| `brief_quality_score` | Independent reasoning quality proxy | Should correlate with WRR — the research finding |
| `anti_hack_violations` | Guard trigger count per episode | Rising = reward hacking in progress |
| `curriculum_level` | Current scenario (0-4) | Steps up on promotion; stuck = learning plateau |
| `episodes_in_level` | Episodes since last promotion | > 100 without promotion = investigate |

---

## 7. Verification Commands

Run all four checks before every training run.

### Check 1: All Imports

```bash
python -c "
from freshprice_env.enums import *
from freshprice_env.constants import *
from freshprice_env.entities import *
from freshprice_env.market_state import MarketStateBuilder
from freshprice_env.engines.pricing_engine import PricingEngine
from freshprice_env.engines.farmer_engine import FarmerEngine
from freshprice_env.engines.trend_engine import TrendEngine
from freshprice_env.reward import WRRRewardEngine
from freshprice_env.brief_pipeline.prompt_builder import OperatingBriefPromptBuilder
from freshprice_env.brief_pipeline.parser import BriefParser
from freshprice_env.brief_pipeline.validator import BriefValidator
from freshprice_env.brief_pipeline.quality_scorer import BriefQualityScorer
from freshprice_env.brief_pipeline.rule_executor import RuleExecutor
from freshprice_env.freshprice_env import FreshPriceEnv
from training.curriculum import CurriculumManager, EpisodeResult
from training.trajectory_buffer import TrajectoryBuffer
from training.counterfactual import CounterfactualEngine
print('ALL IMPORTS OK')
"
```

Expected: `ALL IMPORTS OK`

### Check 2: Gym Compliance

```bash
python -c "
from gymnasium.utils.env_checker import check_env
from freshprice_env.freshprice_env import FreshPriceEnv
env = FreshPriceEnv()
check_env(env)
print('GYM CHECK PASSED')
"
```

Expected: `GYM CHECK PASSED` (two warnings about render_fps and spec are informational — ignore them)

### Check 3: Episode Smoke Test

```bash
python -c "
from freshprice_env.freshprice_env import FreshPriceEnv
from freshprice_env.enums import CurriculumScenario

class DummyLLM:
    def generate(self, prompt):
        return '''SITUATION: Testing.
SIGNAL ANALYSIS: N/A
VIABILITY CHECK: N/A
RECOMMENDATION: Hold prices.
DIRECTIVE: {\"engine\": \"PRICING\", \"actions\": []}
CONFIDENCE: LOW'''

env = FreshPriceEnv(scenario=CurriculumScenario.STABLE_WEEK, seed=42, llm_client=DummyLLM())
obs, info = env.reset()
done = False
steps = 0
while not done:
    obs, reward, done, truncated, info = env.step(DummyLLM().generate(obs))
    steps += 1
print(f'SMOKE TEST PASSED — {steps} steps')
print(f'Final WRR: {info.get(\"final_reward\", {}).get(\"wrr\", 0.0):.4f}')
"
```

Expected: `SMOKE TEST PASSED — 84 steps` / `Final WRR: 0.0000`
WRR is 0.0 because the dummy LLM sends empty actions — the agent does nothing, so no revenue is recovered. This is correct behaviour.

### Check 4: WRR Weights Sum to 1.0

```bash
python -c "
from freshprice_env.constants import WRR_WEIGHT_R1, WRR_WEIGHT_R2, WRR_WEIGHT_R3
total = WRR_WEIGHT_R1 + WRR_WEIGHT_R2 + WRR_WEIGHT_R3
assert abs(total - 1.0) < 1e-9
print(f'WRR WEIGHTS OK: {WRR_WEIGHT_R1} + {WRR_WEIGHT_R2} + {WRR_WEIGHT_R3} = {total}')
"
```

Expected: `WRR WEIGHTS OK: 0.4 + 0.3 + 0.3 = 1.0`

---

## 8. Full Training Walkthrough

### On Your Local Machine (No GPU)

Run the 4 verification commands from Section 7. No training runs locally — training requires a GPU.

### On Google Colab (T4 GPU)

#### Cell 1: Install Dependencies (~5 min)

```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install torch transformers trl peft accelerate bitsandbytes
!pip install datasets tokenizers gymnasium wandb numpy pandas scipy tqdm
```

Expected: All packages install without errors. Ignore deprecation warnings.

#### Cell 2: Clone Repo

```python
!git clone https://github.com/nandeshkanagaraju/QStorePrice.git
%cd QStorePrice
```

#### Cell 3: Authenticate

```python
from huggingface_hub import login
login()  # Paste your HF token when prompted

import wandb
wandb.login()  # Paste your WandB API key when prompted
```

#### Cell 4: Verify GPU and Imports

```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

from freshprice_env.freshprice_env import FreshPriceEnv
from freshprice_env.enums import CurriculumScenario
print("Imports OK")
```

Expected: GPU name (e.g., Tesla T4), ~15 GB VRAM, "Imports OK".

#### Cell 5: Run Baseline (~30-60 min)

```python
!python eval/baseline.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --episodes 5 \
  --output eval/baseline_results.json
```

Expected: Table showing zero-shot WRR around 0.03-0.10 per scenario. This is the "before" measurement. Save the output — you need it for the before/after comparison.

Time: ~30-60 min depending on GPU. Each episode is 84 LLM generations.

#### Cell 6: SFT Warm-Start (~45 min)

```python
!python training/sft_trainer.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --output-dir checkpoints/sft_v1 \
  --epochs 2 \
  --seed 42
```

**What success looks like**: Training loss drops from ~2.5 to ~0.8 over 2 epochs. At the end, you see `CHECKPOINT VERIFIED — all 6 sections present`. This means the model learned the Operating Brief format.

**If CHECKPOINT VERIFICATION FAILED**: The model might not have converged. Continue anyway — GRPO training will improve it. The SFT checkpoint is a warm start, not a finished model.

#### Cell 7: GRPO Training (~2-3 hrs for 100 episodes)

```python
!python training/train.py \
  --resume-from checkpoints/sft_v1 \
  --output-dir checkpoints \
  --start-scenario STABLE_WEEK \
  --episodes-per-level 100 \
  --wandb-project qstoreprice-ai \
  --seed 42
```

**What to watch in WandB**:
- `wrr` should trend upward from ~0.1 toward 0.5-0.7 over 80 episodes
- `brief_quality_score` should rise alongside WRR — this is the research finding
- `anti_hack_violations` should stay near zero

**If WRR is flat after 30 episodes**: The LLM is not improving its briefs. Check that SFT data covers the CRITICAL urgency tier. Consider lowering the learning rate from 5e-6 to 2e-6.

**If anti_hack_violations are rising**: The agent found a shortcut. Lower learning rate. Check which guard is firing in the WandB logs.

Time: ~2-3 hours for 100 episodes on T4. Each episode = 84 brief generations.

#### Cell 8: DPO Fine-Tuning (~30 min)

DPO runs automatically within `train.py` every 25 episodes if the trajectory buffer has enough pairs. If you see `WARNING: Too few DPO pairs` — this is normal in early training. The buffer needs at least 10 valid+constitutional episodes before DPO can generate pairs.

**What the pre/post comparison table shows**: Pre-DPO WRR vs post-DPO WRR. If post-DPO WRR drops > 3%, the trainer prints a WARNING and you should consider reverting to the pre-DPO checkpoint.

#### Cell 9: Evaluate Trained vs Baseline

```python
!python eval/evaluator.py \
  --checkpoint checkpoints/promoted_level1 \
  --scenarios STABLE_WEEK \
  --episodes 10
```

Compare the output table against baseline_results.json from Cell 5.

#### Cell 10: Save WRR Chart

```python
import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
run = api.run("YOUR_WANDB_PROJECT/YOUR_RUN_ID")
history = run.history(keys=["wrr", "brief_quality_score"])

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(history["wrr"], color="blue", label="WRR")
ax1.set_ylabel("WRR", color="blue")
ax2 = ax1.twinx()
ax2.plot(history["brief_quality_score"], color="green", label="Brief Quality")
ax2.set_ylabel("Brief Quality", color="green")
ax1.set_xlabel("Episode")
ax1.set_title("WRR vs Brief Quality Over Training")
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
plt.savefig("eval/plots/wrr_vs_quality_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
```

#### Cell 11: Push Results to GitHub

```python
!git add eval/baseline_results.json eval/plots/ checkpoints/
!git commit -m "results: baseline + trained checkpoint + WRR chart"
!git push origin main
```

---

## 9. How to Read WandB During Training

### wrr Chart

Should trend upward from ~0.1 toward 0.7+ over 80 episodes. A flat line means the LLM is not improving its briefs — check SFT data coverage and learning rate. A line that goes up then crashes means reward hacking started — the agent found a shortcut that initially increases WRR but triggers anti-hack guards that penalise it later. Watch `anti_hack_violations` to confirm.

### brief_quality_score Chart

Should rise alongside WRR. This is the research finding: better reasoning produces better outcomes. If WRR rises but quality stays flat, the agent found a shortcut that bypasses reasoning — the briefs are formulaic but the DIRECTIVE is effective. If quality rises but WRR stays flat, the briefs are well-written but the decisions are wrong — the quality scorer may need recalibration.

### anti_hack_violations Chart

Should stay near zero throughout training. Any sustained rise (more than 2-3 per episode for 10+ consecutive episodes) means the agent found a way to get reward without solving the actual problem. Common hacks: deep-discounting all fresh stock to inflate the "near-expiry clearance" bonus, declining all farmer offers to avoid risk, approving every trend signal regardless of score. Check which guard is firing and consider lowering the learning rate.

### curriculum_level Chart

Step function — jumps from 0 to 1, then 1 to 2, etc. On STABLE_WEEK (level 0), expect the first promotion around episode 70-100. If the agent hasn't promoted by episode 150, it is plateaued. Check WRR trajectory — is it approaching 0.70 but not quite reaching it, or is it stuck below 0.50?

---

## 10. How to Evaluate a Checkpoint

### Run Evaluation

```bash
python eval/evaluator.py \
  --checkpoint checkpoints/promoted_level1 \
  --scenarios STABLE_WEEK BUSY_WEEKEND \
  --episodes 10
```

### Compare Two Checkpoints

```bash
python eval/evaluator.py \
  --checkpoint checkpoints/grpo_level0 \
  --compare-with checkpoints/dpo_level0_ep25 \
  --episodes 5
```

The comparison table shows per-scenario WRR for both checkpoints on the same seeds, the delta, and which checkpoint is better.

### Anti-Hack Buffer Scan

```python
from eval.anti_hack_checker import AntiHackChecker
report = AntiHackChecker.scan_trajectory_buffer(trainer.trajectory_buffer._buffer)
```

Output: count of clean/flagged/excluded trajectories and pattern frequency. "Exclude from DPO" means the trajectory showed HIGH or CRITICAL hacking patterns and should not be used for DPO preference pair generation, even if its WRR is high.

### Eval Report Columns

- **WRR mean +/- std [min -> max]**: Average WRR across episodes with standard deviation and range. Higher mean is better. Low std means consistent performance.
- **Brief Quality**: Average quality score. Tracks reasoning quality independent of reward.
- **Violations mean per episode**: Average anti-hack guard triggers. Should be < 1.0 for a well-trained model.
- **Constitutional Pass Rate**: X/N episodes that passed the 4-check constitutional audit. Should be 100% for a model ready for DPO.

---

## 11. Common Errors and Fixes

| Error / Symptom | Cause | Fix |
|-----------------|-------|-----|
| `CUDA out of memory` during GRPO | Batch size or sequence length too large | Set `--batch-size 1`, reduce `gradient_accumulation_steps` to 4, use `max_seq_length=2048` |
| `torch==2.2.1 not found` on HF Space | Space uses Python 3.13; requirements.txt has old pins | Fixed: `requirements.txt` is now Space-safe. Training deps are in `requirements_training.txt` |
| `CHECKPOINT VERIFICATION FAILED` after SFT | Model didn't converge in 2 epochs — missing section headers | Continue to GRPO anyway. If persistent: increase SFT epochs to 3, add more SFT examples |
| WRR stuck at 0.0 for 30+ episodes | Normal early — WRR needs at-risk batches to have non-zero denominator | Wait. If still 0.0 after 50 episodes: check that CRITICAL batches exist in STABLE_WEEK scenario |
| `WARNING: Too few DPO pairs` | Trajectory buffer has < 10 valid episodes | Run more GRPO episodes first. DPO needs data — it cannot improve from nothing |
| Model outputs garbage in first 20 GRPO episodes | SFT warm-start not run or too few examples | Run `training/sft_trainer.py` with at least 50 examples. SFT teaches the brief format |
| `anti_hack_violations` rising steadily | Learning rate too high; agent gaming reward | Lower LR to 1e-6. Check which guard is firing. Consider adjusting reward weights |
| Colab disconnects mid-training | Session timeout or runtime limit | Use `training/train.py --resume-from` to continue from last checkpoint. KeyboardInterrupt handler saves emergency checkpoint |
| HF Space build fails | requirements.txt includes torch or other heavy training deps | Ensure `requirements.txt` has only: gradio, matplotlib, gymnasium, numpy |
| BriefParser returns `success=False` on every step | Model outputs non-brief text (chat preamble, questions) | SFT data needs more examples. Check prompt builder includes the SYSTEM_PROMPT |
| KeyboardInterrupt does not save checkpoint | grpo_trainer was not yet initialised when interrupt hit | The handler checks `if grpo_trainer is not None` — if interrupted during SFT, there is no GRPO checkpoint to save |
| Gym check fails with charset error | Observation text contains characters not in the Text space charset | Fixed: charset now includes all printable ASCII, emoji, and Unicode used in prompts |

---

## 12. The Research Question

**The question**: Does LLM reasoning quality (measured by `brief_quality_score`) improve through RL training, and does that improvement causally drive WRR improvement?

**The two metrics**:
- `wrr`: The primary reward signal. Are the agent's decisions actually reducing waste?
- `brief_quality_score`: An independent proxy for reasoning quality. Does the agent's situation assessment, viability reasoning, and confidence calibration improve alongside its operational performance?

**What correlation would mean**:
- Pearson r > 0.65: Strong evidence that reasoning quality and operational performance are coupled. The model is not just finding shortcuts — it is genuinely learning to reason about perishable goods.
- Pearson r < 0.30: The agent found reward pathways that bypass the reasoning process. WRR improved, but the briefs did not get better. The quality scorer may need recalibration, or the reward function has a loophole.
- Brief quality *leads* WRR by a few episodes: The strongest possible signal. The model learns to reason better first, and the operational metric follows. This would suggest that RL is genuinely improving the model's world understanding.

**How to compute the correlation after training**:

```python
import wandb
import numpy as np
from scipy import stats

api = wandb.Api()
run = api.run("YOUR_PROJECT/YOUR_RUN_ID")
history = run.history(keys=["wrr", "brief_quality_score"])

wrr = history["wrr"].dropna().values
quality = history["brief_quality_score"].dropna().values
min_len = min(len(wrr), len(quality))

r, p_value = stats.pearsonr(wrr[:min_len], quality[:min_len])
print(f"Pearson r = {r:.3f} (p = {p_value:.4f})")
if r > 0.65:
    print("Strong correlation: reasoning quality drives WRR improvement")
elif r > 0.30:
    print("Moderate correlation: partial coupling between reasoning and outcome")
else:
    print("Weak correlation: agent may be finding shortcuts")
```

**Honest assessment**: This is one environment, one LLM family (Qwen-2.5), one reward function, and one quality scoring methodology. The finding is encouraging — it establishes a methodology for measuring reasoning-reward correlation in language agent training. Whether the finding generalises to other domains, models, and reward structures is why the environment should be extended, not replaced. The value of QStorePrice AI is not the specific correlation number — it is the infrastructure that makes the question testable.

---

## 13. Submission Checklist

- [ ] All 4 local checks pass (import, gym, smoke test, WRR weights)
- [ ] HF Space is live and accessible at the submission URL
- [ ] `eval/baseline_results.json` committed to repo
- [ ] `eval/plots/wrr_vs_quality_correlation.png` committed and embedded in README
- [ ] WandB run is public — URL in README
- [ ] README links updated: HF Space, WandB, blog, video
- [ ] `openenv.yaml` in repo root with correct class path
- [ ] `requirements.txt` is Space-safe (no torch, no pinned versions that break Python 3.13)
- [ ] `requirements_training.txt` has the full training stack
- [ ] HuggingFace mini-blog posted (problem, brief example, before/after table, WRR chart)
- [ ] Colab notebook committed as `training/QStorePrice_Training.ipynb`
- [ ] All placeholder URLs in README replaced with real URLs
- [ ] `DEVELOPER_GUIDE.md` committed (this file)
