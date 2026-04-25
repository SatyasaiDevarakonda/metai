---
title: FreshPrice AI - Perishable Goods Intelligence
emoji: 🥭
colorFrom: green
colorTo: red
sdk: docker
app_file: app.py
pinned: false
---

# FreshPrice AI — Perishable Goods Intelligence

**OpenEnv Hackathon (India 2026) — Round 2 submission**
**Repo:** https://github.com/SatyasaiDevarakonda/metai

> Can an LLM learn to make every perishable goods decision a small business
> owner faces — pricing, procurement, trend-reading, multi-store rebalancing,
> dead-stock routing, event pre-positioning, and weekly clearance —
> better than they can alone?

---

## Priya's story (the problem we are solving)

Priya runs an organic grocery in Chennai. She stocks 40 fresh products. Every
week she throws away **22% of what she buys — roughly Rs 18,000 in wasted
inventory**. She misses demand spikes because she did not see the viral food
trend coming. She turns away farmers with good surplus because pricing it
quickly feels risky. She has three stores but no system to move excess stock
between them. Every Friday she discounts frantically, hoping to clear the
week's remaining produce before the weekend.

**FreshPrice AI is built for Priya.** Not for Blinkit. Not for enterprise
retail. For the small online grocery owner losing money to perishable waste
every single week, with no intelligent system to help.

---

## The seven engines

The agent's job is the full stack of decisions a perishable-goods seller
faces every week. Each engine produces its own reward component (`r_i`)
which combines into a single composite **Store Efficiency Score (SES)**:

| # | Engine | Decision | Reward (r_i) | SES weight |
|---|---|---|---|---|
| 1 | **Dynamic Pricing** | Price multiplier + flash-sale + bundle per SKU | `r1_pricing` | **0.28** |
| 2 | **Farmer Offer** | Accept / counter-offer / decline pending farmer batches | `r2_farmer` | 0.18 |
| 3 | **Social Trend** | Approve / decline trend-driven purchase orders | `r3_trend` | 0.15 |
| 4 | **Intra-Fleet Rebalancing** | TRANSFER excess stock between stores | `r4_intrafleet` | 0.12 |
| 5 | **Micro-Manufacturer Pipeline** | Route near-expiry batches to processors | `r5_micromfg` | 0.10 |
| 6 | **Event Pre-Positioning** | Pre-stock for cricket / festival / weekend events | `r6_event` | 0.10 |
| 7 | **Surplus Box Subscription** | Friday assembly of weekly subscriber box | `r7_surplusbox` | 0.07 |

**SES = Σ wᵢ · rᵢ.** Used as the curriculum-promotion metric: an agent must
reach SES ≥ 0.70 over five consecutive evaluation episodes to advance to
the next training scenario. The weights are computed in
[`freshprice_env/constants.py`](freshprice_env/constants.py) and the SES
math lives in [`freshprice_env/engines/seven_engines_reward.py`](freshprice_env/engines/seven_engines_reward.py).

---

## The Operating Brief — what the LLM actually outputs

The agent does **not** emit raw action values. It writes a 6-section
**Operating Brief**, and a deterministic rule executor parses the
`DIRECTIVE` section into action values. Every decision is therefore
readable, explainable, and overridable by the seller. Sample brief:

```
SITUATION: Farmer Rajan offers 50 kg ripe Alphonso at Rs 35/kg. We have no
current mango inventory. It is Thursday 2pm.

SIGNAL ANALYSIS: N/A

VIABILITY CHECK: Shelf life: 48 hours. At Rs 80/kg, velocity = 3 kg/hr.
Need ~17 hrs of selling time. PASS. No conflicting mango inventory. PASS.
Break-even at Rs 48/kg. Market rate Rs 85-110. HEALTHY. Worst-case
(60% sell-through): Rs 2,250 revenue vs Rs 1,750 cost. BREAK-EVEN at worst.

RECOMMENDATION: ACCEPT. Counter-offer Rs 42/kg. Expected profit Rs 1,100-1,800.

DIRECTIVE:
{"engine": "FARMER", "actions": [{"offer_id": "F001", "decision": "COUNTER",
                                  "counter_price": 42}]}

CONFIDENCE: HIGH
```

A single brief can carry **side directives** for any subset of Engines 4-7
inside the same JSON object — see
[`freshprice_env/brief_pipeline/prompt_builder.py`](freshprice_env/brief_pipeline/prompt_builder.py)
for the full schema.

---

## How it covers the four hackathon themes

| Theme | How FreshPrice earns it |
|---|---|
| **#1 Multi-Agent** | Hero StoreAgent + simulated customer cohorts + farmer pool (5 agents with persistent reputation) + competitor stores + RegulatorAgent + InfluencerAgent + OversightAuditor. All the agent files live under [`freshprice_env/agents/`](freshprice_env/agents/). |
| **#2 Long-Horizon** | 30-day `LongHorizonFreshPriceEnv` (84 briefs / week × 4 weeks) with `AgentNotebook` for `NOTE`/`COMMIT`/`UPDATE_PLAN` directives. Plan-adherence reward (`r_plan_adherence`) tracks honored commitments minus broken ones. |
| **#3 World Modeling** | The agent must build an implicit world model: which trend signals reliably convert to demand, which farmer offers have positive expected value, which events drive which categories. Schema-drift APIs (Patronus AI bonus) force the model to update its understanding of the world mid-episode. |
| **#4 Self-Improvement** | Curriculum across 5 scenarios (STABLE_WEEK → CRISIS_WEEK), `ScenarioComposer` with Thompson-sampling difficulty selection, and self-play `NegotiationEnv` for bilateral training. |

---

## Training pipeline (SFT → REINFORCE+KL → DPO)

Single Colab/Kaggle notebook: [`kaggle_qstoreprice.ipynb`](kaggle_qstoreprice.ipynb)
runs end-to-end on a Kaggle T4. Stages:

1. **SFT warm-start** — 50-100 hand-crafted Operating Briefs (covers the
   format + viability-check pattern). Without this, RL spends the first
   20-30 episodes fixing the brief format.
2. **Rollout collection** — generate briefs in `FreshPriceEnv`, score with
   r1..r7 + SES, push valid trajectories into `TrajectoryBuffer`.
3. **REINFORCE+KL (real on-policy RL)** — see
   [`training/reinforce_trainer.py`](training/reinforce_trainer.py).
   `policy_loss = -advantage · log π_θ + β · KL(π_θ || π_ref)`. Loss runs
   `.backward()` and `optimizer.step()` per grad-accum boundary. KL ref
   is the frozen SFT policy (LoRA off → π_ref).
4. **DPO** — TRL `DPOTrainer` on regret-weighted preference pairs from
   the trajectory buffer. The constitutional anti-hack filter excludes
   trajectories that earned reward by violating the operating constitution.

The SFT data budget is derived (not hardcoded) by
[`training/data_budget.py`](training/data_budget.py): the formula reads
model-parameter count, format complexity, target format-recall, and a
VRAM cap. The notebook prints `budget.rationale` so a judge can see *why*
each value was picked.

---

## Hackathon submission checklist

| Requirement | Status | Where |
|---|---|---|
| Use OpenEnv (latest release) | ✅ | [`server/app.py`](server/app.py), [`openenv.yaml`](openenv.yaml), `gym.Env`-compliant `reset/step/state` |
| Working training script (Unsloth / TRL) | ✅ | [`kaggle_qstoreprice.ipynb`](kaggle_qstoreprice.ipynb) |
| Evidence of training (loss + reward plots) | ✅ | `plots/rl_learning_curve.png`, `plots/reinforce_curve.png` (generated by the notebook) |
| Hugging Face Space hosting | 🟡 | Push this repo to a Space; the Dockerfile is ready |
| Mini-blog OR <2 min video | ⏳ | TODO — link from this README once published |
| README that motivates problem + shows results | ✅ | This file |

---

## Quickstart — run locally

```bash
git clone https://github.com/SatyasaiDevarakonda/metai.git
cd metai
pip install -r requirements.txt

# Run the test suite (64 tests, ~1 sec)
python -m unittest discover tests -v

# Start the FastAPI dashboard server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
# open http://localhost:8000
```

To drive the dashboard with **your trained Kaggle model** (no local GPU
needed), set:

```bash
export AGENT_BACKEND=hf_inference
export HF_REPO_ID="your-hf-username/qstoreprice-sft"
export HF_TOKEN="hf_..."
```

…and click **Run live demo** at the top of the dashboard. The Simulation
Theater below auto-plays the recorded tick frames.

The full Kaggle → VS Code path with all environment variables is in
[`.env.example`](.env.example).

---

## Architecture cheat-sheet

```
freshprice_env/
├── freshprice_env.py             ← single-store gym env (the core)
├── long_horizon_env.py           ← 30-day wrapper (Theme #2)
├── market_commons_env.py         ← 6+1 multi-agent commons (Theme #1)
├── multi_store_env.py            ← Engine 4 transfer execution
├── negotiation_env.py            ← bilateral self-play (Theme #4)
├── multi_agent_env.py            ← reactive consumer + LLM hero
├── engines/
│   ├── pricing_engine.py         ← Engine 1
│   ├── farmer_engine.py          ← Engine 2
│   ├── trend_engine.py           ← Engine 3
│   └── seven_engines_reward.py   ← Engines 4-7 + SES (NEW)
├── agents/
│   ├── farmer_agent.py competitor_store_agent.py consumer_agent.py
│   ├── influencer_agent.py regulator_agent.py oversight_auditor.py
│   └── consumer_cohort_agent.py
├── brief_pipeline/               ← parser + validator + prompt builder
├── notebook/                     ← AgentNotebook for plan adherence
├── persistence/reputation_store.py  ← persistent farmer trust
└── protocol/market_bus.py        ← pub/sub for cross-agent messages

training/
├── data_budget.py                ← formula-driven SFT data budget
├── generate_sft_data.py          ← 270 hand-crafted briefs
├── sft_trainer.py                ← Unsloth SFT
├── reinforce_trainer.py          ← REAL REINFORCE+KL policy gradient
├── dpo_trainer.py                ← TRL DPO
└── trajectory_buffer.py          ← regret-weighted DPO pair generation

server/
├── app.py                        ← FastAPI: /reset /step /state /agent/*
├── agent_runtime.py              ← Local / HF Inference / Scripted backends
└── environment.py                ← OpenEnv adapter

static/v2/                        ← Live multi-agent dashboard + Theater
```

---

## Test coverage

```
$ python -m unittest discover tests
Ran 64 tests in 1.05s
OK (skipped=1)
```

Tests cover: env contract, reward components, anti-hack rules, brief
parser, all 6 scenarios, the Blinkit-style realism layer (rider pool,
cohorts, liquidation), the Market Commons multi-agent flow, and the new
7-engine SES path (`tests/test_seven_engines.py`).

---

## Citation / further reading

- **FreshPrice strategy doc** (the full design): the hackathon write-up bundled in this repo.
- **Newsvendor problem**: Arrow et al. 1951; Whitin 1955. FreshPrice is a
  partially observable multi-product newsvendor with endogenous pricing.
- **GRPO** (DeepSeek): "GRPO is more efficient than PPO for verifiable
  reward tasks" — quoted in the official hackathon guide.
- **Liu et al. SAGE POM 2024**: confirms MARL outperforms classical OR
  heuristics on multi-echelon supply chains.
