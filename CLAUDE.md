# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project Overview

QStorePrice is a **multi-agent perishable goods ecosystem** built on
OpenEnv. v2 reframes the project around a 6+1 actor simulation:

- StoreAgent (hero, the trained LLM)
- CompetitorStoreAgent (rival, frozen ckpt or scripted)
- FarmerAgent pool (5 farmers, persistent SQLite reputation)
- ConsumerCohortAgent (rule-based reactive demand)
- InfluencerAgent (trend signals incl. paid promotion)
- RegulatorAgent (mid-episode schema drift)
- OversightAuditor (7th — reads trajectories, writes audits)

The hero is trained via SFT → GRPO → DPO. Episodes can be 7 days
(legacy `FreshPriceEnv`) or **30 days** (`LongHorizonFreshPriceEnv`).
The headline env is `MarketCommonsEnv`.

## Source of Truth

`FreshPrice_SDD.md` is the original spec for the single-store engines
and the WRR metric. The multi-agent additions are documented in:

- `freshprice_env/persistence/reputation_store.py`
- `freshprice_env/notebook/`
- `freshprice_env/protocol/market_bus.py`
- `freshprice_env/agents/{farmer,competitor_store,oversight_auditor,regulator,influencer}_agent.py`
- `freshprice_env/brief_pipeline/schema_registry.py`
- `freshprice_env/scenario_composer.py`
- `eval/{theory_of_mind_probe,counterfactual_replay}.py`

## Build & Run Commands

```bash
# Install (CPU-only deps for env work)
pip install -r requirements.txt

# Install training extras (GPU required for unsloth)
pip install -r requirements_training.txt

# Tests (29 tests — all should pass on CPU)
python -m unittest discover tests -v

# Run training pipeline
python training/train.py --base-model Qwen/Qwen2.5-7B-Instruct --output-dir checkpoints

# Run SFT only
python training/sft_trainer.py --model-id Qwen/Qwen2.5-7B-Instruct --output-dir checkpoints/sft_v1

# Train the OversightAuditor (small model)
python -c "from training.oversight_trainer import build_examples_from_episodes_jsonl, run_sft; \
           ex = build_examples_from_episodes_jsonl('episode_log.jsonl'); \
           run_sft(ex, base_model='Qwen/Qwen2.5-0.5B-Instruct')"

# Self-play smoke
python -c "from training.self_play import smoke_test; print(smoke_test(4))"

# Server (V2 dashboard at /, legacy at /legacy)
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Lint / type
ruff check .
mypy freshprice_env/ training/ eval/
```

## Architecture Cheatsheet

- **Single-store, gym-style**: `FreshPriceEnv` — 672 ticks, brief-driven step
- **30-day long-horizon**: `LongHorizonFreshPriceEnv` — wraps `FreshPriceEnv`,
  adds `AgentNotebook`, sparse weekly reward, prompt truncation, plan-adherence
- **Multi-agent commons**: `MarketCommonsEnv` — wraps `FreshPriceEnv`, adds
  `MarketBus`, competitor agents, farmer pool, regulator
- **Bilateral negotiation**: `NegotiationEnv` — used by `training/self_play.py`
- **Multi-store cooperative**: `MultiStoreFreshPriceEnv` — TRANSFER directives
- **Hero + reactive consumer**: `MultiAgentFreshPriceEnv`

### Blinkit/Zepto layer (additive — opt-in, all existing tests still pass)

- `freshprice_env/engines/rider_pool_engine.py` — finite courier pool. Each
  brief contributes `r6_delivery_quality`: bonus for on-time deliveries,
  penalty for transit spoilage on queued orders. Saturation events surface
  on the bus.
- `freshprice_env/agents/consumer_cohort_agent.py` — three cohorts
  (PREMIUM / MASS / BARGAIN) with their own price elasticity, freshness
  tolerance and ETA tolerance. Premium cohort walks on slow ETAs or
  CRITICAL stock; this couples pricing with delivery throughput.
- `freshprice_env/engines/liquidation_engine.py` — B2B firesale channel
  for dead stock (`{"action":"LIQUIDATE", "batch_id":..., "channel":"B2B"}`
  inside a PRICING DIRECTIVE). Recovers `LIQUIDATION_RECOVERY_RATIO` of
  original price on CRITICAL stock; reckless attempts on FRESH/WATCH
  stock are anti-hack flagged and contribute the
  `LIQUIDATION_RECKLESS_PENALTY`.
- All three are **additive** — `MarketCommonsEnv` calls them after the
  pricing engine and exposes their snapshots through the bus + the new
  server endpoints (`/commons/rider_pool`, `/commons/cohorts`,
  `/commons/liquidation`).

### SFT data budget (no more magic numbers)

- `training/data_budget.py:compute_data_budget()` derives
  `n_per_difficulty` from model parameter count, format complexity,
  target format-recall, and a VRAM cap. Calibrated once against the
  Qwen-1.5B / 270-example T4 anchor; all other (model, GPU)
  combinations are derived. The Kaggle notebook prints
  `budget.rationale` so a judge can see *why* the value was picked.
- `compute_grpo_episode_budget()` does the same for GRPO episode
  count (driven by wall-clock target + DPO-buffer floor).

### Simulation Theater (frontend / backend)

- Server endpoints under `/commons/`: `rider_pool`, `cohorts`,
  `liquidation`, `sim_frames` (ring buffer of recent tick frames).
- Update via `server.app.update_blinkit_state(...)` — env tick
  callbacks push `tick_frame={tick, batches, cohorts, rider_pool,
  latest_brief, reasoning}` blobs.
- Dashboard at `/` adds a Simulation Theater panel: ▶ / ⏸ / speed,
  per-tick batch grid, rider gauge, cohort retention bars,
  liquidation queue, and a side-by-side reasoning stream. The
  **What-if** button forks the simulation through `/commons/replay`.

## Reward Components

| Component | Source | Role |
|---|---|---|
| r1_pricing | PricingEngine.tick | Discount timing |
| r2_farmer | FarmerEngine | Accept/counter/decline farmer offers |
| r3_trend | TrendEngine | Restock decisions |
| r4_plan_adherence | AgentNotebook auto-resolve | Honored − broken |
| r5_reasoning_tokens | reward.compute_token_reward | Capped, quality-gated |
| r6_delivery_quality | RiderPoolEngine.compute_brief_reward | On-time deliveries minus transit spoilage |
| r7_liquidation | LiquidationEngine.compute_brief_reward | Legit firesale minus reckless dumping |
| cooperation_index | MarketCommonsEnv | Pareto-improving exchanges |

## Critical Rules

- **Schema drift**: Validator consults `default_registry()` for the
  *current* schema version. Briefs against an outdated schema fail.
  Reset registry between episodes via `RegulatorAgent.__init__()`.
- **Reputation persists** across env.reset(). Tests should construct
  their own `ReputationStore(":memory:")` to avoid cross-contamination.
- **Notebook is per-episode**: lives inside the env instance, reset on
  reset(). Pinned notes survive FIFO eviction.
- **Bus is server-scoped**: `server.app.get_server_bus()` returns the
  process singleton. MarketCommonsEnv accepts a `bus=` kwarg —
  pass the singleton in production runs to feed `/commons/ws`.
- **Reward penalties**: Stored as positive in `constants.py`. Applied
  as negative in engine code (`r1 -= penalty`).
- **Anti-hack guards**: RuleExecutor flags violations; env wires to
  reward engine; engines compute rewards. Three-step separation.
- **No bare `random`**: All randomness goes through `rng: random.Random`
  instances for reproducibility.

## Naming Conventions

Constants use `SCREAMING_SNAKE`: `R5_TOKEN_TARGET`,
`COOPERATION_INDEX_PARETO_WEIGHT`, etc. Entities are frozen
dataclasses (except `SimulatedMarketState` which mutates each tick).
Enums are `str, Enum` for JSON serialization except
`CurriculumScenario(int, Enum)` for WandB logging.
