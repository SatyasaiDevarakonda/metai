---
title: QStorePrice Commons
emoji: 🥭
colorFrom: green
colorTo: red
sdk: docker
app_file: app.py
pinned: false
---

# QStorePrice Commons — a 6+1 Multi-Agent Perishable Goods Ecosystem

> Six LLM-driven actors with conflicting incentives, hidden private information,
> and persistent reputation negotiate a shared market over a 30-day horizon.
> A seventh — the Oversight Auditor — watches them all.

QStorePrice was originally pitched as "an RL-trained LLM that writes pricing
briefs." The hackathon-grade version flips that upside down: pricing is one
sub-task inside a *partially observable, multi-agent* simulation that targets
**Theme #1 (multi-agent)** + **Theme #2 (long-horizon)** simultaneously, with
bonus alignment on the **Fleet AI**, **Halluminate**, **Patronus**, and **Mercor**
sub-prizes.

---

## 1. The seven actors

| Agent | Role | Hidden info | LLM? |
|---|---|---|---|
| **StoreAgent** (hero, the trained model) | Writes Operating Briefs + ## NOTEBOOK + ## MESSAGES | private cash, plan | yes |
| **CompetitorStoreAgent** | Same market, different store | competitor inventory + cash | yes (frozen ckpt or scripted) |
| **FarmerAgent** (pool of 5) | Emits BIDs, reacts to COUNTERs, **remembers** every interaction | reserve price, alternative buyers | optional LLM, otherwise persona-based |
| **ConsumerCohortAgent** (3 personas: budget / foodie / festival) | Reactive demand boosts | true willingness-to-pay | rule-based (fast) |
| **InfluencerAgent** | Emits trend signals — half are paid promotion | which signals are paid | procedural |
| **RegulatorAgent** | Mid-episode policy drift (FSSAI, ASCI, DPIIT) | drift schedule | procedural |
| **OversightAuditor** (the 7th) | Reads bus + briefs + notebook → trust score + narrative + recommendation | nothing — that's the point | LLM (or rule-based fallback) |

---

## 2. What's new vs. the v1 single-store env

| Feature | v1 | v2 (now) |
|---|---|---|
| Episode length | 7 days / 84 briefs | **30 days / 360 briefs** in `LongHorizonFreshPriceEnv` |
| Memory | LLM context only | **AgentNotebook**: `NOTE` / `RECALL` / `COMMIT` / `UPDATE_PLAN`, with FIFO eviction + pinned slots |
| Reward | r1+r2+r3 → WRR | + **r4 plan adherence**, **r5 token-scaled** (Mercor), **cooperation_index** |
| Multi-agent | static rule-based consumer | **bus + competitor + farmer pool + auditor + regulator** |
| Farmer behaviour | fixed scheduled offers | **persistent SQLite reputation**, persona-driven counters, trust-modulated reserves |
| Schema | fixed | **versioned schemas** (`PRICING.v1/v2/v3`, `FARMER.v1/v2`, `TREND.v1/v2`); regulator mutates mid-episode |
| Trend signals | all genuine | **half paid promotion** with hidden `is_paid_promotion` flag; corroborants observable |
| Curriculum | 5 hand-crafted levels | + 6th `REGULATORY_WEEK` + **adaptive `ScenarioComposer`** (Thompson sampling) |
| Self-play | none | **rotating frozen-opponent** negotiation arena (Theme #4) |
| Eval | WRR only | + **Theory-of-Mind probe** + **counterfactual replay** + **oversight audit grade** |
| Dashboard | polling HTML | **WebSocket-streamed**, 6+1 agent panes, replay slider, audit narrative card |

---

## 3. Architecture

```
freshprice_env/
  freshprice_env.py             # legacy single-store gym env (still works)
  long_horizon_env.py           # 30-day, sparse reward, NOTEBOOK, plan-adherence
  market_commons_env.py         # ★ headline multi-agent ecosystem
  multi_agent_env.py            # hero + reactive consumer
  multi_store_env.py            # cooperative N-store transfers
  negotiation_env.py            # bilateral self-play arena

  agents/
    farmer_agent.py             # ★ LLM-or-persona farmer with reputation
    competitor_store_agent.py   # ★ rival store with 4 personas
    oversight_auditor.py        # ★ 7th-agent trajectory auditor (Fleet AI)
    regulator_agent.py          # ★ schema-drift broadcaster (Patronus)
    influencer_agent.py         # ★ trend signals incl. paid promotion
    consumer_agent.py           # rule-based reactive consumer

  notebook/
    agent_notebook.py           # ★ durable scratchpad (NOTE/COMMIT/PLAN)
    notebook_directives.py      # ★ parser + executor + commitment evaluator

  protocol/
    market_bus.py               # ★ typed inter-agent message log

  persistence/
    reputation_store.py         # ★ SQLite reputation graph

  brief_pipeline/
    schema_registry.py          # ★ versioned DIRECTIVE schemas

  scenario_composer.py          # ★ Thompson-sampling adaptive curriculum
  reward.py                     # extended w/ r4, r5, cooperation_index

eval/
  theory_of_mind_probe.py       # ★ ToM held-out evaluation
  counterfactual_replay.py      # ★ swap-one-decision replay tool
  evaluator.py                  # WRR-greedy held-out eval
  anti_hack_checker.py          # constitutional audit
  baselines/                    # random + rule-based + run_baselines.py

training/
  train.py                      # SFT → GRPO → DPO orchestration
  self_play.py                  # ★ rotating-opponent negotiation self-play
  oversight_trainer.py          # ★ SFT for the auditor LLM
  grpo_trainer.py / sft_trainer.py / dpo_trainer.py
  curriculum.py / counterfactual.py / trajectory_buffer.py

server/
  app.py                        # FastAPI + ★ /commons/* endpoints + ★ /commons/ws

static/v2/                      # ★ new live dashboard
data/snapshots/                 # ★ real Bangalore mandi prices, food trends, FSSAI excerpts
```

★ = added in v2.

---

## 4. Getting started

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -r requirements_training.txt

# Verify everything wired up (CPU only)
python -c "from freshprice_env.market_commons_env import MarketCommonsEnv; \
           from freshprice_env.enums import CurriculumScenario; \
           env = MarketCommonsEnv(scenario=CurriculumScenario.CRISIS_WEEK); \
           obs, info = env.reset(); print(info['mode'])"

# Run all tests (46 of them, ~1s)
python -m unittest discover tests -v
```

---

## 4b. Use your trained weights in the dashboard

After running `kaggle_qstoreprice.ipynb` end-to-end on Kaggle the SFT
checkpoint sits at `/kaggle/working/checkpoints/final` and (if you set
`HF_TOKEN`) is also pushed to the HF Hub. Tying it back to the local
FastAPI dashboard is one env-var flip:

| Path | When to use | What to set |
|---|---|---|
| **HF Inference API** | no GPU on your laptop | `AGENT_BACKEND=hf_inference`, `HF_REPO_ID=<user/repo>`, `HF_TOKEN=hf_…` |
| **Local checkpoint** | you have an 8 GB+ GPU and downloaded `final/` | `AGENT_BACKEND=local`, `MODEL_PATH=/abs/path/to/final` |
| **Scripted fallback** | first run, no model yet | `AGENT_BACKEND=scripted` (or just leave the others unset) |

Then start the server in a fresh terminal:

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Open <http://localhost:8000>. The **Run live demo** panel at the top of
the dashboard shows the active backend and exposes a one-click episode
run that drives the Simulation Theater with the loaded model. Scripted
mode is what you see before you have a checkpoint — the briefs are
heuristic but well-formed (all 6 sections, valid JSON DIRECTIVE), so
the rest of the dashboard works end-to-end on day one.

Diagnostic endpoints once the server is up:

| Endpoint | What it tells you |
|---|---|
| `GET /agent/info` | which backend was loaded and how |
| `POST /agent/brief` | one-shot brief from a prompt |
| `POST /agent/run_episode` | run an episode, push frames to the theater |
| `GET /commons/sim_frames` | the ring buffer of recent tick frames |
| `GET /commons/rider_pool` | live rider-queue + transit-spoilage stats |
| `GET /commons/cohorts` | per-cohort retention / walk-aways |
| `GET /commons/liquidation` | dead-stock B2B firesale history |

---

## 5. The five reward components

WRR remains the headline KPI. The expanded reward signal exposes more of *why*
the agent behaves the way it does:

| Component | Source | What it measures |
|---|---|---|
| `r1_pricing` | PricingEngine | Discount timing vs. expiry urgency |
| `r2_farmer` | FarmerEngine | Viability + reputation-aware accept/counter/decline |
| `r3_trend` | TrendEngine | Restock decisions; penalises paid-promo gullibility |
| `r4_plan_adherence` | AgentNotebook | Honored commitments − broken commitments |
| `r5_reasoning_tokens` | Reward engine | Capped, quality-gated token-scaled reward (Mercor) |
| `cooperation_index` | MarketCommonsEnv | Pareto-improving exchanges with other agents |

---

## 6. Hackathon theme alignment

| Theme | How we hit it |
|---|---|
| **#1 Multi-Agent Interactions** | 6+1 actors share a partially-observable bus; farmer reputation persists across episodes; cooperation index measures pareto-improving messages |
| **#2 (Super) Long-Horizon Planning** | 30-day episodes (360 briefs), sparse weekly reward, prompt forcibly truncated to ~3.5 KB so the AgentNotebook is the *only* memory, plan-adherence reward grades multi-week commitments |
| **#3 World Modeling** | External shocks (weather, festivals), influencer disinformation, regulator policy drift |
| **#4 Self-Improvement** | `ScenarioComposer` Thompson-sampling adaptive curriculum, rotating-opponent self-play in NegotiationEnv |
| **Fleet AI sub-prize** | OversightAuditor reads trajectories and writes structured audits (TRUST_SCORE / SUSPICIOUS_PATTERNS / NARRATIVE / RECOMMENDATION). Trains via `training/oversight_trainer.py` |
| **Halluminate sub-prize** | StoreAgent manages multiple actors (farmers, competitor, influencer, regulator) to discover and achieve goals |
| **Patronus sub-prize** | `SchemaRegistry` + `RegulatorAgent` mutate the DIRECTIVE schema mid-episode (v1→v2→v3); briefs that don't adapt fail validation |
| **Mercor sub-prize** | `r5_reasoning_tokens` is a capped reward that scales with brief token output, gated by quality floor — falsifiable curve |

---

## 7. Live dashboard

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
# open http://localhost:8000/        ← V2 multi-agent dashboard
# open http://localhost:8000/legacy  ← V1 polling dashboard (still works)
```

The V2 dashboard shows:

- 6+1 live agent panes streaming bus messages in real time
- KPI strip: WRR, Cooperation Index, Plan Adherence, Auditor Trust, Pool Trust, Active Schemas
- Notebook panel: plan, pinned notes, recent notes, commitments (✓ / ✗ / ⏳)
- Latest oversight audit narrative
- Counterfactual replay slider (POSTs to `/commons/replay`)
- Adaptive curriculum hardness posteriors per axis
- Full bus message log (most recent 200)

---

## 8. The agent's brief, expanded

Briefs are the same 6-section document, plus two new optional sections:

```
SITUATION:        ...
SIGNAL ANALYSIS:  ...
VIABILITY CHECK:  ...
RECOMMENDATION:   ...
DIRECTIVE:        {"engine": "PRICING", "actions": [...]}
CONFIDENCE:       HIGH | MEDIUM | LOW

## NOTEBOOK
NOTE: cash_buffer -> 4200
NOTE_PIN: regulator_pricing_v3_required -> after tick 528
COMMIT: inventory_below:dairy:30@800 | clear dairy by midnight day 6
UPDATE_PLAN: protect cash buffer through day 4, then unwind dairy

## MESSAGES
CHAT @farmer.rajan: appreciate the mango offer, considering
BID  @farmer.rajan: 38.0/kg for 50kg, 24h decision window
REVEAL @store_002: we are over-stocked on dairy this week
```

---

## 9. Endpoints

OpenEnv contract (single-store):

`GET /health` · `POST /reset` · `POST /step` · `GET /state` · `WS /ws` · `GET /docs`

Multi-agent commons:

`GET /commons/snapshot` · `GET /commons/bus` · `GET /commons/audit`
`GET /commons/notebook` · `GET /commons/scenario_composer`
`POST /commons/replay` · `WS /commons/ws`

Admin / observability:

`GET /admin/dashboard` · `GET /admin/metrics/scores` · `GET /admin/metrics/reward-curve`
`GET /admin/tasks` · `POST /admin/metrics/reset`

---

## 10. Citation

```bibtex
@software{qstoreprice_commons_2026,
  title   = {QStorePrice Commons: A 6+1 Multi-Agent Perishable-Goods Ecosystem},
  year    = {2026},
  url     = {https://github.com/nandeshkanagaraju/QStorePrice},
}
```
