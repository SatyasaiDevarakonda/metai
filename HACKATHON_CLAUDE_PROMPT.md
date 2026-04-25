# HACKATHON_CLAUDE_PROMPT.md
*A self-contained brief you can paste into a fresh Claude Code session
to make the agent extend QStorePrice into a hackathon-winning submission.*

---

## 0. Who you are, what this repo is

You are working in **QStorePrice** — a multi-agent OpenEnv project that
trains a Qwen-2.5 LLM (the "StoreAgent") to write structured
**Operating Briefs** that manage perishable goods in a quick-commerce
store. Source-of-truth docs live in:

- `CLAUDE.md` — architectural cheat-sheet, every rule that matters
- `FreshPrice_SDD.md` — original single-store spec
- `freshprice_env/market_commons_env.py` — the headline 6+1-actor env
- `freshprice_env/long_horizon_env.py` — 30-day wrapper
- `kaggle_qstoreprice.ipynb` — the end-to-end RL pipeline cell-by-cell
- `static/v2/` — the live multi-agent dashboard

Read those before touching anything else. Every architectural rule in
`CLAUDE.md` ("schema drift", "no bare random", "reputation persists",
the three-step anti-hack separation) is load-bearing — violations will
cascade through the rollouts.

## 1. The hackathon brief in one paragraph

This is OpenEnv Theme #1 ("environments where multiple intelligent
agents share a partially-observable world") and Theme #2 ("long-horizon
agents that maintain commitments"). The hero is trained
**SFT → GRPO → DPO** and judged on **WRR** (Weekly Waste Recovery
Rate), **plan adherence**, **cooperation index**, and **anti-hack
violations**. Sub-prizes (Fleet AI, Halluminate, Patronus, Mercor) reward
the realism of the environment and the rigor of the training signal.
Our edge has to be **the simulation**, not the prompt engineering.

## 2. Hard requirements you must satisfy

1. **Generalized SFT data budget.** Replace every hard-coded sample
   count (`30 / 20 / 15 / 25 per difficulty`, "90 per theme", etc.) with
   `training/data_budget.compute_data_budget(...)`. The formula must
   read off (a) model parameter count, (b) format complexity (number
   and length of required brief sections), (c) a target generalization
   coverage, and (d) available VRAM. No magic numbers in the notebook
   — the notebook should print *why* it picked the value.
2. **An RL-learning visualization cell** in
   `kaggle_qstoreprice.ipynb` that shows, for every GRPO episode and
   DPO update, how the agent is actually improving: WRR curve, the
   three reward components (`r1_pricing`, `r2_farmer`, `r3_trend`),
   brief-quality, format-compliance, anti-hack violation rate, engine
   coverage, and pre-vs-post-DPO delta on a held-out scenario.
3. **Blinkit/Zepto-grade realism in the multi-agent commons.** Quick
   commerce in 2026 lives or dies on three constraints: a finite
   **rider/courier pool**, **cohort heterogeneity** (Blinkit Plus vs
   walk-in, premium vs price-sensitive), and **dead-stock liquidation**
   (the B2B firesale channel that kicks in when discounts can't move
   stock fast enough). Add all three as engines/agents the LLM can see
   and act on, with reward components that reflect their economics.
4. **A simulation experience for judges.** The dashboard at
   `static/v2/index.html` already has 6+1 panes. Add a **Simulation
   Theater** panel that lets a judge replay a recorded episode tick-
   by-tick at 1× / 4× / 16× speed, with the LLM's reasoning streamed
   side-by-side and a "what if I changed *this* directive" replay
   button that calls `/commons/replay`. Make it feel like watching a
   live store.
5. **Train end-to-end on the existing pipeline.** Every change must
   keep `kaggle_qstoreprice.ipynb` runnable on a Kaggle T4 (16 GB
   VRAM). No CUDA-only code paths in the env / SFT data generator.

## 3. The exact tasks (do them in this order)

### Task 3.1 — `training/data_budget.py`

Create a small module exporting `compute_data_budget(model_id,
vram_gb, target_format_recall=0.99, format_sections=6,
avg_completion_chars=900, difficulty_levels=3, engine_count=3) ->
DataBudget`. The dataclass returns `n_per_difficulty`, `total_examples`,
`epochs`, `rationale: str`. The formula (no fudge factors masquerading
as constants):

```
sft_token_target = ALPHA * format_complexity / sqrt(model_params_b)
n_per_difficulty = ceil(sft_token_target / (avg_tokens_per_example *
                                            engine_count *
                                            difficulty_levels))
```

where `format_complexity = format_sections * log(avg_completion_chars)`
and `ALPHA` is calibrated *once* against the existing 270-example T4
run that worked for Qwen-2.5-1.5B. Document the calibration in the
docstring so the next person can re-derive it. Cap the result at
`vram_gb_to_max_examples(vram_gb)` so a 4 GB box doesn't ask for 600
examples.

Replace the auto-tune block in `kaggle_qstoreprice.ipynb` (the cell
with the `if "1.5b" in mid: SFT_N_PER_DIFFICULTY = 30` ladder) with one
call to `compute_data_budget(...)` and `print(budget.rationale)`. Same
treatment for `GRPO_EPISODES_MANUAL` (different formula but same idea —
generalize from `vram_gb` and the trajectory-buffer admission rate).

### Task 3.2 — RL-learning cell

Insert a new cell *after* the DPO cell (around the existing "DPO
pre/post WRR" section) titled **CELL 13b — RL Learning Curves**. It
must produce a single multi-panel matplotlib figure with:

- **Panel A** — episode index vs WRR (line + scenario-colored markers)
- **Panel B** — `r1_pricing` / `r2_farmer` / `r3_trend` stacked over
  episodes (so engine coverage is visible at a glance)
- **Panel C** — brief quality + format-compliance % per episode
- **Panel D** — anti-hack violation count per episode (bar)
- **Panel E** — DPO Δ: held-out WRR before vs after on STABLE_WEEK,
  FARMER_WEEK, TREND_WEEK (grouped bar)
- **Panel F** — buffer-admission funnel: rollouts → valid → constitutional
  → admitted to buffer → used as DPO chosen pair (sankey-ish)

Save the PNG to `PLOTS_DIR / "rl_learning_curve.png"` and *also* upload
it as an artifact to the HF model repo if `HF_AUTHED`. Pull data from
`episode_results`, `metrics.get_episode_scores()`,
`trajectory_buffer.get_stats()`, and the DPO pre/post WRR dict that
already exists in the notebook. **Don't fabricate data**: if a panel's
inputs are missing (e.g. DPO didn't run because VRAM was tight), draw
the panel with a clear "DPO skipped — buffer empty" annotation.

### Task 3.3 — Blinkit/Zepto layer

Three additive modules (existing tests must continue to pass):

#### 3.3.a — `freshprice_env/engines/rider_pool_engine.py`

A `RiderPoolEngine` that the env ticks. State:

- `rider_count: int` — pool capacity (default 6 riders / dark store)
- `active_orders: list[Order]` — orders currently being ferried
- `pending_orders: queue[Order]` — orders waiting for a rider

Each tick: each batch's units sold this tick spawn `Order` objects with
a category, a freshness-clock, and a target ETA (10 min if Blinkit
Express, 30 min otherwise — toggle from the brief). When
`pending_orders` exceeds rider capacity the env emits a
`RIDER_SATURATED` event on the bus, freshness clocks for queued orders
keep ticking, and any order whose freshness expires mid-queue scores a
**spoilage-in-transit penalty** that flows into a new
`r6_delivery_quality` reward component. Default the rider-pool size
from `BLINKIT_DEFAULT_RIDER_COUNT` in `constants.py`. The point of this
engine is that *over-aggressive discounts now have a globally visible
cost* — the agent has to reason about throughput, not just price.

#### 3.3.b — `freshprice_env/agents/consumer_agent.py` cohorts

Replace the single `ConsumerAgent` with a `ConsumerCohortAgent` that
holds a list of `Cohort` dataclasses (`PREMIUM`, `MASS`,
`BARGAIN_HUNTER`). Each cohort has its own:

- `price_elasticity: float` — premium ≈ 0.4, mass ≈ 1.2, bargain ≈ 2.5
- `freshness_tolerance: float` — premium walks away from any URGENT
  batch, bargain hunters happily buy CRITICAL at 60 % off
- `eta_tolerance_minutes: int` — premium ≤ 12, mass ≤ 25, bargain ≤ 60
  (this couples to the rider-pool engine — slow ETAs lose premium
  customers)
- `weight: float` — share of footfall, summing to 1.0

Existing `ConsumerAgent` callers should keep working via a thin compat
shim that delegates to `ConsumerCohortAgent.act()` and aggregates the
per-cohort multipliers. Expose per-cohort observation in `observe()`
so the brief prompt can include "**Premium share lost to slow ETA:
38 %**" — the LLM has to learn to balance discount depth against
delivery promise.

#### 3.3.c — `LIQUIDATE` directive in `pricing_engine.py`

Add a fourth action to the PRICING brief vocabulary:
`{"action": "LIQUIDATE", "batch_id": "...", "channel": "B2B"}`.
Effect: the batch is removed from the active inventory at a recovery
fraction of `LIQUIDATION_RECOVERY_RATIO` (default 0.18) of original
price. Allowed only on batches with `urgency == CRITICAL` and
`hours_to_expiry < 6`. Anti-hack rule: liquidating a `FRESH` or
`WATCH` batch is a `RECKLESS_LIQUIDATION` violation and zeroes that
brief's reward. Reward contribution: positive when WRR-equivalent
recovery exceeds expected discount-driven sell-through, negative
otherwise. Document the formula next to `R1_*` constants in
`constants.py`.

### Task 3.4 — Server + frontend simulation theater

Backend (`server/app.py`):

- New endpoint `GET /commons/rider_pool` → `{rider_count, queue_depth,
  saturation_window_pct, avg_eta_seconds}`
- New endpoint `GET /commons/cohorts` → per-cohort current weight,
  retention pct, recent walk-aways
- New endpoint `GET /commons/liquidation` → list of dead-stock
  candidates (urgency, hours, recoverable Rs)
- Push a new WS message kind `tick_frame` from `commons_ws` carrying
  the minimum state needed to render one tick of the theater:
  `{tick, batches[], cohorts[], rider_pool, latest_brief, reasoning}`

Frontend (`static/v2/`):

- New section under the existing pane grid titled **Simulation
  Theater**. Three controls: ▶ Play, ⏸ Pause, speed (1× / 4× / 16×).
- A canvas/grid showing batches color-coded by urgency, with rider
  dots flowing from store → pin and back. Rider-pool saturation drawn
  as a gauge.
- A **Reasoning Stream** column on the right that streams the
  hero's last brief, section-by-section, as the WS pushes them — so
  judges can read the chain of thought *while* the corresponding tick
  plays.
- A **What-if** button next to each PRICING DIRECTIVE in the
  reasoning stream that opens a modal: edit the directive JSON, hit
  "Run fork", and the result calls `/commons/replay` and overlays the
  forked WRR on top of the baseline curve. The slider already exists —
  reuse `replay-slider`, just wire the new modal to it.

Keep all new frontend code in `static/v2/` and load it from
`index.html`. Don't touch the legacy dashboard.

### Task 3.5 — Documentation

Update `CLAUDE.md` "Architecture Cheatsheet" with the new engine /
cohort agent / liquidation flow. Add a one-paragraph "Why these
mechanics?" pointer to public Blinkit/Zepto operating-model writeups so
the next reader can sanity-check the model. Update the README's
"Architecture" section to mention rider-pool + cohorts + liquidation +
sim theater.

## 4. Decision principles (read these before improvising)

- **Generalize, don't hardcode.** If you find yourself typing a number
  with no derivation comment, ask: "what does this value depend on?"
  and put *that* in the code instead. Magic numbers are a smell that
  loses hackathon points.
- **The simulation is the moat.** Prompt engineering is commoditized;
  a realistic Blinkit-grade env is not. Every change should make the
  simulation *more accurate* to real quick-commerce, not just *more
  complex*.
- **Reward components are a contract.** New components
  (`r6_delivery_quality`, `r7_liquidation`) must follow the existing
  three-step separation: *RuleExecutor flags → env wires → engine
  computes*. Never let an engine read agent state directly.
- **Tests are the safety net.** Before submitting, `python -m unittest
  discover tests -v` must show all green. New mechanics need at least
  one happy-path and one anti-hack test.
- **Don't break the OpenEnv contract.** `/reset`, `/step`, `/state`,
  `/ws` are the judged surface. Everything you add lives under
  `/commons/*` or `/admin/*`.
- **Reproducibility.** All new randomness goes through an injected
  `rng: random.Random`. No bare `random.random()`. No bare `time.time()`
  for seeding.
- **The notebook is the demo.** A judge will open
  `kaggle_qstoreprice.ipynb` and run-all. If a cell fails because a
  new mechanic is missing a fallback, you've lost the demo. Every new
  mechanic must degrade gracefully.

## 5. Definition of done

- [ ] `python -m unittest discover tests -v` — all green
- [ ] `kaggle_qstoreprice.ipynb` runs cell-by-cell on T4 with no
      pre-edited values, calling `compute_data_budget` and printing its
      rationale
- [ ] The new RL-learning-curve cell renders a 6-panel figure for a
      ≥3-episode GRPO run
- [ ] `MarketCommonsEnv` ticks the rider pool, cohort agent, and
      liquidation channel; reward components `r6_delivery_quality` and
      `r7_liquidation` appear in the per-step info dict
- [ ] Dashboard at `/` shows the Simulation Theater + reasoning stream
      and the **What-if** modal can replay a fork
- [ ] `CLAUDE.md` and `README.md` describe the new layer
- [ ] No new magic numbers without a `# why this value` comment

When all of those tick, push to HF Hub, run `validate_submission.py`,
and stop.
