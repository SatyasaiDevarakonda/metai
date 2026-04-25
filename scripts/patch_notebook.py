"""Surgical rewrite of kaggle_qstoreprice.ipynb to use the new APIs.

Targets each cell that the post-mortem flagged as broken. Run from the
repo root:

    python scripts/patch_notebook.py

The script idempotently patches the notebook in-place (no behaviour
change on a second run). Cells are matched by header comment, not by
index, so reordering / inserting cells in the future doesn't break it.

Cells touched:

  Cell 11 (GRPO rollouts)       — scenario rotation + lower max_new_tokens
                                  + per-episode engine coverage + parse-fail
                                  counter
  Cell 12 (DPO)                 — DPO_MIN_PAIRS lowered to 2 + use
                                  trajectory_buffer.dpo_readiness()
  Cell 13 (Evaluation)          — eval all 5 scenarios + n>=5 seeds + suppress
                                  ± when n<5 + report both WRR and constitutional
                                  pass-rate gates
  Cell 14 (Anti-hack)           — switch to scan_all_rollouts so rejected
                                  episodes ARE scanned
  Cell 17 (API test)            — use /gym/reset, /gym/step, /gym/state
                                  (guaranteed gym contract)
  Cell 18 (server kill)         — neutralised; kill moves to end of cell 21
  Cell 19 (HF push)             — add explicit HfApi().repo_info() round-trip
                                  so the URL is only printed if the model
                                  actually exists on the Hub
  Cell 20 (Final summary)       — use trajectory_buffer.dpo_readiness().can_run
                                  as truth for "DPO actually ran"
  Cell 21 (Admin dashboard)     — kills the server at the end so cell ordering
                                  no longer matters
"""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "kaggle_qstoreprice.ipynb"


# --------------------------------------------------------------------------
# New cell sources
# --------------------------------------------------------------------------

CELL_11_GRPO = '''\
# ============================================================
# CELL 11 — GRPO EPISODE ROLLOUTS (scenario-rotating)
# Runs GRPO_EPISODES of the LLM acting in FreshPriceEnv.
#
# Scenario rotation (NEW): instead of pinning to STABLE_WEEK, rollouts
# round-robin across [STABLE_WEEK, FARMER_WEEK, TREND_WEEK] so the
# trajectory buffer sees PRICING + FARMER + TREND briefs. Without this,
# R2-F and R3-T columns silently stay zero because no FARMER / TREND
# episodes were ever run, and DPO would only train PRICING behaviour.
# Override ROTATE_SCENARIOS=False in Cell 1 to keep the legacy single-
# scenario behaviour.
#
# VRAM-based episode count:
#   < 16 GB (T4):   3 episodes
#   16-24 GB:       6 episodes
#   >= 24 GB:       12 episodes
# ============================================================

import sys, os, json, time, random
sys.path.insert(0, REPO_DIR)

from freshprice_env.freshprice_env import FreshPriceEnv
from freshprice_env.enums import CurriculumScenario
from freshprice_env.monitoring import metrics
from training.curriculum import CurriculumManager, EpisodeResult
from training.trajectory_buffer import Trajectory, TrajectoryBuffer
from training.counterfactual import CounterfactualEngine

# ---- Resolve GRPO_EPISODES ----
if AUTO_TUNE:
    if VRAM_GB < 16:
        GRPO_EPISODES = 3
    elif VRAM_GB < 24:
        GRPO_EPISODES = 6
    else:
        GRPO_EPISODES = 12
    print(f"AUTO_TUNE: GRPO_EPISODES = {GRPO_EPISODES} (VRAM_GB={VRAM_GB})")
else:
    GRPO_EPISODES = GRPO_EPISODES_MANUAL
    print(f"MANUAL: GRPO_EPISODES = {GRPO_EPISODES}")

# Rotation switch — defaults ON so engine coverage is real
ROTATE_SCENARIOS = globals().get("ROTATE_SCENARIOS", True)
ROTATION_LIST = [
    CurriculumScenario.STABLE_WEEK,
    CurriculumScenario.FARMER_WEEK,
    CurriculumScenario.TREND_WEEK,
]

rng                 = random.Random(SEED)
curriculum          = CurriculumManager()
trajectory_buffer   = TrajectoryBuffer(rng=rng)
counterfactual_eng  = CounterfactualEngine(rng=rng)

episode_results = []   # ALL rollouts (including buffer-rejected) for honest scans

print(f"\\nStarting GRPO rollouts: {GRPO_EPISODES} episodes "
      f"({'rotating ' + '+'.join(s.name for s in ROTATION_LIST) if ROTATE_SCENARIOS else 'STABLE_WEEK only'})")
print("-" * 70)
print(f"{'Ep':>4} {'Scenario':>14} {'WRR':>6} {'R1-P':>6} {'R2-F':>6} {'R3-T':>6} "
      f"{'Qual':>6} {'Viol':>5} {'Const':>6} {'PFail':>5} {'Time':>6}")
print("-" * 70)

total_start = time.time()

# Engine-coverage tracker across all rollouts in this run
engine_coverage = {"PRICING": 0, "FARMER": 0, "TREND": 0}

for ep_idx in range(GRPO_EPISODES):
    ep_seed = rng.randint(0, 999999)
    ep_start = time.time()

    # Rotate per episode so a 3-episode run hits one of each engine
    scenario = (
        ROTATION_LIST[ep_idx % len(ROTATION_LIST)]
        if ROTATE_SCENARIOS else CurriculumScenario.STABLE_WEEK
    )

    try:
        env = FreshPriceEnv(scenario=scenario, seed=ep_seed, llm_client=llm_client)
        obs, info = env.reset(seed=ep_seed)

        ep_briefs = []
        done = False
        step_count = 0
        parse_failures = 0
        parse_fail_with_reward = 0

        while not done:
            try:
                raw_brief = llm_client.generate(obs)
            except Exception as gen_err:
                print(f"    [ep {ep_idx+1} step {step_count}] generate failed: {gen_err}")
                raw_brief = (
                    "SITUATION: fallback\\n\\nSIGNAL ANALYSIS: N/A\\n\\n"
                    "VIABILITY CHECK: N/A\\n\\nRECOMMENDATION: hold\\n\\n"
                    'DIRECTIVE:\\n{"engine": "PRICING", "actions": []}\\n\\n'
                    "CONFIDENCE: LOW\\n"
                )
            obs, reward, done, truncated, info = env.step(raw_brief)

            engine_t = info.get("engine_type", "PRICING")
            engine_coverage[engine_t] = engine_coverage.get(engine_t, 0) + 1
            if not info.get("parse_success", True):
                parse_failures += 1
                # info["wrr_delta"] is the bare WRR change (added by env.step)
                if info.get("wrr_delta", reward) > 0:
                    parse_fail_with_reward += 1

            ep_briefs.append({
                "tick":         info.get("tick", step_count * 8),
                "engine_type":  engine_t,
                "raw_response": raw_brief,
                "quality_score":info.get("quality_score", 0.0),
                "reward_delta": reward,
                "parse_success":info.get("parse_success", True),
                "wrr_delta":    info.get("wrr_delta", 0.0),
            })

            metrics.record_step(
                scenario=scenario.name,
                tick=info.get("tick", step_count * 8),
                engine_type=engine_t,
                reward=float(reward),
                quality_score=float(info.get("quality_score", 0.0)),
                parse_success=bool(info.get("parse_success", True)),
            )
            step_count += 1

        final_reward = info.get("final_reward", {}) or {}
        audit = info.get("constitutional_audit", {}) or {}

        wrr = float(final_reward.get("wrr", 0.0))
        r1 = float(final_reward.get("r1_pricing", 0.0))
        r2 = float(final_reward.get("r2_farmer", 0.0))
        r3 = float(final_reward.get("r3_trend", 0.0))
        quality = float(final_reward.get("brief_quality_score", 0.0))
        violations = int(final_reward.get("anti_hack_violations", 0))
        ep_valid = bool(final_reward.get("episode_valid", True))
        const_pass = bool(audit.get("passed", True))

        # Always-recorded result, regardless of buffer eligibility
        result = {
            "episode_num":            ep_idx,
            "scenario":               scenario,
            "scenario_name":          scenario.name,
            "wrr":                    wrr,
            "r1_pricing":             r1,
            "r2_farmer":              r2,
            "r3_trend":               r3,
            "brief_quality_score":    quality,
            "anti_hack_violations":   violations,
            "episode_valid":          ep_valid,
            "constitutional_passed":  const_pass,
            "briefs":                 ep_briefs,
            "final_reward":           final_reward,
            "parse_failures":         parse_failures,
            "parse_fail_with_positive_reward": parse_fail_with_reward,
        }
        episode_results.append(result)

        # Buffer admission still gated by validity + constitutional check.
        # Anti-hack scan in cell 14 will see the rejected ones too.
        traj = Trajectory(
            episode_num=ep_idx, scenario=scenario, wrr=wrr,
            brief_quality_score=quality,
            constitutional_passed=const_pass, episode_valid=ep_valid,
            briefs=ep_briefs, reward_engine_snapshot=final_reward,
        )
        trajectory_buffer.add(traj)

        # Curriculum tracker
        curriculum.record_episode(EpisodeResult(
            episode_num=ep_idx, scenario=scenario, wrr=wrr,
            brief_quality_score=quality, anti_hack_violations=violations,
            constitutional_passed=const_pass, episode_valid=ep_valid,
        ))

        metrics.record_episode(
            scenario=scenario.name, wrr=wrr,
            r1_pricing=r1, r2_farmer=r2, r3_trend=r3,
            brief_quality_score=quality,
            anti_hack_violations=violations,
            constitutional_passed=const_pass,
            episode_valid=ep_valid, steps=step_count,
            agent_type="llm_rollout",
        )

        elapsed = time.time() - ep_start
        const_str = "PASS" if const_pass else "FAIL"
        print(
            f"{ep_idx+1:>4} {scenario.name:>14} {wrr:>6.3f} {r1:>6.3f} {r2:>6.3f} "
            f"{r3:>6.3f} {quality:>6.3f} {violations:>5} {const_str:>6} "
            f"{parse_fail_with_reward:>5} {elapsed:>5.0f}s"
        )

    except Exception as ep_err:
        print(f"  Episode {ep_idx+1} crashed: {type(ep_err).__name__}: {ep_err}")

print("-" * 70)
print(f"Total rollout time: {time.time()-total_start:.0f}s")
print(f"Engine coverage   : {engine_coverage}")
print(f"Buffer size       : {trajectory_buffer.get_stats()['buffer_size']}")
print(f"All rollouts kept : {len(episode_results)} (incl. buffer-rejected)")

# Honest DPO readiness check (used by cells 12 + 20 instead of a hardcoded flag)
DPO_READINESS = trajectory_buffer.dpo_readiness(
    min_buffer=globals().get("DPO_MIN_BUFFER", 2),
)
print(f"DPO readiness     : can_run={DPO_READINESS.can_run} | "
      f"reason='{DPO_READINESS.reason}'")
'''


CELL_12_DPO_PATCH = '''\
def _decide_skip_reason():
    if not DPO_ENABLED:
        return "DPO_ENABLED=False in Cell 1"
    if VRAM_GB and VRAM_GB < DPO_MIN_VRAM_GB:
        return f"VRAM {VRAM_GB} GB < DPO_MIN_VRAM_GB ({DPO_MIN_VRAM_GB} GB)"
    if trajectory_buffer is None:
        return "trajectory_buffer is None (rollout cell did not run)"
    # Honest readiness check (uses TrajectoryBuffer.dpo_readiness from
    # the new code) — the prior `< DPO_MIN_PAIRS` test was a hardcoded
    # `>= 4` that silently skipped DPO on every short Kaggle run.
    readiness = trajectory_buffer.dpo_readiness(
        min_buffer=globals().get("DPO_MIN_BUFFER", 2),
    )
    if not readiness.can_run:
        return readiness.reason
    return None'''


CELL_13_EVAL = '''\
# ============================================================
# CELL 13 — DETERMINISTIC EVALUATION
# Greedy decoding on fixed seeds. Fully reproducible.
#
# Now evaluates ALL curriculum scenarios with >= 5 seeds each so the
# std-dev is meaningful. Uses CurriculumManager.is_eval_above_promotion
# which combines WRR threshold AND constitutional pass-rate floor —
# replaces the old WRR-only "ABOVE PROMOTION THRESHOLD" line that hid
# 75% constitutional failures.
# ============================================================

import sys, os, json, math, time, gc
sys.path.insert(0, REPO_DIR)

from freshprice_env.freshprice_env import FreshPriceEnv
from freshprice_env.enums import CurriculumScenario
from freshprice_env.monitoring import metrics
from freshprice_env.brief_pipeline.prompt_builder import OperatingBriefPromptBuilder
from training.curriculum import CurriculumManager

eval_results = {}

# Reload model if Cell 12 (DPO) freed it
need_reload = False
try:
    _ = model, tokenizer
except NameError:
    need_reload = True

if need_reload:
    print(f"Reloading model from {CURRENT_CHECKPOINT} for evaluation...")
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=CURRENT_CHECKPOINT,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            token=HF_TOKEN if HF_TOKEN_SET else None,
        )
    except Exception as e:
        print(f"[!] Could not reload model for eval: {e}")
        print("    Skipping evaluation.")
        model = None
        tokenizer = None


class GreedyClient:
    def __init__(self, model, tokenizer, system_prompt):
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
        self._model = model
        self._tok = tokenizer
        self._sys = system_prompt

    def generate(self, prompt: str) -> str:
        import torch
        full = (
            f"<|im_start|>system\\n{self._sys}<|im_end|>\\n"
            f"<|im_start|>user\\n{prompt}<|im_end|>\\n"
            f"<|im_start|>assistant\\n"
        )
        inputs = self._tok(full, return_tensors="pt", truncation=True,
                           max_length=3800).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=600, do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        return self._tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )


if model is None or tokenizer is None:
    print("Skipping evaluation (model not loaded).")
else:
    greedy_client = GreedyClient(
        model=model, tokenizer=tokenizer,
        system_prompt=OperatingBriefPromptBuilder.SYSTEM_PROMPT,
    )

    # ALL canonical scenarios + REGULATORY_WEEK if available
    eval_scenarios = list(CurriculumScenario)

    # Need n>=5 for std-dev to be meaningful. Override with EVAL_SEEDS=N
    # in cell 1 to bump for paper-grade numbers.
    EVAL_SEEDS_PER_SCENARIO = globals().get("EVAL_SEEDS", 5)
    MIN_N_FOR_STD = 5

    print("=" * 70)
    print(" EVALUATION REPORT")
    print(f" Checkpoint: {CURRENT_CHECKPOINT}")
    print(f" Scenarios:  {[s.name for s in eval_scenarios]}")
    print(f" Seeds/scen: {EVAL_SEEDS_PER_SCENARIO}")
    print("=" * 70)

    all_eval_episodes = []   # for the combined promotion gate

    for scenario in eval_scenarios:
        level = scenario.value
        seeds = [level * 1000 + i for i in range(EVAL_SEEDS_PER_SCENARIO)]
        ep_res = []

        print(f"\\n-- {scenario.name} ({EVAL_SEEDS_PER_SCENARIO} episodes) --")

        for seed in seeds:
            try:
                env = FreshPriceEnv(scenario=scenario, seed=seed, llm_client=greedy_client)
                obs, info = env.reset(seed=seed)
                done = False
                while not done:
                    brief = greedy_client.generate(obs)
                    obs, reward, done, truncated, info = env.step(brief)

                fr = info.get("final_reward", {}) or {}
                audit = info.get("constitutional_audit", {}) or {}

                rec = {
                    "seed": seed,
                    "wrr": float(fr.get("wrr", 0.0)),
                    "r1": float(fr.get("r1_pricing", 0.0)),
                    "r2": float(fr.get("r2_farmer", 0.0)),
                    "r3": float(fr.get("r3_trend", 0.0)),
                    "quality": float(fr.get("brief_quality_score", 0.0)),
                    "violations": int(fr.get("anti_hack_violations", 0)),
                    "constitutional_passed": bool(audit.get("passed", True)),
                }
                ep_res.append(rec)
                all_eval_episodes.append(rec)
                metrics.record_episode(
                    scenario=scenario.name, wrr=rec["wrr"],
                    r1_pricing=rec["r1"], r2_farmer=rec["r2"], r3_trend=rec["r3"],
                    brief_quality_score=rec["quality"],
                    anti_hack_violations=rec["violations"],
                    constitutional_passed=rec["constitutional_passed"],
                    steps=84, agent_type="llm_eval",
                )
                const_str = "PASS" if rec["constitutional_passed"] else "FAIL"
                print(f"  seed={seed:5d} WRR={rec['wrr']:.3f} qual={rec['quality']:.3f} "
                      f"viol={rec['violations']} const={const_str}")
            except Exception as eval_err:
                print(f"  seed={seed} crashed: {type(eval_err).__name__}: {eval_err}")

        if not ep_res:
            continue

        wrrs = [r["wrr"] for r in ep_res]
        qs = [r["quality"] for r in ep_res]
        vs = [r["violations"] for r in ep_res]
        cp = sum(1 for r in ep_res if r["constitutional_passed"])
        n = len(ep_res)
        mean_wrr = sum(wrrs) / n
        std_wrr = (sum((x - mean_wrr) ** 2 for x in wrrs) / n) ** 0.5

        std_meaningful = n >= MIN_N_FOR_STD
        eval_results[scenario.name] = {
            "wrr_mean": round(mean_wrr, 4),
            "wrr_std": round(std_wrr, 4),
            "wrr_min": round(min(wrrs), 4),
            "wrr_max": round(max(wrrs), 4),
            "n": n,
            "std_meaningful": std_meaningful,
            "quality": round(sum(qs) / n, 4),
            "violations_mean": round(sum(vs) / n, 2),
            "constitutional_pass_rate": f"{cp}/{n}",
            "constitutional_pass_fraction": round(cp / n, 3),
        }

        if std_meaningful:
            print(f"  -> WRR {mean_wrr:.4f} +/- {std_wrr:.4f}  "
                  f"const_pass {cp}/{n}")
        else:
            print(f"  -> WRR {mean_wrr:.4f}  const_pass {cp}/{n}  "
                  f"(std not meaningful at n={n}; need n>={MIN_N_FOR_STD})")

    # ------------------------------------------------------------------
    # Combined promotion gate (WRR threshold AND constitutional floor)
    # ------------------------------------------------------------------
    if all_eval_episodes:
        gate_passes, gate_diag = CurriculumManager.is_eval_above_promotion(
            all_eval_episodes,
        )
        print("\\n" + "=" * 70)
        print(" PROMOTION GATE (combined WRR + constitutional pass-rate)")
        print("=" * 70)
        print(f"  Eval episodes              : {gate_diag['n']}")
        print(f"  Mean WRR                   : {gate_diag['wrr_mean']:.4f} "
              f"(threshold {gate_diag['wrr_threshold']})")
        print(f"  Constitutional pass rate   : "
              f"{gate_diag['constitutional_pass_rate']:.0%} "
              f"(floor {gate_diag['constitutional_floor']:.0%})")
        print(f"  WRR ok                     : {gate_diag['wrr_ok']}")
        print(f"  Constitution ok            : {gate_diag['constitution_ok']}")
        status = "ABOVE PROMOTION THRESHOLD" if gate_passes else "BELOW PROMOTION THRESHOLD"
        print(f"  STATUS                     : {status}")
        print(f"  Reason                     : {gate_diag['reason']}")

    # Save eval_results.json with non-trivial content so the summary cell
    # check stops reporting "OK (0 KB)".
    eval_path = os.path.join(WORK_DIR, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump({
            "by_scenario": eval_results,
            "n_episodes_total": len(all_eval_episodes),
            "checkpoint": CURRENT_CHECKPOINT,
        }, f, indent=2)
    print(f"\\nWrote {eval_path} ({os.path.getsize(eval_path)} bytes)")
'''


CELL_14_ANTIHACK = '''\
# ============================================================
# CELL 14 — ANTI-HACK ANALYSIS (scans ALL rollouts, not just buffer)
# Closes the bug where the scan ran on trajectory_buffer.get_top_n(),
# which had already filtered out invalid + constitutionally-failed
# episodes. The scanner could never see the worst cases by construction.
# ============================================================

import sys
sys.path.insert(0, REPO_DIR)

from eval.anti_hack_checker import AntiHackChecker

print("Anti-Hack Pattern Analysis")
print("=" * 60)

# `episode_results` is the FULL list of rollouts collected in cell 11
# — buffer-eligible ones plus the rejects. Pass them all to the new
# scan_all_rollouts helper so rejected episodes are scanned too.
total_rollouts = len(episode_results) if "episode_results" in dir() else 0
print(f"Scanning {total_rollouts} rollouts (incl. buffer-rejected)...")

if total_rollouts == 0:
    print("episode_results empty — cell 11 did not run successfully.")
else:
    summary = AntiHackChecker.scan_all_rollouts(episode_results)

    print(f"\\nSummary:")
    print(f"  Total rollouts scanned : {summary['total_trajectories']}")
    print(f"    Buffer-eligible      : {summary['buffer_eligible']}")
    print(f"    Buffer-rejected      : {summary['buffer_excluded']}")
    print(f"  Clean (DPO-safe)       : {summary['clean']}")
    print(f"  Flagged (review)       : {summary['flagged_for_review']}")
    print(f"  Excluded (hack)        : {summary['excluded']}")

    if summary["pattern_frequency"]:
        print(f"\\nDetected patterns:")
        for ptype, count in sorted(summary["pattern_frequency"].items(),
                                    key=lambda x: -x[1]):
            print(f"  {ptype:35s}: {count}")
        print(f"  Most common: {summary['most_common_pattern']}")

    if summary.get("hidden_pattern_frequency"):
        print(f"\\n  ⚠ Patterns ONLY visible in buffer-rejected rollouts:")
        for ptype, count in summary["hidden_pattern_frequency"].items():
            print(f"    {ptype}: {count}")
        print("    (these would have been missed by the legacy buffer-only scan)")
    else:
        print("\\n  No hidden patterns: rejected rollouts share their patterns "
              "with buffer-eligible ones.")

    if summary["total_trajectories"] > 0:
        dpo_safe_pct = summary['clean'] / summary['total_trajectories'] * 100
        print(f"\\nDPO-safe rate (across ALL rollouts): {dpo_safe_pct:.0f}%")
        if dpo_safe_pct < 50:
            print("WARNING: < 50% DPO-safe. Model may be reward hacking.")
            print("Consider more SFT epochs or reducing DPO learning rate.")
'''


CELL_17_API_TEST = '''\
# ============================================================
# CELL 17 — TEST BACKEND ENDPOINTS via /gym/* (gym-compliant)
# The legacy openenv-core /reset / /step / /state had a contract
# mismatch (no info field, stale episode_id, error payloads). The
# new /gym/reset, /gym/step, /gym/state endpoints in server.app are
# guaranteed-conformant (Gymnasium semantics) and back the Kaggle
# notebook tests reliably.
# ============================================================

import json, urllib.request, urllib.error


def http_get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def http_post(url, data):
    try:
        req = urllib.request.Request(
            url, data=json.dumps(data).encode(), method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


passed_checks = []
failed_checks = []

if server_up:
    print("=" * 60)
    print(" BACKEND API TEST — Live Server (/gym/* endpoints)")
    print("=" * 60)

    # 1. Health
    resp = http_get(f"{SERVER_URL}/health")
    print(f"\\nGET /health        -> {resp}")
    (passed_checks if "error" not in resp else failed_checks).append("/health")

    # 2. /gym/reset — must include `info`
    resp = http_post(f"{SERVER_URL}/gym/reset",
                     {"scenario": "STABLE_WEEK", "seed": 42})
    print(f"POST /gym/reset    -> keys: {list(resp.keys())}")
    obs_excerpt = str(resp.get("observation", ""))[:200]
    print(f"  observation[:200]: {obs_excerpt}")
    if "observation" in resp and "info" in resp:
        passed_checks.append("/gym/reset (has obs+info)")
    else:
        failed_checks.append(f"/gym/reset missing keys: {set(['observation','info'])-set(resp.keys())}")

    # 3. /gym/step — must return 5 keys (gym contract)
    step_payload = {
        "action": (
            "## SITUATION SUMMARY\\nBatch analysis complete.\\n\\n"
            "## SIGNAL ANALYSIS\\nModerate expiry risk.\\n\\n"
            "## VIABILITY CHECK\\nDiscount viable.\\n\\n"
            "## RECOMMENDATION\\nHold prices on FRESH stock.\\n\\n"
            '## DIRECTIVE\\n{"engine": "PRICING", "actions": []}\\n\\n'
            "## CONFIDENCE\\n0.65\\n"
        )
    }
    resp = http_post(f"{SERVER_URL}/gym/step", step_payload)
    print(f"POST /gym/step     -> keys: {list(resp.keys())}")
    print(f"  reward            : {resp.get('reward', 'N/A')}")
    print(f"  terminated        : {resp.get('terminated', 'N/A')}")
    print(f"  truncated         : {resp.get('truncated', 'N/A')}")
    info_keys = list(resp.get("info", {}).keys()) if isinstance(resp.get("info"), dict) else []
    print(f"  info keys         : {info_keys[:8]}...")
    expected = {"observation", "reward", "terminated", "truncated", "info"}
    if expected.issubset(set(resp.keys())):
        passed_checks.append("/gym/step (5-key gym contract)")
    else:
        failed_checks.append(f"/gym/step missing: {expected - set(resp.keys())}")

    # 4. /gym/state — must persist episode_id + step_count
    resp = http_get(f"{SERVER_URL}/gym/state")
    print(f"GET /gym/state     -> episode_id={resp.get('episode_id')} "
          f"step_count={resp.get('step_count')}")
    if resp.get("episode_id") and resp.get("step_count", 0) >= 1:
        passed_checks.append("/gym/state (persistent session)")
    else:
        failed_checks.append("/gym/state lost episode_id or step_count")

    # 5. Docs (HTML, not JSON — just check it's served)
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/docs", timeout=5) as r:
            doc_ok = r.status == 200
    except Exception:
        doc_ok = False
    print(f"GET /docs          -> {'HTML page available' if doc_ok else 'unavailable'}")
    (passed_checks if doc_ok else failed_checks).append("/docs")

    print(f"\\n=== Endpoint check summary ===")
    print(f"  PASSED: {len(passed_checks)} -> {passed_checks}")
    print(f"  FAILED: {len(failed_checks)} -> {failed_checks}")
    print(f"\\nServer URL: {SERVER_URL}    Swagger UI: {SERVER_URL}/docs")

else:
    print("=" * 60)
    print(" BACKEND API — Python Simulation (no live server)")
    print("=" * 60)
    print("Server did not start (likely missing openenv-core).")
    print("Demonstrating equivalent logic via direct Python calls:")

    import sys
    sys.path.insert(0, REPO_DIR)
    from freshprice_env.freshprice_env import FreshPriceEnv
    from freshprice_env.enums import CurriculumScenario

    print('\\nSimulated GET /health -> {"status": "ok"}')
    env = FreshPriceEnv(scenario=CurriculumScenario.STABLE_WEEK, seed=42)
    obs, info = env.reset()
    print(f"Simulated POST /gym/reset -> obs len={len(obs)}, "
          f"info keys={list(info.keys())[:5]}")

    test_brief = (
        "## SITUATION SUMMARY\\nInventory assessed.\\n\\n"
        "## SIGNAL ANALYSIS\\nModerate demand.\\n\\n"
        "## VIABILITY CHECK\\nConservative strategy.\\n\\n"
        "## RECOMMENDATION\\nHold prices.\\n\\n"
        '## DIRECTIVE\\n{"engine": "PRICING", "actions": []}\\n\\n'
        "## CONFIDENCE\\n0.6\\n"
    )
    obs2, reward, done, truncated, info2 = env.step(test_brief)
    print(f"Simulated POST /gym/step -> reward={reward:.4f}, "
          f"terminated={done}, truncated={truncated}")
    state = env.state()
    print(f"Simulated GET /gym/state -> tick={state.get('tick')} "
          f"wrr_so_far={state.get('wrr_so_far'):.4f}")

    print("\\nTo run the live server, ensure server.app:app is reachable at "
          f"{SERVER_URL}.")
'''


CELL_18_SERVER_KILL = '''\
# ============================================================
# CELL 18 — SERVER KILL DEFERRED to end of cell 21
# Cell 21 (admin dashboard) needs the server alive to poll
# /admin/dashboard. Killing here used to cause a "Connection refused"
# error in cell 21. The kill now happens at the *end* of cell 21
# (look for the `_safe_kill_server()` call there) so ordering is
# guaranteed regardless of how the notebook is run.
# ============================================================
print("Server kill deferred to end of cell 21 (admin dashboard).")
'''


CELL_19_HF_PUSH = '''\
# ============================================================
# CELL 19 — PUSH TRAINED MODEL TO HUGGING FACE HUB
# Now verifies the upload via HfApi().repo_info() before printing the
# URL. Previously the URL was hardcoded in cell 20's summary even when
# no push happened.
# ============================================================

import os, json

PUSH_OK = False
PUSH_REASON = ""
PUSH_URL = ""

if not HF_TOKEN_SET or not HF_AUTHED:
    PUSH_REASON = "HF_TOKEN not set or auth failed"
elif "your-hf-username" in HF_REPO_ID:
    PUSH_REASON = f"HF_REPO_ID still placeholder ({HF_REPO_ID})"
else:
    try:
        from huggingface_hub import HfApi, upload_folder

        api = HfApi(token=HF_TOKEN)
        print(f"Creating/verifying HF repo: {HF_REPO_ID}")
        api.create_repo(
            repo_id=HF_REPO_ID, repo_type="model",
            exist_ok=True, private=False,
        )

        push_dir = CURRENT_CHECKPOINT if os.path.exists(CURRENT_CHECKPOINT) else SFT_DIR
        print(f"Pushing from: {push_dir}")

        model_card = f"""---
tags:
  - qstoreprice
  - reinforcement-learning
  - perishable-goods
  - operating-brief
  - wrr
base_model: {MODEL_ID}
---

# QStorePrice AI — Trained Checkpoint

**Base model**: `{MODEL_ID}`
**Training**: SFT warm-start ({SFT_EPOCHS} epochs) + GRPO rollouts ({GRPO_EPISODES} episodes)
**Metric**: WRR (Weekly Waste Recovery Rate)

## Evaluation Results

```json
{json.dumps(eval_results, indent=2)}
```

## Project
- GitHub: https://github.com/nandeshkanagaraju/QStorePrice
- Trained on Kaggle with Unsloth 4-bit + LoRA
"""
        card_path = os.path.join(push_dir, "README.md")
        with open(card_path, "w") as f:
            f.write(model_card)

        print("Uploading folder...")
        upload_folder(
            repo_id=HF_REPO_ID,
            folder_path=push_dir,
            repo_type="model",
            token=HF_TOKEN,
            ignore_patterns=["*.pyc", "__pycache__", "optimizer.pt"],
            commit_message=f"QStorePrice SFT+GRPO checkpoint",
        )

        # Verify the push actually landed before printing the URL.
        # repo_info raises if the repo isn't found.
        info = api.repo_info(HF_REPO_ID, repo_type="model")
        sibling_count = len(info.siblings) if info.siblings else 0
        if sibling_count > 0:
            PUSH_OK = True
            PUSH_URL = f"https://huggingface.co/{HF_REPO_ID}"
            print(f"VERIFIED: {sibling_count} files at {PUSH_URL}")
        else:
            PUSH_REASON = "repo exists but contains 0 files (push may have failed)"

        # Plot uploads (best-effort)
        for plot_file in ["training_metrics.png", "eval_wrr_by_scenario.png"]:
            plot_path = os.path.join(PLOTS_DIR, plot_file)
            if os.path.exists(plot_path):
                try:
                    api.upload_file(
                        path_or_fileobj=plot_path,
                        path_in_repo=f"plots/{plot_file}",
                        repo_id=HF_REPO_ID, repo_type="model", token=HF_TOKEN,
                    )
                    print(f"  Uploaded: {plot_file}")
                except Exception as e:
                    print(f"  {plot_file}: {e}")

    except Exception as e:
        PUSH_REASON = f"{type(e).__name__}: {e}"

if not PUSH_OK:
    print(f"HF push skipped/failed: {PUSH_REASON}")
'''


CELL_20_SUMMARY = '''\
# ============================================================
# CELL 20 — FINAL SUMMARY
# Reports what ACTUALLY happened, not what was configured.
#   - "DPO actually ran" comes from trajectory_buffer.dpo_readiness()
#   - HF repo URL only shown if PUSH_OK was set in cell 19
#   - eval_results.json size reported truthfully (>50 bytes = non-trivial)
# ============================================================

import os, json

print("=" * 70)
print(" QStorePrice AI — Run Complete")
print("=" * 70)

print(f"\\n{'Model':<32}: {MODEL_ID}")
print(f"{'SFT epochs':<32}: {SFT_EPOCHS}")
print(f"{'GRPO episodes (configured)':<32}: {GRPO_EPISODES}")
print(f"{'GRPO episodes (recorded)':<32}: "
      f"{len(episode_results) if 'episode_results' in dir() else 0}")

# Honest DPO status: configured vs actually-ran
dpo_actually_ran = False
dpo_reason = "(not evaluated)"
if "trajectory_buffer" in dir() and trajectory_buffer is not None:
    readiness = trajectory_buffer.dpo_readiness(
        min_buffer=globals().get("DPO_MIN_BUFFER", 2),
    )
    dpo_actually_ran = bool(readiness.can_run) and globals().get("_DPO_RAN", False)
    dpo_reason = readiness.reason
print(f"{'DPO enabled (configured)':<32}: {DPO_ENABLED}")
print(f"{'DPO actually ran':<32}: {dpo_actually_ran}  ({dpo_reason})")
print(f"{'Seed':<32}: {SEED}")
print(f"{'Final checkpoint':<32}: {CURRENT_CHECKPOINT}")

if globals().get("PUSH_OK", False):
    print(f"{'HF repo (verified)':<32}: {PUSH_URL}")
else:
    reason = globals().get("PUSH_REASON", "(skipped)")
    print(f"{'HF repo':<32}: skipped — {reason}")

if eval_results:
    print(f"\\n{'-'*70}")
    print(" Evaluation Results")
    print(f"{'-'*70}")
    print(f"  {'Scenario':<22} {'WRR':>8} {'+/-':>10} {'Quality':>8} "
          f"{'Viol':>5} {'Const':>7}")
    print(f"  {'-'*65}")
    for sc, r in eval_results.items():
        std_disp = (
            f"{r['wrr_std']:>10.4f}" if r.get("std_meaningful", False)
            else f"{'(n='+str(r.get('n','?'))+')':>10}"
        )
        print(
            f"  {sc:<22} {r['wrr_mean']:>8.4f} {std_disp} "
            f"{r['quality']:>8.4f} {r['violations_mean']:>5.1f} "
            f"{r['constitutional_pass_rate']:>7}"
        )
    all_wrrs = [v["wrr_mean"] for v in eval_results.values()]
    overall = sum(all_wrrs) / len(all_wrrs)
    print(f"  {'-'*65}")
    print(f"  {'Overall mean WRR':<22} {overall:>8.4f}")
else:
    print("\\n(no eval results)")

print(f"\\n{'-'*70}")
print(" Output Files")
print(f"{'-'*70}")
outputs = [
    ("SFT checkpoint",        SFT_DIR),
    ("DPO dir",               DPO_DIR),
    ("Episode log",           f"{WORK_DIR}/episode_log.json"),
    ("Eval results",          f"{WORK_DIR}/eval_results.json"),
    ("Training metrics plot", f"{PLOTS_DIR}/training_metrics.png"),
    ("Eval WRR plot",         f"{PLOTS_DIR}/eval_wrr_by_scenario.png"),
]
for label, path in outputs:
    exists = os.path.exists(path)
    if not exists:
        print(f"  {label:<32}: (skipped/missing)")
        continue
    if os.path.isdir(path):
        total = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(path) for f in fns
        )
        print(f"  {label:<32}: OK ({total/1e6:.0f} MB)")
    else:
        size = os.path.getsize(path)
        # eval_results.json must be more than ~50 bytes to count as
        # non-trivial; smaller means the eval cell didn't actually run.
        ok_marker = "OK"
        if path.endswith(".json") and size < 50:
            ok_marker = "EMPTY"
        print(f"  {label:<32}: {ok_marker} ({size/1e3:.1f} KB)")

print("\\n" + "=" * 70)
print(" Run complete.")
print("=" * 70)
'''


CELL_21_DASHBOARD = '''\
# ============================================================
# CELL 21 — POLL ADMIN DASHBOARD + SAFE SERVER SHUTDOWN
# Cell ordering bug fixed: server is killed at the *end* of this cell
# instead of in cell 18, so the dashboard always has a live server to
# poll. If the server was already stopped, falls back to the in-process
# metrics store.
# ============================================================

import json, urllib.request

print("Admin dashboard snapshot")
print("=" * 70)

snap = {}
if not server_up:
    print("Server is not running — pulling metrics directly from the in-process store.")
    try:
        from freshprice_env.monitoring import metrics
        snap = metrics.get_dashboard()
    except Exception as e:
        print(f"Could not load monitoring module: {e}")
else:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/admin/dashboard", timeout=5) as r:
            snap = json.loads(r.read())
    except Exception as e:
        print(f"GET /admin/dashboard failed: {e}; falling back to in-process store.")
        from freshprice_env.monitoring import metrics
        snap = metrics.get_dashboard()

if not snap:
    print("(no metrics recorded)")
else:
    s = snap.get("summary", {})
    print(f"  Episodes total          : {s.get('episodes_total', 0)}")
    print(f"  Steps total             : {s.get('steps_total', 0)}")
    print(f"  WRR mean / max          : {s.get('wrr_mean', 0):.4f} / {s.get('wrr_max', 0):.4f}")
    print(f"  Brief quality mean      : {s.get('quality_mean', 0):.4f}")
    print(f"  Anti-hack violations    : {s.get('violations_total', 0)}")
    pass_rate = s.get('constitutional_pass_rate', 1.0)
    print(f"  Constitutional pass rate: {pass_rate*100:.0f}%")

    by_sc = snap.get("by_scenario", {})
    if by_sc:
        print("\\n  Per scenario:")
        for name, b in by_sc.items():
            print(f"    {name:<15} n={b.get('n',0):<3} WRR={b.get('wrr_mean',0):.4f}")

    print(f"\\n  Recent episodes ({len(snap.get('recent_episodes', []))}):")
    for ep in snap.get("recent_episodes", [])[-5:]:
        const = "PASS" if ep.get("constitutional_passed", True) else "FAIL"
        print(f"    {ep.get('scenario','?'):<15} agent={ep.get('agent_type','?'):<18} "
              f"WRR={ep.get('wrr',0):.4f} viol={ep.get('anti_hack_violations',0)} "
              f"const={const}")

print(f"\\nFull JSON also at: " +
      (f"{SERVER_URL}/admin/dashboard" if server_up else "(in-process only)"))


# ---- Safe shutdown (deferred from cell 18) -----------------------------------
def _safe_kill_server():
    try:
        if "server_proc" in dir() and server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except Exception:
                server_proc.kill()
            print(f"\\nServer process {server_proc.pid} terminated.")
        else:
            print("\\nNo server process to terminate.")
    except NameError:
        print("\\nserver_proc not defined; nothing to terminate.")

_safe_kill_server()
'''


# --------------------------------------------------------------------------
# Patch logic
# --------------------------------------------------------------------------

# Map: header substring (matches the comment line near the top of each cell)
# → new source string
PATCHES = {
    "CELL 11 — GRPO EPISODE ROLLOUTS": CELL_11_GRPO,
    "CELL 13 — DETERMINISTIC EVALUATION": CELL_13_EVAL,
    "CELL 14 — ANTI-HACK ANALYSIS": CELL_14_ANTIHACK,
    "CELL 17 — TEST ALL BACKEND ENDPOINTS": CELL_17_API_TEST,
    "CELL 18 — STOP SERVER": CELL_18_SERVER_KILL,
    "CELL 19 — PUSH TRAINED MODEL TO HUGGING FACE HUB": CELL_19_HF_PUSH,
    "CELL 20 — FINAL SUMMARY": CELL_20_SUMMARY,
    "CELL 21 — POLL ADMIN DASHBOARD": CELL_21_DASHBOARD,
}


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]

    n_patched = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src_str = "".join(cell.get("source", []))
        for header, new_src in PATCHES.items():
            if header in src_str:
                cell["source"] = new_src.splitlines(keepends=True)
                # Wipe stale outputs so the notebook is clean for next run
                cell["outputs"] = []
                cell["execution_count"] = None
                n_patched += 1
                print(f"  patched: {header}")
                break

    # Cell 12 (DPO) gets a smaller surgical patch — replace just the
    # _decide_skip_reason() function so the rest of the cell logic stays
    # intact (it's complex, with TRL-specific imports).
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src_str = "".join(cell.get("source", []))
        if "CELL 12 — DPO FINE-TUNING" in src_str and "def _decide_skip_reason" in src_str:
            old_fn_start = src_str.find("def _decide_skip_reason")
            old_fn_end = src_str.find("\n\n", old_fn_start)
            # The function block ends at the first blank line
            patched = (
                src_str[:old_fn_start]
                + CELL_12_DPO_PATCH
                + src_str[old_fn_end:]
            )
            # Also mark `_DPO_RAN = True` after a successful train.
            # Insert near the end of the success path.
            if "_DPO_RAN" not in patched:
                # Add a sentinel set after DPO trainer.train() — search for
                # `trainer.train()` and inject below it.
                marker = "trainer.train()"
                if marker in patched:
                    patched = patched.replace(
                        marker,
                        marker + "\n        _DPO_RAN = True   # honest summary flag",
                        1,
                    )
            cell["source"] = patched.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
            n_patched += 1
            print("  patched: CELL 12 — DPO FINE-TUNING (skip-reason + _DPO_RAN flag)")
            break

    NOTEBOOK.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nPatched {n_patched} cells in {NOTEBOOK.name}")


if __name__ == "__main__":
    main()
