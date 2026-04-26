"""FastAPI server for the QStorePrice OpenEnv environment.

Exposes the canonical OpenEnv HTTP/WebSocket endpoints required by the
hackathon:

    GET  /health   - liveness probe
    POST /reset    - start a new episode
    POST /step     - submit an Operating Brief, advance the simulation
    GET  /state    - current FreshPriceState snapshot
    WS   /ws       - persistent session used by the async client
    GET  /docs     - OpenAPI documentation

Plus admin / dashboard endpoints (additive — do not affect the OpenEnv
contract):

    GET  /admin/dashboard            - live metrics snapshot (JSON)
    GET  /admin/metrics/scores       - flat list of episode records
    GET  /admin/metrics/reward-curve - flat list of step records
    GET  /admin/tasks                - curriculum scenario list
    POST /admin/metrics/reset        - clear in-memory metrics

Multi-agent commons endpoints (Theme #1):

    GET  /commons/bus                - bus messages from latest market commons run
    GET  /commons/audit              - latest oversight audit report
    GET  /commons/notebook           - hero AgentNotebook snapshot
    GET  /commons/scenario_composer  - hardness posteriors + last sample
    POST /commons/replay             - request a counterfactual fork
    WS   /commons/ws                 - live agent-message stream (multi-pane dashboard)

Static dashboards:

    GET  /                           - V2 multi-agent dashboard
    GET  /legacy                     - original polling dashboard (kept for
                                       backwards compatibility)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

try:
    from openenv.core.env_server import create_fastapi_app
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False
    create_fastapi_app = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _build_app():
    """Build the FastAPI app — falls back to a plain FastAPI() if openenv-core
    is missing so the dashboard still works in dev environments."""
    if _OPENENV_AVAILABLE:
        from server.environment import BriefAction, BriefObservation, FreshPriceOpenEnv
        return create_fastapi_app(
            env=FreshPriceOpenEnv,
            action_cls=BriefAction,
            observation_cls=BriefObservation,
        )

    from fastapi import FastAPI
    fallback = FastAPI(title="QStorePrice (fallback - openenv-core missing)")

    @fallback.get("/health")
    def _health():
        return {"status": "ok", "openenv_core": False}

    return fallback


app = _build_app()


# ---------------------------------------------------------------------------
# Imports for admin + commons endpoints
# ---------------------------------------------------------------------------

from freshprice_env.enums import CurriculumScenario  # noqa: E402
from freshprice_env.monitoring import metrics  # noqa: E402
from freshprice_env.protocol.market_bus import MarketBus  # noqa: E402

# A *server-scoped* MarketBus singleton that any in-process MarketCommonsEnv
# can be wired into. Sessions running on a separate process publish via this
# bus by sharing it explicitly. For the demo path (single FastAPI worker),
# this is enough; for multi-worker Prom-style deployments swap for Redis.
_server_bus = MarketBus()


def get_server_bus() -> MarketBus:
    """Return the process-wide bus singleton."""
    return _server_bus


# ---------------------------------------------------------------------------
# Admin / dashboard endpoints (additive — do not change OpenEnv contract)
# ---------------------------------------------------------------------------

@app.get("/admin/dashboard", tags=["Admin"])
def admin_dashboard():
    """Full metrics snapshot — feeds the live HTML dashboard."""
    return metrics.get_dashboard()


@app.get("/admin/metrics/scores", tags=["Admin"])
def admin_metrics_scores(scenario: str | None = None):
    return {"episodes": metrics.get_episode_scores(scenario=scenario)}


@app.get("/admin/metrics/reward-curve", tags=["Admin"])
def admin_metrics_reward_curve(scenario: str | None = None):
    return {"steps": metrics.get_reward_curve(scenario=scenario)}


@app.get("/admin/tasks", tags=["Admin"])
def admin_tasks():
    """Curriculum scenarios available to the environment."""
    return {
        "tasks": [
            {"level": s.value, "name": s.name}
            for s in CurriculumScenario
        ]
    }


@app.post("/admin/metrics/reset", tags=["Admin"])
def admin_metrics_reset():
    metrics.reset()
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Gym-compliant single-session endpoints
# ---------------------------------------------------------------------------
# The notebook that ran on Kaggle reported broken /reset (no `info`),
# stale /state (episode_id null after step), and a /step that returned
# `{"error": ...}`. The root cause was a contract mismatch with the
# specific openenv-core version installed.
#
# These /gym/* endpoints are an additive, openenv-core-independent
# fallback that ALWAYS conform to Gymnasium semantics:
#
#   POST /gym/reset {scenario, seed}
#       -> {observation, info}
#   POST /gym/step  {action}
#       -> {observation, reward, terminated, truncated, info}
#   GET  /gym/state
#       -> live snapshot of the in-process session

import threading  # noqa: E402

from freshprice_env.freshprice_env import FreshPriceEnv  # noqa: E402

_session_lock = threading.Lock()
_session_env: FreshPriceEnv | None = None
_session_meta: dict = {"episode_id": None, "step_count": 0, "scenario": None}


def _ensure_session(
    scenario: CurriculumScenario | None = None,
    seed: int = 42,
) -> FreshPriceEnv:
    """Return the process-scoped FreshPriceEnv session, building if needed."""
    global _session_env
    with _session_lock:
        if _session_env is None:
            _session_env = FreshPriceEnv(
                scenario=scenario or CurriculumScenario.STABLE_WEEK,
                seed=seed,
            )
        return _session_env


@app.post("/gym/reset", tags=["Gym"])
def gym_reset(payload: dict | None = None):
    """Gymnasium-style reset: returns ``{observation, info}``."""
    global _session_env, _session_meta
    payload = payload or {}
    scenario_name = payload.get("scenario", "STABLE_WEEK")
    seed = int(payload.get("seed", 42))
    try:
        scenario = CurriculumScenario[scenario_name]
    except KeyError:
        return {"error": f"unknown scenario {scenario_name}"}
    with _session_lock:
        _session_env = FreshPriceEnv(scenario=scenario, seed=seed)
        obs, info = _session_env.reset()
        _session_meta = {
            "episode_id": info.get("scenario", scenario_name) + f"_{seed}",
            "step_count": 0,
            "scenario": scenario_name,
            "seed": seed,
        }
    return {"observation": obs, "info": info}


@app.post("/gym/step", tags=["Gym"])
def gym_step(payload: dict):
    """Gymnasium-style step: returns
    ``{observation, reward, terminated, truncated, info}``."""
    global _session_meta
    if _session_env is None:
        return {"error": "call /gym/reset first"}
    action = (payload or {}).get("action")
    if not isinstance(action, str):
        return {"error": "payload must include `action` (string)"}
    with _session_lock:
        obs, reward, terminated, truncated, info = _session_env.step(action)
        _session_meta["step_count"] += 1
    return {
        "observation": obs,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info,
    }


@app.get("/gym/state", tags=["Gym"])
def gym_state():
    """Live state snapshot for the in-process session.

    Unlike the openenv-core /state which lost episode_id between
    requests, this reads directly from the process-scoped FreshPriceEnv.
    """
    if _session_env is None:
        return {
            "status": "not_started",
            "episode_id": None,
            "step_count": 0,
            "hint": "call /gym/reset first",
        }
    snapshot = _session_env.state()
    return {
        **_session_meta,
        **snapshot,
    }


# ---------------------------------------------------------------------------
# Multi-agent commons endpoints
# ---------------------------------------------------------------------------

# Latest in-process commons artifacts (set by the demo / training scripts).
_commons_state: dict = {
    "bus_messages": [],
    "audit_report": None,
    "notebook_snapshot": None,
    "scenario_composer": None,
    "replay_forks": [],
    "active_schemas": None,
    "regulatory_events": [],
    "tom_report": None,
    "competitor_actions": [],
    "farmer_pool_summary": None,
    "current_tick": 0,
    "current_wrr": 0.0,
    "current_cooperation_index": 0.0,
    "episode_id": None,
    "scenario_name": None,
}


def update_commons_state(**fields) -> None:
    """Merge fields into the in-process commons snapshot.

    Env / training scripts call this after each step so the dashboard
    has fresh data without round-tripping through metrics.
    """
    _commons_state.update(fields)


@app.get("/commons/bus", tags=["Commons"])
def commons_bus(limit: int = 200):
    """Return the latest `limit` bus messages (in-process bus)."""
    msgs = [m.to_dict() for m in _server_bus.all_messages()[-limit:]]
    return {"messages": msgs, "total": len(_server_bus.all_messages())}


@app.get("/commons/audit", tags=["Commons"])
def commons_audit():
    return {"audit_report": _commons_state.get("audit_report")}


@app.get("/commons/notebook", tags=["Commons"])
def commons_notebook():
    return {"notebook": _commons_state.get("notebook_snapshot")}


@app.get("/commons/scenario_composer", tags=["Commons"])
def commons_scenario_composer():
    return {"composer": _commons_state.get("scenario_composer")}


@app.get("/commons/snapshot", tags=["Commons"])
def commons_snapshot():
    """Aggregate snapshot for the v2 dashboard."""
    return _commons_state | {
        "bus_total": len(_server_bus.all_messages()),
    }


@app.post("/commons/replay", tags=["Commons"])
def commons_request_replay(payload: dict):
    """Run a counterfactual fork synchronously.

    Body:
        {
          "scenario": "CRISIS_WEEK",
          "seed": 42,
          "brief_log": [{"tick": 0, "engine_type": "PRICING", "brief_text": "..."}, ...],
          "swap_brief_index": 14,
          "new_brief": "...",
          "max_briefs": 50
        }
    """
    from eval.counterfactual_replay import (
        BriefLogEntry, CounterfactualReplay, fork_to_dict,
    )
    try:
        scenario = CurriculumScenario[payload["scenario"]]
    except KeyError:
        return {"error": "scenario must be one of: "
                + ", ".join(s.name for s in CurriculumScenario)}
    seed = int(payload.get("seed", 42))
    brief_log = [
        BriefLogEntry(
            tick=int(e["tick"]),
            engine_type=str(e.get("engine_type", "PRICING")),
            brief_text=str(e.get("brief_text", "")),
        )
        for e in payload.get("brief_log", [])
    ]
    if not brief_log:
        return {"error": "brief_log empty"}
    replay = CounterfactualReplay(scenario, seed=seed, brief_log=brief_log)
    baseline = replay.run_baseline(
        max_briefs=int(payload.get("max_briefs", len(brief_log))),
    )
    fork = replay.swap_decision_at_brief(
        brief_index=int(payload["swap_brief_index"]),
        new_brief=str(payload["new_brief"]),
        max_briefs=int(payload.get("max_briefs", len(brief_log))),
    )
    summary = fork.divergence_summary(baseline)
    out = {
        "summary": summary,
        "baseline": fork_to_dict(baseline),
        "fork": fork_to_dict(fork),
    }
    _commons_state.setdefault("replay_forks", []).append(summary)
    return out


# ---------------------------------------------------------------------------
# WebSocket: live agent-message stream
# ---------------------------------------------------------------------------

try:
    from fastapi import WebSocket, WebSocketDisconnect

    _ws_clients: list[WebSocket] = []
    _ws_lock = asyncio.Lock()

    @app.websocket("/commons/ws")
    async def commons_ws(websocket: WebSocket):
        """Pushes every new MarketBus message (and snapshot updates) live."""
        await websocket.accept()
        async with _ws_lock:
            _ws_clients.append(websocket)
        # Send the current snapshot right away so the client can render
        await websocket.send_json({
            "kind": "snapshot",
            "data": _commons_state | {
                "bus_total": len(_server_bus.all_messages()),
            },
        })
        # Send the last 50 bus messages as backfill
        for m in _server_bus.all_messages()[-50:]:
            await websocket.send_json({"kind": "message", "data": m.to_dict()})
        try:
            # We don't expect inbound traffic; keep the connection open
            while True:
                msg = await websocket.receive_text()
                # echo for debugging — clients can ignore
                await websocket.send_json({"kind": "echo", "data": msg})
        except WebSocketDisconnect:
            pass
        finally:
            async with _ws_lock:
                if websocket in _ws_clients:
                    _ws_clients.remove(websocket)

    def _bus_subscriber(msg) -> None:
        """Sync subscriber that schedules a coroutine to broadcast."""
        if not _ws_clients:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                return
            for client in list(_ws_clients):
                asyncio.run_coroutine_threadsafe(
                    _safe_send(client, {"kind": "message", "data": msg.to_dict()}),
                    loop,
                )
        except RuntimeError:
            # No running loop — log only at debug
            pass

    async def _safe_send(client, payload: dict) -> None:
        try:
            await client.send_json(payload)
        except Exception:  # noqa: BLE001 — client may have disconnected
            pass

    _server_bus.subscribe(_bus_subscriber)

except ImportError:  # pragma: no cover — FastAPI ws optional in older versions
    logger.warning("FastAPI WebSocket support not available; /commons/ws disabled")


# ---------------------------------------------------------------------------
# Blinkit/Zepto-style mechanics endpoints (rider pool, cohorts, liquidation)
#
# These are read-only views over an in-process snapshot that the env (or
# the simulation-theater feeder below) updates as ticks happen. Keep the
# state ephemeral; the source of truth for training is still the env.
# ---------------------------------------------------------------------------

_blinkit_state: dict = {
    "rider_pool": None,        # last RiderPoolEngine.snapshot()
    "cohorts": None,           # last ConsumerCohortAgent.observe()
    "liquidation": None,       # last LiquidationEngine.snapshot()
    "tick_frames": [],         # ring buffer of recent simulation frames
    "max_tick_frames": 240,    # ~1 hour at 15-min ticks; trimmed below
}


def update_blinkit_state(**fields) -> None:
    """Merge fields into the in-process Blinkit snapshot.

    Called from MarketCommonsEnv after each tick so the dashboard /
    simulation theater has fresh data.
    """
    if "tick_frame" in fields:
        frame = fields.pop("tick_frame")
        _blinkit_state["tick_frames"].append(frame)
        if len(_blinkit_state["tick_frames"]) > _blinkit_state["max_tick_frames"]:
            del _blinkit_state["tick_frames"][:-_blinkit_state["max_tick_frames"]]
    _blinkit_state.update(fields)


@app.get("/commons/rider_pool", tags=["Commons"])
def commons_rider_pool():
    """Latest RiderPoolEngine snapshot (queue depth, on-time %, transit spoilage)."""
    snap = _blinkit_state.get("rider_pool")
    if not snap:
        return {"status": "idle",
                "hint": "MarketCommonsEnv has not ticked the rider pool yet"}
    return {"status": "live", "snapshot": snap}


@app.get("/commons/cohorts", tags=["Commons"])
def commons_cohorts():
    """Per-cohort retention + walk-away percentages from the latest tick."""
    snap = _blinkit_state.get("cohorts")
    if not snap:
        return {"status": "idle",
                "hint": "MarketCommonsEnv has not used ConsumerCohortAgent yet"}
    return {"status": "live", "snapshot": snap}


@app.get("/commons/liquidation", tags=["Commons"])
def commons_liquidation():
    """Dead-stock liquidation history (B2B firesale channel)."""
    snap = _blinkit_state.get("liquidation")
    if not snap:
        return {"status": "idle", "results_this_brief": []}
    return {"status": "live", "snapshot": snap}


@app.get("/commons/sim_frames", tags=["Commons"])
def commons_sim_frames(limit: int = 240):
    """Ring buffer of recent simulation frames for the Theater player.

    Each frame is a self-contained tick snapshot:
        {tick, batches, cohorts, rider_pool, latest_brief, reasoning}
    """
    frames = _blinkit_state.get("tick_frames", [])
    return {"frames": frames[-limit:], "total": len(frames)}


# ---------------------------------------------------------------------------
# Agent runtime endpoints — drive the Simulation Theater with a trained model
#
# These endpoints turn the static-state-only `/commons/*` endpoints into a
# live demo: POST /agent/run_episode resets MarketCommonsEnv, asks the agent
# runtime for one brief per step, ticks the env, and pushes a tick_frame
# into the Blinkit ring buffer the dashboard polls. After training on Kaggle,
# point the server at your weights via env vars (see README) and hit the
# "Run live demo" button on the dashboard — that is the bridge from
# trained-model artifact -> visible inference.
# ---------------------------------------------------------------------------

from server.agent_runtime import (  # noqa: E402
    AgentRuntime, get_agent_runtime, reset_agent_runtime,
    MultiAgentRuntime, get_comparison_runtime, reset_comparison_runtime,
)


@app.get("/agent/info", tags=["Agent"])
def agent_info():
    """Diagnostic — which backend is currently loaded and how it was wired."""
    try:
        rt = get_agent_runtime()
        return {"status": "ready", **rt.info()}
    except Exception as e:  # noqa: BLE001 — surface the misconfiguration
        return {"status": "error", "error": str(e),
                "hint": "Set AGENT_BACKEND=local|hf_inference|scripted "
                        "+ MODEL_PATH or HF_REPO_ID/HF_TOKEN. See README."}


@app.post("/agent/reload", tags=["Agent"])
def agent_reload():
    """Drop the cached runtime so the next call picks up new env vars."""
    reset_agent_runtime()
    return {"status": "ok"}


@app.post("/agent/brief", tags=["Agent"])
def agent_brief(payload: dict):
    """Generate one Operating Brief from a prompt string.

    Body: ``{"prompt": "...", "max_new_tokens": 600, "temperature": 0.7}``
    """
    prompt = (payload or {}).get("prompt", "")
    if not isinstance(prompt, str) or not prompt:
        return {"error": "payload must include `prompt` (non-empty string)"}
    rt = get_agent_runtime()
    brief = rt.generate(
        prompt,
        max_new_tokens=int((payload or {}).get("max_new_tokens", 600)),
        temperature=float((payload or {}).get("temperature", 0.7)),
    )
    return {"brief": brief, "agent": rt.info()}


def _build_tick_frame(env_state: dict, brief_text: str, info_dict: dict) -> dict:
    """Project an env state snapshot into the lightweight frame the
    Simulation Theater consumes."""
    batches_raw = env_state.get("batches") or []
    batches = []
    for b in batches_raw[:64]:    # cap to keep frames small
        if not isinstance(b, dict):
            continue
        batches.append({
            "batch_id": b.get("batch_id"),
            "category": b.get("category"),
            "urgency": b.get("urgency"),
            "status": b.get("status", "ACTIVE"),
            "hours_to_expiry": b.get("hours_to_expiry"),
            "quantity_remaining": b.get("quantity_remaining"),
        })
    return {
        "tick": env_state.get("tick", 0),
        "scenario": env_state.get("scenario"),
        "engine_type": info_dict.get("engine_type"),
        "batches": batches,
        "rider_pool": _blinkit_state.get("rider_pool"),
        "cohorts": _blinkit_state.get("cohorts"),
        "liquidation": _blinkit_state.get("liquidation"),
        "latest_brief": brief_text,
        "reasoning": brief_text,
        "wrr_so_far": env_state.get("wrr_so_far", 0.0),
        "reward_this_step": info_dict.get("reward"),
    }


# ---------------------------------------------------------------------------
# /agent/compare — before vs. after RL (judging criterion #3, 20% of score)
# ---------------------------------------------------------------------------

@app.get("/agent/compare/info", tags=["Agent Comparison"])
def agent_compare_info():
    """List the runtimes loaded for comparison (baseline / sft / rl).

    Returns 503 if no runtime could be loaded (env vars missing).
    """
    try:
        rt = get_comparison_runtime()
        return {"status": "ready", "runtimes": rt.info(),
                "names": rt.names}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


@app.post("/agent/compare/reload", tags=["Agent Comparison"])
def agent_compare_reload():
    """Drop the cached comparison runtime so the next call picks up new env vars."""
    reset_comparison_runtime()
    return {"status": "ok"}


@app.post("/agent/compare/brief", tags=["Agent Comparison"])
def agent_compare_brief(payload: dict):
    """Generate the same brief from every loaded runtime.

    Body: ``{"prompt": "...", "max_new_tokens": 600, "temperature": 0.7}``
    Returns: ``{"baseline": "...", "sft": "...", "rl": "..."}``
    """
    prompt = (payload or {}).get("prompt", "")
    if not isinstance(prompt, str) or not prompt:
        return {"error": "payload must include `prompt` (non-empty string)"}
    rt = get_comparison_runtime()
    briefs = rt.generate_all(
        prompt,
        max_new_tokens=int((payload or {}).get("max_new_tokens", 600)),
        temperature=float((payload or {}).get("temperature", 0.7)),
    )
    return {"runtimes": rt.info(), "briefs": briefs}


@app.post("/agent/compare/episode", tags=["Agent Comparison"])
def agent_compare_episode(payload: dict | None = None):
    """Run the same episode with each loaded runtime; report per-runtime SES.

    Body: ``{"scenario": "STABLE_WEEK", "seed": 42, "max_briefs": 8}``
    Returns: ``{
        "scenario": ..., "seed": ...,
        "results": {
            "baseline": {steps_completed, total_reward, final_wrr, mean_ses, briefs[]},
            "sft":      {...},
            "rl":       {...}
        },
        "improvement": {
            "sft_over_baseline_ses_delta": +0.15,
            "rl_over_sft_ses_delta":       +0.08,
            ...
        }
    }``
    """
    payload = payload or {}
    scenario_name = payload.get("scenario", "STABLE_WEEK")
    try:
        scenario = CurriculumScenario[scenario_name]
    except KeyError:
        return {"error": f"unknown scenario {scenario_name}"}
    seed = int(payload.get("seed", 42))
    max_briefs = int(payload.get("max_briefs", 8))

    from freshprice_env.freshprice_env import FreshPriceEnv  # noqa: E402

    rt = get_comparison_runtime()
    results: dict[str, dict] = {}

    for name in rt.names:
        runtime = rt.get(name)
        if runtime is None:
            continue
        env = FreshPriceEnv(scenario=scenario, seed=seed)
        obs, info = env.reset()
        ses_acc, total_reward = 0.0, 0.0
        n_briefs, anti_hack_violations = 0, 0
        sample_briefs = []
        for step in range(max_briefs):
            try:
                brief = runtime.generate(obs)
            except Exception as e:  # noqa: BLE001
                brief = f"[runtime '{name}' failed: {e}]"
            obs, reward, done, truncated, info = env.step(brief)
            total_reward += float(reward)
            ses_acc += float(info.get("store_efficiency_score", 0.0))
            n_briefs += 1
            if step < 2:    # keep only the first 2 briefs as samples
                sample_briefs.append(brief)
            if not info.get("parse_success", True):
                anti_hack_violations += 1
            if done:
                break
        final_reward = info.get("final_reward", {}) or {}
        results[name] = {
            "steps_completed":  n_briefs,
            "total_reward":     round(total_reward, 4),
            "final_wrr":        round(final_reward.get("wrr",
                                       env.state().get("wrr_so_far", 0.0)), 4),
            "mean_ses":         round(ses_acc / max(1, n_briefs), 4),
            "anti_hack_violations": anti_hack_violations,
            "sample_briefs":    sample_briefs,
        }

    improvement: dict[str, float] = {}
    if "baseline" in results and "sft" in results:
        improvement["sft_over_baseline_ses_delta"] = round(
            results["sft"]["mean_ses"] - results["baseline"]["mean_ses"], 4,
        )
    if "sft" in results and "rl" in results:
        improvement["rl_over_sft_ses_delta"] = round(
            results["rl"]["mean_ses"] - results["sft"]["mean_ses"], 4,
        )
    if "baseline" in results and "rl" in results:
        improvement["rl_over_baseline_ses_delta"] = round(
            results["rl"]["mean_ses"] - results["baseline"]["mean_ses"], 4,
        )

    return {
        "scenario": scenario_name,
        "seed":     seed,
        "results":  results,
        "improvement": improvement,
    }


@app.get("/agent/compare/snapshot", tags=["Agent Comparison"])
def agent_compare_snapshot():
    """Return the cached comparison_results.json if the inference comparison
    script has been run. The dashboard uses this for the "Before vs After
    RL" panel without needing the model checkpoints loaded server-side."""
    snap_path = Path(__file__).resolve().parent.parent / "data" / "comparison_results.json"
    if not snap_path.is_file():
        return {"status": "no_snapshot",
                "hint": "run `python inference_comparison.py` first"}
    try:
        return {"status": "ok",
                "results": json.loads(snap_path.read_text(encoding="utf-8"))}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


@app.post("/agent/run_episode", tags=["Agent"])
def agent_run_episode(payload: dict | None = None):
    """Run one episode end-to-end with the loaded agent.

    Body (all optional):
        {
            "scenario": "STABLE_WEEK",   # any CurriculumScenario name
            "seed": 42,
            "max_briefs": 12,            # safety cap; default 12 (1 sim day)
            "store_id": "store_001"
        }

    For each brief:
      - asks the agent runtime for a brief from the env's prompt
      - steps the env
      - records a tick_frame the Simulation Theater polls
    Returns the final WRR and per-step summaries. The dashboard will
    auto-pick up the frames within a few seconds.
    """
    payload = payload or {}
    scenario_name = payload.get("scenario", "STABLE_WEEK")
    try:
        scenario = CurriculumScenario[scenario_name]
    except KeyError:
        return {"error": f"unknown scenario {scenario_name}"}
    seed = int(payload.get("seed", 42))
    max_briefs = int(payload.get("max_briefs", 12))
    store_id = str(payload.get("store_id", "store_001"))

    # Use the same FreshPriceEnv class the gym/* endpoints use; wiring the
    # full MarketCommonsEnv requires bus initialization beyond this scope.
    from freshprice_env.freshprice_env import FreshPriceEnv  # noqa: E402

    rt = get_agent_runtime()
    env = FreshPriceEnv(scenario=scenario, seed=seed)
    obs, info = env.reset()

    update_blinkit_state(tick_frame=_build_tick_frame(env.state(), "", info))

    step_summaries = []
    total_reward = 0.0
    last_brief = ""
    for step in range(max_briefs):
        last_brief = rt.generate(obs)
        obs, reward, done, truncated, info = env.step(last_brief)
        total_reward += float(reward)

        # Publish on the bus so the agent panes update.
        try:
            _server_bus.publish(
                store_id, "BRIEF",
                {"text": last_brief[:600],
                 "engine_type": info.get("engine_type"),
                 "step": step, "reward": float(reward)},
            )
        except Exception:  # noqa: BLE001
            pass

        update_blinkit_state(
            tick_frame=_build_tick_frame(env.state(), last_brief, info),
        )

        step_summaries.append({
            "step": step,
            "engine_type": info.get("engine_type"),
            "reward": round(float(reward), 4),
            "parse_success": info.get("parse_success"),
            "quality_score": info.get("quality_score"),
            "next_engine": info.get("next_engine_type"),
        })

        if done:
            break

    final = info.get("final_reward", {}) or {}
    return {
        "scenario": scenario_name,
        "seed": seed,
        "agent": rt.info(),
        "steps_completed": len(step_summaries),
        "total_reward": round(total_reward, 4),
        "final_wrr": round(final.get("wrr", env.state().get("wrr_so_far", 0.0)), 4),
        "steps": step_summaries,
        "last_brief": last_brief,
    }


# ---------------------------------------------------------------------------
# Static dashboard mounts
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles  # noqa: E402
    from fastapi.responses import FileResponse  # noqa: E402

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    _V2_INDEX = _STATIC_DIR / "v2" / "index.html"
    _V1_INDEX = _STATIC_DIR / "index.html"

    @app.get("/", include_in_schema=False)
    def dashboard_index():
        # Prefer v2 if present
        if _V2_INDEX.is_file():
            return FileResponse(str(_V2_INDEX))
        return FileResponse(str(_V1_INDEX))

    @app.get("/legacy", include_in_schema=False)
    def dashboard_legacy():
        return FileResponse(str(_V1_INDEX))


def main() -> None:
    """CLI entry point used by `python -m server.app` and the Dockerfile."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
