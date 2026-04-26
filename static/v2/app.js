// QStorePrice Commons V2 — live multi-agent dashboard
// Connects to /commons/ws for live bus messages, polls /commons/snapshot
// for slow-moving aggregates (notebook, audit, composer), and exposes
// a counterfactual replay slider that POSTs to /commons/replay.

const SNAPSHOT_POLL_MS = 2500;
const MAX_PANE_MESSAGES = 80;

const $ = (id) => document.getElementById(id);
const els = {
    wsDot: $("ws-dot"),
    wsText: $("ws-text"),
    episodeId: $("episode-id"),
    scenarioName: $("scenario-name"),
    tickCounter: $("tick-counter"),
    kpiWrr: $("kpi-wrr"),
    kpiCoop: $("kpi-coop"),
    kpiAdherence: $("kpi-adherence"),
    kpiTrust: $("kpi-trust"),
    kpiTrustRec: $("kpi-trust-rec"),
    kpiPoolTrust: $("kpi-pool-trust"),
    kpiPoolMeta: $("kpi-pool-meta"),
    kpiSchemas: $("kpi-schemas"),
    audit: $("audit-content"),
    notebook: $("notebook-content"),
    composer: $("composer-content"),
    busLog: $("bus-log-table"),
    replaySlider: $("replay-slider"),
    replayLabel: $("replay-tick-label"),
    replayBtn: $("replay-run"),
    replayResult: $("replay-result"),
};

let cachedBriefLog = null;
let busMessages = [];        // chronological cache for the full bus log
let lastSnapshot = null;

// ───────── WebSocket ─────────
function connectWS() {
    const wsUrl = (location.protocol === "https:" ? "wss://" : "ws://")
        + location.host + "/commons/ws";
    let ws;
    try { ws = new WebSocket(wsUrl); }
    catch (e) { setWS(false, "WS init failed"); scheduleReconnect(); return; }

    ws.onopen = () => setWS(true, "Live");
    ws.onclose = () => { setWS(false, "Disconnected"); scheduleReconnect(); };
    ws.onerror = () => { setWS(false, "WS error"); };
    ws.onmessage = (ev) => {
        try {
            const obj = JSON.parse(ev.data);
            if (obj.kind === "snapshot") onSnapshot(obj.data);
            else if (obj.kind === "message") onBusMessage(obj.data);
        } catch (e) { /* ignore non-JSON */ }
    };
}
function scheduleReconnect() { setTimeout(connectWS, 2500); }

function setWS(ok, label) {
    els.wsDot.classList.toggle("bad", !ok);
    els.wsText.textContent = label;
}

// ───────── Snapshot polling fallback ─────────
async function pollSnapshot() {
    try {
        const r = await fetch("/commons/snapshot");
        if (!r.ok) throw new Error(r.status);
        const data = await r.json();
        onSnapshot(data);
    } catch (e) { /* surface as offline indicator only */ }
}
setInterval(pollSnapshot, SNAPSHOT_POLL_MS);
pollSnapshot();

// ───────── Snapshot rendering ─────────
function onSnapshot(snap) {
    if (!snap) return;
    lastSnapshot = snap;

    if (snap.episode_id) els.episodeId.textContent = snap.episode_id;
    if (snap.scenario_name) els.scenarioName.textContent = snap.scenario_name;
    if (snap.current_tick !== undefined) els.tickCounter.textContent = `tick ${snap.current_tick}`;

    if (snap.current_wrr !== undefined) els.kpiWrr.textContent = fmt(snap.current_wrr, 3);
    if (snap.current_cooperation_index !== undefined)
        els.kpiCoop.textContent = fmt(snap.current_cooperation_index, 3);

    if (snap.notebook_snapshot) renderNotebook(snap.notebook_snapshot);
    if (snap.audit_report) renderAudit(snap.audit_report);
    if (snap.scenario_composer) renderComposer(snap.scenario_composer);
    if (snap.active_schemas) {
        const a = snap.active_schemas;
        els.kpiSchemas.textContent = `${a.PRICING || "-"} · ${a.FARMER || "-"} · ${a.TREND || "-"}`;
    }
    if (snap.farmer_pool_summary) {
        const p = snap.farmer_pool_summary;
        els.kpiPoolTrust.textContent = fmt(p.mean_trust, 2);
        els.kpiPoolMeta.textContent = `${p.n_farmers} farmers · min ${fmt(p.min_trust,2)} · max ${fmt(p.max_trust,2)}`;
    }
    if (snap.notebook_snapshot && snap.notebook_snapshot.adherence_score !== undefined) {
        els.kpiAdherence.textContent = fmt(snap.notebook_snapshot.adherence_score, 2);
    }
}

function fmt(v, d = 2) {
    if (v === null || v === undefined || Number.isNaN(v)) return "--";
    return Number(v).toFixed(d);
}

// ───────── Bus message → routed pane ─────────
function onBusMessage(m) {
    busMessages.push(m);
    if (busMessages.length > 800) busMessages.shift();
    renderBusRow(m);
    routeToPane(m);
}

function paneIdForSender(sender) {
    if (!sender) return null;
    if (sender === "store_001") return "store_001";
    if (sender.startsWith("store_") && sender !== "store_001") return "store_002";
    if (sender.startsWith("farmer")) return "farmer";
    if (sender === "consumer" || sender.startsWith("cohort")) return "consumer";
    if (sender === "influencer") return "influencer";
    if (sender === "regulator") return "regulator";
    if (sender === "oversight" || sender === "auditor") return "oversight";
    return null;
}

function routeToPane(msg) {
    const paneId = paneIdForSender(msg.sender);
    if (!paneId) return;
    const stream = document.querySelector(`[data-actor-stream="${paneId}"]`);
    if (!stream) return;
    const div = document.createElement("div");
    div.className = "msg " + (msg.verb || "").toLowerCase();
    div.innerHTML = `
        <div class="meta">[t${msg.tick} seq${msg.seq}] → ${escapeHTML(msg.receiver || "*all*")}</div>
        <div><span class="verb">${escapeHTML(msg.verb)}</span> ${escapeHTML(msg.body || "")}</div>
    `;
    stream.appendChild(div);
    while (stream.children.length > MAX_PANE_MESSAGES) stream.removeChild(stream.firstChild);
    stream.scrollTop = stream.scrollHeight;
}

function renderBusRow(m) {
    const row = document.createElement("div");
    row.className = "bus-row";
    row.innerHTML = `
        <span class="seq">#${m.seq}</span>
        <span class="from">${escapeHTML(m.sender)}</span>
        <span class="to">${escapeHTML(m.receiver || "*all*")}</span>
        <span class="verb">${escapeHTML(m.verb)}</span>
        <span class="body">${escapeHTML(m.body || "")}</span>
    `;
    els.busLog.prepend(row);
    while (els.busLog.children.length > 200) els.busLog.removeChild(els.busLog.lastChild);
}

// ───────── Audit ─────────
function renderAudit(report) {
    const trust = report.trust_score;
    if (trust !== undefined) {
        els.kpiTrust.textContent = fmt(trust, 2);
        els.kpiTrustRec.textContent = report.recommendation || "--";
    }
    const patterns = (report.suspicious_patterns || []).map(p => {
        const tick = p.tick !== undefined && p.tick !== null
            ? ` @ t${p.tick}`
            : (p.ticks && p.ticks.length ? ` @ t${p.ticks[0]}` : "");
        return `<li><span class="pid">${escapeHTML(p.id || "")}</span>${tick} — ${escapeHTML(p.description || "")}</li>`;
    }).join("");

    els.audit.innerHTML = `
        <div class="audit-block">
            <div class="label">Trust score</div><div class="value">${fmt(trust, 2)}</div>
            <div class="label">Recommendation</div><div class="value">
                <span class="audit-rec ${report.recommendation || ""}">${escapeHTML(report.recommendation || "--")}</span>
            </div>
            <div class="label">Patterns</div>
            <div class="value">
                <ul class="audit-pattern-list">${patterns || '<li class="empty">none detected</li>'}</ul>
            </div>
            <div class="narrative">${escapeHTML(report.narrative || "")}</div>
        </div>
    `;
}

// ───────── Notebook ─────────
function renderNotebook(nb) {
    const pinned = (nb.notes || []).filter(n => n.pinned);
    const recent = (nb.notes || []).filter(n => !n.pinned)
        .sort((a, b) => b.tick - a.tick).slice(0, 8);
    const commitments = nb.commitments || [];
    const open = commitments.filter(c => !c.resolved);
    const honored = commitments.filter(c => c.resolved && c.honored);
    const broken = commitments.filter(c => c.resolved && !c.honored);

    els.notebook.innerHTML = `
        <div class="nb-grid">
            <div class="nb-block">
                <h3>Plan</h3>
                <div class="nb-note">${escapeHTML(nb.plan || "(no plan written yet)")}</div>
            </div>
            <div class="nb-block">
                <h3>Pinned notes</h3>
                ${pinned.length ? pinned.map(n =>
                    `<div class="nb-note nb-pinned">[t${n.tick}] ${escapeHTML(n.key)}: ${escapeHTML(n.value)}</div>`
                ).join("") : '<div class="nb-note">(none)</div>'}
            </div>
            <div class="nb-block">
                <h3>Recent notes</h3>
                ${recent.length ? recent.map(n =>
                    `<div class="nb-note">[t${n.tick}] ${escapeHTML(n.key)}: ${escapeHTML(n.value)}</div>`
                ).join("") : '<div class="nb-note">(none)</div>'}
            </div>
            <div class="nb-block">
                <h3>Commitments — adherence ${fmt(nb.adherence_score, 2)}</h3>
                ${commitments.length === 0 ? '<div class="nb-note">(none)</div>' : ""}
                ${honored.map(c =>
                    `<div class="nb-commitment honored">${c.id} ✓ ${escapeHTML(c.kind)}:${escapeHTML(c.target)}</div>`
                ).join("")}
                ${broken.map(c =>
                    `<div class="nb-commitment broken">${c.id} ✗ ${escapeHTML(c.kind)}:${escapeHTML(c.target)}</div>`
                ).join("")}
                ${open.map(c =>
                    `<div class="nb-commitment open">${c.id} ⏳ ${escapeHTML(c.kind)}:${escapeHTML(c.target)} due t${c.due_tick}</div>`
                ).join("")}
            </div>
        </div>
    `;
}

// ───────── Composer ─────────
function renderComposer(c) {
    const blocks = [];
    for (const [axis, rows] of Object.entries(c.hardest_cells || {})) {
        const inner = (rows || []).map(r =>
            `<div class="composer-bar"><span class="name">${escapeHTML(String(r.value))}</span><span class="rate">${fmt(r.failure_rate, 2)}</span></div>`
        ).join("");
        blocks.push(`<div class="composer-block"><h3>${escapeHTML(axis)}</h3>${inner}</div>`);
    }
    if (!blocks.length) {
        els.composer.innerHTML = '<p class="empty">No samples yet.</p>';
    } else {
        els.composer.innerHTML = `<div class="composer-grid">${blocks.join("")}</div>`;
    }
}

// ───────── Counterfactual replay slider ─────────
els.replaySlider.addEventListener("input", () => {
    els.replayLabel.textContent = `brief ${els.replaySlider.value}`;
});
els.replayBtn.addEventListener("click", async () => {
    if (!cachedBriefLog) {
        els.replayResult.textContent = "Load a brief log first (POST it to /commons/replay).";
        return;
    }
    els.replayBtn.disabled = true;
    els.replayResult.textContent = "Running fork...";
    try {
        const swap_idx = parseInt(els.replaySlider.value, 10);
        const replacement = prompt(
            "Replacement brief text for index " + swap_idx + ":",
            cachedBriefLog[swap_idx]?.brief_text?.slice(0, 200) || ""
        );
        if (replacement === null) { els.replayBtn.disabled = false; return; }
        const r = await fetch("/commons/replay", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                scenario: "CRISIS_WEEK",
                seed: 42,
                brief_log: cachedBriefLog,
                swap_brief_index: swap_idx,
                new_brief: replacement,
                max_briefs: cachedBriefLog.length,
            }),
        });
        const data = await r.json();
        if (data.error) {
            els.replayResult.textContent = "Error: " + data.error;
        } else {
            const s = data.summary || {};
            els.replayResult.innerHTML =
                `<strong>Fork ${s.fork_id}:</strong> baseline WRR ${fmt(s.baseline_final_wrr,3)} → fork WRR ${fmt(s.fork_final_wrr,3)} ` +
                `(<span style="color:${s.wrr_delta >= 0 ? 'var(--accent)' : 'var(--danger)'}">${s.wrr_delta >= 0 ? '+' : ''}${fmt(s.wrr_delta,3)}</span>) ` +
                `over ${s.n_points} briefs.`;
        }
    } catch (e) {
        els.replayResult.textContent = "Error: " + e.message;
    } finally {
        els.replayBtn.disabled = false;
    }
});

// ───────── Helpers ─────────
function escapeHTML(s) {
    if (s === null || s === undefined) return "";
    return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

// Allow other scripts to push a brief log into the replay tool
window.QStorePriceCommons = {
    setBriefLog(log) {
        cachedBriefLog = log;
        els.replaySlider.max = String(Math.max(0, log.length - 1));
        els.replaySlider.value = String(Math.min(14, log.length - 1));
        els.replayLabel.textContent = `brief ${els.replaySlider.value}`;
        els.replayBtn.disabled = false;
        els.replayBtn.textContent = "Run fork";
    },
};

// ───────── Simulation Theater (Blinkit/Zepto layer) ─────────
//
// Polls /commons/sim_frames for the recent ring buffer of tick frames and
// plays them back at user-selected speed. Each frame carries the minimum
// state needed to render one tick: batches with urgency, cohort retention,
// rider pool snapshot, latest brief reasoning, optional liquidation events.
// If no frames have been recorded yet (fresh server, no env runs), the
// theater shows a friendly idle hint instead of fabricating data.

const theater = {
    frames: [],
    idx: 0,
    timerId: null,
    speed: 4,
    els: {
        play:        $("theater-play"),
        pause:       $("theater-pause"),
        speed:       $("theater-speed"),
        slider:      $("theater-frame-slider"),
        label:       $("theater-frame-label"),
        tick:        $("theater-tick-label"),
        batches:     $("theater-batches"),
        riderFill:   $("rider-gauge-fill"),
        riderText:   $("rider-gauge-text"),
        riderEta:    $("rider-eta-value"),
        riderSpoil:  $("rider-spoil-value"),
        cohortBars:  $("theater-cohort-bars"),
        liquidation: $("theater-liquidation"),
        reasoning:   $("theater-reasoning-text"),
        whatif:      $("theater-whatif"),
    },
};

async function pollSimFrames() {
    try {
        const r = await fetch("/commons/sim_frames?limit=240");
        if (!r.ok) return;
        const data = await r.json();
        const frames = Array.isArray(data.frames) ? data.frames : [];
        if (frames.length === 0) {
            theater.els.reasoning.textContent =
                "No simulation frames recorded yet. Run an episode in MarketCommonsEnv with sim_theater=True " +
                "(or POST tick frames via update_blinkit_state) to populate the theater.";
            return;
        }
        const wasAtEnd = theater.idx >= theater.frames.length - 1;
        theater.frames = frames;
        theater.els.slider.max = String(Math.max(0, frames.length - 1));
        if (wasAtEnd) {
            theater.idx = frames.length - 1;
            theater.els.slider.value = String(theater.idx);
            renderFrame(frames[theater.idx]);
        }
        updateFrameLabel();
    } catch (e) { /* ignore — server may be cold */ }
}
setInterval(pollSimFrames, SNAPSHOT_POLL_MS);
pollSimFrames();

function updateFrameLabel() {
    theater.els.label.textContent = `frame ${theater.idx + 1} / ${theater.frames.length}`;
}

function renderFrame(frame) {
    if (!frame) return;
    theater.els.tick.textContent = `tick ${frame.tick ?? "--"}`;

    // Batches
    const batches = Array.isArray(frame.batches) ? frame.batches : [];
    theater.els.batches.innerHTML = batches.length
        ? batches.map(renderBatchTile).join("")
        : "<span style=\"color:var(--fg-2);font-size:11px;\">no active batches</span>";

    // Rider pool
    const rp = frame.rider_pool || {};
    const cap = rp.rider_count || 6;
    const busy = (rp.active_orders || 0) + (rp.queue_depth || 0);
    const fillPct = Math.min(100, (busy / cap) * 100);
    theater.els.riderFill.style.width = `${fillPct}%`;
    theater.els.riderText.textContent =
        `${busy} / ${cap} busy  (queue ${rp.queue_depth || 0})`;
    theater.els.riderEta.textContent = rp.avg_eta_minutes != null
        ? `${rp.avg_eta_minutes.toFixed(1)} min` : "--";
    theater.els.riderSpoil.textContent = String(rp.transit_spoiled || 0);

    // Cohorts
    const cohorts = (frame.cohorts && frame.cohorts.cohorts) || [];
    theater.els.cohortBars.innerHTML = cohorts.length
        ? cohorts.map(renderCohortRow).join("")
        : "<span style=\"color:var(--fg-2);font-size:11px;\">no cohort data</span>";

    // Liquidation
    const liq = (frame.liquidation && frame.liquidation.this_brief) || [];
    theater.els.liquidation.innerHTML = liq.length
        ? liq.map(renderLiquidationRow).join("")
        : "<span style=\"color:var(--fg-2);font-size:11px;\">no liquidation activity this brief</span>";

    // Reasoning
    theater.els.reasoning.textContent = (frame.reasoning || frame.latest_brief || "").trim()
        || "(no brief reasoning attached to this frame)";
    theater.els.whatif.disabled = !frame.latest_brief;
}

function renderBatchTile(b) {
    const u = (b.urgency || "FRESH");
    const status = (b.status || "ACTIVE");
    const cls = status === "LIQUIDATED" ? "urg-LIQUIDATED" : `urg-${u}`;
    return `<div class="batch-tile ${cls}">
        <div>${escapeHTML(b.batch_id || "?")}</div>
        <div>${escapeHTML(b.category || "")}</div>
        <div>${b.quantity_remaining ?? b.quantity ?? 0}u · ${(+b.hours_to_expiry || 0).toFixed(1)}h</div>
    </div>`;
}

function renderCohortRow(c) {
    const pct = Math.max(0, Math.min(100, (c.retention_pct || 0)));
    return `<div class="cohort-row">
        <span class="name">${escapeHTML(c.name)}</span>
        <div class="cohort-bar-track">
            <div class="cohort-bar-fill ${escapeHTML(c.name)}" style="width:${pct.toFixed(0)}%"></div>
        </div>
        <span class="pct">${pct.toFixed(0)}%</span>
    </div>`;
}

function renderLiquidationRow(r) {
    const cls = r.reckless ? "reckless" : (r.accepted ? "accepted" : "");
    const tag = r.reckless ? "RECKLESS" : (r.accepted ? "ACCEPTED" : "REJECTED");
    return `<div class="row ${cls}">[${tag}] ${escapeHTML(r.batch_id)} · ${r.units || 0}u · Rs ${(r.rs || 0).toFixed(0)} · ${escapeHTML(r.reason || "")}</div>`;
}

function tickPlayer() {
    if (theater.idx >= theater.frames.length - 1) {
        stopPlayer();
        return;
    }
    theater.idx += 1;
    theater.els.slider.value = String(theater.idx);
    renderFrame(theater.frames[theater.idx]);
    updateFrameLabel();
}

function startPlayer() {
    if (theater.frames.length === 0) return;
    if (theater.timerId) clearInterval(theater.timerId);
    const intervalMs = Math.max(50, Math.round(800 / theater.speed));
    theater.timerId = setInterval(tickPlayer, intervalMs);
    theater.els.play.disabled = true;
    theater.els.pause.disabled = false;
}

function stopPlayer() {
    if (theater.timerId) clearInterval(theater.timerId);
    theater.timerId = null;
    theater.els.play.disabled = false;
    theater.els.pause.disabled = true;
}

theater.els.play.addEventListener("click", startPlayer);
theater.els.pause.addEventListener("click", stopPlayer);
theater.els.speed.addEventListener("change", (ev) => {
    theater.speed = Number(ev.target.value) || 4;
    if (theater.timerId) startPlayer();
});
theater.els.slider.addEventListener("input", (ev) => {
    theater.idx = Number(ev.target.value) || 0;
    renderFrame(theater.frames[theater.idx]);
    updateFrameLabel();
});
theater.els.whatif.addEventListener("click", () => {
    const frame = theater.frames[theater.idx];
    if (!frame || !frame.latest_brief) return;
    const newBrief = window.prompt(
        "Edit the brief for this tick to fork the simulation:",
        frame.latest_brief,
    );
    if (!newBrief) return;
    fetch("/commons/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            scenario: lastSnapshot && lastSnapshot.scenario_name || "STABLE_WEEK",
            seed: 42,
            brief_log: theater.frames.map((f, i) => ({
                tick: f.tick || i,
                engine_type: "PRICING",
                brief_text: f.latest_brief || "",
            })),
            swap_brief_index: theater.idx,
            new_brief: newBrief,
        }),
    }).then(r => r.json()).then(out => {
        const s = out.summary || {};
        if (els.replayResult) {
            els.replayResult.innerHTML = `<strong>What-if @ frame ${theater.idx}:</strong> ` +
                `baseline WRR ${fmt(s.baseline_final_wrr,3)} → fork ${fmt(s.fork_final_wrr,3)} ` +
                `(${(s.wrr_delta >= 0 ? '+' : '') + fmt(s.wrr_delta, 3)})`;
        }
    }).catch(e => {
        if (els.replayResult) els.replayResult.textContent = "What-if error: " + e.message;
    });
});

// ───────── Project Atlas ─────────
//
// Static one-glance inventory: every env, every agent, every reward
// component with weights, every penalty. Rendered once on page load
// from /atlas/inventory.

async function loadAtlas() {
    try {
        const r = await fetch("/atlas/inventory");
        const data = await r.json();
        const envEl = $("atlas-envs");
        const agentEl = $("atlas-agents");
        const rewardEl = $("atlas-rewards");
        const penaltyEl = $("atlas-penalties");
        const scenarioEl = $("atlas-scenarios");
        if (envEl) envEl.innerHTML = (data.environments || []).map(e =>
            `<div class="atlas-row"><span class="name">${escapeHTML(e.name)}<span class="tag">${escapeHTML(e.theme)}</span></span><span class="role">${escapeHTML(e.role)}</span></div>`).join("");
        if (agentEl) agentEl.innerHTML = (data.agents || []).map(a => {
            const tag = a.trained ? `<span class="tag trained">trained</span>` : `<span class="tag">${escapeHTML(a.kind)}</span>`;
            return `<div class="atlas-row"><span class="name">${escapeHTML(a.name)}${tag}</span><span class="role">${escapeHTML(a.role)}</span></div>`;
        }).join("");
        if (rewardEl) rewardEl.innerHTML = (data.reward_components || []).map(r =>
            `<div class="atlas-row reward">
                <span class="name">${escapeHTML(r.id)} <span class="weight">×${(r.weight ?? 0).toFixed(2)}</span></span>
                <span class="role">${escapeHTML(r.engine)}</span>
                <span class="pos">+ ${escapeHTML(r.positive)}</span>
                <span class="neg">− ${escapeHTML(r.negative)}</span>
            </div>`).join("");
        if (penaltyEl) penaltyEl.innerHTML = (data.global_penalties || []).map(p =>
            `<div class="atlas-row"><span class="name">${escapeHTML(p.id)} (${escapeHTML(String(p.magnitude))})</span><span class="role">${escapeHTML(p.fires_on)}</span></div>`).join("");
        if (scenarioEl) {
            scenarioEl.innerHTML = "<div style=\"font-size:11px;color:var(--fg-2);text-transform:uppercase;letter-spacing:0.10em;margin-bottom:4px;\">Scenario → active engines</div>"
                + (data.scenarios || []).map(s =>
                    `<span class="atlas-scenario">${escapeHTML(s.name)} → [${(s.active_engines || []).join(",")}]</span>`).join("");
        }
    } catch (e) { /* ignore */ }
}
loadAtlas();


// ───────── Live Reward Signal ─────────
//
// Every reward gauge listens for "frame" events; when a tick frame
// arrives the gauge updates its value + bar width and (if a penalty
// fired) flashes red. SES is shown bigger and uses the [0..1] range
// against the 0.70 promotion threshold.

const REWARD_KEYS = [
    "store_efficiency_score",
    "r1_pricing", "r2_farmer", "r3_trend",
    "r4_intrafleet", "r5_micromfg", "r6_event", "r7_surplusbox",
];
const REWARD_BAR_RANGE = 1.0;     // gauges normalise against [0, 1.0]

function updateRewardSignal(frame) {
    if (!frame) return;
    document.querySelectorAll(".reward-gauge").forEach(el => {
        const key = el.dataset.key;
        if (!key) return;
        const v = Number(frame[key] ?? 0);
        const valEl = el.querySelector(".gauge-value");
        const fill = el.querySelector(".gauge-fill");
        if (valEl) valEl.textContent = (v >= 0 ? "+" : "") + v.toFixed(3);
        if (fill) {
            const pct = Math.max(2, Math.min(100, (Math.abs(v) / REWARD_BAR_RANGE) * 100));
            fill.style.width = pct + "%";
            fill.classList.toggle("negative", v < 0);
        }
    });
    // Penalty alerts
    const list = $("reward-alerts-list");
    if (list) {
        const events = frame.penalty_events || [];
        if (events.length === 0) {
            list.innerHTML = `<span class="muted">none this brief</span>`;
        } else {
            list.innerHTML = events.map(ev =>
                `<span class="alert-pill">${escapeHTML(ev.kind)}${ev.magnitude ? " · −" + Number(ev.magnitude).toFixed(2) : ""}${ev.reason ? " · " + escapeHTML(String(ev.reason).slice(0, 60)) : ""}</span>`).join("");
            // Flash the gauge that owns the violation type.
            const flashKey = events.some(e => e.kind === "parse_fail") ? "store_efficiency_score" : null;
            if (flashKey) {
                const g = document.querySelector(`.reward-gauge[data-key="${flashKey}"]`);
                if (g) {
                    g.classList.add("flash-red");
                    setTimeout(() => g.classList.remove("flash-red"), 800);
                }
            }
        }
    }
}


// ───────── RL Training Telemetry ─────────
//
// Stage indicator + REINFORCE loss curve. Pulls /training/status every
// 2.5s. The loss curve renders into the canvas with a simple line
// drawing -- no external chart lib so it stays small.

function setStage(stage) {
    document.querySelectorAll(".training-stage").forEach(el => {
        const s = el.dataset.stage;
        const status = el.querySelector(".stage-status");
        el.classList.remove("active", "done");
        if (s === stage) {
            el.classList.add("active");
            if (status) status.textContent = "▶";
        } else if (stageOrder.indexOf(s) < stageOrder.indexOf(stage)) {
            el.classList.add("done");
            if (status) status.textContent = "✓";
        } else {
            if (status) status.textContent = "⏸";
        }
    });
}
const stageOrder = ["sft", "rollout", "reinforce", "dpo"];

function drawTrainingCurve(history) {
    const canvas = $("training-loss-canvas");
    if (!canvas || !history || history.length === 0) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const losses = history.map(h => h.loss);
    const kls    = history.map(h => h.kl);
    const min = Math.min(0, ...losses);
    const max = Math.max(1e-6, ...losses);
    const range = max - min || 1;
    // Loss line
    ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 2;
    ctx.beginPath();
    losses.forEach((v, i) => {
        const x = (i / Math.max(1, losses.length - 1)) * (W - 20) + 10;
        const y = H - 20 - ((v - min) / range) * (H - 40);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    // KL line (rescaled to its own y-axis on the right)
    const klMax = Math.max(0.05, ...kls.map(Math.abs));
    ctx.strokeStyle = "#60a5fa"; ctx.lineWidth = 1.5;
    ctx.beginPath();
    kls.forEach((v, i) => {
        const x = (i / Math.max(1, kls.length - 1)) * (W - 20) + 10;
        const y = H - 20 - (v / klMax) * (H - 40) * 0.4 - 4;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    // Zero line
    const zeroY = H - 20 - ((0 - min) / range) * (H - 40);
    ctx.strokeStyle = "rgba(255,255,255,0.15)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(10, zeroY); ctx.lineTo(W - 10, zeroY); ctx.stroke();
    // Legend
    ctx.fillStyle = "#fbbf24"; ctx.font = "10px JetBrains Mono";
    ctx.fillText("loss (left)", 10, 14);
    ctx.fillStyle = "#60a5fa";
    ctx.fillText("KL (right, scaled)", 80, 14);
}

async function pollTrainingStatus() {
    try {
        const r = await fetch("/training/status");
        const data = await r.json();
        setStage(data.current_stage || "idle");
        const history = data.reinforce_history || [];
        drawTrainingCurve(history);
        const meta = $("training-meta");
        if (meta) {
            if (history.length === 0) {
                meta.innerHTML = data.last_event
                    ? `<span class="muted">${escapeHTML(data.last_event)}</span>`
                    : `<span class="muted">no telemetry yet — POST to /training/event from the notebook</span>`;
            } else {
                const h = history[history.length - 1];
                const losses = history.map(s => s.loss);
                meta.textContent = `step ${h.step}  loss=${h.loss.toFixed(4)}  ` +
                    `policy=${h.policy_loss.toFixed(4)}  KL=${h.kl.toFixed(4)}  ` +
                    `(${history.length} steps; loss range ${Math.min(...losses).toFixed(2)} → ${Math.max(...losses).toFixed(2)})`;
            }
        }
        const buf = data.buffer_quality || {};
        const fill = $("buffer-fill");
        const text = $("buffer-text");
        const tot = Number(buf.total_briefs || 0);
        const sco = Number(buf.scorable_briefs || 0);
        const pct = tot > 0 ? (sco / tot) * 100 : 0;
        if (fill) fill.style.width = pct.toFixed(0) + "%";
        if (text) text.textContent = tot > 0 ? `${sco} / ${tot} (${pct.toFixed(0)}%)` : "—";
    } catch (e) { /* ignore */ }
}
setInterval(pollTrainingStatus, 2500);
pollTrainingStatus();


// ───────── Decision Flow ─────────
//
// For the latest tick frame, render the four-stage RL loop:
// observation → brief sections → reward chain → learning hook.

const SECTION_ORDER = ["SITUATION", "SIGNAL ANALYSIS", "VIABILITY CHECK",
                       "RECOMMENDATION", "DIRECTIVE", "CONFIDENCE"];

function parseBriefSections(briefText) {
    const sections = {};
    if (!briefText) return sections;
    const lines = briefText.split(/\r?\n/);
    let current = null;
    let buf = [];
    for (const line of lines) {
        const m = line.match(/^([A-Z][A-Z ]{3,}):\s*(.*)$/);
        const head = m && SECTION_ORDER.includes(m[1].trim()) ? m[1].trim() : null;
        if (head) {
            if (current) sections[current] = buf.join("\n").trim();
            current = head; buf = [m[2]];
        } else if (current) {
            buf.push(line);
        }
    }
    if (current) sections[current] = buf.join("\n").trim();
    return sections;
}

function updateDecisionFlow(frame, lastObservation) {
    if (!frame) return;
    // 1. Observation (truncated)
    const obsEl = $("flow-observation");
    if (obsEl) {
        const obsText = lastObservation || "(observation not captured this frame)";
        obsEl.textContent = obsText.length > 600 ? obsText.slice(0, 600) + " …" : obsText;
    }
    // 2. Brief sections
    const briefEl = $("flow-brief");
    if (briefEl) {
        const sections = parseBriefSections(frame.latest_brief || frame.reasoning || "");
        const html = SECTION_ORDER.map(sec => {
            const body = sections[sec] || "—";
            return `<div class="section-header">${sec}</div>` +
                   `<div class="section-body">${escapeHTML(body.slice(0, 240))}</div>`;
        }).join("");
        briefEl.innerHTML = html || "<span class='muted'>no brief in this frame</span>";
    }
    // 3. Reward chain
    const rewardsEl = $("flow-rewards");
    if (rewardsEl) {
        const rs = [
            ["r1 Pricing",    frame.r1_pricing,    0.28],
            ["r2 Farmer",     frame.r2_farmer,     0.18],
            ["r3 Trend",      frame.r3_trend,      0.15],
            ["r4 Intra-Fleet",frame.r4_intrafleet, 0.12],
            ["r5 Micro-Mfg",  frame.r5_micromfg,   0.10],
            ["r6 Event",      frame.r6_event,      0.10],
            ["r7 SurplusBox", frame.r7_surplusbox, 0.07],
        ];
        const ses = Number(frame.store_efficiency_score ?? 0);
        const lines = rs.map(([name, v, w]) => {
            const val = Number(v ?? 0);
            const cls = val > 0 ? "positive" : (val < 0 ? "negative" : "zero");
            const contrib = val * w;
            return `<div class="reward-line ${cls}">
                        <span class="name">${escapeHTML(name)}</span>
                        <span class="val">${(val>=0?"+":"")+val.toFixed(3)}  →  ${(contrib>=0?"+":"")+contrib.toFixed(3)}</span>
                    </div>`;
        }).join("");
        const sesLine = `<div class="reward-line ${ses>0?'positive':ses<0?'negative':'zero'}" style="margin-top:6px;border-top:1px dashed rgba(255,255,255,0.10);padding-top:6px;">
                            <span class="name">SES = Σ wᵢ·rᵢ</span>
                            <span class="val">${(ses>=0?"+":"")+ses.toFixed(3)}</span>
                         </div>`;
        const penalties = (frame.penalty_events || []).map(p =>
            `<div class="reward-line negative"><span class="name">PENALTY ${escapeHTML(p.kind)}</span><span class="val">${p.magnitude ? "−" + Number(p.magnitude).toFixed(2) : ""}</span></div>`).join("");
        rewardsEl.innerHTML = lines + sesLine + penalties;
    }
    // 4. Learn step (descriptive — points to the buffer + REINFORCE flow)
    const learnEl = $("flow-learn");
    if (learnEl) {
        const lines = [
            `<div class="flow-learn-line"><span class="key">→ buffer</span>this brief lands in <code>trajectory_buffer</code> with prompt + brief_text + reward.</div>`,
            `<div class="flow-learn-line"><span class="key">→ advantage</span>(reward − mean) / std → REINFORCE loss = −advantage · log π_θ(brief).</div>`,
            `<div class="flow-learn-line"><span class="key">→ KL anchor</span>+ β · KL(π_θ ‖ π_ref) keeps the policy from drifting too far from SFT.</div>`,
            `<div class="flow-learn-line"><span class="key">→ next brief</span>updated weights → better directive on the next observation.</div>`,
        ];
        learnEl.innerHTML = lines.join("");
    }
}


// Hook the new panels into the existing sim_frames poller. Keep the
// previous renderFrame call -- the new updates are additive.
const _origRenderFrame = renderFrame;
let _lastObservation = "";
renderFrame = function(frame) {
    _origRenderFrame(frame);
    try { updateRewardSignal(frame); } catch (e) {}
    try { updateDecisionFlow(frame, _lastObservation); } catch (e) {}
    if (frame && frame.latest_brief) _lastObservation = frame.reasoning || "";
};


// ───────── Before vs After RL comparison ─────────
//
// Drives the "compare" panel: runs the same scenario through every loaded
// runtime (baseline / sft / rl) and renders the SES-per-scenario bar
// chart, per-runtime SES delta stats, and a side-by-side brief grid.
// The "Load saved snapshot" button reads the JSON produced by
// inference_comparison.py from the server -- so the panel works even
// when the heavy local-model backends are not loaded.

const compareEls = {
    badge:    $("compare-runtimes-badge"),
    scenario: $("compare-scenario"),
    maxBriefs:$("compare-max-briefs"),
    seed:     $("compare-seed"),
    runBtn:   $("compare-run-btn"),
    snapBtn:  $("compare-load-snapshot-btn"),
    status:   $("compare-status"),
    statSft:    $("compare-stat-sft"),
    statRl:     $("compare-stat-rl"),
    statTotal:  $("compare-stat-total"),
    chartRows:  $("compare-chart-rows"),
    briefsGrid: $("compare-briefs-grid"),
};

const RUNTIME_COLOURS = ["baseline", "sft", "rl"];

async function refreshCompareInfo() {
    if (!compareEls.badge) return;
    try {
        const r = await fetch("/agent/compare/info");
        const data = await r.json();
        if (data.status === "ready") {
            compareEls.badge.textContent =
                `runtimes loaded: ${data.names.join(" + ") || "(none)"}`;
        } else {
            compareEls.badge.textContent = `error: ${data.error || "not ready"}`;
        }
    } catch (e) { compareEls.badge.textContent = "offline"; }
}
refreshCompareInfo();

function renderImprovement(imp) {
    const setStat = (el, val) => {
        if (!el) return;
        if (val == null) { el.textContent = "—"; el.classList.remove("positive","negative"); return; }
        el.textContent = (val >= 0 ? "+" : "") + val.toFixed(3);
        el.classList.toggle("positive", val >= 0);
        el.classList.toggle("negative", val < 0);
    };
    setStat(compareEls.statSft,   imp?.sft_over_baseline_ses_delta);
    setStat(compareEls.statRl,    imp?.rl_over_sft_ses_delta);
    setStat(compareEls.statTotal, imp?.rl_over_baseline_ses_delta);
}

function renderCompareChart(perScenario) {
    if (!compareEls.chartRows) return;
    // Find max SES across all rows so bars share a scale; min 0.05 so empty runs still render.
    let maxSes = 0.05;
    for (const scen of Object.keys(perScenario)) {
        for (const name of Object.keys(perScenario[scen] || {})) {
            const v = perScenario[scen][name]?.mean_ses;
            if (typeof v === "number" && v > maxSes) maxSes = v;
        }
    }
    const rows = [];
    for (const scen of Object.keys(perScenario)) {
        for (const name of RUNTIME_COLOURS) {
            const r = perScenario[scen]?.[name];
            if (!r) continue;
            const ses = Number(r.mean_ses ?? 0);
            const widthPct = Math.max(2, Math.min(100, (ses / maxSes) * 100));
            rows.push(`<div class="compare-row">
                <span class="scen">${escapeHTML(scen)}</span>
                <span class="name">${escapeHTML(name)}</span>
                <div class="bar-track"><div class="bar-fill ${escapeHTML(name)}" style="width:${widthPct.toFixed(0)}%"></div></div>
                <span class="pct">${(ses >= 0 ? "+" : "") + ses.toFixed(3)}</span>
            </div>`);
        }
    }
    compareEls.chartRows.innerHTML = rows.join("") ||
        "<span style=\"color:var(--fg-2);font-size:11px;\">no comparison data yet</span>";
}

function renderBriefsGrid(perScenario, scenarioName) {
    if (!compareEls.briefsGrid) return;
    const data = perScenario?.[scenarioName] || {};
    const cells = [];
    for (const name of RUNTIME_COLOURS) {
        const r = data[name];
        if (!r) continue;
        const samples = r.sample_briefs || [];
        const sample = samples[0] || "(no sample brief recorded)";
        cells.push(`<div class="compare-brief-cell">
            <span class="runtime-tag ${escapeHTML(name)}">${escapeHTML(name)}</span>
            <pre>${escapeHTML(sample)}</pre>
        </div>`);
    }
    compareEls.briefsGrid.innerHTML = cells.join("") ||
        "<span style=\"color:var(--fg-2);font-size:11px;\">run a comparison or load a snapshot to populate this</span>";
}

if (compareEls.runBtn) {
    compareEls.runBtn.addEventListener("click", async () => {
        compareEls.runBtn.disabled = true;
        compareEls.status.textContent = "running comparison...";
        try {
            const r = await fetch("/agent/compare/episode", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    scenario:   compareEls.scenario.value,
                    seed:       Number(compareEls.seed.value) || 42,
                    max_briefs: Number(compareEls.maxBriefs.value) || 6,
                }),
            });
            const data = await r.json();
            if (data.error) {
                compareEls.status.textContent = "error: " + data.error;
            } else {
                const scen = data.scenario;
                const perScenario = { [scen]: data.results };
                renderImprovement(data.improvement);
                renderCompareChart(perScenario);
                renderBriefsGrid(perScenario, scen);
                compareEls.status.textContent = "live: " + scen;
            }
        } catch (e) {
            compareEls.status.textContent = "error: " + e.message;
        } finally {
            compareEls.runBtn.disabled = false;
        }
    });
}

if (compareEls.snapBtn) {
    compareEls.snapBtn.addEventListener("click", async () => {
        compareEls.snapBtn.disabled = true;
        compareEls.status.textContent = "loading snapshot...";
        try {
            const r = await fetch("/agent/compare/snapshot");
            const data = await r.json();
            if (data.status !== "ok") {
                compareEls.status.textContent = "snapshot: " + (data.hint || data.error || data.status);
                return;
            }
            const snap = data.results;
            renderImprovement(snap.improvement);
            renderCompareChart(snap.per_scenario || {});
            // Render briefs from the currently-selected scenario, falling
            // back to the first scenario in the snapshot.
            const want = compareEls.scenario.value;
            const scenarios = snap.scenarios || Object.keys(snap.per_scenario || {});
            const chosen = scenarios.includes(want) ? want : scenarios[0];
            if (chosen) renderBriefsGrid(snap.per_scenario || {}, chosen);
            compareEls.status.textContent = `snapshot loaded (${scenarios.length} scenarios, ` +
                `produced ${snap.produced_at || "—"})`;
        } catch (e) {
            compareEls.status.textContent = "error: " + e.message;
        } finally {
            compareEls.snapBtn.disabled = false;
        }
    });
}

// ───────── Agent live-demo runner ─────────
const agentEls = {
    badge:    $("agent-info-badge"),
    scenario: $("agent-scenario"),
    maxBriefs:$("agent-max-briefs"),
    seed:     $("agent-seed"),
    runBtn:   $("agent-run-btn"),
    status:   $("agent-run-status"),
};

async function refreshAgentInfo() {
    try {
        const r = await fetch("/agent/info");
        const data = await r.json();
        if (data.status === "ready") {
            const detail = data.model_path || data.repo_id || data.model_id || "";
            agentEls.badge.textContent = `${data.backend}${detail ? " — " + detail : ""}`;
        } else {
            agentEls.badge.textContent = `error: ${data.error || "?"}`;
        }
    } catch (e) {
        agentEls.badge.textContent = "offline";
    }
}
refreshAgentInfo();

if (agentEls.runBtn) {
    agentEls.runBtn.addEventListener("click", async () => {
        agentEls.runBtn.disabled = true;
        agentEls.status.textContent = "running...";
        try {
            const r = await fetch("/agent/run_episode", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    scenario: agentEls.scenario.value,
                    max_briefs: Number(agentEls.maxBriefs.value) || 8,
                    seed: Number(agentEls.seed.value) || 42,
                }),
            });
            const data = await r.json();
            if (data.error) {
                agentEls.status.textContent = "error: " + data.error;
            } else {
                agentEls.status.textContent =
                    `done: ${data.steps_completed} briefs, WRR=${(data.final_wrr ?? 0).toFixed(3)} ` +
                    `(reward=${(data.total_reward ?? 0).toFixed(3)})`;
                pollSimFrames();   // pull the freshly recorded frames
            }
        } catch (e) {
            agentEls.status.textContent = "error: " + e.message;
        } finally {
            agentEls.runBtn.disabled = false;
        }
    });
}

connectWS();
