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
