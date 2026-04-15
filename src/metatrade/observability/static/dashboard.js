/* MetaTrade Dashboard — main script */
"use strict";

const cfg = window.METATRADE_DASHBOARD;

// ── DOM refs ───────────────────────────────────────────────────────────────
const symbolInput     = document.getElementById("symbol-input");
const timeframeInput  = document.getElementById("timeframe-input");
const activeRunInfo   = document.getElementById("active-run-info");

let equityChart, barsChart;
let hasInitializedActiveRun = false;
let decisionProgressTickStarted = false;
let refreshTimer = null;
let lastUiPollSec = null;

// ── Formatters ─────────────────────────────────────────────────────────────
function fmt2(v)   { return (v == null || isNaN(+v)) ? "—" : (+v).toFixed(2); }
function fmtPct(v) { return (v == null || isNaN(+v)) ? "—" : `${(+v * 100).toFixed(1)}%`; }
function fmtConf(v){ return (v == null || isNaN(+v)) ? "—" : (+v).toFixed(2); }
function fmtTs(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString("it-IT", {
    day: "2-digit", month: "2-digit", year: "numeric",
    hour: "2-digit", minute: "2-digit", second: "2-digit"
  });
}

function setText(id, v) {
  const el = document.getElementById(id);
  if (el) el.textContent = v;
}

function escHtml(v) {
  return String(v ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function parseDetails(v) {
  if (!v) return {};
  if (typeof v === "object") return v;
  try { return JSON.parse(v); } catch { return {}; }
}

// ── Badges ─────────────────────────────────────────────────────────────────
function dirBadge(dir) {
  const d = String(dir || "HOLD").toUpperCase();
  const cls = d === "BUY" ? "badge-buy" : d === "SELL" ? "badge-sell" : "badge-hold";
  return `<span class="badge ${cls}">${escHtml(d)}</span>`;
}

function orderEventBadge(type) {
  const t = String(type || "").toLowerCase();
  if (t === "order_failed" || t === "order_rejected") {
    return `<span class="badge badge-fail">FALLITO</span>`;
  }
  if (t === "order_submitted") {
    return `<span class="badge badge-ok">INVIATO</span>`;
  }
  if (t === "order_filled") {
    return `<span class="badge badge-ok">ESEGUITO</span>`;
  }
  return `<span class="badge badge-neutral">${escHtml(type)}</span>`;
}

function statusBadge(status) {
  const s = String(status || "").toLowerCase();
  if (s === "running") return `<span class="badge badge-ok">running</span>`;
  if (s === "completed") return `<span class="badge badge-neutral">completed</span>`;
  return `<span class="badge badge-hold">${escHtml(status)}</span>`;
}

// ── Decision helpers ────────────────────────────────────────────────────────
function decisionEsito(row) {
  if (!row) return "—";
  if (!row.actionable) return `<span class="muted">Nessuna operazione</span>`;
  if (row.approved === true || row.approved === 1)
    return `<span class="badge badge-ok">Approvata</span>`;
  if (row.approved === false || row.approved === 0)
    return row.veto_code
      ? `<span class="badge badge-fail">${escHtml(row.veto_code)}</span>`
      : `<span class="badge badge-fail">Bloccata</span>`;
  return "—";
}

function summarizeHoldReason(row, details) {
  if (!row) return "Nessuna decisione disponibile.";
  const signals = Array.isArray(details.signals) ? details.signals : [];
  if (row.direction !== "HOLD") {
    if (row.approved === false || row.approved === 0)
      return row.veto_reason || row.explanation || "Segnale presente ma bloccato dal risk manager.";
    return row.explanation || "L'ultima decisione non era HOLD.";
  }
  if (!row.actionable)
    return row.explanation || `Consenso sotto soglia (${fmtConf(row.threshold_used)}).`;
  const nonHold = signals.filter(s => s.direction && s.direction !== "HOLD");
  if (nonHold.length === 0) return "Tutti i moduli hanno votato HOLD.";
  if (row.approved === false || row.approved === 0)
    return row.veto_reason || "Segnale presente, ma il risk manager lo ha bloccato.";
  return row.explanation || "Consenso insufficiente per BUY o SELL.";
}

// ── Signal breakdown ────────────────────────────────────────────────────────
function renderSignalBreakdown(details) {
  const el = document.getElementById("signal-breakdown");
  const signals = Array.isArray(details.signals) ? [...details.signals] : [];
  if (!signals.length) {
    el.innerHTML = `<p class="muted">Nessun segnale disponibile.</p>`;
    return;
  }
  signals.sort((a, b) => +b.confidence - +a.confidence);
  el.innerHTML = signals.map(s => `
    <div class="signal-item">
      <div class="signal-head">
        <div class="signal-title"><code>${escHtml(s.module_id || "?")}</code></div>
        <div class="signal-meta">
          ${dirBadge(s.direction)}
          <span class="badge badge-neutral">conf ${fmtConf(s.confidence)}</span>
        </div>
      </div>
      <div class="signal-reason">${escHtml(s.reason || "—")}</div>
    </div>`).join("");
}

function renderLatestDecision(row) {
  const det = parseDetails(row?.details);
  setText("latest-direction", row?.direction || "—");
  setText("latest-confidence", fmtConf(row?.aggregate_confidence));
  setText("latest-actionable", row ? (row.actionable ? "Sì" : "No") : "—");
  setText("latest-approved", row
    ? (row.approved === null || row.approved === undefined ? "—" : row.approved ? "Sì" : "No")
    : "—");
  setText("latest-explanation", row?.explanation || "—");
  setText("latest-hold-reason", summarizeHoldReason(row, det));
  renderSignalBreakdown(det);
  // Signal badge on accordion header
  const badge = document.getElementById("acc-signals-badge");
  if (badge && row) badge.textContent = row.direction || "";
}

// ── Decision progress (sidebar) ─────────────────────────────────────────────
function fmtDurSec(sec) {
  if (sec == null || isNaN(+sec)) return "—";
  const x = +sec;
  if (x < 1) return "< 1 s";
  const s = Math.ceil(x);
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60), r = s % 60;
  return r ? `${m} min ${r} s` : `${m} min`;
}

function tickDecisionProgress() {
  const s = window.__dpState;
  const fill = document.getElementById("dp-fill");
  const track = document.getElementById("dp-track");
  if (!fill || !track) return;
  if (!s?.available) {
    fill.style.width = "0%";
    track.setAttribute("aria-valuenow", "0");
    setText("dp-timeframe", s?.message || "—");
    setText("dp-remaining", "—");
    setText("dp-last", "—");
    return;
  }
  setText("dp-timeframe", `${s.timeframe || "—"}`);
  if (s.last_event_ts == null) {
    fill.style.width = "0%";
    setText("dp-remaining", "In attesa della prima barra…");
    setText("dp-last", "—");
    return;
  }
  const interval = +s.interval_seconds;
  const now = Date.now() / 1000;
  const elapsed = now - s.last_event_ts;
  const k = Math.floor(elapsed / interval);
  const phase = elapsed - k * interval;
  const progress = Math.min(1, Math.max(0, phase / interval));
  const remaining = Math.max(0, s.last_event_ts + (k + 1) * interval - now);
  fill.style.width = `${(progress * 100).toFixed(1)}%`;
  track.setAttribute("aria-valuenow", String(Math.round(progress * 100)));
  setText("dp-remaining", `≈ ${fmtDurSec(remaining)}`);
  setText("dp-last", fmtTs(s.last_event_ts));
}

function startProgressTick() {
  if (decisionProgressTickStarted) return;
  decisionProgressTickStarted = true;
  setInterval(tickDecisionProgress, 250);
}

// ── Tables ──────────────────────────────────────────────────────────────────
function renderTable(id, rows, mapper) {
  const tbody = document.querySelector(`#${id} tbody`);
  if (!tbody) return;
  tbody.innerHTML = rows.length
    ? rows.map(mapper).join("")
    : `<tr><td colspan="99" class="muted tc">Nessun dato</td></tr>`;
}

// ── Fetch helpers ───────────────────────────────────────────────────────────
async function getJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} ${url}`);
  return r.json();
}

// ── Charts ──────────────────────────────────────────────────────────────────
function upsertLine(chart, ctx, labels, data, label, color) {
  if (chart) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.data.datasets[0].label = label;
    chart.update("none");
    return chart;
  }
  return new Chart(ctx, {
    type: "line",
    data: { labels, datasets: [{ label, data, borderColor: color, backgroundColor: "transparent", tension: 0.2, pointRadius: 1 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { labels: { color: "#94a3b8", font: { size: 11 } } },
        zoom: {
          pan: { enabled: true, mode: "x", modifierKey: "shift" },
          zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: "x" }
        }
      },
      scales: {
        x: { ticks: { color: "#475569", font: { size: 10 }, maxTicksLimit: 10 }, grid: { color: "#111827" } },
        y: { ticks: { color: "#475569", font: { size: 10 } }, grid: { color: "#111827" } }
      }
    }
  });
}

// ── Threshold bar ────────────────────────────────────────────────────────────
const TH_MIN = 0.45, TH_MAX = 0.90, TH_DEF = 0.60;

function thresholdBar(t) {
  const pct = Math.round(((t - TH_MIN) / (TH_MAX - TH_MIN)) * 100);
  const col = t < TH_DEF ? "#34d399" : t > TH_DEF ? "#f87171" : "#64748b";
  return `<div class="threshold-bar-wrap" title="${t.toFixed(3)}">
    <div class="threshold-bar-track">
      <div class="threshold-bar-fill" style="width:${pct}%;background:${col}"></div>
      <div class="threshold-bar-default"></div>
    </div>
    <span class="threshold-val">${t.toFixed(3)}</span>
  </div>`;
}

function trendArrow(t) {
  if (t < TH_DEF - 0.02) return `<span class="trend-down" title="Affidabile">▼</span>`;
  if (t > TH_DEF + 0.02) return `<span class="trend-up" title="Poco affidabile">▲</span>`;
  return `<span class="trend-neutral">■</span>`;
}

function weightBar(w) {
  const pct = Math.max(0, Math.min(100, w));
  const col = w >= 65 ? "#22c55e" : w >= 50 ? "#86efac" : w >= 35 ? "#fb923c" : "#ef4444";
  return `<div class="threshold-bar-wrap" title="peso ${w.toFixed(1)}">
    <div class="threshold-bar-track">
      <div class="threshold-bar-fill" style="width:${pct}%;background:${col}"></div>
      <div class="threshold-bar-default" style="left:50%"></div>
    </div>
    <span class="threshold-val">${w.toFixed(1)}</span>
  </div>`;
}

// ── Kill Switch ──────────────────────────────────────────────────────────────
const KS_LABELS = { 0: "NONE", 1: "TRADE GATE", 2: "SESSION GATE", 3: "EMERGENCY HALT", 4: "HARD KILL" };

function applyKsState(data) {
  const level = +(data.level ?? 0);
  const label = KS_LABELS[level] ?? "UNKNOWN";
  ["ks-badge", "sidebar-ks-badge"].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = label;
    el.className = `badge ks-badge ks-level-${level}`;
  });
  const bar = document.getElementById("kill-switch-bar");
  if (bar) bar.classList.toggle("ks-active", level > 0);
  const reason = document.getElementById("ks-reason");
  if (reason) {
    reason.textContent = level > 0 && data.reason
      ? `${data.reason}${data.activated_by ? ` — ${data.activated_by}` : ""}`
      : "Trading attivo";
  }
}

async function refreshKillSwitch() {
  const data = await getJson("/api/kill-switch");
  applyKsState(data);
}

function wireKillSwitchButtons() {
  document.querySelectorAll(".btn-ks[data-level]").forEach(btn => {
    btn.addEventListener("click", async () => {
      const level = +btn.dataset.level;
      const label = btn.dataset.label || btn.textContent.trim();
      if (level >= 3 && !confirm(`Confermare attivazione ${label}?\n\nQuesto blocca il sistema.`)) return;
      await fetch("/api/kill-switch/activate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ level, reason: `${label} — dashboard` })
      });
      await refreshKillSwitch();
    });
  });
  document.getElementById("ks-reset-btn")?.addEventListener("click", async () => {
    await fetch("/api/kill-switch/reset", { method: "POST" });
    await refreshKillSwitch();
  });
}

// ── Accordion state (localStorage) ──────────────────────────────────────────
function saveAccordionState() {
  const state = {};
  document.querySelectorAll("details.accordion[id]").forEach(d => {
    state[d.id] = d.open;
  });
  try { localStorage.setItem("mt_accordion", JSON.stringify(state)); } catch {}
}

function restoreAccordionState() {
  try {
    const state = JSON.parse(localStorage.getItem("mt_accordion") || "{}");
    document.querySelectorAll("details.accordion[id]").forEach(d => {
      if (d.id in state) d.open = state[d.id];
    });
  } catch {}
  document.querySelectorAll("details.accordion").forEach(d => {
    d.addEventListener("toggle", saveAccordionState);
  });
}

// ── UI polling + runner status ──────────────────────────────────────────────

function scheduleUiRefresh(sec) {
  const s = Math.max(3, Math.min(Number(sec) || 10, 3600));
  if (lastUiPollSec === s && refreshTimer) return;
  lastUiPollSec = s;
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = setInterval(refreshAll, s * 1000);
}

function applyRunnerState(ov) {
  const r = ov.runner || {};
  const line = document.getElementById("runner-status-line");
  const btn = document.getElementById("btn-runner-restart");
  const msg = document.getElementById("runner-restart-msg");
  const hint = document.getElementById("ui-refresh-hint");

  if (line) {
    let t = r.process_running
      ? `● Runner attivo (PID ${(r.pids || []).join(", ") || "—"})`
      : "○ Nessun processo run_live / run_paper";
    if (r.db_orphan) t += " — DB: sessione aperta senza processo";
    line.textContent = t;
    line.className = r.process_running ? "sidebar-session runner-ok" : "sidebar-session runner-down";
  }
  if (btn) {
    btn.disabled = !r.can_restart;
    btn.title = r.can_restart
      ? "Riavvia l'ultimo comando salvato (nuova console)"
      : r.process_running
        ? "Disponibile quando il runner non è in esecuzione"
        : "Esegui run_live o run_paper una volta per salvare il comando";
  }

  if (hint && ov.refresh_seconds != null) {
    hint.textContent =
      `Aggiornamento dati UI: ogni ${ov.refresh_seconds}s. La barra “Prossima decisione” segue il timeframe (es. M1 = 60s).`;
  }
  if (ov.refresh_seconds != null) scheduleUiRefresh(ov.refresh_seconds);
}

// ── Refresh functions ────────────────────────────────────────────────────────

async function refreshOverview() {
  const ov = await getJson("/api/overview");
  const mt5 = ov.mt5_account || {};
  const model = ov.active_model || {};
  const tr = ov.latest_training || {};
  const sessions = ov.sessions || [];
  const decision = ov.latest_decision || null;

  window.__dpState = ov.decision_progress || null;
  tickDecisionProgress();

  setText("kpi-balance",   fmt2(mt5.balance));
  setText("kpi-equity",    fmt2(mt5.equity));
  setText("kpi-profit",    fmt2(mt5.profit));
  setText("kpi-positions", String((ov.mt5_positions || []).length));
  setText("kpi-model",     model.version || "—");
  setText("kpi-training",  tr.model_version || tr.status || "—");

  const activeSession = sessions[0] || null;
  if (activeSession) {
    activeRunInfo.textContent =
      `${activeSession.run_mode} · ${activeSession.symbol}/${activeSession.timeframe} · ${activeSession.model_version || "no ML"}`;
    if (!hasInitializedActiveRun) {
      if (activeSession.symbol)    symbolInput.value    = activeSession.symbol;
      if (activeSession.timeframe) timeframeInput.value = activeSession.timeframe;
      hasInitializedActiveRun = true;
    }
  } else {
    activeRunInfo.textContent = "Nessuna sessione attiva";
  }

  // Sessions count badge
  const sessBadge = document.getElementById("acc-sessions-badge");
  if (sessBadge) sessBadge.textContent = sessions.length ? String(sessions.length) : "";

  renderLatestDecision(decision);

  // MT5 positions
  const positions = ov.mt5_positions || [];
  renderTable("positions-table", positions, row => `
    <tr>
      <td><strong>${escHtml(row.symbol ?? "—")}</strong></td>
      <td>${dirBadge(row.type === 0 ? "BUY" : "SELL")}</td>
      <td>${fmt2(row.volume)}</td>
      <td>${fmt2(row.price_open)}</td>
      <td>${fmt2(row.price_current)}</td>
      <td class="${+row.profit >= 0 ? "positive" : "negative"}">${fmt2(row.profit)}</td>
      <td class="muted">${fmt2(row.sl)}</td>
      <td class="muted">${fmt2(row.tp)}</td>
    </tr>`);

  // Positions badge
  const ordBadge = document.getElementById("acc-orders-badge");
  if (ordBadge) ordBadge.textContent = positions.length ? `${positions.length} pos.` : "";

  // Sessions table
  renderTable("sessions-table", sessions, row => `
    <tr>
      <td><code>${escHtml(row.session_id)}</code></td>
      <td>${escHtml(row.run_mode)}</td>
      <td>${escHtml(row.symbol ?? "—")}</td>
      <td>${escHtml(row.timeframe ?? "—")}</td>
      <td>${row.ml_enabled ? "✓" : "—"}</td>
      <td class="muted">${escHtml(row.model_version ?? "—")}</td>
      <td>${statusBadge(row.status)}</td>
    </tr>`);

  applyRunnerState(ov);
}

async function refreshDecisions() {
  const rows = await getJson("/api/decisions?limit=20");
  renderTable("decisions-table", rows, row => {
    const voteInfo = row.explanation
      ? escHtml(row.explanation.length > 50 ? row.explanation.slice(0, 50) + "…" : row.explanation)
      : "—";
    return `
    <tr>
      <td class="muted" style="white-space:nowrap">${fmtTs(row.ts)}</td>
      <td>${dirBadge(row.direction)}</td>
      <td>${fmtConf(row.aggregate_confidence)}</td>
      <td>${decisionEsito(row)}</td>
      <td class="muted" style="font-size:0.75rem">${voteInfo}</td>
      <td class="muted" style="font-size:0.75rem">${escHtml(row.veto_reason || row.explanation?.slice(0,60) || "—")}</td>
      <td class="muted" style="font-size:0.72rem">${escHtml(row.model_version ?? "—")}</td>
    </tr>`;
  });
}

async function refreshOrders() {
  const rows = await getJson("/api/orders?limit=30");
  // Sort newest first (they come desc already)
  let failCount = 0;
  renderTable("orders-table", rows, row => {
    const isFailed = /fail|reject/i.test(row.event_type || "");
    if (isFailed) failCount++;
    const det = parseDetails(row.details);
    const detailStr = isFailed && det.error
      ? `<span style="color:#fca5a5;font-size:0.75rem">${escHtml(String(det.error).slice(0, 80))}</span>`
      : escHtml(row.status || "—");
    return `
    <tr class="${isFailed ? "row-failed" : "row-submitted"}">
      <td class="muted" style="white-space:nowrap;font-size:0.75rem">${fmtTs(row.ts)}</td>
      <td>${orderEventBadge(row.event_type)}</td>
      <td><strong>${escHtml(row.symbol ?? "—")}</strong></td>
      <td>${row.side ? dirBadge(row.side) : "—"}</td>
      <td>${row.lot_size != null ? fmt2(row.lot_size) : "—"}</td>
      <td>${escHtml(row.status || "—")}</td>
      <td style="max-width:220px;word-break:break-word">${detailStr}</td>
    </tr>`;
  });
  // Update orders badge with fail count
  const badge = document.getElementById("acc-orders-badge");
  if (badge) {
    if (failCount > 0) {
      badge.textContent = `${failCount} errori`;
      badge.style.background = "#7f1d1d";
      badge.style.color = "#fca5a5";
    } else if (rows.length > 0) {
      badge.textContent = `${rows.length} ordini`;
      badge.style.background = "";
      badge.style.color = "";
    }
  }
}

async function refreshEquityCurve() {
  const rows = await getJson("/api/account/equity-curve?limit=200");
  const asc = [...rows].reverse();
  equityChart = upsertLine(
    equityChart, document.getElementById("equity-chart"),
    asc.map(r => new Date(r.ts * 1000).toLocaleTimeString("it-IT")),
    asc.map(r => r.equity),
    "Equity", "#38bdf8"
  );
}

async function refreshBars() {
  const sym = encodeURIComponent(symbolInput.value || cfg.defaultSymbol);
  const tf  = encodeURIComponent(timeframeInput.value || cfg.defaultTimeframe);
  const rows = await getJson(`/api/bars?symbol=${sym}&timeframe=${tf}&limit=180`);
  barsChart = upsertLine(
    barsChart, document.getElementById("bars-chart"),
    rows.map(r => new Date(r.ts).toLocaleTimeString("it-IT")),
    rows.map(r => r.close),
    `${symbolInput.value} close`, "#818cf8"
  );
}

async function refreshTraining() {
  const rows = await getJson("/api/training/runs?limit=20");
  renderTable("training-table", rows, row => `
    <tr>
      <td><code>${escHtml(row.model_version ?? "—")}</code></td>
      <td>${escHtml(row.symbol)}</td>
      <td>${escHtml(row.timeframe)}</td>
      <td class="${+row.best_test_accuracy >= 0.55 ? "positive" : "negative"}">${fmt2(row.best_test_accuracy)}</td>
      <td>${fmt2(row.mean_test_accuracy)}</td>
      <td class="muted">${row.bars_fetched ?? "—"}</td>
      <td>${statusBadge(row.status)}</td>
    </tr>`);
}

async function refreshThresholds() {
  const rows = await getJson("/api/module-thresholds");
  rows.forEach(r => {
    const n = r.eval_count || 0, c = r.correct_count || 0;
    r.accuracy_pct = n > 0 ? +(c / n * 100).toFixed(1) : null;
  });
  renderTable("thresholds-table", rows, row => `
    <tr>
      <td><code>${escHtml(row.module_id)}</code></td>
      <td>${thresholdBar(+row.threshold)}</td>
      <td>${row.eval_count ?? 0}</td>
      <td>${row.correct_count ?? 0}</td>
      <td>${row.accuracy_pct != null ? `${row.accuracy_pct}%` : `<span class="muted">—</span>`}</td>
      <td class="muted">${row.mean_score != null ? (+row.mean_score).toFixed(3) : "—"}</td>
      <td>${trendArrow(+row.threshold)}</td>
      <td class="muted" style="font-size:0.75rem;white-space:nowrap">${fmtTs(row.updated_at)}</td>
    </tr>`);
  const badge = document.getElementById("acc-thresholds-badge");
  if (badge) badge.textContent = rows.length ? `${rows.length} moduli` : "";
}

async function refreshReputations() {
  const rows = await getJson("/api/rule-reputations");
  renderTable("reputations-table", rows, row => `
    <tr>
      <td><code>${escHtml(row.rule_id)}</code></td>
      <td>${row.symbol === "*" ? `<span class="muted">globale</span>` : escHtml(row.symbol)}</td>
      <td>${weightBar(+(row.weight ?? 50))}</td>
      <td>${row.eval_count ?? 0}</td>
      <td class="muted">${row.mean_score != null ? (+row.mean_score).toFixed(3) : "—"}</td>
      <td class="muted" style="font-size:0.75rem;white-space:nowrap">${row.last_eval_ts > 0 ? fmtTs(row.last_eval_ts) : "—"}</td>
    </tr>`);
}

// ── Main refresh ─────────────────────────────────────────────────────────────
async function refreshAll() {
  try {
    await refreshOverview();
    await Promise.all([
      refreshDecisions(),
      refreshOrders(),
      refreshEquityCurve(),
      refreshThresholds(),
      refreshReputations(),
      refreshTraining(),
      refreshKillSwitch(),
    ]);
    setText("last-refresh", new Date().toLocaleTimeString("it-IT"));
  } catch (err) {
    setText("last-refresh", `Errore: ${err.message}`);
  }
}

// ── Event listeners ──────────────────────────────────────────────────────────
document.getElementById("reload-bars")?.addEventListener("click", refreshBars);
document.getElementById("reset-bars-zoom")?.addEventListener("click", () => barsChart?.resetZoom());

// Ops panel
document.getElementById("btn-copy-stop")?.addEventListener("click", () => {
  const pre = document.querySelector(".command-block");
  if (pre) navigator.clipboard.writeText(pre.textContent).catch(() => {});
});

document.getElementById("btn-close-db-sessions")?.addEventListener("click", async () => {
  const msg = document.getElementById("close-sessions-msg");
  try {
    const r = await fetch("/api/sessions/close-open?status=interrupted", { method: "POST" });
    const d = await r.json();
    if (msg) msg.textContent = `${d.closed} sessioni chiuse.`;
  } catch (e) {
    if (msg) msg.textContent = `Errore: ${e.message}`;
  }
});

document.getElementById("btn-runner-restart")?.addEventListener("click", async () => {
  const msg = document.getElementById("runner-restart-msg");
  if (msg) {
    msg.dataset.busy = "1";
    msg.textContent = "Riavvio in corso…";
  }
  try {
    const r = await fetch("/api/runner/restart", { method: "POST" });
    const j = await r.json();
    if (!r.ok) {
      const detail = j.detail;
      const errText = Array.isArray(detail)
        ? detail.map((x) => x.msg || x).join(" ")
        : (detail || r.statusText);
      throw new Error(errText);
    }
    if (msg) msg.textContent = j.detail || "Avviato.";
    await refreshOverview();
  } catch (e) {
    if (msg) msg.textContent = `Errore: ${e.message}`;
  } finally {
    if (msg) delete msg.dataset.busy;
  }
});

// ── Init ─────────────────────────────────────────────────────────────────────
restoreAccordionState();
wireKillSwitchButtons();
startProgressTick();
scheduleUiRefresh(cfg.refreshSeconds);
refreshAll().then(refreshBars);
