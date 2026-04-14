const cfg = window.METATRADE_DASHBOARD;

const equityCtx = document.getElementById("equity-chart");
const barsCtx = document.getElementById("bars-chart");
const symbolInput = document.getElementById("symbol-input");
const timeframeInput = document.getElementById("timeframe-input");
const activeRunInfo = document.getElementById("active-run-info");
const resetBarsZoomButton = document.getElementById("reset-bars-zoom");

let equityChart;
let barsChart;
let hasInitializedActiveRun = false;
let decisionProgressTickStarted = false;

function fmtNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(2);
}

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function parseJsonDetails(value) {
  if (!value) return {};
  if (typeof value === "object") return value;
  try {
    return JSON.parse(value);
  } catch {
    return {};
  }
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function fmtConfidence(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${Number(value).toFixed(2)}`;
}

function directionBadge(direction) {
  const normalized = String(direction || "HOLD").toUpperCase();
  const cssClass = normalized === "BUY"
    ? "badge-buy"
    : normalized === "SELL"
      ? "badge-sell"
      : "badge-hold";
  return `<span class="badge ${cssClass}">${escapeHtml(normalized)}</span>`;
}

function decisionStatus(row) {
  if (!row) return "-";
  if (!row.actionable) return "Nessuna operazione";
  if (row.approved === true || row.approved === 1) return "Operazione approvata";
  if (row.approved === false || row.approved === 0) {
    return row.veto_code ? `Bloccata: ${row.veto_code}` : "Bloccata dal risk";
  }
  return "In valutazione";
}

function summarizeHoldReason(row, details) {
  if (!row) return "Nessuna decisione disponibile.";
  const signals = Array.isArray(details.signals) ? details.signals : [];
  if (row.direction !== "HOLD") {
    if (row.approved === false || row.approved === 0) {
      return row.veto_reason || row.explanation || "Segnale presente ma fermato dal modulo di rischio.";
    }
    return row.explanation || "L'ultima decisione non era HOLD.";
  }
  if (!row.actionable) {
    return row.explanation || `Il consenso non ha superato la soglia minima (${fmtConfidence(row.threshold_used)}).`;
  }
  const nonHoldSignals = signals.filter((signal) => signal.direction && signal.direction !== "HOLD");
  if (nonHoldSignals.length === 0) {
    return "Tutti i moduli hanno votato HOLD, quindi non e' stato aperto nessun trade.";
  }
  if (row.approved === false || row.approved === 0) {
    return row.veto_reason || "C'era un segnale, ma il risk manager lo ha bloccato.";
  }
  return row.explanation || "Il sistema non ha trovato abbastanza evidenza per comprare o vendere.";
}

function renderSignalBreakdown(details) {
  const container = document.getElementById("signal-breakdown");
  const signals = Array.isArray(details.signals) ? [...details.signals] : [];
  if (!signals.length) {
    container.innerHTML = `<p class="muted">Nessun segnale ancora disponibile.</p>`;
    return;
  }
  signals.sort((a, b) => Number(b.confidence || 0) - Number(a.confidence || 0));
  container.innerHTML = signals.map((signal) => `
    <div class="signal-item">
      <div class="signal-head">
        <div class="signal-title">${escapeHtml(signal.module_id || "module")}</div>
        <div class="signal-meta">
          ${directionBadge(signal.direction)}
          <span class="badge badge-neutral">conf ${fmtConfidence(signal.confidence)}</span>
        </div>
      </div>
      <div class="signal-reason">${escapeHtml(signal.reason || "Nessuna motivazione disponibile.")}</div>
    </div>
  `).join("");
}

function renderLatestDecision(row) {
  const details = parseJsonDetails(row?.details);
  setText("latest-direction", row?.direction || "-");
  setText("latest-confidence", fmtConfidence(row?.aggregate_confidence));
  setText("latest-actionable", row ? (row.actionable ? "Si" : "No") : "-");
  setText("latest-approved", row ? (row.approved === null ? "-" : (row.approved ? "Si" : "No")) : "-");
  setText("latest-explanation", row?.explanation || "Nessuna spiegazione disponibile.");
  setText("latest-hold-reason", summarizeHoldReason(row, details));
  renderSignalBreakdown(details);
}

function fmtDurationSeconds(sec) {
  if (sec === null || sec === undefined || Number.isNaN(Number(sec))) return "—";
  const x = Number(sec);
  if (x < 1) return "meno di 1 s";
  const s = Math.ceil(x);
  if (s < 60) return `circa ${s} s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  return r ? `circa ${m} min ${r} s` : `circa ${m} min`;
}

function basisLabel(basis) {
  if (basis === "decision") return "ultima decisione";
  if (basis === "snapshot") return "ultimo snapshot account";
  return "telemetria";
}

function tickDecisionProgress() {
  const s = window.__dpState;
  const fill = document.getElementById("dp-fill");
  const track = document.getElementById("dp-track");
  if (!fill || !track) return;

  if (!s || !s.available) {
    fill.style.width = "0%";
    track.setAttribute("aria-valuenow", "0");
    setText("dp-timeframe", s?.message || "Timeframe non disponibile");
    setText("dp-remaining", "—");
    setText("dp-last", "Ultimo evento: —");
    return;
  }

  const tf = s.timeframe || "—";
  const basis = basisLabel(s.basis);
  setText("dp-timeframe", `Timeframe: ${tf} · riferimento: ${basis}`);

  if (s.last_event_ts == null) {
    fill.style.width = "0%";
    track.setAttribute("aria-valuenow", "0");
    setText("dp-remaining", "In attesa della prima barra processata...");
    setText("dp-last", "Ultimo evento: —");
    return;
  }

  const interval = Number(s.interval_seconds);
  if (!interval || interval <= 0) {
    fill.style.width = "0%";
    track.setAttribute("aria-valuenow", "0");
    setText("dp-remaining", "—");
    return;
  }

  const now = Date.now() / 1000;
  const elapsed = now - s.last_event_ts;
  const k = Math.floor(elapsed / interval);
  const phase = elapsed - k * interval;
  const progress = Math.min(1, Math.max(0, phase / interval));
  const nextTs = s.last_event_ts + (k + 1) * interval;
  const remaining = Math.max(0, nextTs - now);

  fill.style.width = `${(progress * 100).toFixed(1)}%`;
  track.setAttribute("aria-valuenow", String(Math.round(progress * 100)));
  setText("dp-remaining", `Stima: ${fmtDurationSeconds(remaining)} alla prossima valutazione`);
  const lastDate = new Date(s.last_event_ts * 1000).toLocaleString();
  setText("dp-last", `Ultimo aggiornamento: ${lastDate}`);
}

function startDecisionProgressTick() {
  if (decisionProgressTickStarted) return;
  decisionProgressTickStarted = true;
  setInterval(tickDecisionProgress, 250);
}

function renderTable(id, rows, mapper) {
  const tbody = document.querySelector(`#${id} tbody`);
  tbody.innerHTML = rows.map(mapper).join("");
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function upsertLineChart(chart, ctx, labels, data, label, color) {
  if (chart) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.data.datasets[0].label = label;
    chart.update();
    return chart;
  }
  return new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{ label, data, borderColor: color, backgroundColor: color, tension: 0.2 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#e2e8f0" } },
        zoom: {
          pan: {
            enabled: true,
            mode: "x",
            modifierKey: "shift"
          },
          zoom: {
            wheel: { enabled: true },
            pinch: { enabled: true },
            mode: "x"
          }
        }
      },
      scales: {
        x: { ticks: { color: "#94a3b8" } },
        y: { ticks: { color: "#94a3b8" } }
      }
    }
  });
}

async function refreshOverview() {
  const overview = await getJson("/api/overview");
  const mt5 = overview.mt5_account || {};
  const latestTraining = overview.latest_training || {};
  const model = overview.active_model || {};
  const activeSession = (overview.sessions || [])[0] || null;
  const latestDecision = overview.latest_decision || null;
  window.__dpState = overview.decision_progress || null;
  tickDecisionProgress();

  setText("kpi-balance", fmtNumber(mt5.balance));
  setText("kpi-equity", fmtNumber(mt5.equity));
  setText("kpi-profit", fmtNumber(mt5.profit));
  setText("kpi-positions", String((overview.mt5_positions || []).length));
  setText("kpi-model", model.version || "-");
  setText("kpi-training", latestTraining.model_version || latestTraining.status || "-");
  renderLatestDecision(latestDecision);

  if (activeSession) {
    activeRunInfo.textContent = `Run attivo: ${activeSession.run_mode} ${activeSession.symbol}/${activeSession.timeframe} - modello ${activeSession.model_version || "n/a"}`;
    if (!hasInitializedActiveRun) {
      if (activeSession.symbol) symbolInput.value = activeSession.symbol;
      if (activeSession.timeframe) timeframeInput.value = activeSession.timeframe;
      hasInitializedActiveRun = true;
    }
  } else {
    activeRunInfo.textContent = "Run attivo: nessuna sessione attiva";
  }

  renderTable("positions-table", overview.mt5_positions || [], (row) => `
    <tr>
      <td>${row.symbol ?? "-"}</td>
      <td>${row.type ?? "-"}</td>
      <td>${fmtNumber(row.volume)}</td>
      <td>${fmtNumber(row.price_open)}</td>
      <td>${fmtNumber(row.price_current)}</td>
      <td class="${Number(row.profit) >= 0 ? "positive" : "negative"}">${fmtNumber(row.profit)}</td>
      <td>${fmtNumber(row.sl)}</td>
      <td>${fmtNumber(row.tp)}</td>
    </tr>
  `);

  renderTable("sessions-table", overview.sessions || [], (row) => `
    <tr>
      <td>${row.session_id}</td>
      <td>${row.run_mode}</td>
      <td>${row.symbol ?? "-"}</td>
      <td>${row.timeframe ?? "-"}</td>
      <td>${row.ml_enabled ? "on" : "off"}</td>
      <td>${row.model_version ?? "-"}</td>
      <td>${row.status}</td>
    </tr>
  `);
}

async function refreshDecisions() {
  const decisions = await getJson("/api/decisions?limit=20");
  renderTable("decisions-table", decisions, (row) => `
    <tr>
      <td>${new Date(row.ts * 1000).toLocaleString()}</td>
      <td>${directionBadge(row.direction)}</td>
      <td>${fmtConfidence(row.aggregate_confidence)}</td>
      <td>${escapeHtml(decisionStatus(row))}</td>
      <td class="decision-cell">${escapeHtml(row.explanation || row.veto_reason || "-")}</td>
      <td>${row.model_version ?? "-"}</td>
    </tr>
  `);
}

async function refreshTraining() {
  const training = await getJson("/api/training/runs?limit=20");
  renderTable("training-table", training, (row) => `
    <tr>
      <td>${row.model_version ?? "-"}</td>
      <td>${row.symbol}</td>
      <td>${row.timeframe}</td>
      <td>${fmtNumber(row.best_test_accuracy)}</td>
      <td>${fmtNumber(row.mean_test_accuracy)}</td>
      <td>${row.bars_fetched ?? "-"}</td>
      <td>${row.status}</td>
    </tr>
  `);
}

async function refreshEquityCurve() {
  const rows = await getJson("/api/account/equity-curve?limit=200");
  const chronological = [...rows];
  equityChart = upsertLineChart(
    equityChart,
    equityCtx,
    chronological.map((row) => new Date(row.ts * 1000).toLocaleTimeString()),
    chronological.map((row) => row.equity),
    "Equity",
    "#38bdf8"
  );
}

async function refreshBars() {
  const symbol = encodeURIComponent(symbolInput.value || cfg.defaultSymbol);
  const timeframe = encodeURIComponent(timeframeInput.value || cfg.defaultTimeframe);
  const rows = await getJson(`/api/bars?symbol=${symbol}&timeframe=${timeframe}&limit=180`);
  barsChart = upsertLineChart(
    barsChart,
    barsCtx,
    rows.map((row) => new Date(row.ts).toLocaleTimeString()),
    rows.map((row) => row.close),
    `${symbolInput.value}/${timeframeInput.value} close`,
    "#a78bfa"
  );
}

async function refreshAll() {
  try {
    await Promise.all([
      refreshOverview(),
      refreshDecisions(),
      refreshTraining(),
      refreshEquityCurve(),
    ]);
    setText("last-refresh", `Aggiornato ${new Date().toLocaleTimeString()}`);
  } catch (error) {
    setText("last-refresh", `Errore refresh: ${error.message}`);
  }
}

document.getElementById("reload-bars").addEventListener("click", refreshBars);
resetBarsZoomButton.addEventListener("click", () => {
  if (barsChart) barsChart.resetZoom();
});

refreshAll().then(refreshBars);
setInterval(refreshAll, cfg.refreshSeconds * 1000);
startDecisionProgressTick();
