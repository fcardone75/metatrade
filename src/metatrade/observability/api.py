"""FastAPI app for the MetaTrade dashboard."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from metatrade.core.enums import Timeframe
from metatrade.market_data.config import MarketDataConfig
from metatrade.market_data.errors import BarStoreError
from metatrade.market_data.models import BarQuery
from metatrade.market_data.storage.duckdb_store import DuckDBBarStore
from metatrade.observability.config import ObservabilityConfig
from metatrade.observability.last_launch import load_last_launch
from metatrade.observability.mt5_runtime import MT5RuntimeReader
from metatrade.observability.runner_probe import probe_trader_processes
from metatrade.observability.store import TelemetryStore

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _build_decision_progress(
    *,
    sessions: list[dict[str, Any]],
    latest_decision: dict[str, Any] | None,
    latest_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    """Hints for the dashboard sidebar: one pipeline decision per closed bar of ``timeframe``."""
    active = sessions[0] if sessions else None
    tf_str: str | None = None
    if active and active.get("timeframe"):
        tf_str = str(active["timeframe"])
    if not tf_str and latest_decision and latest_decision.get("timeframe"):
        tf_str = str(latest_decision["timeframe"])
    if not tf_str and latest_snapshot and latest_snapshot.get("timeframe"):
        tf_str = str(latest_snapshot["timeframe"])

    if not tf_str:
        return {
            "available": False,
            "reason": "no_timeframe",
            "message": "Avvia una sessione paper o live per vedere l'avanzamento.",
            "timeframe": None,
            "interval_seconds": None,
            "last_event_ts": None,
            "basis": None,
        }

    try:
        tf = Timeframe(tf_str)
    except ValueError:
        return {
            "available": False,
            "reason": "bad_timeframe",
            "message": f"Timeframe sconosciuto: {tf_str}",
            "timeframe": tf_str,
            "interval_seconds": None,
            "last_event_ts": None,
            "basis": None,
        }

    interval = int(tf.seconds)
    last_ts: float | None = None
    basis: str | None = None
    if latest_decision and latest_decision.get("ts") is not None:
        last_ts = float(latest_decision["ts"])
        basis = "decision"
    elif latest_snapshot and latest_snapshot.get("ts") is not None:
        last_ts = float(latest_snapshot["ts"])
        basis = "snapshot"

    return {
        "available": True,
        "reason": None,
        "message": None,
        "timeframe": tf_str,
        "interval_seconds": interval,
        "last_event_ts": last_ts,
        "basis": basis,
    }


def create_app() -> FastAPI:
    obs_cfg = ObservabilityConfig.load()
    market_cfg = MarketDataConfig.load()

    app = FastAPI(title=obs_cfg.title)
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    telemetry = TelemetryStore.from_env()
    mt5 = MT5RuntimeReader()

    def runner_payload() -> dict[str, Any]:
        trader = probe_trader_processes()
        last_launch = load_last_launch()
        open_n = telemetry.count_open_sessions()
        return {
            "process_running": trader.running,
            "pids": list(trader.pids),
            "last_launch": last_launch,
            "can_restart": bool(last_launch) and not trader.running,
            "db_orphan": open_n > 0 and not trader.running,
        }

    def client_is_local(request: Request) -> bool:
        c = request.client
        if c is None:
            return True
        host = (c.host or "").lower()
        if host in ("127.0.0.1", "::1", "localhost"):
            return True
        return host.startswith("::ffff:127.0.0.1")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        active_sessions = telemetry.list_sessions(limit=1, active_only=True)
        active_session = active_sessions[0] if active_sessions else None
        return TEMPLATES.TemplateResponse(
            request,
            "dashboard.html",
            {
                "title": obs_cfg.title,
                "refresh_seconds": obs_cfg.refresh_seconds,
                "observability_port": obs_cfg.port,
                "default_symbol": (
                    active_session.get("symbol")
                    if active_session and active_session.get("symbol")
                    else (market_cfg.symbols[0] if market_cfg.symbols else "EURUSD")
                ),
                "default_timeframe": (
                    active_session.get("timeframe")
                    if active_session and active_session.get("timeframe")
                    else market_cfg.primary_timeframe.value
                ),
            },
        )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "title": obs_cfg.title,
            "refresh_seconds": obs_cfg.refresh_seconds,
            "time_utc": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/overview")
    async def overview() -> dict[str, Any]:
        sessions = telemetry.list_sessions(limit=5, active_only=True)
        latest_snapshot = telemetry.get_latest_account_snapshot()
        latest_decisions = telemetry.list_recent_decisions(limit=1)
        latest_training = telemetry.latest_training_run()
        models = telemetry.list_model_artifacts(limit=5)
        active_model = next((m for m in models if m.get("is_active")), models[0] if models else None)
        mt5_account = mt5.get_account()
        mt5_positions = mt5.get_open_positions()
        latest_decision = latest_decisions[0] if latest_decisions else None
        decision_progress = _build_decision_progress(
            sessions=sessions,
            latest_decision=latest_decision,
            latest_snapshot=latest_snapshot,
        )
        open_sessions_count = telemetry.count_open_sessions()
        return {
            "sessions": sessions,
            "open_sessions_count": open_sessions_count,
            "latest_snapshot": latest_snapshot,
            "latest_decision": latest_decision,
            "latest_training": latest_training,
            "active_model": active_model,
            "mt5_account": mt5_account,
            "mt5_positions": mt5_positions,
            "decision_progress": decision_progress,
            "refresh_seconds": obs_cfg.refresh_seconds,
            "runner": runner_payload(),
        }

    @app.get("/api/runner/status")
    async def runner_status() -> dict[str, Any]:
        """Stato processo trader (run_live/run_paper) + ultimo comando salvato."""
        return runner_payload()

    @app.post("/api/runner/restart")
    async def runner_restart(request: Request) -> dict[str, Any]:
        """Riavvia l'ultimo run_live/run_paper salvato in ``data/last_runner_launch.json``."""
        if not client_is_local(request):
            raise HTTPException(status_code=403, detail="Riavvio consentito solo da localhost")
        trader = probe_trader_processes()
        if trader.running:
            raise HTTPException(status_code=409, detail="Runner già in esecuzione")
        data = load_last_launch()
        if not data:
            raise HTTPException(
                status_code=404,
                detail="Nessun comando salvato. Avvia run_live o run_paper una volta dalla CLI.",
            )
        exe = str(data["executable"])
        script = str(data["script"])
        argv = [str(x) for x in (data.get("argv") or [])]
        cwd = data.get("cwd") or None
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_CONSOLE
        subprocess.Popen(
            [exe, script, *argv],
            cwd=cwd,
            creationflags=creationflags,
        )
        return {"ok": True, "detail": "Processo avviato in una nuova finestra console."}

    @app.post("/api/sessions/close-open")
    async def close_open_sessions(
        status: str = Query(default="interrupted", description="Nuovo status per le sessioni chiuse"),
    ) -> dict[str, Any]:
        """Chiude in DB tutte le sessioni ancora senza ``ended_at`` (es. dopo kill del processo)."""
        if status not in ("interrupted", "completed", "stopped"):
            raise HTTPException(status_code=400, detail="status non consentito")
        n = telemetry.close_open_sessions(status=status)
        return {"closed": n, "status": status}

    @app.get("/api/sessions")
    async def sessions(limit: int = 20, active_only: bool = False) -> list[dict[str, Any]]:
        return telemetry.list_sessions(limit=limit, active_only=active_only)

    @app.get("/api/account/equity-curve")
    async def equity_curve(limit: int = 200, session_id: str | None = None) -> list[dict[str, Any]]:
        rows = telemetry.list_account_snapshots(limit=limit, session_id=session_id)
        return list(reversed(rows))

    @app.get("/api/decisions")
    async def decisions(limit: int = 100, session_id: str | None = None) -> list[dict[str, Any]]:
        return telemetry.list_recent_decisions(limit=limit, session_id=session_id)

    @app.get("/api/orders")
    async def orders(limit: int = 100, session_id: str | None = None) -> list[dict[str, Any]]:
        return telemetry.list_order_events(limit=limit, session_id=session_id)

    @app.get("/api/training/runs")
    async def training_runs(limit: int = 20) -> list[dict[str, Any]]:
        return telemetry.list_training_runs(limit=limit)

    @app.get("/api/models")
    async def models(limit: int = 20) -> list[dict[str, Any]]:
        return telemetry.list_model_artifacts(limit=limit)

    @app.get("/api/mt5/account")
    async def mt5_account() -> dict[str, Any]:
        account = mt5.get_account()
        if account is None:
            raise HTTPException(status_code=503, detail="MT5 account not available")
        return account

    @app.get("/api/mt5/positions")
    async def mt5_positions() -> list[dict[str, Any]]:
        return mt5.get_open_positions()

    @app.get("/api/module-thresholds")
    async def module_thresholds() -> list[dict[str, Any]]:
        """Return the current adaptive threshold for every tracked module.

        Each entry contains:
          module_id      — stable module identifier
          threshold      — current confidence threshold in [0.45, 0.90]
          eval_count     — total number of evaluations since tracking started
          correct_count  — evaluations where market moved in signal direction
          mean_score     — EMA of market-accuracy scores (0.5 = neutral)
          accuracy_pct   — correct_count / eval_count × 100 (null if no evals)
          updated_at     — Unix timestamp of last update
        """
        rows = telemetry.list_module_thresholds()
        for row in rows:
            evals = row.get("eval_count") or 0
            correct = row.get("correct_count") or 0
            row["accuracy_pct"] = round(correct / evals * 100, 1) if evals > 0 else None
        return rows

    @app.get("/api/rule-reputations")
    async def rule_reputations() -> list[dict[str, Any]]:
        """Return the current reputation state for every exit rule.

        Each entry contains:
          rule_id        — stable rule identifier (e.g. "trailing_stop")
          symbol         — "*" for global, or a specific Forex pair
          weight         — current raw weight in [5, 95]
          eval_count     — number of closed trades used to update this rule
          mean_score     — EMA of contribution scores (0.5 = neutral)
          last_eval_ts   — Unix timestamp of the last update (0 if never)
        """
        return telemetry.list_rule_reputations()

    @app.get("/api/kill-switch")
    async def kill_switch_status() -> dict[str, Any]:
        """Return the current kill switch command state.

        level 0 = no active command (trading enabled).
        level 1 = TRADE_GATE, 2 = SESSION_GATE, 3 = EMERGENCY_HALT, 4 = HARD_KILL.
        """
        return telemetry.read_kill_command()

    @app.post("/api/kill-switch/activate")
    async def kill_switch_activate(
        level: int = Body(..., ge=1, le=4),
        reason: str = Body(default="Activated from dashboard"),
    ) -> dict[str, Any]:
        """Activate the kill switch at the requested level.

        The runner will pick this up within one bar.
        """
        telemetry.write_kill_command(level=level, reason=reason, activated_by="dashboard")
        return telemetry.read_kill_command()

    @app.post("/api/kill-switch/reset")
    async def kill_switch_reset() -> dict[str, Any]:
        """Reset the kill switch — re-enables trading on the next bar."""
        telemetry.write_kill_command(level=0, reason="Reset from dashboard", activated_by="dashboard")
        return telemetry.read_kill_command()

    @app.get("/api/bars")
    async def bars(
        symbol: str = Query(default=market_cfg.symbols[0] if market_cfg.symbols else "EURUSD"),
        timeframe: str = Query(default=market_cfg.primary_timeframe.value),
        limit: int = Query(default=200, ge=20, le=5000),
    ) -> list[dict[str, Any]]:
        tf = Timeframe(timeframe)
        try:
            store = DuckDBBarStore(db_path=market_cfg.duckdb_path, read_only=True)
            try:
                newest = store.get_latest_bar_time(symbol, tf)
                if newest is not None:
                    date_from = newest - timedelta(seconds=tf.seconds * max(limit * 2, 200))
                    query = BarQuery(
                        symbol=symbol,
                        timeframe=tf,
                        date_from=date_from,
                        date_to=newest + timedelta(seconds=tf.seconds),
                    )
                    rows = store.get_bars(query)
                    rows = rows[-limit:]
                    return [
                        {
                            "ts": bar.timestamp_utc.isoformat(),
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": float(bar.volume),
                        }
                        for bar in rows
                    ]
            finally:
                store.close()
        except BarStoreError:
            pass

        try:
            from metatrade.market_data.collectors.mt5_collector import MT5Collector

            collector = MT5Collector(store=None)
            collector.initialize(
                login=market_cfg.mt5_login,
                password=market_cfg.mt5_password,
                server=market_cfg.mt5_server,
                path=market_cfg.mt5_path,
                timeout=market_cfg.mt5_timeout_ms,
            )
            try:
                date_to = datetime.now(timezone.utc)
                date_from = date_to - timedelta(seconds=tf.seconds * max(limit + 10, 50))
                rows = collector.collect(symbol, tf, date_from, date_to)
                rows = rows[-limit:]
                return [
                    {
                        "ts": bar.timestamp_utc.isoformat(),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                    }
                    for bar in rows
                ]
            finally:
                collector.shutdown()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Market data unavailable: {exc}") from exc

    return app
