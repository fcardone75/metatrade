"""Base runner — shared logic across backtest, paper, and live modes.

The BaseRunner implements the core signal→decision→execution pipeline:
1. Receive bar data (via process_bar())
2. Run each technical/ML module to produce AnalysisSignals
3. Feed signals to the consensus engine
4. Evaluate risk (kill switch, limits, position sizing)
5. Return the RiskDecision (approved or vetoed) to the caller

Each runner subclass calls process_bar() with appropriate account data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from metatrade.alerting.command_handler import CommandHandler
from metatrade.alerting.command_receiver import TelegramCommandReceiver
from metatrade.alerting.telegram_alerter import TelegramAlerter
from metatrade.consensus.adaptive_threshold import AdaptiveThresholdManager
from metatrade.core.utils.session import is_in_session
from metatrade.ml.live_tracker import LiveAccuracyTracker
from metatrade.ml.model_watcher import ModelWatcher
from metatrade.ml.retrain_scheduler import RetrainScheduler
from metatrade.consensus.config import ConsensusConfig
from metatrade.consensus.engine import ConsensusEngine
from metatrade.consensus.market_accuracy_tracker import MarketAccuracyTracker
from metatrade.core.contracts.account import AccountState
from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.risk import RiskDecision, RiskVeto
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import ConsensusMode, OrderSide, RunMode, SignalDirection
from metatrade.core.log import get_logger
from metatrade.observability.store import TelemetryStore
from metatrade.risk.config import RiskConfig
from metatrade.risk.manager import RiskManager
from metatrade.runner.config import RunnerConfig
from metatrade.technical_analysis.interface import ITechnicalModule

log = get_logger(__name__)


@dataclass
class RunStats:
    """Accumulated statistics for a runner session."""

    bars_processed: int = 0
    signals_generated: int = 0
    trades_attempted: int = 0
    trades_executed: int = 0
    trades_vetoed: int = 0
    kill_switch_activations: int = 0
    weight_updates: int = 0           # signals evaluated against market


def _format_adaptive_progress(data: dict, fold_data: dict | None = None) -> str:
    """Format adaptive_progress.json into a human-readable Telegram message.

    Args:
        data:      Contents of adaptive_progress.json.
        fold_data: Optional contents of training_progress.json (fold-level detail
                   for the currently running attempt).
    """
    status_icon = {
        "running": "🔁",
        "completed": "✅",
        "exhausted": "⚠️",
        "failed": "❌",
    }.get(data.get("status", ""), "❓")

    symbol = data.get("symbol", "?")
    tf = data.get("timeframe", "?")
    target = data.get("target")
    current = data.get("current_model_acc")
    done = data.get("attempts_done", 0)
    max_att = data.get("max_attempts", "?")
    best_h = data.get("best_holdout")
    updated = data.get("updated_at_utc", "")[:16].replace("T", " ")

    # Elapsed since training started
    elapsed_str = ""
    started_raw = data.get("started_at_utc")
    if started_raw:
        try:
            from datetime import datetime as _dt, timezone as _tz
            started = _dt.fromisoformat(started_raw)
            elapsed_td = _dt.now(_tz.utc) - started
            total_sec = int(elapsed_td.total_seconds())
            days, rem = divmod(total_sec, 86400)
            hours, rem = divmod(rem, 3600)
            mins, secs = divmod(rem, 60)
            parts = []
            if days:
                parts.append(f"{days}g")
            if hours or days:
                parts.append(f"{hours}h")
            parts.append(f"{mins}m {secs}s")
            elapsed_str = " ".join(parts)
        except Exception:
            elapsed_str = ""

    lines = [
        f"{status_icon} <b>{symbol} {tf}</b>  [{data.get('status', '?').upper()}]",
        f"  Target:   {target:.2%}" if target else "",
        f"  Modello attuale: {current:.2%}" if current else "  Nessun modello precedente",
        f"  Tentativi: {done}/{max_att}",
        f"  Miglior holdout: {best_h:.2%}" if best_h else "  Miglior holdout: —",
        f"  In corso da: {elapsed_str}" if elapsed_str else "",
        f"  Aggiornato: {updated} UTC",
    ]

    # ── Fold-level progress for the current attempt ───────────────────────────
    if fold_data and fold_data.get("status") == "walk_forward":
        folds_done = fold_data.get("folds_completed", 0)
        folds_est = fold_data.get("estimated_folds_upper_bound", "?")
        pct = fold_data.get("progress_pct_approx")
        last_acc = fold_data.get("test_accuracy_last")
        elapsed = fold_data.get("elapsed_sec")
        eta = fold_data.get("eta_sec")

        pct_str = f"  {pct:.1f}%" if pct is not None else ""
        acc_str = f"  last acc={last_acc:.2%}" if last_acc is not None else ""

        if elapsed is not None and elapsed >= 60:
            elapsed_str = f"{elapsed/60:.1f}m"
        elif elapsed is not None:
            elapsed_str = f"{elapsed:.0f}s"
        else:
            elapsed_str = ""

        if eta is not None and eta >= 60:
            eta_str = f"ETA ~{eta/60:.0f}m"
        elif eta is not None:
            eta_str = f"ETA ~{eta:.0f}s"
        else:
            eta_str = ""

        fold_line = f"  Fold corrente: {folds_done}/{folds_est}{pct_str}{acc_str}"
        time_line = "  " + "  ".join(filter(None, [elapsed_str and f"Elapsed: {elapsed_str}", eta_str]))
        lines.append("")
        lines.append("<b>Tentativo in corso:</b>")
        lines.append(fold_line)
        if time_line.strip():
            lines.append(time_line)

    attempts = data.get("attempts", [])
    if attempts:
        lines.append("")
        lines.append("<b>Dettaglio tentativi:</b>")
        for a in attempts:
            att_n = a.get("attempt", "?")
            h = a.get("holdout")
            h_str = f"{h:.2%}" if h is not None else "n/a"
            ok = "✓" if a.get("success") else "✗"
            depth = a.get("max_depth", "?")
            itr = a.get("max_iter", "?")
            dur = a.get("duration_sec")
            dur_str = f"{dur:.0f}s" if dur else ""
            lines.append(
                f"  [{ok}] #{att_n} — holdout={h_str}  depth={depth}  iter={itr}  {dur_str}"
            )

    return "\n".join(l for l in lines if l != "")


class BaseRunner:
    """Core orchestration logic shared by all runner modes.

    Subclasses call `process_bar()` to run the full pipeline for one tick.

    Args:
        config:  RunnerConfig for this session.
        modules: List of ITechnicalModule instances to query each bar.
    """

    def __init__(
        self,
        config: RunnerConfig,
        modules: list[ITechnicalModule],
        auto_weight: bool = True,
        weight_forward_bars: int = 5,
        alerter: TelegramAlerter | None = None,
        telemetry: TelemetryStore | None = None,
        session_id: str | None = None,
        timeframe: str | None = None,
        intermarket_engine: object | None = None,  # IntermarketEngine | None
        exit_profile_generator: object | None = None,  # ExitProfileCandidateGenerator | None
        exit_profile_selector: object | None = None,   # ExitProfileSelector | None
        live_tracker: LiveAccuracyTracker | None = None,
        model_watcher: ModelWatcher | None = None,
        retrain_scheduler: RetrainScheduler | None = None,
        command_handler: CommandHandler | None = None,
        run_mode: RunMode = RunMode.PAPER,
        symbol: str | None = None,
    ) -> None:
        """
        Args:
            config:                 RunnerConfig for this session.
            modules:                List of ITechnicalModule instances.
            auto_weight:            If True (default), use DYNAMIC_VOTE consensus
                                    and automatically update module weights based
                                    on how well each module predicted market
                                    direction over the next ``weight_forward_bars``
                                    bars.  Weights are proportional to accuracy:
                                    a large error reduces the weight more than a
                                    small one.
            weight_forward_bars:    Bars ahead used to evaluate signal accuracy.
                                    Must match the ML labelling horizon (default 5).
            alerter:                Optional TelegramAlerter for operational alerts.
                                    None (default) = silent, no alerts sent.
            intermarket_engine:     Optional ``IntermarketEngine`` for cross-pair
                                    correlation and exposure control.  When set,
                                    it is evaluated between the consensus step and
                                    the risk manager.  ``None`` = disabled
                                    (default, fully backward-compatible).
            exit_profile_generator: Optional ``ExitProfileCandidateGenerator``.
                                    When set together with ``exit_profile_selector``,
                                    steps 6-7 of the pipeline run after risk
                                    approval and override the default Chandelier
                                    SL/TP with the selected exit profile.
                                    ``None`` = disabled (default, fully
                                    backward-compatible).
            exit_profile_selector:  Optional ``ExitProfileSelector``.  Required
                                    together with ``exit_profile_generator`` to
                                    enable exit profile selection.
            live_tracker:           Optional ``LiveAccuracyTracker`` that records
                                    ML predictions bar-by-bar and evaluates them
                                    once ``forward_bars`` bars have elapsed.
            model_watcher:          Optional ``ModelWatcher`` that polls the
                                    registry directory for better trained models
                                    and switches the active model automatically.
            retrain_scheduler:      Optional ``RetrainScheduler`` that launches
                                    ``scripts/train.py`` in the background at the
                                    configured interval (hours or bars).
        """
        self._config = config
        self._modules = modules

        consensus_cfg = ConsensusConfig(
            mode=ConsensusMode.DYNAMIC_VOTE if auto_weight else ConsensusMode.SIMPLE_VOTE,
            threshold=config.consensus_threshold,
            min_signals=config.min_signals,
        )
        self._consensus = ConsensusEngine(consensus_cfg)

        # Adaptive threshold manager — per-module confidence thresholds that
        # self-adjust based on market accuracy (lower when right, raise when wrong).
        # Persisted to the TelemetryStore so thresholds survive restarts.
        self._threshold_mgr = AdaptiveThresholdManager(
            store=telemetry,
        )

        # Market accuracy tracker — evaluates each signal N bars later and
        # (1) updates the dynamic weight of the module in the consensus engine,
        # (2) updates the adaptive threshold for the module.
        def _on_eval(module_id: str, score: float) -> None:
            self._consensus.update_performance_scored(module_id, score)
            self._threshold_mgr.on_eval(module_id, score)

        if auto_weight:
            self._tracker: MarketAccuracyTracker | None = MarketAccuracyTracker(
                update_callback=_on_eval,
                forward_bars=weight_forward_bars,
            )
        else:
            self._tracker = None

        risk_cfg = RiskConfig(
            max_risk_pct=config.max_risk_pct,
            daily_loss_limit_pct=config.daily_loss_limit_pct,
            auto_kill_drawdown_pct=config.auto_kill_drawdown_pct,
            max_open_positions=config.max_open_positions,
        )
        self._risk = RiskManager(risk_cfg)
        self._stats = RunStats()

        # ── Cooldown counter ──────────────────────────────────────────────────
        # Counts down the remaining bars to skip after an approved trade.
        self._cooldown_remaining: int = 0

        # ── Drawdown recovery ─────────────────────────────────────────────────
        # Tracks the highest equity seen so far to measure peak-to-trough DD.
        self._peak_equity: Decimal = Decimal("0")

        # ── Daily circuit breaker ─────────────────────────────────────────────
        # Resets each day; blocks new entries once intraday loss >= limit.
        self._today_date: date | None = None
        self._session_start_equity: Decimal = Decimal("0")

        # ── Alerter ───────────────────────────────────────────────────────────
        self._alerter: TelegramAlerter | None = alerter
        self._telemetry = telemetry
        self._session_id = session_id
        self._timeframe = timeframe

        # ── Intermarket engine (optional) ──────────────────────────────────────
        # Type hint kept as object to avoid hard import at module level.
        self._intermarket = intermarket_engine

        # ── Exit profile selection (optional, Steps 6-7) ───────────────────────
        # Both generator and selector must be set together.  When only one is
        # provided the feature is effectively disabled (no hard error).
        self._exit_profile_generator = exit_profile_generator
        self._exit_profile_selector = exit_profile_selector

        # ── ML lifecycle components (all optional) ─────────────────────────────
        self._live_tracker = live_tracker
        self._model_watcher = model_watcher
        self._retrain_scheduler = retrain_scheduler

        # ── Command handler and receiver (optional) ────────────────────────────
        self._command_handler = command_handler
        self._command_receiver: TelegramCommandReceiver | None = None
        if command_handler is not None and alerter is not None:
            self._command_receiver = TelegramCommandReceiver(
                config=alerter._cfg,
                handler=command_handler,
            )
            self._register_commands()
            # Start immediately so commands work even before the first bar
            # (e.g. weekends, market closed, startup phase).
            self._command_receiver.start()

        # ── Session / pause state ──────────────────────────────────────────────
        self._paused: bool = False
        self._run_mode: RunMode = run_mode
        self._symbol: str = symbol or config.symbol
        self._session_started: bool = False
        self._session_start_time: datetime | None = None
        self._last_account: AccountState | None = None

        # ── Daily summary deduplication ────────────────────────────────────────
        # Store the last date for which the daily summary was sent so we
        # fire it exactly once per UTC day at daily_summary_hour_utc.
        self._daily_summary_sent_date: date | None = None

    @property
    def stats(self) -> RunStats:
        return self._stats

    @property
    def risk_manager(self) -> RiskManager:
        return self._risk

    @property
    def accuracy_tracker(self) -> MarketAccuracyTracker | None:
        """The market accuracy tracker, or None if auto_weight=False."""
        return self._tracker

    @property
    def threshold_manager(self) -> AdaptiveThresholdManager:
        """The adaptive threshold manager for per-module confidence thresholds."""
        return self._threshold_mgr

    def get_module_weight(self, module_id: str) -> float | None:
        """Current dynamic weight for a module (None if auto_weight=False)."""
        return self._consensus.get_module_weight(module_id)

    def system_confidence(self) -> float:
        """Aggregate system confidence in [0.0, 1.0].

        Derived from the average dynamic module weight, normalized from the
        [5, 95] weight range to [0, 1].  Returns 0.5 (neutral) when no
        modules are loaded or when auto_weight is disabled (weights are None).
        """
        weights = [
            w
            for m in self._modules
            if (w := self.get_module_weight(m.module_id)) is not None
        ]
        if not weights:
            return 0.5
        avg = sum(weights) / len(weights)
        # 5 → 0.0, 50 → 0.5, 95 → 1.0
        return max(0.0, min(1.0, (avg - 5.0) / 90.0))

    # ── Session lifecycle ──────────────────────────────────────────────────────

    def start(self, balance: Decimal | None = None, model_version: str | None = None) -> None:
        """Notify session start and begin background services."""
        if self._command_receiver is not None:
            self._command_receiver.start()
        if self._alerter is not None and not self._session_started:
            acc_balance = balance or (self._last_account.balance if self._last_account else Decimal("0"))
            sess_filter = None
            start_h = self._config.session_filter_utc_start
            end_h = self._config.session_filter_utc_end
            if start_h is not None and end_h is not None:
                sess_filter = f"{start_h:02d}:00–{end_h:02d}:00 UTC"
            self._alerter.alert_session_start(
                mode=self._run_mode.value,
                symbol=self._symbol,
                timeframe=self._timeframe or "",
                balance=acc_balance,
                model_version=model_version,
                session_filter=sess_filter,
            )
        self._session_started = True
        self._session_start_time = datetime.now(UTC)

    def stop(
        self,
        n_trades: int | None = None,
        total_pnl: float | Decimal = Decimal("0"),
        win_rate: float | None = None,
    ) -> None:
        """Notify session end and stop background services."""
        if self._command_receiver is not None:
            self._command_receiver.stop()
        if self._alerter is not None:
            balance = self._last_account.balance if self._last_account else Decimal("0")
            duration_min = None
            if self._session_start_time is not None:
                elapsed = datetime.now(UTC) - self._session_start_time
                duration_min = int(elapsed.total_seconds() / 60)
            self._alerter.alert_session_end(
                mode=self._run_mode.value,
                symbol=self._symbol,
                n_trades=n_trades if n_trades is not None else self._stats.trades_executed,
                total_pnl=total_pnl,
                win_rate=win_rate,
                balance=balance,
                duration_min=duration_min,
            )
        self._session_started = False

    # ── Pause / resume ─────────────────────────────────────────────────────────

    def pause(self) -> None:
        """Pause new trade entries (exits continue unaffected)."""
        self._paused = True
        log.info("runner_paused", symbol=self._symbol)

    def resume(self) -> None:
        """Resume new trade entries after a pause."""
        self._paused = False
        log.info("runner_resumed", symbol=self._symbol)

    # ── Command registration ───────────────────────────────────────────────────

    def _register_commands(self) -> None:
        """Register runner command callbacks in the CommandHandler."""
        if self._command_handler is None:
            return
        ch = self._command_handler
        ch.register("/status",    self._cmd_status)
        ch.register("/positions", self._cmd_positions)
        ch.register("/balance",   self._cmd_balance)
        ch.register("/model",     self._cmd_model)
        ch.register("/accuracy",  self._cmd_accuracy)
        ch.register("/daily",     self._cmd_daily)
        ch.register("/retrain",      self._cmd_retrain)
        ch.register("/training",     self._cmd_training)
        ch.register("/stoptraining", self._cmd_stop_training)
        ch.register("/pause",        self._cmd_pause)
        ch.register("/resume",    self._cmd_resume)
        ch.register("/stop",      self._cmd_stop)

    # ── Command callbacks ──────────────────────────────────────────────────────

    def _cmd_status(self, _args: str) -> str:
        acc = self._last_account
        if acc is None:
            return "🤖 MetaTrade — nessun dato disponibile (nessuna barra processata)."
        uptime_str = ""
        if self._session_start_time is not None:
            elapsed = datetime.now(UTC) - self._session_start_time
            h, rem = divmod(int(elapsed.total_seconds()), 3600)
            m = rem // 60
            uptime_str = f"\n  Uptime:    {h}h {m}m"
        paused_str = "  ⏸ IN PAUSA\n" if self._paused else ""
        mv = self._model_watcher.state.current_version if self._model_watcher else None
        model_str = f"\n  Modello:   {mv}" if mv else ""
        retrain_str = ""
        if self._retrain_scheduler is not None:
            if self._retrain_scheduler.is_training:
                retrain_str = "\n  Training:  🔁 in corso"
            else:
                nxt = self._retrain_scheduler.next_slot
                retrain_str = f"\n  Prossimo training: {nxt.strftime('%H:%M UTC')}"
        return (
            f"🤖 <b>MetaTrade — {self._symbol} {self._timeframe or ''}</b>  "
            f"[{self._run_mode.value.upper()}]\n"
            f"{paused_str}"
            f"  Balance:   ${float(acc.balance):.2f}\n"
            f"  Equity:    ${float(acc.equity):.2f}\n"
            f"  Posizioni: {acc.open_positions_count}\n"
            f"  Trade:     {self._stats.trades_executed}"
            f"{uptime_str}"
            f"{model_str}"
            f"{retrain_str}"
        )

    def _cmd_positions(self, _args: str) -> str:
        acc = self._last_account
        n = acc.open_positions_count if acc else 0
        return f"📋 <b>Posizioni aperte ({n})</b>\n\nDettagli disponibili tramite il broker."

    def _cmd_balance(self, _args: str) -> str:
        acc = self._last_account
        if acc is None:
            return "⚠️ Nessun dato account disponibile."
        dd_str = ""
        if self._peak_equity > Decimal("0"):
            dd_pct = float((self._peak_equity - acc.equity) / self._peak_equity) * 100
            dd_str = f"\n  DD curr:   {dd_pct:.2f}%"
        return (
            f"💰 <b>Account snapshot</b>\n"
            f"  Balance:   ${float(acc.balance):.2f}\n"
            f"  Equity:    ${float(acc.equity):.2f}\n"
            f"  Free margin: ${float(acc.free_margin):.2f}\n"
            f"  Used margin: ${float(acc.used_margin):.2f}"
            f"{dd_str}"
        )

    def _cmd_model(self, _args: str) -> str:
        if self._model_watcher is None:
            return "🧠 ModelWatcher non configurato."
        state = self._model_watcher.state
        mv = state.current_version or "nessuno"
        h_str = f"{state.current_holdout:.1%}" if state.current_holdout else "n/a"
        live_str = ""
        if self._live_tracker is not None and state.current_version:
            stats = self._live_tracker.live_accuracy_stats(state.current_version)
            reliable = "✅" if stats.is_reliable else "⏳"
            live_str = (
                f"\n  Live acc:  {stats.accuracy:.1%}  "
                f"({stats.n_correct}/{stats.n_evaluated}) {reliable}"
            )
        return (
            f"🧠 <b>Modello attivo</b> — {self._symbol}\n"
            f"  Versione:  {mv}\n"
            f"  Holdout:   {h_str}"
            f"{live_str}"
        )

    def _cmd_accuracy(self, _args: str) -> str:
        if self._live_tracker is None:
            return "📊 LiveAccuracyTracker non configurato."
        mv = (
            self._model_watcher.state.current_version
            if self._model_watcher else None
        )
        if not mv:
            return "📊 Nessun modello attivo."
        stats = self._live_tracker.live_accuracy_stats(mv)
        reliable = "✅ affidabile" if stats.is_reliable else f"⏳ ({stats.n_evaluated}/{stats.min_samples})"
        return (
            f"📊 <b>Accuracy live</b> — {self._symbol}\n"
            f"  Modello:   {mv}\n"
            f"  Valutati:  {stats.n_evaluated}  {reliable}\n"
            f"  Corretti:  {stats.n_correct}\n"
            f"  Pending:   {stats.n_pending}\n"
            f"  Accuracy:  {stats.accuracy:.1%}"
        )

    def _cmd_daily(self, _args: str) -> str:
        if self._alerter is None:
            return "⚠️ Alerter non configurato."
        if self._last_account is None:
            return "⚠️ Nessun dato disponibile."
        mv = (
            self._model_watcher.state.current_version
            if self._model_watcher else None
        )
        live_acc = None
        live_n = None
        if self._live_tracker is not None and mv:
            stats = self._live_tracker.live_accuracy_stats(mv)
            live_acc = stats.accuracy
            live_n = stats.n_evaluated
        self._alerter.alert_daily_summary(
            symbol=self._symbol,
            timeframe=self._timeframe or "",
            n_trades=self._stats.trades_executed,
            total_pnl=Decimal("0"),
            pnl_pips=None,
            win_rate=0.0,
            max_dd=Decimal("0"),
            balance=self._last_account.balance,
            model_version=mv,
            live_accuracy=live_acc,
            live_samples=live_n,
        )
        return "📊 Riepilogo giornaliero inviato."

    def _cmd_retrain(self, _args: str) -> str:
        if self._retrain_scheduler is None:
            return "⚠️ RetrainScheduler non configurato (ML_RETRAIN_ENABLED=false)."
        if self._retrain_scheduler.is_training:
            return "🔁 Training già in corso — attendi che finisca."
        launched = self._retrain_scheduler.trigger_now()
        if launched:
            return "🚀 Training avviato manualmente."
        return "❌ Impossibile avviare il training (script non trovato o errore di sistema)."

    def _cmd_training(self, _args: str) -> str:
        if self._retrain_scheduler is None:
            return "⚠️ Retraining non abilitato (ML_RETRAIN_ENABLED=false)."

        # Find adaptive_progress.json in the model registry dir
        try:
            from metatrade.ml.config import MLConfig as _MLCfg
            model_dir = Path(_MLCfg().model_registry_dir)
        except Exception:
            model_dir = Path("data/models")

        progress_file = model_dir / "adaptive_progress.json"

        if not self._retrain_scheduler.is_training:
            nxt = self._retrain_scheduler.next_slot
            idle_msg = f"⏸ Non in training.\nProssimo slot: {nxt.strftime('%Y-%m-%d %H:%M UTC')}"
            if not progress_file.exists():
                return idle_msg
            # Show results of last completed run
            try:
                data = json.loads(progress_file.read_text(encoding="utf-8"))
                return idle_msg + "\n\n" + _format_adaptive_progress(data)
            except Exception:
                return idle_msg

        # Training in corso — leggi progress file
        if not progress_file.exists():
            return "🔁 Training in corso (nessun dato di progresso ancora disponibile)."
        try:
            data = json.loads(progress_file.read_text(encoding="utf-8"))
            fold_data: dict | None = None
            fold_file = model_dir / "training_progress.json"
            if fold_file.exists():
                try:
                    fold_data = json.loads(fold_file.read_text(encoding="utf-8"))
                except Exception:
                    fold_data = None
            return "🔁 <b>Training in corso</b>\n\n" + _format_adaptive_progress(data, fold_data)
        except Exception as exc:
            return f"🔁 Training in corso (errore lettura progress: {exc})."

    def _cmd_stop_training(self, _args: str) -> str:
        if self._retrain_scheduler is None:
            return "⚠️ Retraining non abilitato (ML_RETRAIN_ENABLED=false)."
        if not self._retrain_scheduler.is_training:
            return "ℹ️ Nessun training in corso."
        stopped = self._retrain_scheduler.stop_training()
        if stopped:
            log.warning("training_stopped_via_telegram")
            return "🛑 <b>Training interrotto.</b>\nIl processo è stato terminato. Usa /retrain per riavviarlo."
        return "❌ Impossibile fermare il training (processo non trovato o già terminato)."

    def _cmd_pause(self, _args: str) -> str:
        self.pause()
        from datetime import datetime as _dt
        now = _dt.now(UTC).strftime("%H:%M UTC")
        return f"⏸ Runner in pausa — nuove entry bloccate.\nOra: {now}\nUsa /resume per riprendere."

    def _cmd_resume(self, _args: str) -> str:
        self.resume()
        from datetime import datetime as _dt
        now = _dt.now(UTC).strftime("%H:%M UTC")
        return f"▶️ Runner ripreso — nuove entry abilitate.\nOra: {now}"

    def _cmd_stop(self, _args: str) -> str:
        self.pause()
        log.warning("runner_stop_via_telegram")
        return (
            "🛑 <b>Stop confermato.</b>\n"
            "Il runner è in pausa. Chiudi manualmente il processo per uno stop completo."
        )

    def _compute_dynamic_position_cap(self, account: AccountState) -> int:
        """Compute the effective max-open-positions cap for this bar.

        Combines three signals to produce a value in [0, config.max_open_positions]:

        1. **Margin health** — ``free_margin / equity`` ratio.
           Critical  (< 5 %)  → 0 (block).
           Tight     (< 20 %) → ×0.33.
           Moderate  (< 50 %) → ×0.67.
           Healthy   (≥ 50 %) → ×1.0.

        2. **Drawdown** — peak-to-trough vs ``drawdown_recovery_threshold_pct``.
           Below threshold          → ×1.0.
           threshold … 2×threshold  → ×0.5.
           ≥ 2×threshold            → 0 (block).

        3. **System confidence** — average module weight normalised to [0, 1].
           < 0.30 (declining)  → ×0.33.
           < 0.45 (neutral)    → ×0.67.
           ≥ 0.45 (confident)  → ×1.0.

        Returns:
            0  → block all new entries (margin critical or severe drawdown).
            1…max_open_positions → dynamic effective limit.
        """
        base = self._config.max_open_positions

        # ── 1. Margin health ─────────────────────────────────────────────────
        if account.equity > Decimal("0"):
            margin_ratio = float(account.free_margin / account.equity)
        else:
            margin_ratio = 0.0

        if margin_ratio < 0.05:
            log.warning(
                "dynamic_cap_margin_critical",
                margin_ratio=round(margin_ratio, 4),
                free_margin=float(account.free_margin),
                equity=float(account.equity),
            )
            return 0
        elif margin_ratio < 0.20:
            margin_factor = 0.33
        elif margin_ratio < 0.50:
            margin_factor = 0.67
        else:
            margin_factor = 1.0

        # ── 2. Drawdown ──────────────────────────────────────────────────────
        threshold = self._config.drawdown_recovery_threshold_pct
        if self._peak_equity > Decimal("0"):
            dd_pct = float(
                (self._peak_equity - account.equity) / self._peak_equity
            )
        else:
            dd_pct = 0.0

        if dd_pct >= 2 * threshold:
            log.warning(
                "dynamic_cap_severe_drawdown",
                dd_pct=round(dd_pct * 100, 2),
                threshold_pct=round(threshold * 100, 2),
            )
            return 0
        elif dd_pct >= threshold:
            dd_factor = 0.5
        else:
            dd_factor = 1.0

        # ── 3. System confidence ─────────────────────────────────────────────
        conf = self.system_confidence()
        if conf < 0.30:
            conf_factor = 0.33
        elif conf < 0.45:
            conf_factor = 0.67
        else:
            conf_factor = 1.0

        effective = max(1, int(base * margin_factor * dd_factor * conf_factor))
        cap = min(effective, base)

        if cap < base:
            log.debug(
                "dynamic_cap_reduced",
                base=base,
                cap=cap,
                margin_factor=round(margin_factor, 2),
                dd_factor=round(dd_factor, 2),
                conf_factor=round(conf_factor, 2),
                confidence=round(conf, 3),
            )

        return cap

    # ── Kill switch DB sync ───────────────────────────────────────────────────

    def _sync_kill_switch_from_db(self) -> None:
        """Read the shared kill_switch_command from SQLite and sync in-memory state.

        Called once per bar so that a dashboard (or CLI) signal is picked up
        within one candle.  If level > 0, escalates the in-memory kill switch;
        if level == 0 and the switch is active, resets it.
        """
        if self._telemetry is None:
            return
        try:
            cmd = self._telemetry.read_kill_command()
        except Exception as exc:
            log.warning("kill_switch_db_sync_failed", error=str(exc))
            return

        level = int(cmd.get("level", 0))
        reason = str(cmd.get("reason", "") or "Kill switch activated via dashboard")
        activated_by = str(cmd.get("activated_by", "") or "dashboard")

        from metatrade.core.enums import KillSwitchLevel
        if level > 0:
            try:
                ks_level = KillSwitchLevel(level)
            except ValueError:
                return
            if ks_level > self._risk.kill_switch.level:
                self._risk.kill_switch.activate(
                    ks_level,
                    reason=reason,
                    activated_by=activated_by,
                )
                if self._alerter is not None:
                    self._alerter.alert_kill_switch(
                        level=str(ks_level.name),
                        reason=reason,
                        activated_by=activated_by,
                    )
        elif self._risk.kill_switch.is_active():
            try:
                self._risk.kill_switch.reset(activated_by=activated_by)
            except Exception:
                # HARD_KILL requires force_reset; that must be done deliberately
                pass

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def process_bar(
        self,
        bars: list[Bar],
        account_balance: Decimal,
        account_equity: Decimal,
        free_margin: Decimal,
        timestamp_utc: datetime | None = None,
        open_positions: int = 0,
        current_spread_pips: float | None = None,
        run_mode: RunMode = RunMode.PAPER,
    ) -> RiskDecision | None:
        """Run the full pipeline for one bar tick.

        Args:
            bars:                 Recent bars (oldest first, newest last).
            account_balance:      Current account balance.
            account_equity:       Balance + unrealized PnL.
            free_margin:          Capital available for new positions.
            timestamp_utc:        UTC timestamp; defaults to now.
            open_positions:       Number of currently open positions.
            current_spread_pips:  Current spread in pips (for spread check).
            run_mode:             Execution context (PAPER, LIVE, BACKTEST).

        Returns:
            RiskDecision (approved=True or approved=False with veto) if consensus
            produced an actionable signal. None if consensus is HOLD.
        """
        if timestamp_utc is None:
            timestamp_utc = datetime.now(UTC)

        # Sync kill switch from shared DB (cross-process signal from dashboard).
        self._sync_kill_switch_from_db()

        self._stats.bars_processed += 1

        # ── Background tasks (run every bar, before all guards) ──────────────
        # These tasks are intentionally placed before session/cooldown guards so
        # they always advance regardless of whether a new entry is allowed.
        _current_bar = bars[-1]
        if self._retrain_scheduler is not None:
            self._retrain_scheduler.on_bar(_current_bar)
        if self._live_tracker is not None:
            self._live_tracker.evaluate_pending(_current_bar)
        if self._model_watcher is not None:
            _live_stats = None
            if self._live_tracker is not None:
                _mv = self._model_watcher.state.current_version
                if _mv:
                    _live_stats = self._live_tracker.live_accuracy_stats(_mv)
            self._model_watcher.on_bar(_current_bar, open_positions, _live_stats)

        used_margin = (
            account_balance - free_margin
            if account_balance >= free_margin
            else Decimal("0")
        )
        account = AccountState(
            balance=account_balance,
            equity=account_equity,
            free_margin=free_margin,
            used_margin=used_margin,
            timestamp_utc=timestamp_utc,
            open_positions_count=open_positions,
            run_mode=run_mode,
        )
        self._last_account = account
        self._record_account_snapshot(account)

        # ── Auto session start (first bar) ────────────────────────────────────
        if not self._session_started:
            self.start(balance=account_balance)

        # ── Daily summary trigger ─────────────────────────────────────────────
        if self._alerter is not None:
            cfg_hour = getattr(self._alerter._cfg, "daily_summary_hour_utc", None)
            if (
                cfg_hour is not None
                and timestamp_utc.hour == cfg_hour
                and self._daily_summary_sent_date != timestamp_utc.date()
            ):
                self._daily_summary_sent_date = timestamp_utc.date()
                mv = (
                    self._model_watcher.state.current_version
                    if self._model_watcher else None
                )
                live_acc = None
                live_n = None
                if self._live_tracker is not None and mv:
                    _s = self._live_tracker.live_accuracy_stats(mv)
                    live_acc = _s.accuracy
                    live_n = _s.n_evaluated
                self._alerter.alert_daily_summary(
                    symbol=self._symbol,
                    timeframe=self._timeframe or "",
                    n_trades=self._stats.trades_executed,
                    total_pnl=Decimal("0"),
                    pnl_pips=None,
                    win_rate=0.0,
                    max_dd=Decimal("0"),
                    balance=account_balance,
                    model_version=mv,
                    live_accuracy=live_acc,
                    live_samples=live_n,
                )

        # Compute the dynamic position cap once per bar so the risk manager
        # can use margin health, drawdown, and system confidence together.
        dynamic_cap = self._compute_dynamic_position_cap(account)

        # ── Guard: spread filter ──────────────────────────────────────────────
        # Block new entries when the broker spread is too wide.
        max_spread = self._config.max_spread_pips
        if (
            max_spread > 0
            and current_spread_pips is not None
            and current_spread_pips > max_spread
        ):
            log.debug(
                "spread_filter: %.2f pips > max %.2f — skipping bar",
                current_spread_pips,
                max_spread,
            )
            return None

        # ── Guard: daily circuit breaker (absolute USD) ───────────────────────
        # Resets the daily reference equity at the start of each new calendar day.
        # Once intraday losses exceed daily_loss_limit_usd, all new entries are blocked.
        bar_date = timestamp_utc.date()
        if self._today_date != bar_date:
            self._today_date = bar_date
            self._session_start_equity = account_equity
        daily_limit = Decimal(str(self._config.daily_loss_limit_usd))
        if daily_limit > Decimal("0"):
            daily_loss = self._session_start_equity - account_equity
            if daily_loss >= daily_limit:
                log.debug(
                    "daily_circuit_breaker: loss=%.2f >= limit=%.2f — skipping bar",
                    float(daily_loss),
                    float(daily_limit),
                )
                if self._alerter is not None:
                    self._alerter.alert_daily_loss_limit(
                        loss=daily_loss,
                        limit=daily_limit,
                        balance=account_balance,
                    )
                return None

        # ── Guard: signal cooldown ────────────────────────────────────────────
        # After any approved trade, stay out of the market for N bars.
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            log.debug(
                "signal_cooldown: %d bars remaining — skipping bar",
                self._cooldown_remaining + 1,
            )
            return None

        # ── Guard: session filter ─────────────────────────────────────────────
        # Block new entries outside the configured trading session.
        # Exits and stop-losses are handled by the exit engine / broker and are
        # never subject to this check — only new trade entries are blocked here.
        _sess_start = self._config.session_filter_utc_start
        _sess_end = self._config.session_filter_utc_end
        if _sess_start is not None and _sess_end is not None:
            if not is_in_session(timestamp_utc, _sess_start, _sess_end):
                log.debug(
                    "session_gate: hour=%d outside [%d, %d) — skipping bar",
                    timestamp_utc.hour,
                    _sess_start,
                    _sess_end,
                )
                return None

        # ── Guard: manual pause ───────────────────────────────────────────────
        # /pause command blocks new entries; exits continue unaffected.
        if self._paused:
            log.debug("runner_paused: skipping new entry for this bar")
            return None

        current_close = bars[-1].close

        # ── Step 1a: Evaluate past signals against current price ──────────────
        # Must run BEFORE collecting new signals so that weights are already
        # updated when the consensus engine votes on the current bar.
        if self._tracker is not None:
            evaluations = self._tracker.on_bar(current_close, timestamp_utc)
            self._stats.weight_updates += len(evaluations)

        # ── Step 1b: Collect signals ──────────────────────────────────────────
        signals: list[AnalysisSignal] = []
        for module in self._modules:
            if len(bars) < module.min_bars:
                continue
            try:
                sig = module.analyse(bars, timestamp_utc)
                signals.append(sig)
                self._stats.signals_generated += 1
            except Exception as exc:
                log.warning(
                    "Module %s raised during analyse(): %s",
                    module.module_id,
                    exc,
                )
                if self._alerter is not None:
                    self._alerter.alert_error(
                        source=f"module:{module.module_id}",
                        error=str(exc),
                    )

        if not signals:
            return None

        # ── Step 1b+: Record ML module predictions for live accuracy tracking ──
        # Runs after signal collection, before any filtering.  Records every
        # BUY/SELL prediction from ml_* modules so the LiveAccuracyTracker can
        # evaluate directional accuracy forward_bars bars later.
        if self._live_tracker is not None:
            for _sig in signals:
                if (
                    _sig.module_id.startswith("ml_")
                    and _sig.direction != SignalDirection.HOLD
                ):
                    _direction_int = 1 if _sig.direction == SignalDirection.BUY else -1
                    _mv = str(_sig.metadata.get("model_version", "unknown"))
                    self._live_tracker.record_prediction(
                        _current_bar, _direction_int, float(_sig.confidence), _mv
                    )

        # ── Step 1c: Record ALL signals for future threshold/weight evaluation ─
        # We record signals BEFORE filtering so that even excluded signals are
        # evaluated — this is what drives the threshold to recover when a module
        # that was filtered out would have been right.
        if self._tracker is not None:
            self._tracker.record_signals(signals, current_close, timestamp_utc)

        # ── Step 1d: Apply per-module adaptive confidence threshold filter ────
        # Signals whose confidence is below their module's current threshold are
        # excluded from the consensus vote (but still evaluated in Step 1c above).
        signals = self._threshold_mgr.filter_signals(signals)

        if not signals:
            return None

        # ── Step 2: Consensus ─────────────────────────────────────────────────
        consensus_result = self._consensus.evaluate(
            symbol=self._config.symbol,
            signals=signals,
            timestamp_utc=timestamp_utc,
        )

        if not consensus_result.is_actionable:
            self._record_decision(signals, consensus_result, None, run_mode)
            return None

        direction = consensus_result.final_direction
        if direction == SignalDirection.BUY:
            side = OrderSide.BUY
        elif direction == SignalDirection.SELL:
            side = OrderSide.SELL
        else:
            return None

        # ── Step 3b: Intermarket evaluation ──────────────────────────────────
        # Inserted between consensus and risk.  When no engine is configured
        # (the common single-symbol case), this block is a no-op.
        im_risk_mult: Decimal | None = None
        if self._intermarket is not None:
            try:
                from metatrade.core.contracts.position import Position as _Position
                from metatrade.intermarket.engine import IntermarketEngine
                if isinstance(self._intermarket, IntermarketEngine):
                    # Collect open positions from the broker/paper state if
                    # available.  BaseRunner does not own positions directly,
                    # so we pass an empty list; subclasses may override via
                    # _get_open_positions() if they need richer integration.
                    open_positions: list[_Position] = self._get_open_positions()
                    # Estimate a nominal lot size for exposure-delta computation.
                    # We use 0.01 lots as a placeholder — actual sizing happens
                    # in the risk manager; the intermarket engine only needs the
                    # direction and a representative magnitude.
                    _nominal_lot = Decimal("0.01")
                    im_decision = self._intermarket.evaluate(
                        symbol=self._config.symbol,
                        side=side,
                        lot_size=_nominal_lot,
                        open_positions=open_positions,
                        timestamp_utc=timestamp_utc,
                        session_id=self._session_id,
                    )
                    if not im_decision.approved:
                        self._stats.trades_vetoed += 1
                        log.info(
                            "intermarket_veto",
                            symbol=self._config.symbol,
                            side=side.value,
                            reason=im_decision.reason,
                        )
                        vetoed = RiskDecision(
                            approved=False,
                            symbol=self._config.symbol,
                            side=side,
                            timestamp_utc=timestamp_utc,
                            veto=RiskVeto(
                                reason=im_decision.reason,
                                veto_code="INTERMARKET_VETO",
                                timestamp_utc=timestamp_utc,
                                context={
                                    "correlated_positions": list(
                                        im_decision.correlated_positions
                                    ),
                                    "warnings": list(im_decision.warnings),
                                },
                            ),
                        )
                        self._record_decision(signals, consensus_result, vetoed, run_mode)
                        return vetoed
                    im_risk_mult = im_decision.risk_multiplier
            except Exception as exc:
                # Never let intermarket evaluation break the trading pipeline.
                log.warning("intermarket_eval_error", error=str(exc))

        # ── Step 3c: Drawdown-recovery tracking ───────────────────────────────
        # Keep a running peak-equity and detect when we're in recovery mode.
        if account_equity > self._peak_equity:
            self._peak_equity = account_equity

        in_recovery = False
        if self._peak_equity > Decimal("0"):
            dd_pct = float(
                (self._peak_equity - account_equity) / self._peak_equity
            )
            if dd_pct >= self._config.drawdown_recovery_threshold_pct:
                in_recovery = True
                if self._alerter is not None:
                    self._alerter.alert_drawdown(
                        equity=account_equity,
                        peak=self._peak_equity,
                        dd_pct=dd_pct,
                        recovery_mode=True,
                    )
                log.debug(
                    "drawdown_recovery: equity=%.2f peak=%.2f dd=%.2f%% — "
                    "scaling risk by %.2f",
                    float(account_equity),
                    float(self._peak_equity),
                    dd_pct * 100,
                    self._config.drawdown_recovery_risk_mult,
                )

        # ── Step 4: Risk evaluation ───────────────────────────────────────────
        self._stats.trades_attempted += 1

        entry_price = bars[-1].close
        sl_price = self._estimate_sl(bars, entry_price, side)
        current_atr = self._compute_current_atr(bars)

        decision = self._risk.evaluate(
            symbol=self._config.symbol,
            side=side,
            entry_price=entry_price,
            sl_price=sl_price,
            account=account,
            timestamp_utc=timestamp_utc,
            spread_pips=current_spread_pips,
            current_atr=current_atr,
            intermarket_risk_mult=im_risk_mult,
            max_positions_override=dynamic_cap,
        )

        # ── Step 4b: Apply drawdown-recovery lot-size reduction ──────────────
        if decision.approved and in_recovery and decision.position_size is not None:
            mult = Decimal(str(self._config.drawdown_recovery_risk_mult))
            scaled_ps = replace(
                decision.position_size,
                lot_size=decision.position_size.lot_size * mult,
            )
            decision = replace(decision, position_size=scaled_ps)

        # ── Steps 6-7: Exit profile selection ────────────────────────────────
        # Runs only when both generator and selector are configured.
        # Overrides the default Chandelier SL/TP computed in Step 4 with the
        # exit profile chosen by the selector (ML or deterministic fallback).
        # The lot size is recomputed to preserve the configured risk_pct given
        # the new SL distance.
        if decision.approved and self._exit_profile_generator is not None and self._exit_profile_selector is not None:
            decision = self._apply_exit_profile(
                decision=decision,
                bars=bars,
                account=account,
                timestamp_utc=timestamp_utc,
                current_spread_pips=current_spread_pips,
            )

        # ── Step 4c: Update stats, alerter, and cooldown ─────────────────────
        if decision.approved:
            self._stats.trades_executed += 1
            # Fire operational alert if alerter is configured
            if self._alerter is not None and decision.position_size is not None:
                ps = decision.position_size
                self._alerter.alert_trade_opened(
                    symbol=self._config.symbol,
                    side=side.value,
                    lot_size=float(ps.lot_size),
                    entry=float(entry_price),
                    sl=float(ps.stop_loss_price),
                    tp=float(ps.take_profit_price) if ps.take_profit_price else None,
                )
            # Activate cooldown — prevents re-entry for N bars after a trade.
            if self._config.signal_cooldown_bars > 0:
                self._cooldown_remaining = self._config.signal_cooldown_bars
        else:
            self._stats.trades_vetoed += 1
            if (
                decision.veto is not None
                and "KILL_SWITCH" in decision.veto.veto_code
            ):
                self._stats.kill_switch_activations += 1
                if self._alerter is not None:
                    self._alerter.alert_kill_switch(
                        level=decision.veto.veto_code,
                        reason=decision.veto.reason,
                        activated_by="risk_manager",
                    )

        self._record_decision(signals, consensus_result, decision, run_mode)
        return decision

    def _record_account_snapshot(self, account: AccountState) -> None:
        if self._telemetry is None or self._session_id is None:
            return
        try:
            self._telemetry.record_account_snapshot(
                session_id=self._session_id,
                symbol=self._config.symbol,
                timeframe=self._timeframe,
                account=account,
                details={"bars_processed": self._stats.bars_processed},
            )
        except Exception as exc:
            log.warning("account_snapshot_telemetry_failed: %s", exc)

    def _record_decision(
        self,
        signals: list[AnalysisSignal],
        consensus_result,
        decision: RiskDecision | None,
        run_mode: RunMode,
    ) -> None:
        if self._telemetry is None or self._session_id is None:
            return
        try:
            self._telemetry.record_decision(
                session_id=self._session_id,
                symbol=self._config.symbol,
                timeframe=self._timeframe,
                run_mode=run_mode.value,
                signals=signals,
                consensus_result=consensus_result,
                decision=decision,
            )
        except Exception as exc:
            log.warning("decision_telemetry_failed: %s", exc)

    def _get_open_positions(self) -> list:
        """Return currently open positions for intermarket exposure accounting.

        The base implementation returns an empty list, which means the
        intermarket engine operates with no existing-position context —
        only the current candidate trade's currency delta is evaluated.

        Subclasses that maintain a position registry (e.g. PaperRunner)
        can override this to return their actual open-position list.
        This is a deliberate extension point, not a required override.
        """
        return []

    def _estimate_sl(
        self,
        bars: list[Bar],
        entry_price: Decimal,
        side: OrderSide,
    ) -> Decimal:
        """Estimate a stop-loss price using the Chandelier Exit method.

        SL = entry ± ATR(chandelier_atr_period) × chandelier_atr_mult

        The Chandelier Exit uses actual volatility to set the stop distance,
        adapting to current market conditions rather than a fixed pip value.

        Falls back to a 20-pip default (0.0020) when there are insufficient
        bars to compute ATR.
        """
        period = self._config.chandelier_atr_period
        mult = Decimal(str(self._config.chandelier_atr_mult))
        fallback_distance = Decimal("0.0020")

        if len(bars) >= period + 1:
            try:
                from metatrade.technical_analysis.indicators.atr import atr as compute_atr
                highs  = [b.high  for b in bars]
                lows   = [b.low   for b in bars]
                closes = [b.close for b in bars]
                atr_vals = compute_atr(highs, lows, closes, period)
                curr_atr = atr_vals[-1]
                if curr_atr > Decimal("0"):
                    distance = curr_atr * mult
                    if side == OrderSide.BUY:
                        return entry_price - distance
                    return entry_price + distance
            except Exception as exc:
                log.debug("chandelier_sl fallback: %s", exc)

        # Fallback: fixed 20-pip distance
        if side == OrderSide.BUY:
            return entry_price - fallback_distance
        return entry_price + fallback_distance

    def _apply_exit_profile(
        self,
        decision: RiskDecision,
        bars: list[Bar],
        account: AccountState,
        timestamp_utc: datetime,
        current_spread_pips: float | None,
    ) -> RiskDecision:
        """Apply exit profile selection (Steps 6-7) to an approved decision.

        Generates exit profile candidates for the current context, selects the
        best one via the configured selector, then recomputes the lot size to
        preserve the configured risk fraction with the new SL distance.

        If no valid candidate can be selected, the original decision is returned
        unchanged — the Chandelier-derived SL/TP from Step 4 remains in effect.

        Args:
            decision:            Approved RiskDecision from Step 4/5.
            bars:                Recent bar history.
            account:             Current AccountState.
            timestamp_utc:       Bar timestamp.
            current_spread_pips: Spread in pips (may be None).

        Returns:
            Updated RiskDecision with the selected exit profile's SL/TP and
            recomputed lot size, or the original decision when selection fails.
        """
        try:
            from metatrade.ml.exit_profile_candidate_generator import (
                ExitProfileCandidateGenerator,
            )
            from metatrade.ml.exit_profile_contracts import ExitProfileContext
            from metatrade.ml.exit_profile_selector import ExitProfileSelector

            if not isinstance(self._exit_profile_generator, ExitProfileCandidateGenerator):
                return decision
            if not isinstance(self._exit_profile_selector, ExitProfileSelector):
                return decision

            assert decision.position_size is not None  # guarded by decision.approved

            entry_price = bars[-1].close
            current_atr = self._compute_current_atr(bars)
            if current_atr is None:
                return decision  # not enough bars to compute ATR

            # Determine pip_digits from symbol name (JPY pairs = 2, others = 4)
            symbol = self._config.symbol
            pip_digits = 2 if symbol.upper().endswith("JPY") else 4

            start = max(0, len(bars) - 51)
            recent_highs = tuple(b.high for b in bars[start:])
            recent_lows  = tuple(b.low  for b in bars[start:])

            ctx = ExitProfileContext(
                symbol=symbol,
                side=decision.side,
                entry_price=entry_price,
                atr=current_atr,
                spread_pips=float(current_spread_pips or 0.0),
                account_balance=account.balance,
                timestamp_utc=timestamp_utc,
                pip_digits=pip_digits,
                recent_highs=recent_highs,
                recent_lows=recent_lows,
            )

            candidates = self._exit_profile_generator.generate(ctx)
            if not candidates:
                log.debug("exit_profile_no_candidates", symbol=symbol)
                return decision

            selected = self._exit_profile_selector.select(candidates, ctx)
            if selected is None:
                log.debug("exit_profile_selector_returned_none", symbol=symbol)
                return decision

            profile = selected.candidate

            # ── Enforce minimum SL pip distance ──────────────────────────────
            # Exit-profile candidates can produce very tight SLs (e.g. 1-3 pip
            # swing levels on M1).  A 2-pip SL at 1% risk yields 2+ lots on a
            # $4k account, which then gets clamped by the broker and creates
            # volume / margin errors.  Widening the SL here prevents over-sizing
            # BEFORE the lot calculation, which is the only safe place to fix it.
            min_sl_pips_val = Decimal(str(self._config.min_sl_pips))
            if min_sl_pips_val > 0 and profile.sl_pips < min_sl_pips_val:
                pip_size = Decimal("10") ** -pip_digits
                new_sl_dist = min_sl_pips_val * pip_size
                if decision.side == OrderSide.BUY:
                    adjusted_sl_price = entry_price - new_sl_dist
                else:
                    adjusted_sl_price = entry_price + new_sl_dist
                # Rescale TP proportionally if one was set
                if profile.tp_price is not None and profile.tp_pips is not None:
                    rr = profile.risk_reward
                    new_tp_dist = new_sl_dist * rr
                    if decision.side == OrderSide.BUY:
                        adjusted_tp_price = entry_price + new_tp_dist
                    else:
                        adjusted_tp_price = entry_price - new_tp_dist
                    new_tp_pips = min_sl_pips_val * rr
                else:
                    adjusted_tp_price = profile.tp_price
                    new_tp_pips = profile.tp_pips
                log.warning(
                    "exit_profile_sl_widened_to_min",
                    symbol=symbol,
                    profile_id=profile.profile_id,
                    original_sl_pips=str(profile.sl_pips),
                    min_sl_pips=str(min_sl_pips_val),
                    adjusted_sl=str(adjusted_sl_price),
                )
                profile = replace(
                    profile,
                    sl_price=adjusted_sl_price,
                    sl_pips=min_sl_pips_val,
                    tp_price=adjusted_tp_price,
                    tp_pips=new_tp_pips,
                )

            # Recompute lot size so the configured risk_pct is preserved
            # with the newly selected SL distance.
            sizing_capital = max(account.equity, account.balance)
            try:
                new_ps = self._risk.recompute_sizing(
                    balance=sizing_capital,
                    entry_price=entry_price,
                    sl_price=profile.sl_price,
                    side=decision.side,
                )
            except ValueError as exc:
                log.warning(
                    "exit_profile_recompute_sizing_failed",
                    error=str(exc),
                    profile_id=profile.profile_id,
                )
                return decision

            # Attach the selected TP.
            # Trailing profiles have tp_price=None (ExitEngine manages exit
            # dynamically).  Fall back to the standard TP from the original
            # PositionSizer result so that MT5 holds a safety-net order: if the
            # ExitEngine ever stops or crashes, the broker will still close the
            # position at the configured risk:reward level.
            fallback_tp = decision.position_size.take_profit_price if decision.position_size else None
            effective_tp = profile.tp_price if profile.tp_price is not None else fallback_tp
            new_ps = replace(new_ps, take_profit_price=effective_tp)

            log.info(
                "exit_profile_selected",
                symbol=symbol,
                profile_id=profile.profile_id,
                exit_mode=profile.exit_mode,
                sl_pips=str(profile.sl_pips),
                tp_pips=str(profile.tp_pips) if profile.tp_pips else "none (fallback to standard)",
                rr=str(profile.risk_reward),
                confidence=selected.confidence,
                method=selected.method,
                lot_size=str(new_ps.lot_size),
                tp_is_fallback=profile.tp_price is None,
            )

            return replace(decision, position_size=new_ps)

        except Exception as exc:
            # Never let exit profile selection break the trading pipeline.
            log.warning("exit_profile_apply_failed", error=str(exc))
            return decision

    def _compute_current_atr(self, bars: list[Bar]) -> Decimal | None:
        """Return the most recent ATR value, or None if insufficient data.

        Used to supply vol-scaled position sizing with the current market
        volatility without recomputing ATR a second time.
        """
        period = self._config.chandelier_atr_period
        if len(bars) < period + 1:
            return None
        try:
            from metatrade.technical_analysis.indicators.atr import atr as compute_atr
            highs  = [b.high  for b in bars]
            lows   = [b.low   for b in bars]
            closes = [b.close for b in bars]
            atr_vals = compute_atr(highs, lows, closes, period)
            val = atr_vals[-1]
            return val if val > Decimal("0") else None
        except Exception as exc:
            log.debug("_compute_current_atr failed: %s", exc)
            return None
