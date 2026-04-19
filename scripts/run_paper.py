"""Paper trading runner — live MT5 data, simulated fills, no real money.

Usage:
    python scripts/run_paper.py --symbol EURUSD --timeframe H1

    # Use a specific pre-trained model
    python scripts/run_paper.py --symbol EURUSD --model-version v20240601_1430

    # Increase position risk
    python scripts/run_paper.py --symbol EURUSD --max-risk-pct 0.02

The script:
  1. Loads the active ML model from disk (trained by scripts/train.py)
  2. Connects to MetaTrader 5 (must be open and logged in)
  3. Streams live bars on a polling loop (configurable interval)
  4. Passes each closed bar through the full signal pipeline
  5. Logs every signal, decision, and simulated order
  6. Logs a P&L summary on Ctrl+C

Requires: Windows + MetaTrader 5 terminal open and logged in.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metatrade.alerting.command_handler import CommandHandler
from metatrade.alerting.config import AlertConfig
from metatrade.alerting.telegram_alerter import TelegramAlerter
from metatrade.core.enums import Timeframe
from metatrade.core.log import configure_logging, get_logger
from metatrade.ml.config import MLConfig
from metatrade.ml.live_tracker import LiveAccuracyTracker
from metatrade.ml.model_watcher import ModelWatcher
from metatrade.ml.registry import ModelRegistry
from metatrade.ml.retrain_scheduler import RetrainScheduler
from metatrade.market_data.config import MarketDataConfig
from metatrade.runner.config import RunnerConfig
from metatrade.runner.module_config import ModuleConfig
from metatrade.runner.module_builder import build_modules
from metatrade.runner.paper_runner import PaperRunner
from metatrade.observability.last_launch import save_last_launch
from metatrade.observability.store import TelemetryStore

log = get_logger("run_paper")

_TIMEFRAME_MAP: dict[str, Timeframe] = {
    "M1": Timeframe.M1,
    "M5": Timeframe.M5,
    "M15": Timeframe.M15,
    "M30": Timeframe.M30,
    "H1": Timeframe.H1,
    "H4": Timeframe.H4,
    "D1": Timeframe.D1,
}

# Timeframe -> bar duration in seconds (for polling interval)
_TF_SECONDS: dict[Timeframe, int] = {
    Timeframe.M1: 60,
    Timeframe.M5: 300,
    Timeframe.M15: 900,
    Timeframe.M30: 1800,
    Timeframe.H1: 3600,
    Timeframe.H4: 14400,
    Timeframe.D1: 86400,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forex paper trading via MetaTrader 5")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--timeframe", default="H1", choices=list(_TIMEFRAME_MAP))
    p.add_argument("--warmup-bars", type=int, default=200, help="Bars to load on startup")
    p.add_argument("--poll-interval", type=float, default=0, help="Poll interval in seconds (0 = auto from timeframe)")
    p.add_argument("--max-risk-pct", type=float, default=0.01)
    p.add_argument("--model-dir", type=Path, default=Path("data/models"))
    p.add_argument("--model-version", default=None, help="Specific model version to use (default: active)")
    p.add_argument("--mt5-login", type=int, default=0)
    p.add_argument("--mt5-password", default="")
    p.add_argument("--mt5-server", default="")
    p.add_argument("--mt5-path", default="", help="Path to terminal64.exe if MT5 is not auto-detected")
    p.add_argument("--mt5-timeout-ms", type=int, default=60000, help="MT5 initialization timeout in milliseconds")
    p.add_argument("--no-ml", action="store_true", help="Disable ML module, use only technical indicators")
    # Module toggles
    p.add_argument("--no-news", action="store_true", help="Disable news/calendar filter")
    p.add_argument("--no-d1", action="store_true", help="Disable D1 multi-timeframe module (needs 500+ bars)")
    p.add_argument("--finnhub-key", default=None, help="Finnhub API key for live economic calendar")
    return p.parse_args()


def load_registry(args: argparse.Namespace) -> ModelRegistry | None:
    """Load model registry from disk. Returns None if no model found."""
    ml_cfg = MLConfig(model_registry_dir=str(args.model_dir))
    registry = ModelRegistry(config=ml_cfg, persist=True)

    if args.model_version:
        snap = registry.load_from_disk(args.symbol, args.model_version)
        if snap is None:
            log.error("model_not_found", version=args.model_version, model_dir=str(args.model_dir))
            sys.exit(1)
        log.info("model_loaded", version=snap.version, test_acc=snap.tags.get("best_test_acc", "?"))
    else:
        active = registry.get_active()
        if active is None:
            latest = sorted(args.model_dir.glob(f"{args.symbol}_v*.pkl"), reverse=True)
            if latest:
                version = latest[0].stem.removeprefix(f"{args.symbol}_")
                active = registry.load_from_disk(args.symbol, version)
            if args.no_ml:
                log.info("ml_disabled", msg="running with technical indicators only")
                return None
            log.warning("no_active_model", msg="no active model; run scripts/train.py first or pass --no-ml")
            return None
        log.info("active_model", version=active.version, test_acc=active.tags.get("best_test_acc", "?"))

    return registry


def make_module_cfg(args: argparse.Namespace) -> ModuleConfig:
    """Build a ModuleConfig from CLI flags."""
    return ModuleConfig(
        multi_tf_d1=not args.no_d1,
        news_calendar=not args.no_news,
        ml=not args.no_ml,
        finnhub_api_key=args.finnhub_key,
    )


def apply_mt5_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill MT5 connection args from .env when CLI values are omitted."""
    cfg = MarketDataConfig.load()
    if not args.mt5_login:
        args.mt5_login = cfg.mt5_login
    if not args.mt5_password:
        args.mt5_password = cfg.mt5_password
    if not args.mt5_server:
        args.mt5_server = cfg.mt5_server
    if not args.mt5_path:
        args.mt5_path = cfg.mt5_path
    if not args.mt5_timeout_ms:
        args.mt5_timeout_ms = cfg.mt5_timeout_ms
    return args


def build_ml_stack(
    args: argparse.Namespace,
    registry: ModelRegistry | None,
    telemetry: TelemetryStore,
) -> tuple[TelegramAlerter | None, LiveAccuracyTracker | None, ModelWatcher | None, RetrainScheduler | None, CommandHandler | None]:
    """Build alerter + ML lifecycle stack from env config.

    Returns (alerter, live_tracker, model_watcher, retrain_scheduler, command_handler).
    All components are None when their respective env flags are disabled.
    """
    alert_cfg = AlertConfig()
    alerter: TelegramAlerter | None = None
    if alert_cfg.bot_token and alert_cfg.chat_id and alert_cfg.enabled:
        alerter = TelegramAlerter(alert_cfg)
        log.info("telegram_alerts_enabled", chat_id=alert_cfg.chat_id)
    else:
        log.info("telegram_alerts_disabled")

    ml_cfg = MLConfig(model_registry_dir=str(args.model_dir))

    live_tracker: LiveAccuracyTracker | None = None
    tf_minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
    live_tracker = LiveAccuracyTracker(
        symbol=args.symbol,
        timeframe=args.timeframe,
        forward_bars=5,
        tf_minutes=tf_minutes.get(args.timeframe, 60),
        telemetry=telemetry,
        min_samples=ml_cfg.live_accuracy_min_samples,
    )

    model_watcher: ModelWatcher | None = None
    if ml_cfg.model_watcher_enabled and registry is not None:
        model_watcher = ModelWatcher(
            registry=registry,
            symbol=args.symbol,
            config=ml_cfg,
            alerter=alerter,
        )
        log.info("model_watcher_enabled", poll_sec=ml_cfg.model_watcher_poll_sec, min_holdout=f"{ml_cfg.candidate_min_holdout:.0%}")
    else:
        log.info("model_watcher_disabled")

    retrain_scheduler: RetrainScheduler | None = None
    if ml_cfg.retrain_enabled:
        train_args = [
            "--source", "mt5",
            "--symbol", args.symbol,
            "--timeframe", args.timeframe,
            "--model-dir", str(args.model_dir),
            "--holdout-fraction", "0.2",
        ]
        retrain_scheduler = RetrainScheduler(
            config=ml_cfg,
            train_args=train_args,
            alerter=alerter,
        )
        unit = f"every {ml_cfg.retrain_every_hours}h" if ml_cfg.retrain_trigger == "hours" else f"every {ml_cfg.retrain_every_bars} bars"
        log.info("retrain_scheduler_enabled", schedule=unit)
    else:
        log.info("retrain_scheduler_disabled")

    command_handler: CommandHandler | None = None
    if alerter is not None and alert_cfg.commands_enabled:
        command_handler = CommandHandler()
        log.info("telegram_commands_enabled", poll_sec=alert_cfg.poll_interval_sec)

    return alerter, live_tracker, model_watcher, retrain_scheduler, command_handler


def main() -> None:
    configure_logging()
    args = parse_args()
    args = apply_mt5_defaults(args)
    save_last_launch(mode="paper", script_path=Path(__file__).resolve())
    tf = _TIMEFRAME_MAP[args.timeframe]
    telemetry = TelemetryStore.from_env()
    session_id: str | None = None
    collector = None
    runner = None

    try:
        # ── 1. Connect to MT5 ─────────────────────────────────────────────────
        try:
            from metatrade.market_data.collectors.mt5_collector import MT5Collector
        except Exception as exc:
            log.error("mt5_import_error", error=str(exc))
            sys.exit(1)

        collector = MT5Collector(store=None)
        collector.initialize(
            login=args.mt5_login,
            password=args.mt5_password,
            server=args.mt5_server,
            path=args.mt5_path,
            timeout=args.mt5_timeout_ms,
        )

        # ── 2. Load warmup bars ───────────────────────────────────────────────
        tf_secs = _TF_SECONDS[tf]
        date_to = datetime.now(timezone.utc)
        date_from = date_to - timedelta(seconds=tf_secs * (args.warmup_bars + 10))

        log.info("loading_warmup_bars", symbol=args.symbol, timeframe=args.timeframe, bars=args.warmup_bars)
        bars = collector.collect(args.symbol, tf, date_from, date_to)
        log.info("warmup_bars_loaded", count=len(bars))

        # ── 3. Build runner ───────────────────────────────────────────────────
        registry = load_registry(args)
        active_model = registry.get_active().version if registry and registry.get_active() else None
        session_id = telemetry.start_session(
            run_mode="PAPER",
            symbol=args.symbol,
            timeframe=args.timeframe,
            ml_enabled=not args.no_ml,
            model_version=active_model,
            details={"mt5_server": args.mt5_server},
        )
        module_cfg = make_module_cfg(args)
        modules = build_modules(args.timeframe.lower(), module_cfg=module_cfg, registry=registry)

        log.info("building_ml_stack")
        alerter, live_tracker, model_watcher, retrain_scheduler, command_handler = (
            build_ml_stack(args, registry, telemetry)
        )

        runner_cfg = RunnerConfig(
            symbol=args.symbol,
            max_risk_pct=args.max_risk_pct,
            min_signals=1 if args.no_ml else 2,
        )
        runner = PaperRunner(
            config=runner_cfg,
            modules=modules,
            telemetry=telemetry,
            session_id=session_id,
            timeframe=args.timeframe,
            alerter=alerter,
            live_tracker=live_tracker,
            model_watcher=model_watcher,
            retrain_scheduler=retrain_scheduler,
            command_handler=command_handler,
        )

        # Pre-fill buffer with warmup bars (without triggering signals)
        for bar in bars:
            runner._bar_buffer.append(bar)
        if len(runner._bar_buffer) > 500:
            runner._bar_buffer = runner._bar_buffer[-500:]

        log.info(
            "paper_trading_started",
            symbol=args.symbol,
            timeframe=args.timeframe,
            modules=[m.__class__.__name__ for m in modules],
            balance=str(runner.broker.get_account().balance),
        )

        poll_interval = args.poll_interval if args.poll_interval > 0 else tf_secs
        last_bar_time: datetime | None = bars[-1].timestamp_utc if bars else None

        # Graceful shutdown on Ctrl+C
        _running = [True]
        def _stop(sig, frame):
            _running[0] = False
        signal.signal(signal.SIGINT, _stop)

        # ── 4. Live polling loop ──────────────────────────────────────────────
        while _running[0]:
            date_to = datetime.now(timezone.utc)
            date_from = date_to - timedelta(seconds=tf_secs * 5)

            try:
                new_bars = collector.collect(args.symbol, tf, date_from, date_to)
            except Exception as exc:
                log.warning("mt5_fetch_error", error=str(exc))
                time.sleep(5)
                continue

            # Only process bars newer than the last seen one
            if last_bar_time is not None:
                new_bars = [b for b in new_bars if b.timestamp_utc > last_bar_time]

            for bar in new_bars:
                last_bar_time = bar.timestamp_utc
                ts_str = bar.timestamp_utc.strftime("%Y-%m-%d %H:%M")

                decision = runner.on_bar(bar)

                if decision is None:
                    log.info("bar_hold", ts=ts_str)
                elif hasattr(decision, "approved") and decision.approved:
                    ps = decision.position_size
                    log.info(
                        "bar_trade",
                        ts=ts_str,
                        side=decision.side.value,
                        lot=float(ps.lot_size),
                        sl=str(ps.stop_loss_price),
                        tp=str(ps.take_profit_price),
                    )
                else:
                    veto_code = decision.veto.veto_code if decision.veto else "?"
                    log.info("bar_vetoed", ts=ts_str, veto_code=veto_code)

            time.sleep(poll_interval)

        # ── 5. Summary ────────────────────────────────────────────────────────
        if runner is not None:
            account_final = runner.broker.get_account()
            runner.stop(
                n_trades=runner.stats.trades_executed,
                total_pnl=account_final.equity - account_final.balance,
            )
        account = runner.broker.get_account()
        stats = runner.stats
        log.info(
            "paper_session_summary",
            bars_processed=stats.bars_processed,
            trades_executed=stats.trades_executed,
            trades_vetoed=stats.trades_vetoed,
            final_balance=str(account.balance),
            final_equity=str(account.equity),
        )
    finally:
        if session_id is not None:
            summary = None
            if runner is not None:
                summary = {
                    "bars_processed": runner.stats.bars_processed,
                    "trades_executed": runner.stats.trades_executed,
                    "trades_vetoed": runner.stats.trades_vetoed,
                }
            telemetry.finish_session(session_id, status="completed", details=summary)
        if collector is not None:
            collector.shutdown()


if __name__ == "__main__":
    main()
