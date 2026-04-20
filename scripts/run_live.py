"""Live trading runner - real broker, real fills, real money.

WARNING: This script places REAL ORDERS on a LIVE account.
    Test thoroughly on a DEMO account using scripts/run_paper.py first.

Usage:
    python scripts/run_live.py --symbol EURUSD --timeframe H1

    # Demo account
    python scripts/run_live.py --symbol EURUSD --mt5-login 12345 \\
        --mt5-password "..." --mt5-server "BrokerName-Demo"

Safety checks built in:
  - Risk per trade capped at max_risk_pct (default 1%)
  - Daily loss limit kills new orders for the rest of the session
  - Drawdown kill switch triggers a hard stop at auto_kill_drawdown_pct
  - All orders tagged with magic_number for easy identification in MT5

Requires: Windows + MetaTrader 5 terminal open and logged in.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metatrade.alerting.command_handler import CommandHandler
from metatrade.alerting.config import AlertConfig
from metatrade.alerting.telegram_alerter import TelegramAlerter
from metatrade.core.enums import RunMode, Timeframe
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
from metatrade.runner.live_runner import LiveRunner
from metatrade.observability.last_launch import save_last_launch
from metatrade.observability.store import TelemetryStore

log = get_logger("run_live")

_TIMEFRAME_MAP: dict[str, Timeframe] = {
    "M1": Timeframe.M1,
    "M5": Timeframe.M5,
    "M15": Timeframe.M15,
    "M30": Timeframe.M30,
    "H1": Timeframe.H1,
    "H4": Timeframe.H4,
    "D1": Timeframe.D1,
}

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
    p = argparse.ArgumentParser(
        description="Forex LIVE trading via MetaTrader 5 - REAL MONEY"
    )
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--timeframe", default="H1", choices=list(_TIMEFRAME_MAP))
    p.add_argument("--warmup-bars", type=int, default=200, help="Bars to preload on startup")
    p.add_argument("--poll-interval", type=float, default=0, help="Seconds between MT5 polls (0 = auto)")
    # Risk parameters (None = use RUNNER_* from .env via apply_runner_defaults)
    p.add_argument("--max-risk-pct", type=float, default=None, help="Max risk per trade (.env RUNNER_MAX_RISK_PCT if omitted)")
    p.add_argument("--daily-loss-limit-pct", type=float, default=None, help="Daily loss limit (.env RUNNER_DAILY_LOSS_LIMIT_PCT if omitted)")
    p.add_argument("--kill-drawdown-pct", type=float, default=None, help="Hard kill drawdown (.env RUNNER_AUTO_KILL_DRAWDOWN_PCT if omitted)")
    p.add_argument("--max-open-positions", type=int, default=None, help="Max concurrent positions (.env RUNNER_MAX_OPEN_POSITIONS if omitted)")
    # MT5 connection
    p.add_argument("--mt5-login", type=int, default=0)
    p.add_argument("--mt5-password", default="")
    p.add_argument("--mt5-server", default="")
    p.add_argument("--mt5-path", default="", help="Path to terminal64.exe if MT5 is not auto-detected")
    p.add_argument("--mt5-timeout-ms", type=int, default=60000, help="MT5 initialization timeout in milliseconds")
    p.add_argument("--magic-number", type=int, default=None, help="MT5 magic number (.env RUNNER_MAGIC_NUMBER if omitted)")
    p.add_argument(
        "--min-stop-pips",
        type=float,
        default=10.0,
        help="Minimum SL/TP distance from entry in pips (default 10). "
             "Prevents immediate stop-out due to tight ATR on short timeframes.",
    )
    # Model
    p.add_argument("--model-dir", type=Path, default=Path("data/models"))
    p.add_argument("--model-version", default=None, help="Specific model version (default: active)")
    p.add_argument("--no-ml", action="store_true", help="Technical indicators only (no ML)")
    # Module toggles
    p.add_argument("--no-news", action="store_true", help="Disable news/calendar filter")
    p.add_argument("--no-d1", action="store_true", help="Disable D1 multi-timeframe module (needs 500+ bars)")
    p.add_argument("--finnhub-key", default=None, help="Finnhub API key for live economic calendar")
    p.add_argument(
        "--no-exit-profile",
        action="store_true",
        help="Disable exit-profile candidate pipeline (use default Chandelier SL/TP only).",
    )
    # Position monitoring
    p.add_argument(
        "--position-monitor-interval",
        type=float,
        default=5.0,
        help="Seconds between near-real-time position checks (trailing SL/TP updates, exits). "
             "0 = tick mode: evaluate on every new MT5 tick (~50 ms polling). "
             "Negative = disable position monitoring between bars.",
    )
    # Safety
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Required flag to confirm you intend to trade with real money",
    )
    return p.parse_args()


def require_confirmation(args: argparse.Namespace) -> None:
    """Force explicit confirmation before going live."""
    if args.confirm:
        return
    log.warning(
        "live_confirmation_required",
        msg="Add --confirm to the command to proceed. For paper trading use scripts/run_paper.py",
    )
    sys.exit(0)


def validate_mt5_access_mode(args: argparse.Namespace) -> None:
    """Accept either explicit credentials or an already logged-in MT5 terminal."""
    if args.mt5_login and args.mt5_password and args.mt5_server:
        return
    log.warning("mt5_using_terminal_session", msg="no explicit MT5 credentials provided — using account currently logged into the MT5 terminal")


def apply_runner_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill risk / magic from .env (RUNNER_*) when CLI flags are omitted."""
    rc = RunnerConfig.load()
    if args.max_risk_pct is None:
        args.max_risk_pct = rc.max_risk_pct
    if args.daily_loss_limit_pct is None:
        args.daily_loss_limit_pct = rc.daily_loss_limit_pct
    if args.kill_drawdown_pct is None:
        args.kill_drawdown_pct = rc.auto_kill_drawdown_pct
    if args.max_open_positions is None:
        args.max_open_positions = rc.max_open_positions
    if args.magic_number is None:
        args.magic_number = rc.magic_number
    return args


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
    return args


def load_registry(args: argparse.Namespace) -> ModelRegistry | None:
    ml_cfg = MLConfig(model_registry_dir=str(args.model_dir))
    registry = ModelRegistry(config=ml_cfg, persist=True)

    if args.model_version:
        snap = registry.load_from_disk(args.symbol, args.model_version)
        if snap is None:
            log.error("model_not_found", version=args.model_version, model_dir=str(args.model_dir))
            sys.exit(1)
        log.info("model_loaded", version=snap.version)
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
        log.info("active_model", version=active.version)

    return registry


def make_module_cfg(args: argparse.Namespace) -> ModuleConfig:
    """Build a ModuleConfig from CLI flags."""
    return ModuleConfig(
        multi_tf_d1=not args.no_d1,
        news_calendar=not args.no_news,
        ml=not args.no_ml,
        finnhub_api_key=args.finnhub_key,
    )


_TICK_POLL_SECS = 0.05   # 50 ms polling in tick mode (interval == 0)
_MARKET_CLOSED_SLEEP_SEC = 900   # 15 min between checks when market is closed
_RECONNECT_SLEEP_SEC = 30        # wait before MT5 reconnect attempt
_MAX_RECONNECT_ATTEMPTS = 5
# Spezzare le attese così l'interprete rientra spesso in Python: su Windows Ctrl+C
# viene gestito solo tra una istruzione e l'altra; le chiamate MT5 native possono
# comunque ritardare l'arresto finché non ritornano.
_INTERRUPTIBLE_CHUNK_SEC = 0.25


def _interruptible_wait(
    total_seconds: float,
    stop_event: threading.Event,
    running_ref: list[bool],
    chunk_sec: float = _INTERRUPTIBLE_CHUNK_SEC,
) -> None:
    """Attende fino a ``total_seconds`` ma esce subito se shutdown o stop richiesto."""
    if total_seconds <= 0:
        return
    end = time.time() + total_seconds
    while running_ref[0] and not stop_event.is_set():
        remaining = end - time.time()
        if remaining <= 0:
            return
        time.sleep(min(chunk_sec, remaining))


def _is_market_closed(dt: datetime) -> bool:
    """Return True when the forex market is closed (Saturday all day, or Sunday before 22:00 UTC).

    Forex closes Friday ~22:00 UTC and reopens Sunday ~22:00 UTC.
    This is a conservative check: treats Saturday and Sunday-before-22h as closed.
    """
    wd = dt.weekday()  # 0=Mon … 6=Sun
    if wd == 5:  # Saturday: always closed
        return True
    if wd == 6 and dt.hour < 22:  # Sunday before 22:00 UTC
        return True
    return False


def _log_monitor_action(act: dict) -> None:
    log.info(
        "position_monitor_action",
        action=act["action"],
        ticket=act["ticket"],
        new_sl=act.get("new_sl"),
        reason=str(act.get("reason") or ""),
    )


def _run_position_monitor(
    runner,
    broker,
    exit_engine,
    symbol: str,
    poll_interval: float,
    mon_interval: float,
    stop_event: threading.Event | None = None,
    running_ref: list[bool] | None = None,
) -> None:
    """Block for ``poll_interval`` seconds while running position monitoring.

    mon_interval behaviour:
      > 0  — evaluate every mon_interval seconds (timer mode).
      == 0 — evaluate on every new MT5 tick; polls at 50 ms to detect tick
              changes.  This is the highest-frequency mode.
      < 0  — disable position monitoring; just sleep.
    """
    if mon_interval < 0:
        if stop_event is not None and running_ref is not None:
            _interruptible_wait(poll_interval, stop_event, running_ref)
        else:
            time.sleep(poll_interval)
        return

    deadline = time.time() + poll_interval
    last_tick_msc: int = 0

    while time.time() < deadline:
        if running_ref is not None and not running_ref[0]:
            return
        if stop_event is not None and stop_event.is_set():
            return

        do_monitor = False

        if mon_interval == 0:
            # Tick mode: run only when a new tick arrived
            tick = broker.get_latest_tick(symbol)
            if tick is not None:
                tick_msc = tick[0]
                if tick_msc != last_tick_msc:
                    last_tick_msc = tick_msc
                    do_monitor = True
            sleep_secs = _TICK_POLL_SECS
        else:
            # Timer mode: run every mon_interval seconds
            do_monitor = True
            sleep_secs = mon_interval

        if do_monitor:
            try:
                actions = runner.monitor_positions_once(exit_engine, symbol=symbol)
                for act in actions:
                    _log_monitor_action(act)
            except Exception as exc:
                log.warning("position_monitor_error", error=str(exc))

        remaining = deadline - time.time()
        slice_len = min(sleep_secs, max(0.0, remaining))
        if stop_event is not None and running_ref is not None:
            _interruptible_wait(slice_len, stop_event, running_ref)
        else:
            time.sleep(slice_len)


def build_ml_stack(
    args: argparse.Namespace,
    registry: ModelRegistry | None,
    telemetry: TelemetryStore,
) -> tuple[TelegramAlerter | None, LiveAccuracyTracker | None, ModelWatcher | None, RetrainScheduler | None, CommandHandler | None]:
    """Build alerter + ML lifecycle stack from env config."""
    alert_cfg = AlertConfig()
    alerter: TelegramAlerter | None = None
    if alert_cfg.bot_token and alert_cfg.chat_id and alert_cfg.enabled:
        alerter = TelegramAlerter(alert_cfg)
        log.info("telegram_alerts_enabled", chat_id=alert_cfg.chat_id)
    else:
        log.info("telegram_alerts_disabled")

    ml_cfg = MLConfig(model_registry_dir=str(args.model_dir))

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
        log.info("model_watcher_enabled", poll_sec=ml_cfg.model_watcher_poll_sec)

    retrain_scheduler: RetrainScheduler | None = None
    if ml_cfg.retrain_enabled:
        # Note: --source/--symbol/--timeframe/--model-dir are overridden by the
        # worker using job metadata and its own local paths. They are kept here
        # only for the local (non-distributed) training path.
        train_args = [
            "--source", "mt5",
            "--symbol", args.symbol,
            "--timeframe", args.timeframe,
            "--model-dir", str(args.model_dir),
            "--holdout-fraction", "0.2",
            "--adaptive",
        ]
        retrain_scheduler = RetrainScheduler(
            config=ml_cfg,
            train_args=train_args,
            alerter=alerter,
        )
        unit = f"ogni {ml_cfg.retrain_every_hours}h" if ml_cfg.retrain_trigger == "hours" else f"ogni {ml_cfg.retrain_every_bars} barre"
        log.info("retrain_scheduler_enabled", schedule=unit)

    command_handler: CommandHandler | None = None
    if alerter is not None and alert_cfg.commands_enabled:
        command_handler = CommandHandler()
        log.info("telegram_commands_enabled", poll_sec=alert_cfg.poll_interval_sec)

    return alerter, live_tracker, model_watcher, retrain_scheduler, command_handler


def main() -> None:
    configure_logging()
    args = parse_args()
    args = apply_runner_defaults(args)
    args = apply_mt5_defaults(args)
    require_confirmation(args)
    validate_mt5_access_mode(args)
    save_last_launch(mode="live", script_path=Path(__file__).resolve())
    telemetry = TelemetryStore.from_env()

    tf = _TIMEFRAME_MAP[args.timeframe]
    tf_secs = _TF_SECONDS[tf]

    # 1. Connect to MT5
    try:
        from metatrade.market_data.collectors.mt5_collector import MT5Collector
        from metatrade.broker.mt5_adapter import MT5BrokerAdapter
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

    broker = MT5BrokerAdapter(
        login=args.mt5_login,
        password=args.mt5_password,
        server=args.mt5_server,
        path=args.mt5_path or None,
        timeout=args.mt5_timeout_ms,
        run_mode=RunMode.LIVE,
        magic_number=args.magic_number,
        min_stop_pips=args.min_stop_pips,
    )

    # 2. Load warmup bars
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(seconds=tf_secs * (args.warmup_bars + 10))

    log.info("loading_warmup_bars", symbol=args.symbol, timeframe=args.timeframe, bars=args.warmup_bars)
    bars = collector.collect(args.symbol, tf, date_from, date_to)
    log.info("warmup_bars_loaded", count=len(bars))

    # 3. Build runner
    registry = load_registry(args)
    active_model = registry.get_active().version if registry and registry.get_active() else None
    session_id = telemetry.start_session(
        run_mode="LIVE",
        symbol=args.symbol,
        timeframe=args.timeframe,
        ml_enabled=not args.no_ml,
        model_version=active_model,
        details={"mt5_server": args.mt5_server, "confirm": args.confirm},
    )
    module_cfg = make_module_cfg(args)
    modules = build_modules(args.timeframe.lower(), module_cfg=module_cfg, registry=registry)

    log.info("building_ml_stack")
    alerter, live_tracker, model_watcher, retrain_scheduler, command_handler = (
        build_ml_stack(args, registry, telemetry)
    )

    # Seed the retrain scheduler bar buffer with warmup bars so that a manual
    # /retrain immediately after startup has enough data to upload to GridFS.
    if retrain_scheduler is not None and bars:
        retrain_scheduler._bar_buffer.extend(bars)
        log.info("retrain_scheduler_bar_buffer_seeded", count=len(bars))

    # In remote mongo mode, fetch a larger dedicated batch for training.
    # warmup_bars (200) is enough for the runner modules but not for walk-forward
    # training which needs at least train_window + test_window bars per fold.
    if retrain_scheduler is not None and retrain_scheduler.get_mongo_db() is not None:
        _ml_cfg = MLConfig(model_registry_dir=str(args.model_dir))
        n = (_ml_cfg.train_window_bars + _ml_cfg.test_window_bars) * 3
        if len(retrain_scheduler._bar_buffer) < n:
            date_to = datetime.now(timezone.utc)
            date_from = date_to - timedelta(seconds=tf_secs * (n + 10))
            log.info("fetching_training_bars", symbol=args.symbol, timeframe=args.timeframe, bars=n)
            training_bars = collector.collect(args.symbol, tf, date_from, date_to)
            retrain_scheduler._bar_buffer.clear()
            retrain_scheduler._bar_buffer.extend(training_bars)
            log.info("retrain_scheduler_training_buffer_ready", count=len(training_bars))

    runner_cfg = RunnerConfig(
        symbol=args.symbol,
        max_risk_pct=args.max_risk_pct,
        daily_loss_limit_pct=args.daily_loss_limit_pct,
        auto_kill_drawdown_pct=args.kill_drawdown_pct,
        max_open_positions=args.max_open_positions,
        magic_number=args.magic_number,
        min_signals=1 if args.no_ml else 2,
    )

    exit_gen = None
    exit_sel = None
    if not args.no_exit_profile:
        from metatrade.ml.exit_profile_candidate_generator import ExitProfileCandidateGenerator
        from metatrade.ml.exit_profile_selector import ExitProfileSelector

        exit_gen = ExitProfileCandidateGenerator()
        exit_sel = ExitProfileSelector()

    # Exit engine — wired to alerter for trade-closed notifications
    from metatrade.exit_engine.engine import ExitEngine
    exit_engine = ExitEngine(alerter=alerter)

    runner = LiveRunner(
        config=runner_cfg,
        modules=modules,
        broker=broker,
        telemetry=telemetry,
        session_id=session_id,
        timeframe=args.timeframe,
        exit_profile_generator=exit_gen,
        exit_profile_selector=exit_sel,
        alerter=alerter,
        live_tracker=live_tracker,
        model_watcher=model_watcher,
        retrain_scheduler=retrain_scheduler,
        command_handler=command_handler,
    )
    runner.connect()

    # Pre-fill buffer — exclude the current (not yet closed) bar so that
    # ATR is computed only from confirmed closed candles.
    now_utc_init = datetime.now(timezone.utc)
    closed_bars = [
        b for b in bars
        if b.timestamp_utc.timestamp() + tf_secs <= now_utc_init.timestamp()
    ]
    for bar in closed_bars:
        runner._bar_buffer.append(bar)
    if len(runner._bar_buffer) > 500:
        runner._bar_buffer = runner._bar_buffer[-500:]
    # Ultima barra chiusa nota (None se warmup vuoto / nessuna candela ancora chiusa)
    last_bar_time: datetime | None = (
        closed_bars[-1].timestamp_utc if closed_bars else None
    )

    account = broker.get_account()
    log.info(
        "live_trading_started",
        symbol=args.symbol,
        timeframe=args.timeframe,
        exit_profile=exit_gen is not None,
        account=account.login if hasattr(account, "login") else "N/A",
        balance=str(account.balance),
        equity=str(account.equity),
        modules=[m.__class__.__name__ for m in modules],
        max_risk_pct=f"{args.max_risk_pct:.1%}",
        daily_loss_limit_pct=f"{args.daily_loss_limit_pct:.1%}",
        kill_drawdown_pct=f"{args.kill_drawdown_pct:.1%}",
    )

    poll_interval = args.poll_interval if args.poll_interval > 0 else tf_secs
    _running = [True]
    _consecutive_errors = [0]
    _stop_event = threading.Event()

    def _stop(sig, frame):
        _running[0] = False
        _stop_event.set()
        log.info("shutdown_requested")

    signal.signal(signal.SIGINT, _stop)
    if sys.platform == "win32" and hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _stop)

    def _try_reconnect() -> bool:
        """Attempt to re-initialize the MT5 collector. Returns True on success."""
        for attempt in range(1, _MAX_RECONNECT_ATTEMPTS + 1):
            if not _running[0] or _stop_event.is_set():
                return False
            log.info("mt5_reconnect_attempt", attempt=attempt, max_attempts=_MAX_RECONNECT_ATTEMPTS)
            try:
                collector.initialize(
                    login=args.mt5_login,
                    password=args.mt5_password,
                    server=args.mt5_server,
                    path=args.mt5_path,
                    timeout=args.mt5_timeout_ms,
                )
                log.info("mt5_reconnected", attempt=attempt)
                if alerter:
                    alerter.alert_error("MT5", f"Riconnesso dopo {attempt} tentativo/i.")
                return True
            except Exception as exc:
                log.warning("mt5_reconnect_failed", attempt=attempt, error=str(exc))
                _interruptible_wait(float(_RECONNECT_SLEEP_SEC), _stop_event, _running)
        return False

    # 4. Live loop — runs continuously even when the market is closed (Windows service mode)
    while _running[0]:
        now_utc = datetime.now(timezone.utc)

        # ── Market-closed guard ───────────────────────────────────────────────
        if _is_market_closed(now_utc):
            log.info("market_closed", sleep_min=_MARKET_CLOSED_SLEEP_SEC // 60)
            _interruptible_wait(float(_MARKET_CLOSED_SLEEP_SEC), _stop_event, _running)
            continue

        date_to = now_utc
        date_from = date_to - timedelta(seconds=tf_secs * 5)

        try:
            new_bars = collector.collect(args.symbol, tf, date_from, date_to)
            _consecutive_errors[0] = 0
        except Exception as exc:
            _consecutive_errors[0] += 1
            log.warning("mt5_fetch_error", error=str(exc), consecutive=_consecutive_errors[0])
            if _consecutive_errors[0] >= 3:
                log.warning("mt5_fetch_failed_reconnecting", consecutive=_consecutive_errors[0])
                if not _try_reconnect():
                    log.error("mt5_reconnect_exhausted", msg="sleeping 5 min before retrying")
                    if alerter:
                        alerter.alert_error("MT5", "Disconnesso — reconnect fallito.")
                    _interruptible_wait(300.0, _stop_event, _running)
                _consecutive_errors[0] = 0
            else:
                _interruptible_wait(5.0, _stop_event, _running)
            continue

        if last_bar_time is not None:
            new_bars = [b for b in new_bars if b.timestamp_utc > last_bar_time]

        # Drop the current (still-forming) bar: its H/L range is tiny (only
        # seconds old), which collapses ATR → inflates lot size → Invalid stops.
        # A bar is confirmed closed when its period has fully elapsed.
        now_utc = datetime.now(timezone.utc)
        new_bars = [
            b for b in new_bars
            if b.timestamp_utc.timestamp() + tf_secs <= now_utc.timestamp()
        ]

        for bar in new_bars:
            last_bar_time = bar.timestamp_utc
            ts_str = bar.timestamp_utc.strftime("%Y-%m-%d %H:%M")

            try:
                decision = runner.on_bar(bar)
            except RuntimeError as exc:
                # Kill switch or disconnect
                log.error("runner_error", error=str(exc))
                _running[0] = False
                break

            if decision is None:
                log.info("bar_hold", ts=ts_str)
            elif hasattr(decision, "approved") and decision.approved:
                ps = decision.position_size
                log.info(
                    "bar_order_submitted",
                    ts=ts_str,
                    side=decision.side.value,
                    lot=float(ps.lot_size),
                    sl=str(ps.stop_loss_price),
                    tp=str(ps.take_profit_price),
                )
            else:
                veto_code = decision.veto.veto_code if decision.veto else "?"
                log.info("bar_vetoed", ts=ts_str, veto_code=veto_code)
                if "KILL" in veto_code:
                    log.warning("kill_switch_triggered", veto_code=veto_code)
                    _running[0] = False
                    break

        if _running[0]:
            _run_position_monitor(
                runner=runner,
                broker=broker,
                exit_engine=exit_engine,
                symbol=args.symbol,
                poll_interval=poll_interval,
                mon_interval=args.position_monitor_interval,
                stop_event=_stop_event,
                running_ref=_running,
            )

    # 5. Summary
    runner.disconnect()
    stats = runner.stats
    account = broker.get_account()
    runner.stop(
        n_trades=stats.trades_executed,
        total_pnl=account.equity - account.balance,
    )

    account = broker.get_account()
    log.info(
        "live_session_summary",
        bars_processed=stats.bars_processed,
        trades_executed=stats.trades_executed,
        trades_vetoed=stats.trades_vetoed,
        final_balance=str(account.balance),
        final_equity=str(account.equity),
    )

    telemetry.finish_session(
        session_id,
        status="completed",
        details={
            "bars_processed": stats.bars_processed,
            "trades_executed": stats.trades_executed,
            "trades_vetoed": stats.trades_vetoed,
        },
    )
    collector.shutdown()


if __name__ == "__main__":
    main()
