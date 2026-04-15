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
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metatrade.core.enums import RunMode, Timeframe
from metatrade.core.log import get_logger
from metatrade.ml.config import MLConfig
from metatrade.ml.registry import ModelRegistry
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
    # Risk parameters
    p.add_argument("--max-risk-pct", type=float, default=0.01, help="Max risk per trade (default 1%%)")
    p.add_argument("--daily-loss-limit-pct", type=float, default=0.05, help="Daily loss limit (default 5%%)")
    p.add_argument("--kill-drawdown-pct", type=float, default=0.10, help="Hard kill drawdown (default 10%%)")
    p.add_argument("--max-open-positions", type=int, default=1, help="Max concurrent positions (default 1)")
    # MT5 connection
    p.add_argument("--mt5-login", type=int, default=0)
    p.add_argument("--mt5-password", default="")
    p.add_argument("--mt5-server", default="")
    p.add_argument("--mt5-path", default="", help="Path to terminal64.exe if MT5 is not auto-detected")
    p.add_argument("--mt5-timeout-ms", type=int, default=60000, help="MT5 initialization timeout in milliseconds")
    p.add_argument("--magic-number", type=int, default=20240601, help="MT5 magic number for order tagging")
    # Model
    p.add_argument("--model-dir", type=Path, default=Path("data/models"))
    p.add_argument("--model-version", default=None, help="Specific model version (default: active)")
    p.add_argument("--no-ml", action="store_true", help="Technical indicators only (no ML)")
    # Module toggles
    p.add_argument("--no-news", action="store_true", help="Disable news/calendar filter")
    p.add_argument("--no-d1", action="store_true", help="Disable D1 multi-timeframe module (needs 500+ bars)")
    p.add_argument("--finnhub-key", default=None, help="Finnhub API key for live economic calendar")
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
    print("WARNING: You are about to start LIVE trading with REAL MONEY.")
    print("   Add --confirm to the command to proceed.")
    print()
    print("   For paper trading (no real money) use: scripts/run_paper.py")
    sys.exit(0)


def validate_mt5_access_mode(args: argparse.Namespace) -> None:
    """Accept either explicit credentials or an already logged-in MT5 terminal."""
    if args.mt5_login and args.mt5_password and args.mt5_server:
        return
    print(
        "No explicit MT5 credentials provided - using the account currently logged into the MT5 terminal.",
        file=sys.stderr,
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
    return args


def load_registry(args: argparse.Namespace) -> ModelRegistry | None:
    ml_cfg = MLConfig(model_registry_dir=str(args.model_dir))
    registry = ModelRegistry(config=ml_cfg, persist=True)

    if args.model_version:
        snap = registry.load_from_disk(args.symbol, args.model_version)
        if snap is None:
            print(f"ERROR: Model {args.model_version!r} not found in {args.model_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Model loaded: {snap.version}")
    else:
        active = registry.get_active()
        if active is None:
            latest = sorted(args.model_dir.glob(f"{args.symbol}_v*.pkl"), reverse=True)
            if latest:
                version = latest[0].stem.removeprefix(f"{args.symbol}_")
                active = registry.load_from_disk(args.symbol, version)
            if args.no_ml:
                print("No ML model - running with technical indicators only.")
                return None
            print(
                "WARNING: No active model. Run scripts/train.py first, or pass --no-ml.",
                file=sys.stderr,
            )
            return None
        print(f"Active model: {active.version}")

    return registry


def make_module_cfg(args: argparse.Namespace) -> ModuleConfig:
    """Build a ModuleConfig from CLI flags."""
    return ModuleConfig(
        multi_tf_d1=not args.no_d1,
        news_calendar=not args.no_news,
        ml=not args.no_ml,
        finnhub_api_key=args.finnhub_key,
    )


def main() -> None:
    args = parse_args()
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
        print(f"ERROR: {exc}", file=sys.stderr)
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
    )

    # 2. Load warmup bars
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(seconds=tf_secs * (args.warmup_bars + 10))

    print(f"Loading {args.warmup_bars} warmup bars ...")
    bars = collector.collect(args.symbol, tf, date_from, date_to)
    print(f"  Got {len(bars)} bars.")

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

    runner_cfg = RunnerConfig(
        symbol=args.symbol,
        max_risk_pct=args.max_risk_pct,
        daily_loss_limit_pct=args.daily_loss_limit_pct,
        auto_kill_drawdown_pct=args.kill_drawdown_pct,
        max_open_positions=args.max_open_positions,
        magic_number=args.magic_number,
        min_signals=1 if args.no_ml else 2,
    )

    runner = LiveRunner(
        config=runner_cfg,
        modules=modules,
        broker=broker,
        telemetry=telemetry,
        session_id=session_id,
        timeframe=args.timeframe,
    )
    runner.connect()

    # Pre-fill buffer
    for bar in bars:
        runner._bar_buffer.append(bar)
    if len(runner._bar_buffer) > 500:
        runner._bar_buffer = runner._bar_buffer[-500:]

    account = broker.get_account()
    print(f"\nLIVE trading started - {args.symbol}/{args.timeframe}")
    print(f"  Account  : {account.login if hasattr(account, 'login') else 'N/A'}")
    print(f"  Balance  : {account.balance}")
    print(f"  Equity   : {account.equity}")
    print(f"  Modules  : {[m.__class__.__name__ for m in modules]}")
    print(f"  Risk/trade: {args.max_risk_pct:.1%}  Daily limit: {args.daily_loss_limit_pct:.1%}  Kill: {args.kill_drawdown_pct:.1%}")
    print("  Press Ctrl+C to stop.\n")

    poll_interval = args.poll_interval if args.poll_interval > 0 else tf_secs
    last_bar_time: datetime | None = bars[-1].timestamp_utc if bars else None
    _running = [True]

    def _stop(sig, frame):
        _running[0] = False
        print("\nStopping ...")

    signal.signal(signal.SIGINT, _stop)

    # 4. Live loop
    while _running[0]:
        date_to = datetime.now(timezone.utc)
        date_from = date_to - timedelta(seconds=tf_secs * 5)

        try:
            new_bars = collector.collect(args.symbol, tf, date_from, date_to)
        except Exception as exc:
            log.warning("mt5_fetch_error", error=str(exc))
            time.sleep(5)
            continue

        if last_bar_time is not None:
            new_bars = [b for b in new_bars if b.timestamp_utc > last_bar_time]

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
                print(f"[{ts_str}] HOLD")
            elif hasattr(decision, "approved") and decision.approved:
                ps = decision.position_size
                print(
                    f"[{ts_str}] ORDER SUBMITTED  side={decision.side.value}  "
                    f"lot={float(ps.lot_size):.2f}  sl={ps.stop_loss_price}  tp={ps.take_profit_price}"
                )
            else:
                veto_code = decision.veto.veto_code if decision.veto else "?"
                # Use ASCII output so Windows consoles do not fail on Unicode.
                icon = "KILL" if "KILL" in veto_code else "VETO"
                print(f"[{ts_str}] {icon} VETOED  ({veto_code})")
                if "KILL" in veto_code:
                    print("  Kill switch triggered - stopping runner.")
                    _running[0] = False
                    break

        if _running[0]:
            time.sleep(poll_interval)

    # 5. Summary
    runner.disconnect()

    stats = runner.stats
    account = broker.get_account()
    print("\n" + "=" * 55)
    print("LIVE TRADING SESSION SUMMARY")
    print("=" * 55)
    print(f"  Bars processed  : {stats.bars_processed}")
    print(f"  Trades executed : {stats.trades_executed}")
    print(f"  Trades vetoed   : {stats.trades_vetoed}")
    print(f"  Final balance   : {account.balance}")
    print(f"  Final equity    : {account.equity}")
    print("=" * 55)

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
