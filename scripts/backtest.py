"""Backtest runner — replay historical data through the full signal pipeline.

Usage examples:

    # From CSV file
    python scripts/backtest.py --source csv --file data/EURUSD_H1.csv --symbol EURUSD

    # From MetaTrader 5 (Windows only, MT5 must be open)
    python scripts/backtest.py --source mt5 --symbol EURUSD --timeframe H1 --bars 5000

    # No ML, technical indicators only
    python scripts/backtest.py --source csv --file data/EURUSD_H1.csv --no-ml

    # Custom risk settings
    python scripts/backtest.py --source csv --file data/EURUSD_H1.csv \\
        --max-risk-pct 0.02 --initial-balance 50000

The script:
  1. Loads historical bars (CSV or MT5)
  2. Replays them bar-by-bar through the signal → consensus → risk pipeline
  3. Simulates fills at next-bar open price (no look-ahead)
  4. Applies SL/TP exits based on bar high/low
  5. Prints a detailed P&L summary

Requires ML model: run scripts/train.py first (or pass --no-ml).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metatrade.core.log import configure_logging, get_logger
from metatrade.core.enums import Timeframe
from metatrade.ml.config import MLConfig
from metatrade.ml.registry import ModelRegistry
from metatrade.runner.backtest_runner import BacktestRunner
from metatrade.runner.config import RunnerConfig
from metatrade.runner.module_config import ModuleConfig
from metatrade.runner.module_builder import build_modules

log = get_logger(__name__)

_TIMEFRAME_MAP: dict[str, Timeframe] = {
    "M1": Timeframe.M1,
    "M5": Timeframe.M5,
    "M15": Timeframe.M15,
    "M30": Timeframe.M30,
    "H1": Timeframe.H1,
    "H4": Timeframe.H4,
    "D1": Timeframe.D1,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forex backtest via historical data replay")
    p.add_argument(
        "--source",
        choices=["csv", "mt5"],
        required=True,
        help="Data source: csv (local file) or mt5 (MetaTrader 5, Windows only)",
    )
    # Data source options
    p.add_argument("--file", type=Path, help="Path to CSV file (required for --source csv)")
    p.add_argument("--symbol", default="EURUSD", help="Symbol name (e.g. EURUSD)")
    p.add_argument(
        "--timeframe",
        default="H1",
        choices=list(_TIMEFRAME_MAP),
        help="Bar timeframe (default: H1)",
    )
    p.add_argument("--bars", type=int, default=5000, help="Number of bars to fetch from MT5")
    p.add_argument("--mt5-login", type=int, default=0)
    p.add_argument("--mt5-password", default="")
    p.add_argument("--mt5-server", default="")
    # Risk / runner parameters
    p.add_argument("--max-risk-pct", type=float, default=0.01, help="Risk per trade (default 1%%)")
    p.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="Starting balance in account currency (default 10000)",
    )
    p.add_argument(
        "--pip-value",
        type=float,
        default=10.0,
        help="P&L per lot per pip in account currency (default 10.0)",
    )
    p.add_argument(
        "--commission",
        type=float,
        default=7.0,
        help="Round-trip commission per standard lot (default 7.0)",
    )
    # Model
    p.add_argument("--model-dir", type=Path, default=Path("data/models"))
    p.add_argument("--model-version", default=None, help="Specific model version (default: active)")
    p.add_argument("--no-ml", action="store_true", help="Technical indicators only (no ML)")
    # Module toggles
    p.add_argument("--no-news", action="store_true", help="Disable news/calendar filter (recommended for backtests)")
    p.add_argument("--no-d1", action="store_true", help="Disable D1 multi-timeframe module (needs 500+ bars)")
    p.add_argument("--finnhub-key", default=None, help="Finnhub API key for live economic calendar")
    return p.parse_args()


def load_bars(args: argparse.Namespace) -> list:
    tf = _TIMEFRAME_MAP[args.timeframe]

    if args.source == "csv":
        from metatrade.market_data.collectors.csv_collector import CsvCollector

        if args.file is None:
            log.error("missing_csv_file", msg="--file is required when --source csv")
            sys.exit(1)
        collector = CsvCollector()
        log.info("loading_bars_csv", file=str(args.file))
        bars = collector.collect_file(
            file_path=args.file,
            symbol=args.symbol,
            timeframe=tf,
        )
    else:
        from datetime import timedelta
        from metatrade.market_data.collectors.mt5_collector import MT5Collector

        collector = MT5Collector(store=None)
        collector.initialize(
            login=args.mt5_login,
            password=args.mt5_password,
            server=args.mt5_server,
        )
        tf_minutes = {
            Timeframe.M1: 1, Timeframe.M5: 5, Timeframe.M15: 15, Timeframe.M30: 30,
            Timeframe.H1: 60, Timeframe.H4: 240, Timeframe.D1: 1440,
        }
        date_to = datetime.now(timezone.utc)
        date_from = date_to - timedelta(minutes=tf_minutes.get(tf, 60) * args.bars)
        log.info("loading_bars_mt5", symbol=args.symbol, timeframe=args.timeframe, bars=args.bars)
        bars = collector.collect(args.symbol, tf, date_from, date_to)
        collector.shutdown()

    log.info("bars_loaded", count=len(bars))
    return bars


def load_registry(args: argparse.Namespace) -> ModelRegistry | None:
    ml_cfg = MLConfig(model_registry_dir=str(args.model_dir))
    registry = ModelRegistry(config=ml_cfg, persist=True)

    if args.model_version:
        snap = registry.load_from_disk(args.symbol, args.model_version)
        if snap is None:
            log.error("model_not_found", version=args.model_version, model_dir=str(args.model_dir))
            sys.exit(1)
        registry.register(snap.classifier, snap.symbol, version=snap.version, tags=snap.tags)
        registry.promote(snap.version)
        log.info("model_loaded", version=snap.version)
    else:
        active = registry.get_active()
        if active is None:
            if args.no_ml:
                log.info("no_ml_model", msg="running with technical indicators only")
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


def main() -> None:
    configure_logging()
    args = parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    bars = load_bars(args)

    if len(bars) < 50:
        log.error("insufficient_bars", count=len(bars), msg="not enough bars to run a backtest")
        sys.exit(1)

    # ── 2. Load model ─────────────────────────────────────────────────────────
    registry = load_registry(args)
    module_cfg = make_module_cfg(args)
    modules = build_modules(args.timeframe.lower(), module_cfg=module_cfg, registry=registry)

    # ── 3. Build and run backtest ─────────────────────────────────────────────
    runner_cfg = RunnerConfig(
        symbol=args.symbol,
        max_risk_pct=args.max_risk_pct,
        initial_balance=args.initial_balance,
        min_signals=1 if args.no_ml else 2,
    )

    log.info(
        "backtest_starting",
        symbol=args.symbol,
        timeframe=args.timeframe,
        bars=len(bars),
        modules=[m.__class__.__name__ for m in modules],
        initial_balance=args.initial_balance,
        max_risk_pct=args.max_risk_pct,
    )

    runner = BacktestRunner(
        config=runner_cfg,
        modules=modules,
        bars=bars,
        pip_value=args.pip_value,
        commission_per_lot=args.commission,
    )
    result = runner.run()

    # ── 4. Log summary ────────────────────────────────────────────────────────
    log.info(
        "backtest_results",
        symbol=args.symbol,
        timeframe=args.timeframe,
        bars_processed=result.stats.bars_processed,
        n_trades=result.n_trades,
        n_winning=result.n_winning,
        n_losing=result.n_losing,
        win_rate=f"{result.win_rate:.1%}" if result.n_trades > 0 else "n/a",
        total_pnl=f"{float(result.total_pnl):+,.2f}",
        return_pct=f"{result.return_pct:+.2%}",
        initial_balance=f"{float(result.initial_balance):,.2f}",
        final_balance=f"{float(result.final_balance):,.2f}",
        weight_updates=result.stats.weight_updates,
    )

    # Trade-level breakdown (last 10 if many)
    if result.trades:
        display = result.trades[-10:] if len(result.trades) > 10 else result.trades
        if len(result.trades) > 10:
            log.info("trade_breakdown_truncated", showing=10, total=len(result.trades))
        for t in display:
            log.info(
                "trade_result",
                bar_index=t.bar_index,
                side=t.side.value,
                entry_price=float(t.entry_price),
                exit_price=float(t.exit_price),
                lot_size=float(t.lot_size),
                pnl=float(t.pnl),
                exit_reason=t.exit_reason,
            )


if __name__ == "__main__":
    main()
