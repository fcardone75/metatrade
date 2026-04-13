"""Train the ML model on historical data using walk-forward validation.

Usage examples:

    # From CSV file
    python scripts/train.py --source csv --file data/EURUSD_H1.csv --symbol EURUSD

    # From MetaTrader 5 (Windows only, MT5 must be open)
    python scripts/train.py --source mt5 --symbol EURUSD --timeframe H1 --bars 5000

    # Override ML config
    python scripts/train.py --source csv --file data/EURUSD_H1.csv \\
        --symbol EURUSD --n-estimators 300 --min-accuracy 0.53

The script:
  1. Loads historical bars (CSV or MT5)
  2. Runs walk-forward training (expanding window)
  3. Prints fold-by-fold accuracy
  4. Registers the best model to disk
  5. Promotes it as the active model in the registry
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make sure src/ is on the path when running as a plain script
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metatrade.core.enums import Timeframe
from metatrade.core.log import get_logger
from metatrade.ml.config import MLConfig
from metatrade.ml.registry import ModelRegistry
from metatrade.ml.walk_forward import WalkForwardTrainer

log = get_logger("train")

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
    p = argparse.ArgumentParser(
        description="Train Forex ML model with walk-forward validation"
    )
    p.add_argument(
        "--source",
        choices=["csv", "mt5"],
        required=True,
        help="Data source: csv (local file) or mt5 (MetaTrader 5, Windows only)",
    )
    # CSV options
    p.add_argument("--file", type=Path, help="Path to CSV file (required for --source csv)")
    p.add_argument("--symbol", default="EURUSD", help="Symbol name (e.g. EURUSD)")
    p.add_argument(
        "--timeframe",
        default="H1",
        choices=list(_TIMEFRAME_MAP),
        help="Timeframe for MT5 collection (default: H1)",
    )
    # MT5 options
    p.add_argument("--bars", type=int, default=5000, help="Number of bars to fetch from MT5")
    p.add_argument("--mt5-login", type=int, default=0, help="MT5 account login")
    p.add_argument("--mt5-password", default="", help="MT5 account password")
    p.add_argument("--mt5-server", default="", help="MT5 broker server name")
    # ML config overrides
    p.add_argument("--train-window", type=int, default=2000, help="Training window in bars")
    p.add_argument("--test-window", type=int, default=500, help="Test window in bars")
    p.add_argument("--step", type=int, default=250, help="Step size in bars between folds")
    p.add_argument("--forward-bars", type=int, default=5, help="Bars ahead for label generation")
    p.add_argument("--n-estimators", type=int, default=200, help="RandomForest trees")
    p.add_argument("--max-depth", type=int, default=5, help="RandomForest max depth")
    p.add_argument("--min-accuracy", type=float, default=0.52, help="Min accuracy to accept model")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/models"),
        help="Directory for model persistence (default: data/models)",
    )
    return p.parse_args()


def load_from_csv(args: argparse.Namespace):
    from metatrade.market_data.collectors.csv_collector import CsvCollector

    if args.file is None:
        print("ERROR: --file is required when --source csv", file=sys.stderr)
        sys.exit(1)

    tf = _TIMEFRAME_MAP[args.timeframe]
    collector = CsvCollector()
    print(f"Loading bars from {args.file} …")
    bars = collector.collect_file(
        file_path=args.file,
        symbol=args.symbol,
        timeframe=tf,
    )
    print(f"  Loaded {len(bars)} bars.")
    return bars


def load_from_mt5(args: argparse.Namespace):
    from datetime import timedelta
    from metatrade.market_data.collectors.mt5_collector import MT5Collector

    collector = MT5Collector(store=None)  # no persistence needed here
    collector.initialize(
        login=args.mt5_login,
        password=args.mt5_password,
        server=args.mt5_server,
    )

    tf = _TIMEFRAME_MAP[args.timeframe]
    date_to = datetime.now(timezone.utc)

    # Estimate date_from from bar count × timeframe minutes
    tf_minutes = {
        Timeframe.M1: 1, Timeframe.M5: 5, Timeframe.M15: 15, Timeframe.M30: 30,
        Timeframe.H1: 60, Timeframe.H4: 240, Timeframe.D1: 1440,
    }
    minutes = tf_minutes.get(tf, 60) * args.bars
    date_from = date_to - timedelta(minutes=minutes)

    print(f"Fetching {args.bars} bars of {args.symbol}/{args.timeframe} from MT5 …")
    bars = collector.collect(args.symbol, tf, date_from, date_to)
    collector.shutdown()
    print(f"  Fetched {len(bars)} bars.")
    return bars


def main() -> None:
    args = parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if args.source == "csv":
        bars = load_from_csv(args)
    else:
        bars = load_from_mt5(args)

    if len(bars) < args.train_window + args.test_window:
        print(
            f"ERROR: Not enough bars ({len(bars)}) for the configured windows "
            f"({args.train_window} train + {args.test_window} test).",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 2. Configure and run walk-forward training ────────────────────────────
    cfg = MLConfig(
        train_window_bars=args.train_window,
        test_window_bars=args.test_window,
        step_bars=args.step,
        forward_bars=args.forward_bars,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_accuracy=args.min_accuracy,
        model_registry_dir=str(args.model_dir),
    )

    trainer = WalkForwardTrainer(cfg)

    print(f"\nStarting walk-forward training on {len(bars)} bars …")
    print(f"  train_window={cfg.train_window_bars}  test_window={cfg.test_window_bars}  step={cfg.step_bars}")
    print()

    result, model = trainer.run(bars)

    # ── 3. Print fold summary ─────────────────────────────────────────────────
    if not result.folds:
        print("ERROR: No folds produced — not enough data.", file=sys.stderr)
        sys.exit(1)

    print(f"{'Fold':>5}  {'Train bars':>11}  {'Test bars':>10}  {'Train acc':>10}  {'Test acc':>9}")
    print("-" * 55)
    for fold in result.folds:
        print(
            f"{fold.fold_index:>5}  "
            f"{fold.train_end - fold.train_start:>11}  "
            f"{fold.n_test_samples:>10}  "
            f"{fold.train_metrics.accuracy:>10.2%}  "
            f"{fold.test_accuracy:>9.2%}"
        )
    print("-" * 55)
    print(f"{'Mean':>5}  {'':>11}  {'':>10}  {'':>10}  {result.mean_test_accuracy:>9.2%}")
    print(f"{'Best':>5}  {'':>11}  {'':>10}  {'':>10}  {result.best_test_accuracy:>9.2%}")
    print()

    # ── 4. Register and promote ───────────────────────────────────────────────
    if model is None or not model.is_trained:
        print(
            f"WARNING: No model met the minimum accuracy threshold ({cfg.min_accuracy:.0%}).\n"
            "Increase --bars, decrease --min-accuracy, or review data quality.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(config=cfg, persist=True)

    best_fold = result.folds[result.best_fold_index]
    snapshot = registry.register(
        classifier=model,
        symbol=args.symbol,
        tags={
            "source": args.source,
            "n_folds": len(result.folds),
            "mean_test_acc": round(result.mean_test_accuracy, 4),
            "best_test_acc": round(result.best_test_accuracy, 4),
            "best_fold": best_fold.fold_index,
            "train_bars": len(bars),
            "timeframe": args.timeframe,
        },
    )
    registry.promote(snapshot.version)

    print(f"Model saved and promoted.")
    print(f"  Version : {snapshot.version}")
    print(f"  Symbol  : {snapshot.symbol}")
    print(f"  Accuracy: {result.best_test_accuracy:.2%}")
    print(f"  Location: {args.model_dir}/{args.symbol}/")
    print()
    print("Done. Ready for paper or live trading.")


if __name__ == "__main__":
    main()
