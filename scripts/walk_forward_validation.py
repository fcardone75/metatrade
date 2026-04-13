"""Walk-forward validation — tests the ENTIRE pipeline across rolling folds.

Unlike scripts/train.py which validates only the ML model, this script
validates the full system: all TA modules + consensus engine + risk manager +
backtest runner across N rolling time windows.

Usage:
    # TA-only (no ML) — fast, no model needed:
    python scripts/walk_forward_validation.py \\
        --source csv --file data/EURUSD_H1.csv --folds 5 --no-ml

    # With ML (retrains model on each fold's train window):
    python scripts/walk_forward_validation.py \\
        --source csv --file data/EURUSD_H1.csv --folds 5

    # Custom fold split ratio and risk:
    python scripts/walk_forward_validation.py \\
        --source csv --file data/EURUSD_H1.csv --folds 5 --train-pct 0.70 \\
        --max-risk-pct 0.01 --initial-balance 10000

Walk-forward structure (rolling, non-anchored):
    Total bars  = N
    Fold width  = N // folds
    Train width = fold_width × train_pct
    Test width  = fold_width × (1 - train_pct)

    Fold 1: bars[0           : fold_width]
    Fold 2: bars[fold_width  : 2×fold_width]
    ...

Each fold:
  1. (Optionally) retrain ML on the train window
  2. Run BacktestRunner on the test window only
  3. Compute: win_rate, return_pct, sharpe_ratio, max_drawdown, n_trades

Output:
    Fold  Bars  Trades  Win%   Return    Sharpe    MaxDD
       1  1000      47  53.2%   +2.14%    0.82    -1.23%
       2  1000      51  51.0%   +1.87%    0.71    -1.45%
    ─────────────────────────────────────────────────────────
    Mean          49.2  52.1%   +1.95%    0.76    -1.34%

Stability check: if ≥ 2 folds produce negative returns → print UNSTABLE warning.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metatrade.core.enums import Timeframe
from metatrade.runner.backtest_runner import BacktestResult, BacktestRunner
from metatrade.runner.config import RunnerConfig
from metatrade.runner.module_config import ModuleConfig
from metatrade.runner.module_builder import build_modules


_TIMEFRAME_MAP: dict[str, Timeframe] = {
    "M1": Timeframe.M1,
    "M5": Timeframe.M5,
    "M15": Timeframe.M15,
    "M30": Timeframe.M30,
    "H1": Timeframe.H1,
    "H4": Timeframe.H4,
    "D1": Timeframe.D1,
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward validation of the entire metatrade pipeline"
    )
    p.add_argument("--source", choices=["csv", "mt5"], required=True)
    p.add_argument("--file", type=Path, help="CSV file path (required for --source csv)")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--timeframe", default="H1", choices=list(_TIMEFRAME_MAP))
    p.add_argument("--bars", type=int, default=5000, help="MT5 bar count")
    p.add_argument("--mt5-login", type=int, default=0)
    p.add_argument("--mt5-password", default="")
    p.add_argument("--mt5-server", default="")
    p.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds (default 5)")
    p.add_argument(
        "--train-pct",
        type=float,
        default=0.70,
        help="Fraction of each fold window used for training (default 0.70)",
    )
    p.add_argument("--max-risk-pct", type=float, default=0.01)
    p.add_argument("--initial-balance", type=float, default=10000.0)
    p.add_argument("--pip-value", type=float, default=10.0)
    p.add_argument("--commission", type=float, default=7.0)
    p.add_argument("--no-ml", action="store_true", help="Skip ML; TA indicators only")
    p.add_argument("--no-news", action="store_true", help="Disable news calendar module")
    p.add_argument("--no-d1", action="store_true", help="Disable D1 multi-tf module")
    p.add_argument("--model-dir", type=Path, default=Path("data/models"))
    p.add_argument("--finnhub-key", default=None)
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_bars(args: argparse.Namespace) -> list:
    tf = _TIMEFRAME_MAP[args.timeframe]
    if args.source == "csv":
        from metatrade.market_data.collectors.csv_collector import CsvCollector
        if args.file is None:
            print("ERROR: --file is required when --source csv", file=sys.stderr)
            sys.exit(1)
        collector = CsvCollector()
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
        bars = collector.collect(args.symbol, tf, date_from, date_to)
        collector.shutdown()
    return bars


# ── Per-fold metrics ───────────────────────────────────────────────────────────

def compute_max_drawdown(result: BacktestResult) -> float:
    """Maximum peak-to-trough equity drawdown as a fraction of initial balance."""
    if not result.trades:
        return 0.0
    balance = float(result.initial_balance)
    peak = balance
    max_dd = 0.0
    for trade in result.trades:
        balance += float(trade.pnl)
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_sharpe(result: BacktestResult) -> float:
    """Annualised Sharpe ratio from per-trade returns (252 trading days)."""
    if len(result.trades) < 2:
        return 0.0
    initial = float(result.initial_balance)
    trade_returns = [float(t.pnl) / initial for t in result.trades]
    mean_r = statistics.mean(trade_returns)
    std_r = statistics.stdev(trade_returns)
    if std_r == 0.0:
        return 0.0
    # Annualise assuming ~252 trades/year (rough for H1)
    return (mean_r / std_r) * math.sqrt(252)


# ── ML retraining (optional) ───────────────────────────────────────────────────

def train_on_window(bars: list, args: argparse.Namespace) -> object | None:
    """Train and return a model registry for the given bar window.

    Returns None if training fails or --no-ml is set.
    """
    if args.no_ml:
        return None
    try:
        from metatrade.ml.config import MLConfig
        from metatrade.ml.features import extract_features, MIN_FEATURE_BARS
        from metatrade.ml.labeller import label_bars
        from metatrade.ml.trainer import ModelTrainer
        from metatrade.ml.registry import ModelRegistry

        ml_cfg = MLConfig(model_registry_dir=str(args.model_dir))
        registry = ModelRegistry(config=ml_cfg, persist=False)

        labelled = label_bars(bars)
        if len(labelled) < MIN_FEATURE_BARS + 20:
            return None

        trainer = ModelTrainer(config=ml_cfg)
        snap = trainer.train(symbol=args.symbol, bars=bars)
        if snap is None:
            return None
        registry.register(snap.classifier, snap.symbol, version=snap.version)
        registry.promote(snap.version)
        return registry
    except Exception as exc:
        print(f"  [ML training skipped: {exc}]")
        return None


# ── Single fold ────────────────────────────────────────────────────────────────

def run_fold(
    fold_idx: int,
    train_bars: list,
    test_bars: list,
    args: argparse.Namespace,
) -> dict:
    """Run one fold: optionally retrain, then backtest on test window."""
    registry = train_on_window(train_bars, args)

    module_cfg = ModuleConfig(
        multi_tf_d1=not args.no_d1,
        news_calendar=not args.no_news,
        ml=not args.no_ml and registry is not None,
        finnhub_api_key=args.finnhub_key,
    )
    modules = build_modules(args.timeframe.lower(), module_cfg=module_cfg, registry=registry)

    runner_cfg = RunnerConfig(
        symbol=args.symbol,
        max_risk_pct=args.max_risk_pct,
        initial_balance=args.initial_balance,
        min_signals=1 if (args.no_ml or registry is None) else 2,
    )
    runner = BacktestRunner(
        config=runner_cfg,
        modules=modules,
        bars=test_bars,
        pip_value=args.pip_value,
        commission_per_lot=args.commission,
    )
    result = runner.run()

    return {
        "fold": fold_idx,
        "test_bars": len(test_bars),
        "n_trades": result.n_trades,
        "win_rate": result.win_rate,
        "return_pct": result.return_pct,
        "sharpe": compute_sharpe(result),
        "max_dd": compute_max_drawdown(result),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"Loading bars ({args.source}) …")
    all_bars = load_bars(args)
    n = len(all_bars)
    print(f"  {n} bars loaded.")

    if n < args.folds * 100:
        print(
            f"ERROR: Not enough bars ({n}) for {args.folds} folds "
            f"(need at least {args.folds * 100}).",
            file=sys.stderr,
        )
        sys.exit(1)

    fold_width = n // args.folds
    train_width = int(fold_width * args.train_pct)
    test_width = fold_width - train_width

    print(f"\nWalk-forward configuration:")
    print(f"  Folds       : {args.folds}")
    print(f"  Fold width  : {fold_width} bars")
    print(f"  Train window: {train_width} bars ({args.train_pct:.0%})")
    print(f"  Test window : {test_width} bars ({1 - args.train_pct:.0%})")
    print(f"  ML          : {'disabled' if args.no_ml else 'enabled (retrain each fold)'}")
    print()

    # ── Header ────────────────────────────────────────────────────────────────
    col = "{:>5}  {:>5}  {:>7}  {:>6}  {:>9}  {:>8}  {:>8}"
    print(col.format("Fold", "Bars", "Trades", "Win%", "Return", "Sharpe", "MaxDD"))
    print("─" * 58)

    fold_results: list[dict] = []

    for fold_idx in range(1, args.folds + 1):
        start = (fold_idx - 1) * fold_width
        train_bars = all_bars[start : start + train_width]
        test_bars  = all_bars[start + train_width : start + fold_width]

        if len(test_bars) < 50:
            print(f"  Fold {fold_idx}: skipped (insufficient test bars: {len(test_bars)})")
            continue

        metrics = run_fold(fold_idx, train_bars, test_bars, args)
        fold_results.append(metrics)

        win_str = f"{metrics['win_rate']:.1%}" if metrics["n_trades"] > 0 else "  n/a"
        print(
            col.format(
                metrics["fold"],
                metrics["test_bars"],
                metrics["n_trades"],
                win_str,
                f"{metrics['return_pct']:+.2%}",
                f"{metrics['sharpe']:.2f}",
                f"{-metrics['max_dd']:.2%}",
            )
        )

    if not fold_results:
        print("No folds completed.")
        return

    # ── Aggregates ────────────────────────────────────────────────────────────
    print("─" * 58)
    avg_trades  = sum(r["n_trades"] for r in fold_results) / len(fold_results)
    avg_win     = sum(r["win_rate"] for r in fold_results if r["n_trades"] > 0)
    n_with_trades = sum(1 for r in fold_results if r["n_trades"] > 0)
    avg_win_str = f"{avg_win / n_with_trades:.1%}" if n_with_trades > 0 else "  n/a"
    avg_return  = sum(r["return_pct"] for r in fold_results) / len(fold_results)
    avg_sharpe  = sum(r["sharpe"] for r in fold_results) / len(fold_results)
    avg_maxdd   = sum(r["max_dd"] for r in fold_results) / len(fold_results)

    print(
        col.format(
            "Mean",
            "—",
            f"{avg_trades:.1f}",
            avg_win_str,
            f"{avg_return:+.2%}",
            f"{avg_sharpe:.2f}",
            f"{-avg_maxdd:.2%}",
        )
    )
    print()

    # ── Stability check ───────────────────────────────────────────────────────
    n_negative = sum(1 for r in fold_results if r["return_pct"] < 0)
    if n_negative >= 2:
        print(
            f"⚠️  UNSTABLE: {n_negative}/{len(fold_results)} folds produced negative returns.\n"
            f"   The system does not generalise consistently across time windows.\n"
            f"   Consider: tighter risk settings, different consensus threshold,\n"
            f"   or gathering more training data before going live."
        )
    elif n_negative == 1:
        print(
            f"⚡ MARGINAL: 1 fold produced a negative return. "
            f"Monitor closely before going live."
        )
    else:
        print(
            f"✅ STABLE: all {len(fold_results)} folds produced positive returns "
            f"(avg Sharpe {avg_sharpe:.2f})."
        )

    print()
    print(f"Avg return : {avg_return:+.2%}")
    print(f"Avg Sharpe : {avg_sharpe:.2f}")
    print(f"Avg max DD : {avg_maxdd:.2%}")


if __name__ == "__main__":
    main()
