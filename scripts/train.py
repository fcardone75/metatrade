"""Train the ML model on historical data using walk-forward validation.

Usage examples:

    # From CSV file
    python scripts/train.py --source csv --file data/EURUSD_H1.csv --symbol EURUSD

    # From MetaTrader 5 (Windows only, MT5 must be open)
    python scripts/train.py --source mt5 --symbol EURUSD --timeframe H1

    # Piu' timeframe da MT5 (M5, M15, M30) con report JSON e telemetria
    python scripts/train.py --source mt5 --symbol EURUSD --timeframes M5,M15,M30

    # CSV multi-timeframe: usa segnaposto nel path file
    python scripts/train.py --source csv --file data/EURUSD_{TIMEFRAME}.csv --symbol EURUSD \\
        --timeframes M5,M15,M30

    # Quale modello risulta attivo in dashboard (default: ultimo della lista)
    python scripts/train.py --source mt5 --symbol EURUSD --timeframes M5,M15,M30 \\
        --promote-timeframe M15

The script:
  1. Loads historical bars (CSV or MT5)
  2. Runs walk-forward training (expanding window)
  3. Prints fold-by-fold accuracy e riepilogo strutturato (log JSON se configurato)
  4. Registers the best model to disk
  5. Promotes one model as active in registry (ultimo timeframe o --promote-timeframe)
  6. Scrive un report JSON in data/models/training_reports/ (disattiva con --no-report-json)

Monitoraggio avanzamento walk-forward (mentre gira il training):

    # File aggiornato ad ogni fold completato (MODEL_DIR/training_progress.json)
    Get-Content data\\models\\training_progress.json

    # PowerShell: aggiorna ogni 3 secondi
    while ($true) { Clear-Host; Get-Content data\\models\\training_progress.json; Start-Sleep 3 }
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metatrade.core.enums import Timeframe  # noqa: E402
from metatrade.core.log import configure_logging, get_logger  # noqa: E402
from metatrade.market_data.config import MarketDataConfig  # noqa: E402
from metatrade.ml.config import MLConfig  # noqa: E402
from metatrade.ml.registry import ModelRegistry  # noqa: E402
from metatrade.ml.walk_forward import (  # noqa: E402
    WalkForwardFold,
    WalkForwardResult,
    WalkForwardTrainer,
)
from metatrade.observability.store import TelemetryStore  # noqa: E402

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


@dataclass
class TimeframeTrainReport:
    """Esito training per un singolo timeframe (serializzabile in JSON)."""

    timeframe: str
    success: bool
    error: str | None
    bars_used: int
    duration_sec: float
    model_version: str | None
    mean_test_accuracy: float | None
    best_test_accuracy: float | None
    best_fold_index: int | None
    n_folds: int
    folds: list[dict[str, Any]]
    artifact_path: str | None
    holdout_accuracy: float | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Forex ML model with walk-forward validation",
    )
    p.add_argument(
        "--source",
        choices=["csv", "mt5"],
        required=True,
        help="Data source: csv (local file) or mt5 (MetaTrader 5, Windows only)",
    )
    p.add_argument("--file", type=Path, help="Path to CSV file (required for --source csv)")
    p.add_argument("--symbol", default="EURUSD", help="Symbol name (e.g. EURUSD)")
    p.add_argument(
        "--timeframe",
        default="H1",
        choices=list(_TIMEFRAME_MAP),
        help="Timeframe singolo (default: H1). Ignorato se --timeframes e' impostato.",
    )
    p.add_argument(
        "--timeframes",
        default=None,
        metavar="M5,M15,M30",
        help="Piu' timeframe separati da virgola o spazio (es. M5,M15,M30). Sostituisce --timeframe.",
    )
    p.add_argument(
        "--promote-timeframe",
        default=None,
        metavar="M15",
        help="Dopo training multiplo, quale timeframe e' attivo in telemetria (default: ultimo in --timeframes).",
    )
    p.add_argument(
        "--no-report-json",
        action="store_true",
        help="Non scrivere il file JSON di riepilogo in MODEL_DIR/training_reports/.",
    )
    p.add_argument(
        "--bars",
        type=int,
        default=30_000,
        help="Numero di barre da scaricare da MT5 (default: 30000; piu' barre = piu' storia per walk-forward)",
    )
    p.add_argument("--mt5-login", type=int, default=0, help="MT5 account login")
    p.add_argument("--mt5-password", default="", help="MT5 account password")
    p.add_argument("--mt5-server", default="", help="MT5 broker server name")
    p.add_argument("--mt5-path", default="", help="Path to terminal64.exe if MT5 is not auto-detected")
    p.add_argument("--mt5-timeout-ms", type=int, default=0, help="MT5 initialization timeout in milliseconds")
    p.add_argument("--train-window", type=int, default=2000, help="Training window in bars")
    p.add_argument("--test-window", type=int, default=500, help="Test window in bars")
    p.add_argument("--step", type=int, default=250, help="Step size in bars between folds")
    p.add_argument("--forward-bars", type=int, default=5, help="Bars ahead for label generation")
    p.add_argument("--max-iter", type=int, default=200, help="HistGradientBoosting max iterations")
    p.add_argument("--max-depth", type=int, default=5, help="RandomForest max depth")
    p.add_argument("--min-accuracy", type=float, default=0.52, help="Min accuracy to accept model")
    p.add_argument(
        "--atr-threshold-mult",
        type=float,
        default=0.8,
        help="ATR multiplier for label threshold (no-trade zone). Use 0.7-0.9 for M1/M5, 1.0-1.5 for H1+",
    )
    p.add_argument(
        "--multires",
        action="store_true",
        default=False,
        help=(
            "Multi-resolution mode: load M5 and M15 bars alongside the primary "
            "timeframe and use 43-feature MultiResFeatureVector. "
            "Recommended for M1 training. Requires --source mt5 or a CSV path "
            "with {TIMEFRAME} placeholder."
        ),
    )
    p.add_argument(
        "--session-start-utc",
        type=int,
        default=None,
        metavar="HOUR",
        help=(
            "Session filter start (UTC hour 0-23). Combined with --session-end-utc, "
            "only bars in [start, end) are used. E.g. --session-start-utc 7 "
            "--session-end-utc 21 keeps London + NY sessions."
        ),
    )
    p.add_argument(
        "--session-end-utc",
        type=int,
        default=None,
        metavar="HOUR",
        help="Session filter end (UTC hour 0-23, exclusive). See --session-start-utc.",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/models"),
        help="Directory for model persistence (default: data/models)",
    )
    p.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        metavar="FRACTION",
        help=(
            "Fraction of bars (from the end) reserved as final out-of-sample holdout. "
            "E.g. 0.2 = last 20%% used only for post-training evaluation. "
            "Walk-forward trains only on the remaining bars. Default: 0 (disabled)."
        ),
    )
    p.add_argument(
        "--auto-tune",
        action="store_true",
        default=False,
        help=(
            "Automatically search over forward_bars × atr_threshold_mult combinations "
            "to maximise holdout accuracy. Requires --holdout-fraction > 0. "
            "Feature cache is precomputed once and reused across all trials."
        ),
    )
    p.add_argument(
        "--auto-tune-target",
        type=float,
        default=0.52,
        metavar="ACCURACY",
        help="Stop auto-tune early when holdout accuracy reaches this threshold (default: 0.52).",
    )
    p.add_argument(
        "--auto-tune-max-trials",
        type=int,
        default=20,
        metavar="N",
        help="Maximum number of parameter combinations to try during auto-tune (default: 20).",
    )
    p.add_argument(
        "--adaptive",
        action="store_true",
        default=False,
        help=(
            "Adaptive training loop: retries with escalating complexity-reduction until a "
            "model beats the target accuracy. Target = min_accuracy if no model exists, "
            "else max(current_holdout + adaptive-min-improvement, adaptive-absolute-min). "
            "Implies --auto-tune and --holdout-fraction 0.2 if not already set."
        ),
    )
    p.add_argument(
        "--adaptive-max-attempts",
        type=int,
        default=20,
        metavar="N",
        help="Max training attempts in adaptive mode (default: 20).",
    )
    p.add_argument(
        "--adaptive-min-improvement",
        type=float,
        default=0.01,
        metavar="PP",
        help="Minimum improvement over current model holdout to accept (default: 0.01 = 1pp).",
    )
    p.add_argument(
        "--adaptive-absolute-min",
        type=float,
        default=0.53,
        metavar="ACC",
        help="Absolute minimum holdout required even when improving over current model (default: 0.53).",
    )
    return p.parse_args()


def apply_mt5_defaults(args: argparse.Namespace) -> argparse.Namespace:
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


def parse_timeframe_list(args: argparse.Namespace) -> list[str]:
    if args.timeframes:
        raw = args.timeframes.replace(",", " ").split()
        out = [t.strip().upper() for t in raw if t.strip()]
        if not out:
            print("ERROR: --timeframes e' vuoto.", file=sys.stderr)
            sys.exit(1)
        for t in out:
            if t not in _TIMEFRAME_MAP:
                print(
                    f"ERROR: timeframe sconosciuto {t!r}. Validi: {', '.join(_TIMEFRAME_MAP)}.",
                    file=sys.stderr,
                )
                sys.exit(1)
        return out
    return [args.timeframe]


def resolve_promote_timeframe(timeframes: list[str], args: argparse.Namespace) -> str:
    if args.promote_timeframe:
        target = args.promote_timeframe.strip().upper()
        if target not in timeframes:
            print(
                f"ERROR: --promote-timeframe={target!r} non e' tra i timeframe addestrati: {timeframes}.",
                file=sys.stderr,
            )
            sys.exit(1)
        return target
    return timeframes[-1]


def resolve_csv_path(file_arg: Path, timeframe: str, *, multi: bool) -> Path:
    s = str(file_arg)
    if "{TIMEFRAME}" in s or "{TF}" in s or "{timeframe}" in s:
        return Path(
            s.replace("{TIMEFRAME}", timeframe)
            .replace("{TF}", timeframe)
            .replace("{timeframe}", timeframe)
        )
    if multi:
        print(
            "ERROR: con piu' timeframe da CSV usa un segnaposto in --file, "
            "es. data/EURUSD_{TIMEFRAME}.csv",
            file=sys.stderr,
        )
        sys.exit(1)
    return file_arg


def fold_to_dict(fold: WalkForwardFold) -> dict[str, Any]:
    return {
        "fold_index": fold.fold_index,
        "train_bars": fold.train_end - fold.train_start,
        "n_test_samples": fold.n_test_samples,
        "train_accuracy": float(fold.train_metrics.accuracy),
        "test_accuracy": float(fold.test_accuracy),
    }


def load_from_csv(path: Path, symbol: str, timeframe: str):
    from metatrade.market_data.collectors.csv_collector import CsvCollector

    if not path.exists():
        print(f"ERROR: file CSV non trovato: {path}", file=sys.stderr)
        sys.exit(1)

    tf = _TIMEFRAME_MAP[timeframe]
    collector = CsvCollector()
    print(f"Caricamento barre da {path} ...")
    bars = collector.collect_file(
        file_path=path,
        symbol=symbol,
        timeframe=tf,
    )
    print(f"  Caricate {len(bars)} barre ({timeframe}).")
    return bars


_TF_MINUTES: dict[str, int] = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}


def load_from_mt5(args: argparse.Namespace, timeframe: str, *, bars_count: int | None = None):
    from metatrade.market_data.collectors.mt5_collector import MT5Collector

    n_bars = bars_count if bars_count is not None else args.bars

    collector = MT5Collector(store=None)
    collector.initialize(
        login=args.mt5_login,
        password=args.mt5_password,
        server=args.mt5_server,
        path=args.mt5_path,
        timeout=args.mt5_timeout_ms,
    )

    tf = _TIMEFRAME_MAP[timeframe]
    date_to = datetime.now(UTC)
    minutes = _TF_MINUTES.get(timeframe, 60) * n_bars
    date_from = date_to - timedelta(minutes=minutes)

    print(f"Scarico {n_bars} barre {args.symbol}/{timeframe} da MT5 ...")
    bars = collector.collect(args.symbol, tf, date_from, date_to)
    collector.shutdown()
    print(f"  Scaricate {len(bars)} barre.")
    return bars


def load_higher_tf_bars_mt5(args: argparse.Namespace, primary_tf: str) -> dict[str, list]:
    """Load M5 and M15 context bars for multi-resolution training.

    Bar counts are derived from the primary M1 bar count so the time window
    is equivalent across all timeframes.
    """
    primary_minutes = _TF_MINUTES.get(primary_tf, 1)
    total_minutes = primary_minutes * args.bars

    result: dict[str, list] = {}
    for htf in ("M5", "M15"):
        htf_min = _TF_MINUTES[htf]
        n = max(200, total_minutes // htf_min + 100)
        result[htf] = load_from_mt5(args, htf, bars_count=n)
    return result


def load_higher_tf_bars_csv(
    file_arg: Path, symbol: str, primary_tf: str
) -> dict[str, list]:
    """Load M5 and M15 context bars from CSV files (requires {TIMEFRAME} placeholder)."""
    result: dict[str, list] = {}
    for htf in ("M5", "M15"):
        path = resolve_csv_path(file_arg, htf, multi=True)
        result[htf] = load_from_csv(path, symbol, htf)
    return result


TRAINING_PROGRESS_JSON = "training_progress.json"


def training_progress_path(model_dir: Path) -> Path:
    return model_dir / TRAINING_PROGRESS_JSON


def _estimate_max_walk_forward_folds(n_bars: int, train_w: int, test_w: int, step: int) -> int:
    """Limite superiore grezzo sul numero di fold (alcuni step possono essere saltati)."""
    if n_bars <= train_w + test_w or step <= 0:
        return 1
    return max(1, (n_bars - train_w - test_w) // step + 1)


def _atomic_write_training_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _print_fold_table(result: WalkForwardResult) -> None:
    print(f"{'Fold':>5}  {'Train bars':>11}  {'Test bars':>10}  {'Train acc':>10}  {'Test acc':>9}")
    print("-" * 55)
    for fold in result.folds:
        print(
            f"{fold.fold_index:>5}  "
            f"{fold.train_end - fold.train_start:>11}  "
            f"{fold.n_test_samples:>10}  "
            f"{fold.train_metrics.accuracy:>10.2%}  "
            f"{fold.test_accuracy:>9.2%}",
        )
    print("-" * 55)
    print(f"{'Mean':>5}  {'':>11}  {'':>10}  {'':>10}  {result.mean_test_accuracy:>9.2%}")
    print(f"{'Best':>5}  {'':>11}  {'':>10}  {'':>10}  {result.best_test_accuracy:>9.2%}")


def train_single_timeframe(
    *,
    args: argparse.Namespace,
    timeframe: str,
    bars: list,
    ml_cfg: MLConfig,
    telemetry: TelemetryStore,
    telemetry_is_active: bool,
    higher_tf_bars: dict[str, list] | None = None,
) -> TimeframeTrainReport:
    t0 = time.perf_counter()
    log.info(
        "train_timeframe_start",
        symbol=args.symbol,
        timeframe=timeframe,
        n_bars=len(bars),
        source=args.source,
        train_window=ml_cfg.train_window_bars,
        test_window=ml_cfg.test_window_bars,
        step=ml_cfg.step_bars,
    )

    if len(bars) < args.train_window + args.test_window:
        msg = (
            f"Barre insufficienti ({len(bars)}) per train={args.train_window} + test={args.test_window}."
        )
        log.warning("train_timeframe_skipped", symbol=args.symbol, timeframe=timeframe, reason=msg)
        _atomic_write_training_progress(
            training_progress_path(args.model_dir),
            {
                "status": "error",
                "symbol": args.symbol,
                "timeframe": timeframe,
                "phase": "skipped",
                "message": msg,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        return TimeframeTrainReport(
            timeframe=timeframe,
            success=False,
            error=msg,
            bars_used=len(bars),
            duration_sec=round(time.perf_counter() - t0, 3),
            model_version=None,
            mean_test_accuracy=None,
            best_test_accuracy=None,
            best_fold_index=None,
            n_folds=0,
            folds=[],
            artifact_path=None,
        )

    trainer = WalkForwardTrainer(ml_cfg)
    multires_active = higher_tf_bars is not None and bool(higher_tf_bars)
    multires_label = (
        f"  [multires: {', '.join(f'{k}={len(v)}bar' for k,v in higher_tf_bars.items())}]"
        if multires_active else ""
    )
    session_label = (
        f"  [session: {ml_cfg.session_filter_utc_start}–{ml_cfg.session_filter_utc_end} UTC]"
        if ml_cfg.session_filter_utc_start is not None else ""
    )
    print(f"\n=== Walk-forward: {args.symbol} {timeframe} ({len(bars)} barre) ===")
    print(f"  train_window={ml_cfg.train_window_bars}  test_window={ml_cfg.test_window_bars}  step={ml_cfg.step_bars}")
    print(f"  forward_bars={ml_cfg.forward_bars}  atr_threshold_mult={ml_cfg.atr_threshold_mult}  class_weight={ml_cfg.class_weight}")
    if multires_label:
        print(multires_label)
    if session_label:
        print(session_label)
    print()

    n_bars = len(bars)
    est_folds = _estimate_max_walk_forward_folds(
        n_bars,
        ml_cfg.train_window_bars,
        ml_cfg.test_window_bars,
        ml_cfg.step_bars,
    )
    print(f"  Fold stimati: ~{est_folds}  |  Calcolo feature + label in corso...")
    print(f"  (Il primo fold puo' richiedere 20-60s per il feature engineering)")
    print()

    prog_path = training_progress_path(args.model_dir)
    _atomic_write_training_progress(
        prog_path,
        {
            "status": "walk_forward",
            "symbol": args.symbol,
            "timeframe": timeframe,
            "phase": "starting",
            "total_bars": n_bars,
            "estimated_folds_upper_bound": est_folds,
            "folds_completed": 0,
            "progress_pct_approx": 0.0,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )

    # Rolling stats for ETA and running accuracy
    _fold_times: list[float] = []
    _fold_last_ts = [time.perf_counter()]  # mutable container for closure

    def on_fold_complete(fold: WalkForwardFold) -> None:
        done = fold.fold_index + 1
        pct = min(100.0, 100.0 * done / max(est_folds, 1))
        elapsed_total = time.perf_counter() - t0

        # Per-fold timing for ETA
        now = time.perf_counter()
        fold_dur = now - _fold_last_ts[0]
        _fold_last_ts[0] = now
        _fold_times.append(fold_dur)
        avg_fold_sec = sum(_fold_times[-10:]) / len(_fold_times[-10:])  # last 10
        remaining_folds = max(0, est_folds - done)
        eta_sec = avg_fold_sec * remaining_folds

        # Running mean test accuracy
        all_test_acc = [f.test_accuracy for f in [fold]]  # only current available here
        test_acc = float(fold.test_accuracy)

        # Class distribution from this fold's training metrics
        cd = fold.train_metrics.class_distribution
        total_cd = sum(cd.values()) or 1
        labels_map = {1: "B", -1: "S", 0: "H"}
        dist_str = " ".join(
            f"{labels_map.get(k, str(k))}={v}({v*100//total_cd}%)"
            for k, v in sorted(cd.items())
        )

        # ETA formatting
        if eta_sec < 60:
            eta_str = f"{eta_sec:.0f}s"
        else:
            eta_str = f"{eta_sec/60:.1f}m"

        # Console line — overwrite same line for compact output
        status_char = "+" if test_acc >= ml_cfg.min_accuracy else "-"
        print(
            f"  [{status_char}] Fold {done:>3}/{est_folds}"
            f"  acc={test_acc:.2%}"
            f"  {dist_str}"
            f"  fold={fold_dur:.1f}s"
            f"  elapsed={elapsed_total:.0f}s"
            f"  ETA~{eta_str}",
            flush=True,
        )

        payload: dict[str, Any] = {
            "status": "walk_forward",
            "symbol": args.symbol,
            "timeframe": timeframe,
            "phase": "fold",
            "total_bars": n_bars,
            "fold_index": fold.fold_index,
            "folds_completed": done,
            "estimated_folds_upper_bound": est_folds,
            "progress_pct_approx": round(pct, 1),
            "train_end_bar": fold.train_end,
            "test_accuracy_last": round(test_acc, 4),
            "elapsed_sec": round(elapsed_total, 1),
            "eta_sec": round(eta_sec, 0),
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_write_training_progress(prog_path, payload)
        log.info(
            "train_walk_forward_fold",
            symbol=args.symbol,
            timeframe=timeframe,
            fold_index=fold.fold_index,
            folds_completed=done,
            test_accuracy=round(test_acc, 4),
            fold_sec=round(fold_dur, 2),
            eta_sec=round(eta_sec, 0),
        )

    def on_precompute_progress(current: int, total: int) -> None:
        pct = 100.0 * current / max(total, 1)
        print(f"  Precomputing features: {pct:.0f}%  ({current}/{total} bars)", end="\r", flush=True)
        _atomic_write_training_progress(
            prog_path,
            {
                "status": "walk_forward",
                "symbol": args.symbol,
                "timeframe": timeframe,
                "phase": "precomputing_features",
                "precompute_pct": round(pct, 1),
                "precompute_bar": current,
                "precompute_total": total,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )

    result, model = trainer.run(
        bars,
        on_fold_complete=on_fold_complete,
        higher_tf_bars=higher_tf_bars or None,
        on_precompute_progress=on_precompute_progress,
    )

    # Summary after all folds
    print()
    print(f"  Walk-forward completato: {len(result.folds)} fold in {time.perf_counter()-t0:.0f}s")
    if result.folds:
        above = sum(1 for f in result.folds if f.test_accuracy >= ml_cfg.min_accuracy)
        print(f"  Fold sopra soglia ({ml_cfg.min_accuracy:.0%}): {above}/{len(result.folds)}")
        print(f"  Test accuracy — mean: {result.mean_test_accuracy:.2%}  best: {result.best_test_accuracy:.2%}")
    if result.holdout_accuracy is not None:
        print(f"  Holdout accuracy ({result.holdout_n_samples} campioni): {result.holdout_accuracy:.2%}")
    print()

    if not result.folds:
        msg = "Nessun fold prodotto: dati insufficienti."
        log.error("train_timeframe_failed", symbol=args.symbol, timeframe=timeframe, reason=msg)
        print(f"ERROR: {msg}", file=sys.stderr)
        _atomic_write_training_progress(
            prog_path,
            {
                "status": "error",
                "symbol": args.symbol,
                "timeframe": timeframe,
                "phase": "no_folds",
                "message": msg,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        return TimeframeTrainReport(
            timeframe=timeframe,
            success=False,
            error=msg,
            bars_used=len(bars),
            duration_sec=round(time.perf_counter() - t0, 3),
            model_version=None,
            mean_test_accuracy=None,
            best_test_accuracy=None,
            best_fold_index=None,
            n_folds=0,
            folds=[],
            artifact_path=None,
        )

    _print_fold_table(result)
    print()

    # Print label distribution from the best fold's training metrics
    if model is not None and model.metrics is not None:
        cd = model.metrics.class_distribution
        total = sum(cd.values()) or 1
        labels = {1: "BUY", -1: "SELL", 0: "HOLD"}
        dist_parts = [
            f"{labels.get(k, str(k))}={v} ({v/total:.0%})"
            for k, v in sorted(cd.items())
        ]
        print(f"  Class distribution (best fold train): {', '.join(dist_parts)}")
        print()

    # Gate on OUT-OF-SAMPLE (test) accuracy, not in-sample.
    # model.is_trained uses in-sample accuracy from fit() — a model can have
    # 95% in-sample and 43% test accuracy and pass that check easily.
    # We require best_test_accuracy >= min_accuracy before saving anything.
    test_acc_ok = result.best_test_accuracy >= ml_cfg.min_accuracy
    if model is None or not model.is_trained or not test_acc_ok:
        if not test_acc_ok and model is not None and model.is_trained:
            msg = (
                f"Test accuracy ({result.best_test_accuracy:.0%}) sotto la soglia min_accuracy "
                f"({ml_cfg.min_accuracy:.0%}) — modello non salvato (overfit o dati insufficienti). "
                "Prova --bars piu' alto o abbassa --min-accuracy."
            )
        else:
            msg = (
                f"Nessun modello sopra la soglia min_accuracy ({ml_cfg.min_accuracy:.0%}). "
                "Prova --bars piu' alto, --min-accuracy piu' basso, o verifica i dati."
            )
        # Compute overfitting ratio: best train acc / best test acc
        best_fold = result.folds[result.best_fold_index] if result.folds else None
        overfit_ratio = (
            round(best_fold.train_accuracy / max(best_fold.test_accuracy, 0.001), 3)
            if best_fold else None
        )
        fold_accuracies = [round(f.test_accuracy, 4) for f in result.folds]
        class_dist: dict | None = None
        if model is not None and model.metrics is not None:
            class_dist = {str(k): v for k, v in model.metrics.class_distribution.items()}
        log.warning(
            "train_timeframe_below_threshold",
            symbol=args.symbol,
            timeframe=timeframe,
            best_test_accuracy=round(float(result.best_test_accuracy), 4),
            mean_test_accuracy=round(float(result.mean_test_accuracy), 4),
            min_accuracy=ml_cfg.min_accuracy,
            n_folds=len(result.folds),
            n_folds_above_threshold=sum(
                1 for f in result.folds if f.test_accuracy >= ml_cfg.min_accuracy
            ),
            best_fold_train_accuracy=round(best_fold.train_accuracy, 4) if best_fold else None,
            overfit_ratio=overfit_ratio,
            fold_test_accuracies=fold_accuracies,
            class_distribution=class_dist,
            forward_bars=ml_cfg.forward_bars,
            atr_threshold_mult=ml_cfg.atr_threshold_mult,
            max_depth=ml_cfg.max_depth,
            max_iter=ml_cfg.max_iter,
            holdout_accuracy=round(result.holdout_accuracy, 4) if result.holdout_accuracy else None,
            bars_used=len(bars),
            message=msg,
        )
        print(f"WARNING: {msg}", file=sys.stderr)
        _atomic_write_training_progress(
            prog_path,
            {
                "status": "warning",
                "symbol": args.symbol,
                "timeframe": timeframe,
                "phase": "below_min_accuracy",
                "message": msg,
                "n_folds": len(result.folds),
                "mean_test_accuracy": float(result.mean_test_accuracy),
                "best_test_accuracy": float(result.best_test_accuracy),
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        return TimeframeTrainReport(
            timeframe=timeframe,
            success=False,
            error=msg,
            bars_used=len(bars),
            duration_sec=round(time.perf_counter() - t0, 3),
            model_version=None,
            mean_test_accuracy=float(result.mean_test_accuracy),
            best_test_accuracy=float(result.best_test_accuracy),
            best_fold_index=int(result.best_fold_index),
            n_folds=len(result.folds),
            folds=[fold_to_dict(f) for f in result.folds],
            artifact_path=None,
        )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(config=ml_cfg, persist=True)
    best_fold = result.folds[result.best_fold_index]
    version = f"v{time.strftime('%Y%m%d_%H%M%S')}_{timeframe}"
    snapshot = registry.register(
        classifier=model,
        symbol=args.symbol,
        version=version,
        tags={
            "source": args.source,
            "n_folds": len(result.folds),
            "mean_test_acc": round(result.mean_test_accuracy, 4),
            "best_test_acc": round(result.best_test_accuracy, 4),
            "best_fold": best_fold.fold_index,
            "train_bars": len(bars),
            "timeframe": timeframe,
        },
    )
    registry.promote(snapshot.version)

    artifact_path = args.model_dir / f"{args.symbol}_{snapshot.version}.pkl"
    duration_sec = round(time.perf_counter() - t0, 3)

    print(f"Modello salvato e registrato ({timeframe}).")
    print(f"  Versione : {snapshot.version}")
    print(f"  Symbol   : {snapshot.symbol}")
    print(f"  Accuracy : {result.best_test_accuracy:.2%} (miglior fold)")
    print(f"  File     : {artifact_path}")
    print(f"  Durata   : {duration_sec}s")
    print()

    fold_details = [fold_to_dict(f) for f in result.folds]
    telemetry.record_training_run(
        symbol=args.symbol,
        timeframe=timeframe,
        source=args.source,
        bars_requested=args.bars if args.source == "mt5" else len(bars),
        bars_fetched=len(bars),
        train_window=args.train_window,
        test_window=args.test_window,
        step=args.step,
        forward_bars=args.forward_bars,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        min_accuracy=args.min_accuracy,
        mean_test_accuracy=result.mean_test_accuracy,
        best_test_accuracy=result.best_test_accuracy,
        best_fold=best_fold.fold_index,
        model_version=snapshot.version,
        status="completed",
        details={
            "artifact_path": str(artifact_path),
            "duration_sec": duration_sec,
            "telemetry_is_active": telemetry_is_active,
            "folds": fold_details,
        },
    )
    telemetry.record_model_artifact(
        version=snapshot.version,
        symbol=args.symbol,
        timeframe=timeframe,
        source=args.source,
        train_bars=len(bars),
        n_folds=len(result.folds),
        mean_test_accuracy=result.mean_test_accuracy,
        best_test_accuracy=result.best_test_accuracy,
        artifact_path=str(artifact_path),
        is_active=telemetry_is_active,
        details={
            "best_fold": best_fold.fold_index,
            "mean_test_acc": result.mean_test_accuracy,
            "best_test_acc": result.best_test_accuracy,
            "duration_sec": duration_sec,
        },
    )

    log.info(
        "train_timeframe_done",
        symbol=args.symbol,
        timeframe=timeframe,
        version=snapshot.version,
        n_folds=len(result.folds),
        mean_test_accuracy=round(result.mean_test_accuracy, 4),
        best_test_accuracy=round(result.best_test_accuracy, 4),
        duration_sec=duration_sec,
        telemetry_is_active=telemetry_is_active,
    )

    _atomic_write_training_progress(
        prog_path,
        {
            "status": "timeframe_complete",
            "symbol": args.symbol,
            "timeframe": timeframe,
            "phase": "model_saved",
            "model_version": snapshot.version,
            "artifact_path": str(artifact_path),
            "n_folds": len(result.folds),
            "mean_test_accuracy": float(result.mean_test_accuracy),
            "best_test_accuracy": float(result.best_test_accuracy),
            "duration_sec": duration_sec,
            "telemetry_is_active": telemetry_is_active,
            "progress_pct_approx": 100.0,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )

    return TimeframeTrainReport(
        timeframe=timeframe,
        success=True,
        error=None,
        bars_used=len(bars),
        duration_sec=duration_sec,
        model_version=snapshot.version,
        mean_test_accuracy=float(result.mean_test_accuracy),
        best_test_accuracy=float(result.best_test_accuracy),
        best_fold_index=int(result.best_fold_index),
        n_folds=len(result.folds),
        folds=fold_details,
        artifact_path=str(artifact_path),
        holdout_accuracy=float(result.holdout_accuracy) if result.holdout_accuracy is not None else None,
    )


_AUTO_TUNE_GRID = list(itertools.product(
    [15, 20, 10, 25, 30],           # forward_bars
    [0.5, 0.6, 0.4, 0.7, 0.8],     # atr_threshold_mult
))


# ─── Tried-params cache ───────────────────────────────────────────────────────

class TriedParamsCache:
    """Persist tried (forward_bars, atr_mult) combos so restarts skip them.

    File: ``{model_dir}/{symbol}_tune_cache.json``

    Each entry records the holdout accuracy and when it was tried.  On restart
    the cache is loaded and already-tried combos are skipped in the grid search.
    Delete the file to reset the cache and retry all combinations.
    """

    def __init__(self, model_dir: Path, symbol: str) -> None:
        self._path = model_dir / f"{symbol}_tune_cache.json"
        self._data: dict[str, dict] = {}
        self._load()

    def _key(self, forward_bars: int, atr_mult: float) -> str:
        return f"{forward_bars}_{atr_mult:.2f}"

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as exc:
                log.warning(
                    "tune_cache_load_failed",
                    path=str(self._path),
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
                self._data = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as exc:
            log.warning(
                "tune_cache_save_failed",
                path=str(self._path),
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            print(f"WARNING: tune cache save failed: {exc}", file=sys.stderr)

    def was_tried(self, forward_bars: int, atr_mult: float) -> bool:
        return self._key(forward_bars, atr_mult) in self._data

    def get_result(self, forward_bars: int, atr_mult: float) -> dict | None:
        return self._data.get(self._key(forward_bars, atr_mult))

    def record(
        self,
        forward_bars: int,
        atr_mult: float,
        holdout: float | None,
        version: str | None,
    ) -> None:
        self._data[self._key(forward_bars, atr_mult)] = {
            "forward_bars": forward_bars,
            "atr_mult": atr_mult,
            "holdout": holdout,
            "version": version,
            "tried_at": datetime.now(UTC).isoformat(),
        }
        self._save()

    def best_known(self) -> tuple[int, float, float] | None:
        """Return (forward_bars, atr_mult, holdout) of best cached result, or None."""
        best = max(
            (v for v in self._data.values() if v.get("holdout") is not None),
            key=lambda v: v["holdout"],
            default=None,
        )
        if best is None:
            return None
        return best["forward_bars"], best["atr_mult"], best["holdout"]

    def summary(self) -> str:
        n = len(self._data)
        best = self.best_known()
        if best:
            return f"{n} tried, best holdout {best[2]:.2%} (fwd={best[0]}, atr={best[1]:.2f})"
        return f"{n} tried, no valid results yet"


def auto_tune_single_timeframe(
    *,
    args: argparse.Namespace,
    timeframe: str,
    bars: list,
    base_ml_cfg: MLConfig,
    telemetry: TelemetryStore,
    telemetry_is_active: bool,
    higher_tf_bars: dict[str, list] | None = None,
) -> TimeframeTrainReport:
    """Grid search over forward_bars × atr_threshold_mult, reusing a single feature cache."""
    t0 = time.perf_counter()
    n_bars = len(bars)
    prog_path = training_progress_path(args.model_dir)
    target = args.auto_tune_target
    max_trials = min(args.auto_tune_max_trials, len(_AUTO_TUNE_GRID))

    cache = TriedParamsCache(args.model_dir, args.symbol)

    print(f"\n=== Auto-tune: {args.symbol} {timeframe} ({n_bars} barre) ===")
    print(f"  holdout={base_ml_cfg.holdout_fraction:.0%}  target={target:.0%}  max_trials={max_trials}")
    if cache.summary():
        print(f"  cache: {cache.summary()}")
    print()

    # ── Precompute feature cache once ─────────────────────────────────────────
    trainer_base = WalkForwardTrainer(base_ml_cfg)

    precompute_start = time.perf_counter()
    print("  [1/2] Precomputing feature cache (una sola volta per tutti i trial)...")

    def _precompute_progress(current: int, total: int) -> None:
        pct = 100.0 * current / max(total, 1)
        print(f"    Features: {pct:.0f}%  ({current}/{total})", end="\r", flush=True)

    feature_cache = trainer_base.precompute_features(
        bars,
        higher_tf_bars=higher_tf_bars,
        on_progress=_precompute_progress,
    )
    print(f"    Features: 100%  ({n_bars}/{n_bars}) — {time.perf_counter()-precompute_start:.1f}s")
    print()

    # ── Grid search ───────────────────────────────────────────────────────────
    print(f"  [2/2] Grid search ({max_trials} trial)...")
    print(f"  {'Trial':>5}  {'fwd':>4}  {'atr_mult':>8}  {'folds':>6}  {'mean_test':>10}  {'holdout':>8}  {'elapsed':>8}")
    print("  " + "-" * 62)

    best_holdout: float = -1.0
    best_result: WalkForwardResult | None = None
    best_model = None
    best_cfg: MLConfig | None = None
    best_trial_idx: int = -1

    _atomic_write_training_progress(
        prog_path,
        {
            "status": "auto_tune",
            "symbol": args.symbol,
            "timeframe": timeframe,
            "phase": "grid_search",
            "max_trials": max_trials,
            "trials_done": 0,
            "best_holdout": None,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )

    # Filter out already-tried combos to avoid redundant runs across restarts
    pending_grid = [
        (fwd, atr_mult)
        for fwd, atr_mult in _AUTO_TUNE_GRID[:max_trials]
        if not cache.was_tried(fwd, atr_mult)
    ]
    if len(pending_grid) < len(_AUTO_TUNE_GRID[:max_trials]):
        skipped = len(_AUTO_TUNE_GRID[:max_trials]) - len(pending_grid)
        print(f"  Skipping {skipped} combo(s) già testati (da cache).")
        # If all combos already tried, seed best_* from cache so we can still save
        best_known = cache.best_known()
        if best_known is not None and not pending_grid:
            best_fwd, best_atr, best_holdout_cached = best_known
            print(f"  Tutti i combo già testati. Miglior noto: fwd={best_fwd}  atr={best_atr}  holdout={best_holdout_cached:.2%}")
            print("  Nessun nuovo trial da eseguire.")
            return TimeframeTrainReport(
                timeframe=timeframe,
                success=False,
                error="Auto-tune: tutti i combo già testati in sessioni precedenti.",
                bars_used=n_bars,
                duration_sec=round(time.perf_counter() - t0, 3),
                model_version=None,
                mean_test_accuracy=None,
                best_test_accuracy=None,
                best_fold_index=None,
                n_folds=0,
                folds=[],
                artifact_path=None,
                holdout_accuracy=best_holdout_cached,
            )
        print()

    for trial_idx, (fwd, atr_mult) in enumerate(pending_grid):
        trial_cfg = MLConfig(
            train_window_bars=base_ml_cfg.train_window_bars,
            test_window_bars=base_ml_cfg.test_window_bars,
            step_bars=base_ml_cfg.step_bars,
            forward_bars=fwd,
            atr_threshold_mult=atr_mult,
            max_iter=base_ml_cfg.max_iter,
            max_depth=base_ml_cfg.max_depth,
            min_accuracy=base_ml_cfg.min_accuracy,
            holdout_fraction=base_ml_cfg.holdout_fraction,
            model_registry_dir=base_ml_cfg.model_registry_dir,
            session_filter_utc_start=base_ml_cfg.session_filter_utc_start,
            session_filter_utc_end=base_ml_cfg.session_filter_utc_end,
        )
        trainer = WalkForwardTrainer(trial_cfg)
        trial_t0 = time.perf_counter()

        try:
            result, model = trainer.run(bars, feature_cache=feature_cache)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"  {trial_idx+1:>5}  {fwd:>4}  {atr_mult:>8.2f}  — ERRORE: {exc}")
            log.error(
                "auto_tune_trial_exception",
                symbol=args.symbol,
                timeframe=timeframe,
                trial=trial_idx + 1,
                forward_bars=fwd,
                atr_threshold_mult=atr_mult,
                error=str(exc),
                error_type=type(exc).__name__,
                traceback=tb,
            )
            continue

        holdout_acc = result.holdout_accuracy
        holdout_str = f"{holdout_acc:.2%}" if holdout_acc is not None else "  n/a  "
        marker = " *" if (holdout_acc is not None and holdout_acc > best_holdout) else "  "
        print(
            f"  {trial_idx+1:>5}  {fwd:>4}  {atr_mult:>8.2f}"
            f"  {result.n_folds:>6}  {result.mean_test_accuracy:>10.2%}"
            f"  {holdout_str:>8}  {time.perf_counter()-trial_t0:>7.1f}s{marker}",
            flush=True,
        )

        cache.record(fwd, atr_mult, holdout_acc if holdout_acc is not None else 0.0, None)

        if holdout_acc is not None and holdout_acc > best_holdout:
            best_holdout = holdout_acc
            best_result = result
            best_model = model
            best_cfg = trial_cfg
            best_trial_idx = trial_idx

        _atomic_write_training_progress(
            prog_path,
            {
                "status": "auto_tune",
                "symbol": args.symbol,
                "timeframe": timeframe,
                "phase": "grid_search",
                "max_trials": max_trials,
                "trials_done": trial_idx + 1,
                "best_holdout": round(best_holdout, 4) if best_holdout >= 0 else None,
                "best_forward_bars": best_cfg.forward_bars if best_cfg else None,
                "best_atr_mult": best_cfg.atr_threshold_mult if best_cfg else None,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )

        if holdout_acc is not None and holdout_acc >= target:
            print(f"\n  Target {target:.0%} raggiunto al trial {trial_idx+1} — stop anticipato.")
            break

    print("  " + "-" * 62)
    print()

    if best_result is None or best_model is None or best_cfg is None:
        msg = "Auto-tune: nessun trial ha prodotto folds validi."
        print(f"ERROR: {msg}", file=sys.stderr)
        return TimeframeTrainReport(
            timeframe=timeframe,
            success=False,
            error=msg,
            bars_used=n_bars,
            duration_sec=round(time.perf_counter() - t0, 3),
            model_version=None,
            mean_test_accuracy=None,
            best_test_accuracy=None,
            best_fold_index=None,
            n_folds=0,
            folds=[],
            artifact_path=None,
            holdout_accuracy=None,
        )

    print(f"  Miglior trial #{best_trial_idx+1}: forward_bars={best_cfg.forward_bars}  atr_mult={best_cfg.atr_threshold_mult}")
    print(f"  Holdout accuracy: {best_holdout:.2%}  ({best_result.holdout_n_samples} campioni)")
    print(f"  Walk-forward — mean: {best_result.mean_test_accuracy:.2%}  best: {best_result.best_test_accuracy:.2%}")
    print()

    test_acc_ok = best_result.best_test_accuracy >= best_cfg.min_accuracy
    if not test_acc_ok:
        msg = (
            f"Auto-tune: miglior holdout={best_holdout:.0%} ma best_test_accuracy="
            f"{best_result.best_test_accuracy:.0%} sotto la soglia {best_cfg.min_accuracy:.0%}. "
            "Modello non salvato. Prova --auto-tune-target piu' basso o piu' barre."
        )
        print(f"WARNING: {msg}", file=sys.stderr)
        return TimeframeTrainReport(
            timeframe=timeframe,
            success=False,
            error=msg,
            bars_used=n_bars,
            duration_sec=round(time.perf_counter() - t0, 3),
            model_version=None,
            mean_test_accuracy=float(best_result.mean_test_accuracy),
            best_test_accuracy=float(best_result.best_test_accuracy),
            best_fold_index=int(best_result.best_fold_index),
            n_folds=best_result.n_folds,
            folds=[fold_to_dict(f) for f in best_result.folds],
            artifact_path=None,
            holdout_accuracy=float(best_holdout),
        )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(config=best_cfg, persist=True)
    best_fold = best_result.folds[best_result.best_fold_index]
    version = f"v{time.strftime('%Y%m%d_%H%M%S')}_{timeframe}_at"
    snapshot = registry.register(
        classifier=best_model,
        symbol=args.symbol,
        version=version,
        tags={
            "source": args.source,
            "auto_tune": True,
            "forward_bars": best_cfg.forward_bars,
            "atr_threshold_mult": best_cfg.atr_threshold_mult,
            "n_folds": best_result.n_folds,
            "mean_test_acc": round(best_result.mean_test_accuracy, 4),
            "best_test_acc": round(best_result.best_test_accuracy, 4),
            "holdout_accuracy": round(best_holdout, 4),
            "best_fold": best_fold.fold_index,
            "train_bars": n_bars,
            "timeframe": timeframe,
        },
    )
    registry.promote(snapshot.version)
    cache.record(best_cfg.forward_bars, best_cfg.atr_threshold_mult, best_holdout, snapshot.version)

    artifact_path = args.model_dir / f"{args.symbol}_{snapshot.version}.pkl"
    duration_sec = round(time.perf_counter() - t0, 3)
    fold_details = [fold_to_dict(f) for f in best_result.folds]

    print(f"Modello auto-tune salvato ({timeframe}).")
    print(f"  Versione      : {snapshot.version}")
    print(f"  forward_bars  : {best_cfg.forward_bars}")
    print(f"  atr_mult      : {best_cfg.atr_threshold_mult}")
    print(f"  Holdout acc   : {best_holdout:.2%}")
    print(f"  File          : {artifact_path}")
    print(f"  Durata totale : {duration_sec}s")
    print()

    telemetry.record_training_run(
        symbol=args.symbol,
        timeframe=timeframe,
        source=args.source,
        bars_requested=args.bars if args.source == "mt5" else n_bars,
        bars_fetched=n_bars,
        train_window=args.train_window,
        test_window=args.test_window,
        step=args.step,
        forward_bars=best_cfg.forward_bars,
        max_iter=best_cfg.max_iter,
        max_depth=best_cfg.max_depth,
        min_accuracy=best_cfg.min_accuracy,
        mean_test_accuracy=best_result.mean_test_accuracy,
        best_test_accuracy=best_result.best_test_accuracy,
        best_fold=best_fold.fold_index,
        model_version=snapshot.version,
        status="completed",
        details={
            "auto_tune": True,
            "holdout_accuracy": float(best_holdout),
            "best_atr_threshold_mult": best_cfg.atr_threshold_mult,
            "artifact_path": str(artifact_path),
            "duration_sec": duration_sec,
            "folds": fold_details,
        },
    )
    telemetry.record_model_artifact(
        version=snapshot.version,
        symbol=args.symbol,
        timeframe=timeframe,
        source=args.source,
        train_bars=n_bars,
        n_folds=best_result.n_folds,
        mean_test_accuracy=best_result.mean_test_accuracy,
        best_test_accuracy=best_result.best_test_accuracy,
        artifact_path=str(artifact_path),
        is_active=telemetry_is_active,
        details={
            "auto_tune": True,
            "holdout_accuracy": float(best_holdout),
            "best_forward_bars": best_cfg.forward_bars,
            "best_atr_threshold_mult": best_cfg.atr_threshold_mult,
        },
    )

    _atomic_write_training_progress(
        prog_path,
        {
            "status": "timeframe_complete",
            "symbol": args.symbol,
            "timeframe": timeframe,
            "phase": "auto_tune_done",
            "model_version": snapshot.version,
            "artifact_path": str(artifact_path),
            "best_forward_bars": best_cfg.forward_bars,
            "best_atr_threshold_mult": best_cfg.atr_threshold_mult,
            "holdout_accuracy": float(best_holdout),
            "n_folds": best_result.n_folds,
            "mean_test_accuracy": float(best_result.mean_test_accuracy),
            "best_test_accuracy": float(best_result.best_test_accuracy),
            "duration_sec": duration_sec,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )

    return TimeframeTrainReport(
        timeframe=timeframe,
        success=True,
        error=None,
        bars_used=n_bars,
        duration_sec=duration_sec,
        model_version=snapshot.version,
        mean_test_accuracy=float(best_result.mean_test_accuracy),
        best_test_accuracy=float(best_result.best_test_accuracy),
        best_fold_index=int(best_result.best_fold_index),
        n_folds=best_result.n_folds,
        folds=fold_details,
        artifact_path=str(artifact_path),
        holdout_accuracy=float(best_holdout),
    )


def write_json_report(model_dir: Path, payload: dict[str, Any]) -> Path:
    report_dir = model_dir / "training_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    name = f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path = report_dir / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ─── Adaptive training loop ───────────────────────────────────────────────────

# Each row: (max_depth, max_iter, bar_scale)
# 20-step schedule: 4 phases × 5 depths, reducing model complexity while
# gradually scaling bars. M1 overfits heavily → depth/iter reduction matters
# more than more data. Bars capped at 100k in adaptive_train_loop.
_ADAPTIVE_SCHEDULE: list[tuple[int, int, float]] = [
    # Phase 1 — depth 4 (softer than default 5)
    (4, 150, 1.0),
    (4, 100, 1.1),
    (4,  75, 1.2),
    (4,  50, 1.3),
    (4,  30, 1.4),
    # Phase 2 — depth 3
    (3, 150, 1.3),
    (3, 100, 1.4),
    (3,  75, 1.5),
    (3,  50, 1.6),
    (3,  30, 1.7),
    # Phase 3 — depth 2
    (2, 150, 1.6),
    (2, 100, 1.7),
    (2,  75, 1.8),
    (2,  50, 1.9),
    (2,  30, 2.0),
    # Phase 4 — depth 1 (stumps, massima regolarizzazione)
    (1, 150, 1.8),
    (1, 100, 1.9),
    (1,  75, 2.0),
    (1,  50, 2.0),
    (1,  30, 2.0),
]


def _compute_adaptive_target(
    model_dir: Path,
    symbol: str,
    min_accuracy: float,
    min_improvement_pp: float,
    absolute_min: float,
) -> tuple[float, float | None]:
    """Return (target_accuracy, current_model_holdout_or_None).

    No model → target = min_accuracy.
    Has model → target = max(current_holdout + min_improvement_pp, absolute_min).
    """
    try:
        _cfg = MLConfig(model_registry_dir=str(model_dir))
        _reg = ModelRegistry(config=_cfg, persist=False)
        active = _reg.get_active()
        if active is not None:
            current = active.tags.get("holdout_accuracy")
            if current is not None:
                current = float(current)
                return max(current + min_improvement_pp, absolute_min), current
    except Exception as exc:
        log.warning(
            "adaptive_target_compute_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
    return min_accuracy, None


def adaptive_train_loop(
    *,
    args: argparse.Namespace,
    timeframe: str,
    base_bars: int,
    telemetry: TelemetryStore,
) -> TimeframeTrainReport:
    """Retry auto-tune with escalating complexity reduction until target met."""
    target, current_acc = _compute_adaptive_target(
        args.model_dir,
        args.symbol,
        args.min_accuracy,
        args.adaptive_min_improvement,
        args.adaptive_absolute_min,
    )
    schedule = _ADAPTIVE_SCHEDULE[: args.adaptive_max_attempts]

    print(f"\n{'='*60}")
    print(f" ADAPTIVE TRAINING  |  {args.symbol} {timeframe}")
    if current_acc is not None:
        print(f" Modello attuale: {current_acc:.2%}  →  Target: {target:.2%}")
    else:
        print(f" Nessun modello attivo  |  Target: {target:.2%}")
    print(f" Max tentativi: {len(schedule)}  |  Strategia: riduzione complessità progressiva")
    print("=" * 60)

    best_report: TimeframeTrainReport | None = None
    attempt_history: list[dict] = []
    _adaptive_progress_path = args.model_dir / "adaptive_progress.json"

    def _write_adaptive_progress(status: str) -> None:
        best_h = (best_report.holdout_accuracy or 0.0) if best_report else 0.0
        payload = {
            "status": status,
            "symbol": args.symbol,
            "timeframe": timeframe,
            "target": round(target, 4),
            "current_model_acc": round(current_acc, 4) if current_acc else None,
            "max_attempts": len(schedule),
            "best_holdout": round(best_h, 4) if best_h else None,
            "attempts_done": len(attempt_history),
            "attempts": attempt_history,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_write_training_progress(_adaptive_progress_path, payload)

    _write_adaptive_progress("running")

    for attempt_idx, (max_depth, max_iter, bar_scale) in enumerate(schedule):
        attempt_bars = min(int(base_bars * bar_scale), 100_000)

        print(f"\n── Tentativo {attempt_idx + 1}/{len(schedule)} ──────────────────────────")
        print(f"   max_depth={max_depth}  max_iter={max_iter}  bars={attempt_bars}")
        if best_report and best_report.holdout_accuracy:
            print(f"   Miglior holdout finora: {best_report.holdout_accuracy:.2%}  target={target:.2%}")

        # Build per-attempt args (shallow copy + overrides)
        aargs = argparse.Namespace(**vars(args))
        aargs.max_depth = max_depth
        aargs.max_iter = max_iter
        aargs.bars = attempt_bars
        aargs.auto_tune = True
        aargs.auto_tune_target = target
        aargs.holdout_fraction = max(args.holdout_fraction, 0.2)

        # Reset tried-params cache — different max_depth/iter → combos behave differently
        cache_file = args.model_dir / f"{args.symbol}_tune_cache.json"
        if cache_file.exists():
            cache_file.unlink()

        aml_cfg = MLConfig(
            train_window_bars=aargs.train_window,
            test_window_bars=aargs.test_window,
            step_bars=aargs.step,
            forward_bars=aargs.forward_bars,
            atr_threshold_mult=aargs.atr_threshold_mult,
            max_iter=max_iter,
            max_depth=max_depth,
            min_accuracy=aargs.min_accuracy,
            holdout_fraction=aargs.holdout_fraction,
            model_registry_dir=str(aargs.model_dir),
            session_filter_utc_start=aargs.session_start_utc,
            session_filter_utc_end=aargs.session_end_utc,
        )

        # Reload bars (may be more than previous attempt)
        if args.source == "mt5":
            bars = load_from_mt5(aargs, timeframe)
        else:
            assert aargs.file is not None
            bars = load_from_csv(
                resolve_csv_path(aargs.file, timeframe, multi=False),
                aargs.symbol,
                timeframe,
            )

        higher_tf_bars = None
        if args.multires and args.source == "mt5":
            higher_tf_bars = load_higher_tf_bars_mt5(aargs, timeframe)

        log.info(
            "adaptive_attempt_start",
            symbol=args.symbol,
            timeframe=timeframe,
            attempt=attempt_idx + 1,
            max_attempts=len(schedule),
            max_depth=max_depth,
            max_iter=max_iter,
            bars=attempt_bars,
            target=round(target, 4),
        )

        report = auto_tune_single_timeframe(
            args=aargs,
            timeframe=timeframe,
            bars=bars,
            base_ml_cfg=aml_cfg,
            telemetry=telemetry,
            telemetry_is_active=True,
            higher_tf_bars=higher_tf_bars,
        )

        # Track best result across all attempts
        if best_report is None or (
            report.holdout_accuracy is not None
            and (
                best_report.holdout_accuracy is None
                or report.holdout_accuracy > best_report.holdout_accuracy
            )
        ):
            best_report = report

        holdout = report.holdout_accuracy or 0.0
        attempt_record = {
            "attempt": attempt_idx + 1,
            "max_depth": max_depth,
            "max_iter": max_iter,
            "bars": attempt_bars,
            "holdout": round(holdout, 4),
            "mean_test_accuracy": round(report.mean_test_accuracy, 4) if report.mean_test_accuracy else None,
            "best_test_accuracy": round(report.best_test_accuracy, 4) if report.best_test_accuracy else None,
            "n_folds": report.n_folds,
            "success": report.success,
            "model_version": report.model_version,
            "duration_sec": report.duration_sec,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        attempt_history.append(attempt_record)

        if report.success and holdout >= target:
            print(f"\n✓ Target raggiunto al tentativo {attempt_idx + 1}: {holdout:.2%} >= {target:.2%}")
            log.info(
                "adaptive_target_reached",
                symbol=args.symbol,
                timeframe=timeframe,
                attempt=attempt_idx + 1,
                holdout=round(holdout, 4),
                target=round(target, 4),
            )
            _write_adaptive_progress("completed")
            return report

        gap = target - holdout
        print(f"\n  ✗ holdout={holdout:.2%}  gap={gap:+.2%}  target={target:.2%}")
        log.warning(
            "adaptive_attempt_failed",
            symbol=args.symbol,
            timeframe=timeframe,
            attempt=attempt_idx + 1,
            holdout=round(holdout, 4),
            gap=round(gap, 4),
            target=round(target, 4),
            success=report.success,
        )
        _write_adaptive_progress("running")
        if attempt_idx < len(schedule) - 1:
            nd, ni, ns = schedule[attempt_idx + 1]
            print(f"  → Prossimo: max_depth={nd}  max_iter={ni}  bars={int(base_bars * ns)}")

    best_h = (best_report.holdout_accuracy or 0.0) if best_report else 0.0
    print(f"\n⚠ Tentativi esauriti. Miglior holdout: {best_h:.2%}  target={target:.2%}")
    _write_adaptive_progress("exhausted")
    log.warning(
        "adaptive_all_attempts_failed",
        symbol=args.symbol,
        timeframe=timeframe,
        best_holdout=round(best_h, 4),
        target=round(target, 4),
        attempts=len(schedule),
    )
    if best_report is None:
        return TimeframeTrainReport(
            timeframe=timeframe,
            success=False,
            error="Adaptive training: tutti i tentativi esauriti.",
            bars_used=0,
            duration_sec=0.0,
            model_version=None,
            mean_test_accuracy=None,
            best_test_accuracy=None,
            best_fold_index=None,
            n_folds=0,
            folds=[],
            artifact_path=None,
            holdout_accuracy=None,
        )
    return best_report


def main() -> None:
    configure_logging()
    args = parse_args()
    args = apply_mt5_defaults(args)
    telemetry = TelemetryStore.from_env()

    timeframes = parse_timeframe_list(args)
    promote_tf = resolve_promote_timeframe(timeframes, args)
    multi = len(timeframes) > 1

    if args.source == "csv" and args.file is None:
        print("ERROR: --file obbligatorio con --source csv", file=sys.stderr)
        sys.exit(1)

    log.info(
        "train_batch_start",
        symbol=args.symbol,
        timeframes=timeframes,
        multi=multi,
        promote_timeframe=promote_tf,
        source=args.source,
    )
    print("\n" + "=" * 60)
    print(f" Training ML  |  {args.symbol}  |  timeframe: {', '.join(timeframes)}")
    print(f" Sorgente: {args.source}  |  modello attivo in dashboard: {promote_tf}")
    if multi:
        print(" Nota: ogni run produce un artifact; solo uno e' marcato attivo in telemetria.")
    print("=" * 60 + "\n")

    # --adaptive implies --auto-tune and forces holdout >= 0.2
    if args.adaptive:
        args.auto_tune = True
        if args.holdout_fraction <= 0.0:
            args.holdout_fraction = 0.2

    if args.auto_tune and args.holdout_fraction <= 0.0:
        print(
            "ERROR: --auto-tune richiede --holdout-fraction > 0 "
            "(es. --holdout-fraction 0.2) per valutare i risultati su dati non visti.",
            file=sys.stderr,
        )
        sys.exit(1)

    ml_cfg = MLConfig(
        train_window_bars=args.train_window,
        test_window_bars=args.test_window,
        step_bars=args.step,
        forward_bars=args.forward_bars,
        atr_threshold_mult=args.atr_threshold_mult,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        min_accuracy=args.min_accuracy,
        holdout_fraction=args.holdout_fraction,
        model_registry_dir=str(args.model_dir),
        session_filter_utc_start=args.session_start_utc,
        session_filter_utc_end=args.session_end_utc,
    )

    reports: list[TimeframeTrainReport] = []
    prog_base = training_progress_path(args.model_dir)
    for idx, tf in enumerate(timeframes):
        _atomic_write_training_progress(
            prog_base,
            {
                "status": "batch",
                "phase": "timeframe_loading",
                "symbol": args.symbol,
                "timeframes": timeframes,
                "timeframe_index": idx,
                "timeframes_total": len(timeframes),
                "current_timeframe": tf,
                "promote_timeframe": promote_tf,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        if args.source == "csv":
            assert args.file is not None
            csv_path = resolve_csv_path(args.file, tf, multi=multi)
            bars = load_from_csv(csv_path, args.symbol, tf)
        else:
            bars = load_from_mt5(args, tf)

        higher_tf_bars: dict[str, list] | None = None
        if args.multires:
            if args.source == "mt5":
                higher_tf_bars = load_higher_tf_bars_mt5(args, tf)
            else:
                assert args.file is not None
                higher_tf_bars = load_higher_tf_bars_csv(args.file, args.symbol, tf)

        is_active = tf == promote_tf
        if args.adaptive:
            report = adaptive_train_loop(
                args=args,
                timeframe=tf,
                base_bars=args.bars,
                telemetry=telemetry,
            )
        elif args.auto_tune:
            report = auto_tune_single_timeframe(
                args=args,
                timeframe=tf,
                bars=bars,
                base_ml_cfg=ml_cfg,
                telemetry=telemetry,
                telemetry_is_active=is_active,
                higher_tf_bars=higher_tf_bars,
            )
        else:
            report = train_single_timeframe(
                args=args,
                timeframe=tf,
                bars=bars,
                ml_cfg=ml_cfg,
                telemetry=telemetry,
                telemetry_is_active=is_active,
                higher_tf_bars=higher_tf_bars,
            )
        reports.append(report)

    ok_count = sum(1 for r in reports if r.success)
    fail_count = len(reports) - ok_count

    print("\n" + "=" * 60)
    print(" RIEPILOGO FINALE")
    print("=" * 60)
    print(f"{'TF':<6} {'Esito':<10} {'Barre':>7} {'Mean test':>10} {'Best test':>10} {'Holdout':>8} {'Versione':<22}")
    print("-" * 70)
    for r in reports:
        status = "OK" if r.success else "FALLITO"
        mean_s = f"{r.mean_test_accuracy:.2%}" if r.mean_test_accuracy is not None else "-"
        best_s = f"{r.best_test_accuracy:.2%}" if r.best_test_accuracy is not None else "-"
        holdout_s = f"{r.holdout_accuracy:.2%}" if r.holdout_accuracy is not None else "-"
        ver = (r.model_version or "-")[:20]
        print(f"{r.timeframe:<6} {status:<10} {r.bars_used:>7} {mean_s:>10} {best_s:>10} {holdout_s:>8} {ver:<22}")
        if r.error and not r.success:
            print(f"        ({r.error})")
    print("-" * 70)
    print(f" Completati con successo: {ok_count}/{len(reports)}  |  attivo (telemetria): {promote_tf}")
    print("=" * 60 + "\n")

    log.info(
        "train_batch_done",
        symbol=args.symbol,
        ok=ok_count,
        failed=fail_count,
        promote_timeframe=promote_tf,
    )

    if not args.no_report_json:
        payload: dict[str, Any] = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "symbol": args.symbol,
            "source": args.source,
            "timeframes": timeframes,
            "promote_timeframe": promote_tf,
            "train_window": args.train_window,
            "test_window": args.test_window,
            "step": args.step,
            "forward_bars": args.forward_bars,
            "max_iter": args.max_iter,
            "max_depth": args.max_depth,
            "min_accuracy": args.min_accuracy,
            "multires": args.multires,
            "holdout_fraction": args.holdout_fraction,
            "auto_tune": args.auto_tune,
            "auto_tune_target": args.auto_tune_target,
            "auto_tune_max_trials": args.auto_tune_max_trials,
            "session_start_utc": args.session_start_utc,
            "session_end_utc": args.session_end_utc,
            "bars_requested_mt5": args.bars if args.source == "mt5" else None,
            "runs": [asdict(r) for r in reports],
        }
        out_path = write_json_report(args.model_dir, payload)
        print(f"Report JSON: {out_path.resolve()}")
        log.info("train_report_written", path=str(out_path.resolve()))

    if ok_count == 0:
        _atomic_write_training_progress(
            prog_base,
            {
                "status": "error",
                "phase": "batch_failed",
                "symbol": args.symbol,
                "message": "nessun training completato con successo",
                "timeframes": timeframes,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        print("ERROR: nessun training completato con successo.", file=sys.stderr)
        sys.exit(1)
    if fail_count:
        _atomic_write_training_progress(
            prog_base,
            {
                "status": "warning",
                "phase": "batch_partial",
                "symbol": args.symbol,
                "ok": ok_count,
                "failed": fail_count,
                "timeframes": timeframes,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        sys.exit(1)

    _atomic_write_training_progress(
        prog_base,
        {
            "status": "batch_complete",
            "phase": "done",
            "symbol": args.symbol,
            "timeframes": timeframes,
            "ok": ok_count,
            "promote_timeframe": promote_tf,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    print("Fatto. Puoi avviare paper o live con il modello desiderato (versione in data/models).")


if __name__ == "__main__":
    main()
