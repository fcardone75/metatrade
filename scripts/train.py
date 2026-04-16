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
import json
import sys
import time
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
        "--model-dir",
        type=Path,
        default=Path("data/models"),
        help="Directory for model persistence (default: data/models)",
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


def load_from_mt5(args: argparse.Namespace, timeframe: str):
    from metatrade.market_data.collectors.mt5_collector import MT5Collector

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

    tf_minutes = {
        Timeframe.M1: 1,
        Timeframe.M5: 5,
        Timeframe.M15: 15,
        Timeframe.M30: 30,
        Timeframe.H1: 60,
        Timeframe.H4: 240,
        Timeframe.D1: 1440,
    }
    minutes = tf_minutes.get(tf, 60) * args.bars
    date_from = date_to - timedelta(minutes=minutes)

    print(f"Scarico {args.bars} barre {args.symbol}/{timeframe} da MT5 ...")
    bars = collector.collect(args.symbol, tf, date_from, date_to)
    collector.shutdown()
    print(f"  Scaricate {len(bars)} barre.")
    return bars


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
    print(f"\n=== Walk-forward: {args.symbol} {timeframe} ({len(bars)} barre) ===")
    print(f"  train_window={ml_cfg.train_window_bars}  test_window={ml_cfg.test_window_bars}  step={ml_cfg.step_bars}")
    print(f"  forward_bars={ml_cfg.forward_bars}  atr_threshold_mult={ml_cfg.atr_threshold_mult}  class_weight={ml_cfg.class_weight}")
    print()

    n_bars = len(bars)
    est_folds = _estimate_max_walk_forward_folds(
        n_bars,
        ml_cfg.train_window_bars,
        ml_cfg.test_window_bars,
        ml_cfg.step_bars,
    )
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

    def on_fold_complete(fold: WalkForwardFold) -> None:
        done = fold.fold_index + 1
        pct = min(100.0, 100.0 * done / max(est_folds, 1))
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
            "test_accuracy_last": round(float(fold.test_accuracy), 4),
            "elapsed_sec": round(time.perf_counter() - t0, 1),
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_write_training_progress(prog_path, payload)
        log.info(
            "train_walk_forward_fold",
            symbol=args.symbol,
            timeframe=timeframe,
            fold_index=fold.fold_index,
            folds_completed=done,
            test_accuracy=round(float(fold.test_accuracy), 4),
        )

    result, model = trainer.run(bars, on_fold_complete=on_fold_complete)

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

    if model is None or not model.is_trained:
        msg = (
            f"Nessun modello sopra la soglia min_accuracy ({ml_cfg.min_accuracy:.0%}). "
            "Prova --bars piu' alto, --min-accuracy piu' basso, o verifica i dati."
        )
        log.warning("train_timeframe_below_threshold", symbol=args.symbol, timeframe=timeframe, message=msg)
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
    )


def write_json_report(model_dir: Path, payload: dict[str, Any]) -> Path:
    report_dir = model_dir / "training_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    name = f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path = report_dir / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


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

    ml_cfg = MLConfig(
        train_window_bars=args.train_window,
        test_window_bars=args.test_window,
        step_bars=args.step,
        forward_bars=args.forward_bars,
        atr_threshold_mult=args.atr_threshold_mult,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        min_accuracy=args.min_accuracy,
        model_registry_dir=str(args.model_dir),
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

        is_active = tf == promote_tf
        report = train_single_timeframe(
            args=args,
            timeframe=tf,
            bars=bars,
            ml_cfg=ml_cfg,
            telemetry=telemetry,
            telemetry_is_active=is_active,
        )
        reports.append(report)

    ok_count = sum(1 for r in reports if r.success)
    fail_count = len(reports) - ok_count

    print("\n" + "=" * 60)
    print(" RIEPILOGO FINALE")
    print("=" * 60)
    print(f"{'TF':<6} {'Esito':<10} {'Barre':>7} {'Mean test':>10} {'Best test':>10} {'Versione':<22}")
    print("-" * 60)
    for r in reports:
        status = "OK" if r.success else "FALLITO"
        mean_s = f"{r.mean_test_accuracy:.2%}" if r.mean_test_accuracy is not None else "-"
        best_s = f"{r.best_test_accuracy:.2%}" if r.best_test_accuracy is not None else "-"
        ver = (r.model_version or "-")[:20]
        print(f"{r.timeframe:<6} {status:<10} {r.bars_used:>7} {mean_s:>10} {best_s:>10} {ver:<22}")
        if r.error and not r.success:
            print(f"        ({r.error})")
    print("-" * 60)
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
