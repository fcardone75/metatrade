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
    signal_precision: float | None = None
    precision_buy: float | None = None
    precision_sell: float | None = None
    recall_buy: float | None = None
    recall_sell: float | None = None


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
    p.add_argument("--max-iter", type=int, default=200, help="Max boosting iterations (all backends)")
    p.add_argument("--max-depth", type=int, default=5, help="Max tree depth")
    _env_cfg = MLConfig()
    p.add_argument(
        "--backend",
        choices=["histgbm", "lightgbm", "xgboost"],
        default=_env_cfg.backend,
        help="ML backend (default: ML_BACKEND env or 'histgbm'). CLI takes precedence over env.",
    )
    p.add_argument("--num-leaves", type=int, default=_env_cfg.num_leaves, help="Number of leaves per tree (LightGBM/XGBoost)")
    p.add_argument("--learning-rate", type=float, default=_env_cfg.learning_rate, help="Learning rate / shrinkage (all backends)")
    p.add_argument("--min-child-samples", type=int, default=_env_cfg.min_child_samples, help="Min samples per leaf (regularisation)")
    _env_min_acc = _env_cfg.min_accuracy
    p.add_argument("--min-accuracy", type=float, default=_env_min_acc, help="Min accuracy to accept model")
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
        "--target-buy-precision",
        type=float,
        default=_env_cfg.target_buy_precision,
        metavar="PRECISION",
        help="Stop adaptive loop when BUY precision >= this value (default from ML_TARGET_BUY_PRECISION).",
    )
    p.add_argument(
        "--target-sell-precision",
        type=float,
        default=_env_cfg.target_sell_precision,
        metavar="PRECISION",
        help="Stop adaptive loop when SELL precision >= this value (default from ML_TARGET_SELL_PRECISION).",
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
    p.add_argument(
        "--adaptive-skip-attempts",
        type=int,
        default=0,
        metavar="N",
        help="Skip the first N adaptive attempts (used to resume an interrupted run).",
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
            log.error("timeframes_empty", msg="--timeframes e' vuoto.")
            sys.exit(1)
        for t in out:
            if t not in _TIMEFRAME_MAP:
                log.error(
                    "timeframe_unknown",
                    msg=f"timeframe sconosciuto {t!r}. Validi: {', '.join(_TIMEFRAME_MAP)}.",
                )
                sys.exit(1)
        return out
    return [args.timeframe]


def resolve_promote_timeframe(timeframes: list[str], args: argparse.Namespace) -> str:
    if args.promote_timeframe:
        target = args.promote_timeframe.strip().upper()
        if target not in timeframes:
            log.error(
                "promote_timeframe_invalid",
                promote_timeframe=target,
                trained_timeframes=timeframes,
                msg=f"--promote-timeframe={target!r} non e' tra i timeframe addestrati.",
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
        log.error(
            "csv_path_missing_placeholder",
            msg="con piu' timeframe da CSV usa un segnaposto in --file, es. data/EURUSD_{TIMEFRAME}.csv",
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
        log.error("csv_file_not_found", path=str(path))
        sys.exit(1)

    tf = _TIMEFRAME_MAP[timeframe]
    collector = CsvCollector()
    log.info("csv_loading", path=str(path), symbol=symbol, timeframe=timeframe)
    bars = collector.collect_file(
        file_path=path,
        symbol=symbol,
        timeframe=tf,
    )
    log.info("csv_loaded", n_bars=len(bars), timeframe=timeframe)
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

    log.info("mt5_downloading", symbol=args.symbol, timeframe=timeframe, n_bars=n_bars)
    bars = collector.collect(args.symbol, tf, date_from, date_to)
    collector.shutdown()
    log.info("mt5_downloaded", n_bars=len(bars), symbol=args.symbol, timeframe=timeframe)
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
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(content, encoding="utf-8")
    try:
        tmp.replace(path)
    except PermissionError:
        # Windows: target file may be held open by the runner reading it.
        # Fall back to direct write — non-atomic but safe for this use case.
        path.write_text(content, encoding="utf-8")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _log_fold_table(result: WalkForwardResult) -> None:
    for fold in result.folds:
        log.info(
            "fold_result",
            fold_index=fold.fold_index,
            train_bars=fold.train_end - fold.train_start,
            n_test_samples=fold.n_test_samples,
            train_accuracy=round(fold.train_metrics.accuracy, 4),
            test_accuracy=round(float(fold.test_accuracy), 4),
        )
    log.info(
        "walk_forward_summary",
        mean_test_accuracy=round(float(result.mean_test_accuracy), 4),
        best_test_accuracy=round(float(result.best_test_accuracy), 4),
    )


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
    log.info(
        "walk_forward_start",
        symbol=args.symbol,
        timeframe=timeframe,
        n_bars=len(bars),
        train_window=ml_cfg.train_window_bars,
        test_window=ml_cfg.test_window_bars,
        step=ml_cfg.step_bars,
        forward_bars=ml_cfg.forward_bars,
        atr_threshold_mult=ml_cfg.atr_threshold_mult,
        class_weight=ml_cfg.class_weight,
        multires=multires_label or None,
        session=session_label or None,
    )

    n_bars = len(bars)
    est_folds = _estimate_max_walk_forward_folds(
        n_bars,
        ml_cfg.train_window_bars,
        ml_cfg.test_window_bars,
        ml_cfg.step_bars,
    )
    log.info("walk_forward_estimated_folds", estimated_folds=est_folds)

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

        # Log fold progress
        status_char = "+" if test_acc >= ml_cfg.min_accuracy else "-"
        log.debug(
            "fold_progress",
            status=status_char,
            fold_done=done,
            fold_total=est_folds,
            pct=round(pct, 1),
            test_accuracy=round(test_acc, 4),
            class_distribution=dist_str,
            fold_sec=round(fold_dur, 1),
            elapsed_sec=round(elapsed_total, 0),
            eta=eta_str,
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
        log.debug("precompute_progress", pct=round(pct, 1), current=current, total=total)
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
    _wf_elapsed = round(time.perf_counter() - t0, 0)
    if result.folds:
        above = sum(1 for f in result.folds if f.test_accuracy >= ml_cfg.min_accuracy)
        log.info(
            "walk_forward_complete",
            n_folds=len(result.folds),
            elapsed_sec=_wf_elapsed,
            folds_above_threshold=above,
            min_accuracy=ml_cfg.min_accuracy,
            mean_test_accuracy=round(float(result.mean_test_accuracy), 4),
            best_test_accuracy=round(float(result.best_test_accuracy), 4),
            holdout_accuracy=round(float(result.holdout_accuracy), 4) if result.holdout_accuracy is not None else None,
            holdout_n_samples=result.holdout_n_samples if result.holdout_accuracy is not None else None,
        )
    else:
        log.info("walk_forward_complete", n_folds=0, elapsed_sec=_wf_elapsed)

    if not result.folds:
        msg = "Nessun fold prodotto: dati insufficienti."
        log.error("train_timeframe_failed", symbol=args.symbol, timeframe=timeframe, reason=msg)
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

    _log_fold_table(result)

    # Log label distribution from the best fold's training metrics
    if model is not None and model.metrics is not None:
        cd = model.metrics.class_distribution
        total = sum(cd.values()) or 1
        log.info(
            "class_distribution",
            buy=cd.get(1, 0),
            sell=cd.get(-1, 0),
            hold=cd.get(0, 0),
            buy_pct=round(cd.get(1, 0) / total, 4),
            sell_pct=round(cd.get(-1, 0) / total, 4),
            hold_pct=round(cd.get(0, 0) / total, 4),
        )

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
            round(best_fold.train_metrics.accuracy / max(best_fold.test_accuracy, 0.001), 3)
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
            best_fold_train_accuracy=round(best_fold.train_metrics.accuracy, 4) if best_fold else None,
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
            "backend": ml_cfg.backend,
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

    log.info(
        "model_saved",
        timeframe=timeframe,
        version=snapshot.version,
        symbol=snapshot.symbol,
        best_test_accuracy=round(float(result.best_test_accuracy), 4),
        artifact_path=str(artifact_path),
        duration_sec=duration_sec,
    )

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
        signal_precision=result.holdout_signal_precision,
        precision_buy=result.holdout_precision_by_class.get(1),
        precision_sell=result.holdout_precision_by_class.get(-1),
        recall_buy=result.holdout_recall_by_class.get(1),
        recall_sell=result.holdout_recall_by_class.get(-1),
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
    max_trials = min(args.auto_tune_max_trials, len(_AUTO_TUNE_GRID))

    cache = TriedParamsCache(args.model_dir, args.symbol)

    log.info(
        "auto_tune_start",
        symbol=args.symbol,
        timeframe=timeframe,
        n_bars=n_bars,
        holdout_fraction=base_ml_cfg.holdout_fraction,
        min_accuracy=base_ml_cfg.min_accuracy,
        max_trials=max_trials,
        cache_summary=cache.summary() or None,
    )

    # ── Precompute feature cache once ─────────────────────────────────────────
    trainer_base = WalkForwardTrainer(base_ml_cfg)

    precompute_start = time.perf_counter()
    log.info("feature_precompute_start", symbol=args.symbol, timeframe=timeframe)

    def _precompute_progress(current: int, total: int) -> None:
        pct = 100.0 * current / max(total, 1)
        log.debug("feature_precompute_progress", pct=round(pct, 1), current=current, total=total)

    feature_cache = trainer_base.precompute_features(
        bars,
        higher_tf_bars=higher_tf_bars,
        on_progress=_precompute_progress,
    )
    log.info(
        "feature_precompute_done",
        n_bars=n_bars,
        elapsed_sec=round(time.perf_counter() - precompute_start, 1),
    )

    # ── Grid search ───────────────────────────────────────────────────────────
    log.info("auto_tune_grid_search_start", max_trials=max_trials)

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
        log.info("auto_tune_skipped_cached_combos", skipped=skipped)
        # If all combos already tried, seed best_* from cache so we can still save
        best_known = cache.best_known()
        if best_known is not None and not pending_grid:
            best_fwd, best_atr, best_holdout_cached = best_known
            log.info(
                "auto_tune_all_combos_cached",
                best_forward_bars=best_fwd,
                best_atr_mult=best_atr,
                best_holdout=round(best_holdout_cached, 4),
            )
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

    for trial_idx, (fwd, atr_mult) in enumerate(pending_grid):
        trial_cfg = MLConfig(
            train_window_bars=base_ml_cfg.train_window_bars,
            test_window_bars=base_ml_cfg.test_window_bars,
            step_bars=base_ml_cfg.step_bars,
            forward_bars=fwd,
            atr_threshold_mult=atr_mult,
            max_iter=base_ml_cfg.max_iter,
            max_depth=base_ml_cfg.max_depth,
            backend=base_ml_cfg.backend,
            num_leaves=base_ml_cfg.num_leaves,
            learning_rate=base_ml_cfg.learning_rate,
            min_child_samples=base_ml_cfg.min_child_samples,
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
        signal_prec = result.holdout_signal_precision
        prec_buy = result.holdout_precision_by_class.get(1)
        prec_sell = result.holdout_precision_by_class.get(-1)
        prec_hold = result.holdout_precision_by_class.get(0)
        recall_buy = result.holdout_recall_by_class.get(1)
        recall_sell = result.holdout_recall_by_class.get(-1)

        # Primary metric: signal precision (avg precision on BUY+SELL)
        primary = signal_prec if signal_prec is not None else (holdout_acc or 0.0)
        is_best = primary > best_holdout
        log.info(
            "auto_tune_trial_result",
            trial=trial_idx + 1,
            forward_bars=fwd,
            atr_threshold_mult=atr_mult,
            n_folds=result.n_folds,
            mean_test_accuracy=round(float(result.mean_test_accuracy), 4),
            holdout_accuracy=round(float(holdout_acc), 4) if holdout_acc is not None else None,
            signal_precision=round(signal_prec, 4) if signal_prec is not None else None,
            precision_buy=round(prec_buy, 4) if prec_buy is not None else None,
            precision_sell=round(prec_sell, 4) if prec_sell is not None else None,
            precision_hold=round(prec_hold, 4) if prec_hold is not None else None,
            recall_buy=round(recall_buy, 4) if recall_buy is not None else None,
            recall_sell=round(recall_sell, 4) if recall_sell is not None else None,
            trial_sec=round(time.perf_counter() - trial_t0, 1),
            is_best=is_best,
        )

        cache.record(fwd, atr_mult, primary, None)

        if is_best:
            best_holdout = primary
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

        if holdout_acc is not None and holdout_acc >= base_ml_cfg.min_accuracy:
            log.info(
                "auto_tune_target_reached",
                target=base_ml_cfg.min_accuracy,
                trial=trial_idx + 1,
                holdout_accuracy=round(holdout_acc, 4),
            )
            break

    if best_result is None or best_model is None or best_cfg is None:
        msg = "Auto-tune: nessun trial ha prodotto folds validi."
        log.error("auto_tune_no_valid_folds", symbol=args.symbol, timeframe=timeframe, msg=msg)
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

    log.info(
        "auto_tune_best_trial",
        trial=best_trial_idx + 1,
        forward_bars=best_cfg.forward_bars,
        atr_threshold_mult=best_cfg.atr_threshold_mult,
        holdout_accuracy=round(best_holdout, 4),
        holdout_n_samples=best_result.holdout_n_samples,
        mean_test_accuracy=round(float(best_result.mean_test_accuracy), 4),
        best_test_accuracy=round(float(best_result.best_test_accuracy), 4),
    )

    test_acc_ok = best_result.best_test_accuracy >= best_cfg.min_accuracy
    if not test_acc_ok:
        msg = (
            f"Auto-tune: miglior holdout={best_holdout:.0%} ma best_test_accuracy="
            f"{best_result.best_test_accuracy:.0%} sotto la soglia {best_cfg.min_accuracy:.0%}. "
            "Modello non salvato. Prova --auto-tune-target piu' basso o piu' barre."
        )
        log.warning(
            "auto_tune_below_threshold",
            symbol=args.symbol,
            timeframe=timeframe,
            best_holdout=round(best_holdout, 4),
            best_test_accuracy=round(float(best_result.best_test_accuracy), 4),
            min_accuracy=best_cfg.min_accuracy,
            msg=msg,
        )
        _prec_buy = best_result.holdout_precision_by_class.get(1)
        _prec_sell = best_result.holdout_precision_by_class.get(-1)
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
            signal_precision=best_result.holdout_signal_precision,
            precision_buy=round(_prec_buy, 4) if _prec_buy is not None else None,
            precision_sell=round(_prec_sell, 4) if _prec_sell is not None else None,
            recall_buy=round(r, 4) if (r := best_result.holdout_recall_by_class.get(1)) is not None else None,
            recall_sell=round(r, 4) if (r := best_result.holdout_recall_by_class.get(-1)) is not None else None,
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
            "backend": best_cfg.backend,
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

    log.info(
        "auto_tune_model_saved",
        timeframe=timeframe,
        version=snapshot.version,
        forward_bars=best_cfg.forward_bars,
        atr_threshold_mult=best_cfg.atr_threshold_mult,
        holdout_accuracy=round(best_holdout, 4),
        artifact_path=str(artifact_path),
        duration_sec=duration_sec,
    )

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
# gradually scaling bars. M1 overfits heavily -> depth/iter reduction matters
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

    No model -> target = min_accuracy.
    Has model -> target = max(current_holdout + min_improvement_pp, absolute_min).
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
    skip_attempts = max(0, getattr(args, "adaptive_skip_attempts", 0))

    # ── Resume: reload attempt history from previous run ─────────────────────
    _adaptive_progress_path = args.model_dir / "adaptive_progress.json"
    attempt_history: list[dict] = []
    _adaptive_started_at = datetime.now(UTC).isoformat()

    if skip_attempts > 0 and _adaptive_progress_path.exists():
        try:
            prev_data = json.loads(_adaptive_progress_path.read_text())
            attempt_history = list(prev_data.get("attempts") or [])
            _adaptive_started_at = prev_data.get("started_at_utc") or _adaptive_started_at
            log.info(
                "adaptive_training_resume",
                symbol=args.symbol,
                timeframe=timeframe,
                skip_attempts=skip_attempts,
                previous_attempts=len(attempt_history),
            )
        except Exception as exc:
            log.warning("adaptive_resume_read_failed", error=str(exc))
            skip_attempts = 0
            attempt_history = []

    log.info(
        "adaptive_training_start",
        symbol=args.symbol,
        timeframe=timeframe,
        target=round(target, 4),
        current_model_accuracy=round(current_acc, 4) if current_acc is not None else None,
        max_attempts=len(schedule),
        skip_attempts=skip_attempts,
    )

    best_report: TimeframeTrainReport | None = None

    def _write_adaptive_progress(status: str, *, error_msg: str | None = None) -> None:
        best_h = (best_report.holdout_accuracy or 0.0) if best_report else 0.0
        payload: dict[str, object] = {
            "status": status,
            "symbol": args.symbol,
            "timeframe": timeframe,
            "target": round(target, 4),
            "target_buy_precision": getattr(args, "target_buy_precision", None),
            "target_sell_precision": getattr(args, "target_sell_precision", None),
            "current_model_acc": round(current_acc, 4) if current_acc else None,
            "max_attempts": len(schedule),
            "best_holdout": round(best_h, 4) if best_h else None,
            "attempts_done": len(attempt_history),
            "attempts": attempt_history,
            "started_at_utc": _adaptive_started_at,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        if error_msg is not None:
            payload["error"] = error_msg
        _atomic_write_training_progress(_adaptive_progress_path, payload)

    _write_adaptive_progress("running")

    for attempt_idx, (max_depth, max_iter, bar_scale) in enumerate(schedule):
        if attempt_idx < skip_attempts:
            continue

        attempt_bars = min(int(base_bars * bar_scale), 100_000)

        log.info(
            "adaptive_attempt_begin",
            attempt=attempt_idx + 1,
            max_attempts=len(schedule),
            max_depth=max_depth,
            max_iter=max_iter,
            bars=attempt_bars,
            best_holdout_so_far=round(best_report.holdout_accuracy, 4) if best_report and best_report.holdout_accuracy else None,
            target=round(target, 4),
        )

        # Build per-attempt args (shallow copy + overrides)
        aargs = argparse.Namespace(**vars(args))
        aargs.max_depth = max_depth
        aargs.max_iter = max_iter
        aargs.bars = attempt_bars
        aargs.auto_tune = True
        aargs.holdout_fraction = max(args.holdout_fraction, 0.2)

        # Reset tried-params cache — different max_depth/iter -> combos behave differently
        cache_file = args.model_dir / f"{args.symbol}_tune_cache.json"
        if cache_file.exists():
            cache_file.unlink()

        import random as _random
        attempt_seed = _random.randint(0, 2**31 - 1)
        aml_cfg = MLConfig(
            train_window_bars=aargs.train_window,
            test_window_bars=aargs.test_window,
            step_bars=aargs.step,
            forward_bars=aargs.forward_bars,
            atr_threshold_mult=aargs.atr_threshold_mult,
            max_iter=max_iter,
            max_depth=max_depth,
            backend=aargs.backend,
            num_leaves=aargs.num_leaves,
            learning_rate=aargs.learning_rate,
            min_child_samples=aargs.min_child_samples,
            min_accuracy=aargs.min_accuracy,
            holdout_fraction=aargs.holdout_fraction,
            model_registry_dir=str(aargs.model_dir),
            session_filter_utc_start=aargs.session_start_utc,
            session_filter_utc_end=aargs.session_end_utc,
            random_seed=attempt_seed,
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

        signal_prec = report.signal_precision
        holdout = report.holdout_accuracy or 0.0

        # Track best result across all attempts (rank by signal_precision, fallback to holdout)
        def _score(r: TimeframeTrainReport) -> float:
            return r.signal_precision or r.holdout_accuracy or 0.0

        if best_report is None or _score(report) > _score(best_report):
            best_report = report

        attempt_record = {
            "attempt": attempt_idx + 1,
            "max_depth": max_depth,
            "max_iter": max_iter,
            "bars": attempt_bars,
            "holdout": round(holdout, 4),
            "signal_precision": round(signal_prec, 4) if signal_prec is not None else None,
            "precision_buy": round(report.precision_buy, 4) if report.precision_buy is not None else None,
            "precision_sell": round(report.precision_sell, 4) if report.precision_sell is not None else None,
            "recall_buy": round(report.recall_buy, 4) if report.recall_buy is not None else None,
            "recall_sell": round(report.recall_sell, 4) if report.recall_sell is not None else None,
            "mean_test_accuracy": round(report.mean_test_accuracy, 4) if report.mean_test_accuracy else None,
            "best_test_accuracy": round(report.best_test_accuracy, 4) if report.best_test_accuracy else None,
            "n_folds": report.n_folds,
            "success": report.success,
            "model_version": report.model_version,
            "duration_sec": report.duration_sec,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        attempt_history.append(attempt_record)

        target_buy = getattr(args, "target_buy_precision", 0.0) or 0.0
        target_sell = getattr(args, "target_sell_precision", 0.0) or 0.0
        buy_ok = (report.precision_buy or 0.0) >= target_buy
        sell_ok = (report.precision_sell or 0.0) >= target_sell

        if report.success and buy_ok and sell_ok:
            log.info(
                "adaptive_target_reached",
                symbol=args.symbol,
                timeframe=timeframe,
                attempt=attempt_idx + 1,
                precision_buy=round(report.precision_buy, 4) if report.precision_buy is not None else None,
                precision_sell=round(report.precision_sell, 4) if report.precision_sell is not None else None,
                target_buy=round(target_buy, 4),
                target_sell=round(target_sell, 4),
                holdout=round(holdout, 4),
            )
            _write_adaptive_progress("completed")
            return report

        log.warning(
            "adaptive_attempt_failed",
            symbol=args.symbol,
            timeframe=timeframe,
            attempt=attempt_idx + 1,
            precision_buy=round(report.precision_buy, 4) if report.precision_buy is not None else None,
            precision_sell=round(report.precision_sell, 4) if report.precision_sell is not None else None,
            target_buy=round(target_buy, 4),
            target_sell=round(target_sell, 4),
            holdout=round(holdout, 4),
            success=report.success,
        )
        _write_adaptive_progress("running")
        if attempt_idx < len(schedule) - 1:
            nd, ni, ns = schedule[attempt_idx + 1]
            log.info(
                "adaptive_next_attempt_preview",
                next_max_depth=nd,
                next_max_iter=ni,
                next_bars=int(base_bars * ns),
            )

    best_h = (best_report.holdout_accuracy or 0.0) if best_report else 0.0
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
        log.error("missing_csv_file", msg="--file obbligatorio con --source csv")
        sys.exit(1)

    log.info(
        "train_batch_start",
        symbol=args.symbol,
        timeframes=timeframes,
        multi=multi,
        promote_timeframe=promote_tf,
        source=args.source,
    )
    log.info(
        "training_session_start",
        symbol=args.symbol,
        timeframes=timeframes,
        source=args.source,
        promote_timeframe=promote_tf,
        multi=multi,
    )

    # --adaptive implies --auto-tune and forces holdout >= 0.2
    if args.adaptive:
        args.auto_tune = True
        if args.holdout_fraction <= 0.0:
            args.holdout_fraction = 0.2

    if args.auto_tune and args.holdout_fraction <= 0.0:
        log.error(
            "auto_tune_missing_holdout",
            msg="--auto-tune richiede --holdout-fraction > 0 (es. --holdout-fraction 0.2) per valutare i risultati su dati non visti.",
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
        try:
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
        except Exception as exc:  # noqa: BLE001
            import traceback as _tb
            _atomic_write_training_progress(
                prog_base,
                {
                    "status": "error",
                    "phase": "training_exception",
                    "symbol": args.symbol,
                    "timeframe": tf,
                    "message": str(exc),
                    "traceback": _tb.format_exc(),
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                },
            )
            log.error("training_exception", timeframe=tf, error=str(exc), error_type=type(exc).__name__)
            sys.exit(1)
        reports.append(report)

    ok_count = sum(1 for r in reports if r.success)
    fail_count = len(reports) - ok_count

    for r in reports:
        log.info(
            "timeframe_report",
            timeframe=r.timeframe,
            success=r.success,
            bars_used=r.bars_used,
            mean_test_accuracy=round(r.mean_test_accuracy, 4) if r.mean_test_accuracy is not None else None,
            best_test_accuracy=round(r.best_test_accuracy, 4) if r.best_test_accuracy is not None else None,
            holdout_accuracy=round(r.holdout_accuracy, 4) if r.holdout_accuracy is not None else None,
            model_version=r.model_version,
            error=r.error if not r.success else None,
        )

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
            "auto_tune_max_trials": args.auto_tune_max_trials,
            "target_buy_precision": getattr(args, "target_buy_precision", None),
            "target_sell_precision": getattr(args, "target_sell_precision", None),
            "session_start_utc": args.session_start_utc,
            "session_end_utc": args.session_end_utc,
            "bars_requested_mt5": args.bars if args.source == "mt5" else None,
            "runs": [asdict(r) for r in reports],
        }
        out_path = write_json_report(args.model_dir, payload)
        log.info("report_written", path=str(out_path.resolve()))

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
        log.error("batch_failed", symbol=args.symbol, msg="nessun training completato con successo.")
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
    log.info("training_done", symbol=args.symbol, ok=ok_count, promote_timeframe=promote_tf)


if __name__ == "__main__":
    main()
