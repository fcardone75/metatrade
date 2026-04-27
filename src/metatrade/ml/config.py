"""ML module configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from metatrade.core.config_base import BaseConfig


class MLConfig(BaseConfig):
    """Configuration for the ML analysis module.

    Walk-forward training parameters and model hyper-parameters
    are defined here. Override via environment variables with the
    ``ML_`` prefix (e.g. ``ML_TRAIN_WINDOW_BARS=5000``).
    """

    # ── Walk-forward training ─────────────────────────────────────────────────
    # Training window in bars (how much history the model trains on)
    train_window_bars: int = Field(default=2000, ge=200)

    # Test window in bars (how many bars to evaluate each fold)
    test_window_bars: int = Field(default=500, ge=50)

    # Step size in bars between successive training windows
    step_bars: int = Field(default=250, ge=10)

    # ── Feature / label parameters ────────────────────────────────────────────
    # Number of bars to look ahead when computing labels
    forward_bars: int = Field(default=5, ge=1, le=50)

    # ATR multiplier for label threshold (no-trade zone half-width).
    # 1.5 is appropriate for H1+; on M1/M5 use 0.7-0.9 to avoid
    # labelling most bars as HOLD (M1 rarely moves 1.5×ATR in 5 bars).
    atr_threshold_mult: float = Field(default=0.8, gt=0.0)

    # ── Model ─────────────────────────────────────────────────────────────────
    # Minimum class accuracy before a model is considered "trained"
    min_accuracy: float = Field(default=0.52, ge=0.5, le=1.0)

    # Precision targets for BUY and SELL signals (adaptive loop stopping criteria).
    # The loop stops when BOTH are met. Set to 0.0 to disable a specific check.
    target_buy_precision: float = Field(default=0.55, ge=0.0, le=1.0)
    target_sell_precision: float = Field(default=0.55, ge=0.0, le=1.0)

    # Minimum number of training samples
    min_train_samples: int = Field(default=100, ge=10)

    # Random seed for reproducibility
    random_seed: int = Field(default=42)

    # ML backend: "histgbm" | "lightgbm" | "xgboost"
    # CLI --backend takes precedence over this env value.
    # Falls back to histgbm if the requested library is not installed.
    backend: str = Field(default="histgbm")

    # GPU acceleration for supported backends (xgboost, lightgbm).
    # histgbm (sklearn) does not support GPU and ignores this flag.
    use_gpu: bool = Field(default=False)

    # Max number of boosting iterations / estimators (all backends)
    max_iter: int = Field(default=100, ge=10)

    # Max depth of each tree (None = unlimited for histgbm/xgboost; ignored by lightgbm
    # when num_leaves is set — use num_leaves instead for lightgbm)
    max_depth: int | None = Field(default=5)

    # Number of leaves per tree — primary complexity knob for LightGBM.
    # Ignored by histgbm. For xgboost maps to max_leaves.
    num_leaves: int = Field(default=31, ge=2)

    # Learning rate / shrinkage for gradient boosting (all backends)
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)

    # Minimum number of samples in a leaf (LightGBM: min_child_samples,
    # XGBoost: min_child_weight, HistGBM: min_samples_leaf)
    min_child_samples: int = Field(default=20, ge=1)

    # Class weight for gradient boosting.
    # "balanced" automatically compensates for HOLD-dominated datasets by
    # up-weighting BUY/SELL samples.  None = uniform weights.
    class_weight: str | None = Field(default="balanced")

    # ── Holdout evaluation ────────────────────────────────────────────────────
    # Fraction of bars (from the end of the series) reserved as a final
    # out-of-sample holdout.  Walk-forward training only sees bars[:holdout_start].
    # Set to 0.0 to disable (no holdout reserved).
    holdout_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)

    # ── Embargo ───────────────────────────────────────────────────────────────
    # Number of bars excluded between the end of the training window and the
    # start of the test window.  Prevents leakage from autocorrelated consecutive
    # bars.  Set to forward_bars (same as the label horizon) as a minimum.
    # On M1 with forward_bars=5, an embargo of 5 bars = 5 minutes of gap.
    embargo_bars: int = Field(default=5, ge=0)

    # ── Session filter ────────────────────────────────────────────────────────
    # When both are set, only bars whose UTC hour is in [start, end) are used
    # for training.  Helps remove thin-market noise on M1/M5.
    # Must mirror RUNNER_SESSION_FILTER_UTC_START/END to avoid train/live mismatch.
    # Example: start=7, end=21 keeps London + NY sessions only.
    session_filter_utc_start: int | None = Field(default=7, ge=0, lt=24)
    session_filter_utc_end: int | None = Field(default=21, ge=0, lt=24)

    # ── Live accuracy tracking ────────────────────────────────────────────────
    # Minimum number of evaluated BUY/SELL predictions before the live accuracy
    # percentage is considered statistically reliable for model-switch decisions.
    # At 200 samples the 95% confidence interval is ±3.5 pp.
    live_accuracy_min_samples: int = Field(default=200, ge=10)

    # Emergency: if live accuracy drops below this threshold (and min_samples
    # has been reached), the system flags a degradation warning.
    live_accuracy_warning_below: float = Field(default=0.50, ge=0.0, le=1.0)

    # ── Model watcher + promotion ─────────────────────────────────────────────
    # Enable periodic scanning of registry dir for new/better snapshots.
    model_watcher_enabled: bool = Field(default=False)

    # How often (seconds) the watcher scans for new snapshots (default 2 min).
    model_watcher_poll_sec: int = Field(default=120, ge=10)

    # Minimum holdout accuracy a candidate must reach to be considered for
    # promotion. Prevents promoting models that barely passed training.
    candidate_min_holdout: float = Field(default=0.52, ge=0.0, le=1.0)

    # ── Adaptive fallback floor ───────────────────────────────────────────────
    # If after all adaptive attempts no model hits the precision targets, accept
    # the best model found if its holdout accuracy is >= this floor.
    # Set to 0.0 to always save the best model regardless of quality.
    adaptive_fallback_min: float = Field(default=0.50, ge=0.0, le=1.0)

    # ── Background retraining scheduler ──────────────────────────────────────
    # Whether to automatically launch train.py in the background.
    retrain_enabled: bool = Field(default=False)

    # Number of MT5 bars to fetch for each retrain run (local path).
    # Also sets the bar buffer size for the distributed path.
    # 30_000 M1 bars ≈ 3 weeks; 200_000 M1 bars ≈ 4.5 months.
    retrain_bars: int = Field(default=30_000, ge=1_000)

    # Trigger mode: "hours" = fixed daily slots; "bars" = every N bars.
    retrain_trigger: str = Field(default="hours")

    # Interval between training slots in hours (used when trigger="hours").
    # First slot = retrain_schedule_start_hour, then every N hours.
    # Example: start=9, every=3 → slots at 09:00, 12:00, 15:00, 18:00, 21:00 UTC.
    retrain_every_hours: int = Field(default=3, ge=1)

    # UTC hour for the first daily training slot (0–23).  Default 9 = 09:00 UTC.
    retrain_schedule_start_hour: int = Field(default=9, ge=0, lt=24)

    # Retrain every N bars processed (used when retrain_trigger="bars").
    retrain_every_bars: int = Field(default=5000, ge=100)

    # ── Signal enricher gates ────────────────────────────────────────────────
    # Downgrade directional signals to HOLD when the confidence-margin (top1
    # minus top2 class probability) is below this threshold.
    # Set cm_gate_enabled=False (default) to disable.
    cm_gate_enabled: bool = Field(default=False)
    min_confidence_margin: float = Field(default=0.10, ge=0.0, le=1.0)

    # Downgrade directional signals to HOLD when the EV regressor score
    # contradicts the classifier direction or |EV| < ev_gate_min_r.
    # Only fires when an EV model is loaded; ignored otherwise.
    ev_gate_enabled: bool = Field(default=False)
    ev_gate_min_r: float = Field(default=0.3, gt=0.0)

    # ── Labeling ─────────────────────────────────────────────────────────────
    # Labeling method:
    #   "v1" — original forward-return (close at t+n vs close at t, default)
    #   "v2" — triple barrier (path-aware: checks intrabar highs/lows)
    label_method: str = Field(default="v1")

    # Round-trip cost proxy for v2 triple barrier labeling.
    # Expressed as fraction of price (e.g. 0.0001 = 1 pip on EURUSD at 1.10).
    # Tightens barriers so a move must exceed the estimated spread before being
    # labeled directional.  Has no effect when label_method="v1".
    label_spread_pct: float = Field(default=0.0001, ge=0.0)

    # ── EV regressor ─────────────────────────────────────────────────────────
    # Enable the Expected Value regressor (PR 3).  When True, the training
    # pipeline should call ExpectedValueTrainer.fit() on OOS data and pass
    # ev_trainer to registry.register().
    ev_enabled: bool = Field(default=False)

    # ATR period for the ContinuousTargetBuilder (must match risk manager ATR).
    ev_atr_period: int = Field(default=14, ge=2)

    # ATR multiplier defining 1 R unit (must match Chandelier Exit mult).
    ev_atr_mult: float = Field(default=2.0, gt=0.0)

    # ── Calibration ───────────────────────────────────────────────────────────
    # Enable post-hoc probability calibration.  When True, the training pipeline
    # should call MLClassifier.calibrate() on a holdout set after fit() and
    # before registering the model.
    calibration_enabled: bool = Field(default=False)

    # Calibration algorithm: "isotonic" (non-parametric, needs ≥ 30 samples) or
    # "sigmoid" (Platt scaling, parametric, works with fewer samples).
    calibration_method: str = Field(default="isotonic")

    # ── Registry ─────────────────────────────────────────────────────────────
    # Directory where model snapshots are persisted
    model_registry_dir: str = Field(default="data/models")

    model_config = {
        "env_prefix": "ML_",
        "env_file": str(Path(__file__).parent.parent.parent.parent / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "frozen": True,
    }
