"""ML module configuration."""

from __future__ import annotations

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

    # Minimum number of training samples
    min_train_samples: int = Field(default=100, ge=10)

    # Random seed for reproducibility
    random_seed: int = Field(default=42)

    # Max number of boosting iterations for HistGradientBoostingClassifier
    # (equivalent to n_estimators in RandomForest)
    max_iter: int = Field(default=100, ge=10)

    # Max depth of each tree (None = unlimited)
    max_depth: int | None = Field(default=5)

    # Class weight for HistGradientBoostingClassifier.
    # "balanced" automatically compensates for HOLD-dominated datasets by
    # up-weighting BUY/SELL samples.  None = uniform weights.
    class_weight: str | None = Field(default="balanced")

    # ── Holdout evaluation ────────────────────────────────────────────────────
    # Fraction of bars (from the end of the series) reserved as a final
    # out-of-sample holdout.  Walk-forward training only sees bars[:holdout_start].
    # Set to 0.0 to disable (no holdout reserved).
    holdout_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)

    # ── Session filter ────────────────────────────────────────────────────────
    # When both are set, only bars whose UTC hour is in [start, end) are used
    # for training.  Helps remove thin-market noise on M1/M5.
    # Example: start=7, end=21 keeps London + NY sessions only.
    session_filter_utc_start: int | None = Field(default=None, ge=0, lt=24)
    session_filter_utc_end: int | None = Field(default=None, ge=0, lt=24)

    # ── Registry ─────────────────────────────────────────────────────────────
    # Directory where model snapshots are persisted
    model_registry_dir: str = Field(default="data/models")

    model_config = {
        "env_prefix": "ML_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "frozen": True,
    }
