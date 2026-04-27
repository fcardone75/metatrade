"""Expected Value (EV) trainer — regressor for the R-multiple outcome.

Trains a gradient-boosting regressor on the same feature vectors used
by the classifier, targeting the ATR-scaled R-multiple produced by
``ContinuousTargetBuilder``.

Supported backends (same set as MLClassifier):
  histgbm   — sklearn HistGradientBoostingRegressor (default, no GPU)
  lightgbm  — LGBMRegressor (GPU via device="gpu")
  xgboost   — XGBRegressor  (GPU via device="cuda")
  catboost  — CatBoostRegressor (GPU via task_type="GPU", ≥2 GB VRAM)

Leakage note: the feature vectors and targets must come from separate
out-of-sample periods.  Training the EV regressor on the same bars used to
train the classifier introduces leakage.  The walk-forward pipeline (PR 4)
enforces this; when calling ``fit()`` directly, the caller is responsible.

Merge criterion: R² > 0.05 on OOS data.
"""

from __future__ import annotations

import pickle
import subprocess
from dataclasses import dataclass, field
from typing import Any

from metatrade.core.log import get_logger
from metatrade.ml.config import MLConfig
from metatrade.ml.features import FeatureVector
from metatrade.ml.targets import MlContinuousTarget

log = get_logger(__name__)

_VALID_BACKENDS = {"histgbm", "lightgbm", "xgboost", "catboost"}
_MIN_CATBOOST_GPU_MB = 2048


def _free_gpu_mb() -> int | None:
    """Return free GPU memory in MB for GPU 0 via nvidia-smi, or None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def _build_ev_estimator(config: MLConfig, backend: str) -> Any:
    """Instantiate the sklearn-compatible regressor for the given backend.

    Falls back to histgbm with a warning if the library is not installed.
    No class_weight / sample_weight handling — regression target is continuous.
    """
    backend = backend.lower()
    if backend not in _VALID_BACKENDS:
        log.warning("ev_backend_unknown", backend=backend, fallback="histgbm")
        backend = "histgbm"

    if backend == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore[import]

            lgb_kwargs: dict[str, Any] = dict(
                n_estimators=config.max_iter,
                max_depth=config.max_depth if config.max_depth else -1,
                num_leaves=config.num_leaves,
                learning_rate=config.learning_rate,
                min_child_samples=config.min_child_samples,
                random_state=config.random_seed,
                verbosity=-1,
            )
            if config.use_gpu:
                lgb_kwargs["device"] = "gpu"
                log.info("ev_gpu_enabled", backend="lightgbm")
            return lgb.LGBMRegressor(**lgb_kwargs)
        except ImportError:
            log.warning("ev_backend_lightgbm_not_installed", fallback="histgbm")
            backend = "histgbm"

    if backend == "xgboost":
        try:
            import xgboost as xgb  # type: ignore[import]

            xgb_kwargs: dict[str, Any] = dict(
                n_estimators=config.max_iter,
                max_depth=config.max_depth if config.max_depth else 6,
                max_leaves=config.num_leaves,
                learning_rate=config.learning_rate,
                min_child_weight=config.min_child_samples,
                random_state=config.random_seed,
                objective="reg:squarederror",
                verbosity=0,
            )
            if config.use_gpu:
                xgb_kwargs["tree_method"] = "hist"
                xgb_kwargs["device"] = "cuda"
                log.info("ev_gpu_enabled", backend="xgboost")
            return xgb.XGBRegressor(**xgb_kwargs)
        except ImportError:
            log.warning("ev_backend_xgboost_not_installed", fallback="histgbm")
            backend = "histgbm"

    if backend == "catboost":
        try:
            from catboost import CatBoostRegressor  # type: ignore[import]

            cb_kwargs: dict[str, Any] = dict(
                iterations=config.max_iter,
                depth=config.max_depth if config.max_depth else 6,
                learning_rate=config.learning_rate,
                min_data_in_leaf=config.min_child_samples,
                random_seed=config.random_seed,
                verbose=0,
            )
            if config.use_gpu:
                free_mb = _free_gpu_mb()
                if free_mb is not None and free_mb < _MIN_CATBOOST_GPU_MB:
                    log.warning(
                        "ev_catboost_gpu_skipped_low_memory",
                        free_gpu_mb=free_mb,
                        min_required_mb=_MIN_CATBOOST_GPU_MB,
                    )
                else:
                    cb_kwargs["task_type"] = "GPU"
                    log.info("ev_gpu_enabled", backend="catboost", free_gpu_mb=free_mb)
            return CatBoostRegressor(**cb_kwargs)
        except ImportError:
            log.warning("ev_backend_catboost_not_installed", fallback="histgbm")
            backend = "histgbm"

    # histgbm (default / fallback) — sklearn, no GPU support
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=config.max_iter,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        min_samples_leaf=config.min_child_samples,
        random_state=config.random_seed,
    )


@dataclass
class EvMetrics:
    """Training / evaluation metrics for the EV regressor."""

    r2: float
    mae: float
    n_samples: int
    feature_importances: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "r2": self.r2,
            "mae": self.mae,
            "n_samples": self.n_samples,
            "feature_importances": self.feature_importances,
        }


class ExpectedValueTrainer:
    """Trains and serves a gradient-boosting regressor for EV prediction.

    The model predicts the expected R-multiple for the next ``forward_bars``
    bars given the current bar's feature vector.

    Args:
        config:  ``MLConfig`` for hyper-parameter settings (shares max_iter,
                 max_depth, learning_rate, min_child_samples, random_seed with
                 the classifier).
        backend: ML backend to use.  Accepts the same values as
                 ``MLConfig.backend`` (histgbm | lightgbm | xgboost | catboost).
                 Defaults to ``config.backend`` when omitted.
    """

    def __init__(
        self,
        config: MLConfig | None = None,
        backend: str | None = None,
    ) -> None:
        self._config = config or MLConfig()
        self._backend = (backend or self._config.backend).lower()
        self._model: object = None
        self._feature_names: list[str] = []
        self._metrics: EvMetrics | None = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def metrics(self) -> EvMetrics | None:
        return self._metrics

    def fit(
        self,
        feature_vectors: list[FeatureVector],
        targets: list[MlContinuousTarget],
    ) -> EvMetrics:
        """Fit the EV regressor.

        Feature vectors and targets are aligned by position (index ``k`` in
        ``feature_vectors`` corresponds to ``targets[k]``).

        Args:
            feature_vectors: Feature vectors extracted from training bars.
            targets:         R-multiple targets from the same bars.

        Returns:
            ``EvMetrics`` from in-sample evaluation.

        Raises:
            ValueError: If lengths mismatch or fewer than
                        ``config.min_train_samples`` samples are provided.
        """
        if len(feature_vectors) != len(targets):
            raise ValueError(
                f"feature_vectors ({len(feature_vectors)}) and "
                f"targets ({len(targets)}) must have equal length"
            )
        n = len(feature_vectors)
        if n < self._config.min_train_samples:
            raise ValueError(
                f"Need at least {self._config.min_train_samples} samples, got {n}"
            )

        import numpy as np
        import pandas as pd
        from sklearn.metrics import mean_absolute_error, r2_score

        fv_cls = type(feature_vectors[0])
        self._feature_names = fv_cls.feature_names()

        X = pd.DataFrame(
            [fv.to_list() for fv in feature_vectors],
            columns=self._feature_names,
        )
        y = np.array([t.r_multiple for t in targets], dtype=float)

        model = _build_ev_estimator(self._config, self._backend)
        log.info("ev_trainer_fit", n_samples=n, backend=self._backend)
        model.fit(X, y)

        preds = model.predict(X)
        r2 = float(r2_score(y, preds))
        mae = float(mean_absolute_error(y, preds))

        try:
            importances = {
                name: float(imp)
                for name, imp in zip(self._feature_names, model.feature_importances_)
            }
        except AttributeError:
            importances = {name: 0.0 for name in self._feature_names}

        self._model = model
        self._metrics = EvMetrics(
            r2=r2,
            mae=mae,
            n_samples=n,
            feature_importances=importances,
        )
        self._is_trained = True
        log.info(
            "ev_trainer_fit_done",
            in_sample_r2=round(r2, 4),
            in_sample_mae=round(mae, 4),
        )
        return self._metrics

    def predict(self, feature_vector: FeatureVector) -> float:
        """Predict the expected R-multiple for a single feature vector.

        Args:
            feature_vector: Extracted features for the current bar.

        Returns:
            Predicted R-multiple.  Positive = bullish, negative = bearish.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError(
                "ExpectedValueTrainer has not been trained. Call fit() first."
            )

        import pandas as pd

        X = pd.DataFrame(
            [feature_vector.to_list()],
            columns=self._feature_names or None,
        )
        result = self._model.predict(X)  # type: ignore[union-attr]
        return float(result[0])

    def serialize(self) -> bytes:
        """Serialize the trained regressor to bytes.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if self._model is None:
            raise RuntimeError("No trained EV model to serialize.")
        payload = {
            "model": pickle.dumps(self._model),
            "feature_names": self._feature_names,
            "metrics": self._metrics.to_dict() if self._metrics else {},
            "backend": self._backend,
        }
        return pickle.dumps(payload)

    @classmethod
    def deserialize(
        cls,
        data: bytes,
        config: MLConfig | None = None,
    ) -> ExpectedValueTrainer:
        """Reconstruct an ``ExpectedValueTrainer`` from serialized bytes.

        Args:
            data:   Bytes produced by ``serialize()``.
            config: Optional ``MLConfig`` (defaults to ``MLConfig()``).

        Returns:
            Fitted ``ExpectedValueTrainer`` with metrics restored.
        """
        instance = cls(config)
        payload: dict[str, object] = pickle.loads(data)  # noqa: S301
        instance._model = pickle.loads(payload["model"])  # type: ignore[arg-type]  # noqa: S301
        instance._feature_names = list(payload.get("feature_names") or [])  # type: ignore[arg-type]
        instance._backend = str(payload.get("backend") or "histgbm")

        metrics_raw = payload.get("metrics") or {}
        if isinstance(metrics_raw, dict) and "r2" in metrics_raw:
            instance._metrics = EvMetrics(
                r2=float(metrics_raw.get("r2") or 0.0),
                mae=float(metrics_raw.get("mae") or 0.0),
                n_samples=int(metrics_raw.get("n_samples") or 0),
                feature_importances={
                    str(k): float(v)
                    for k, v in (metrics_raw.get("feature_importances") or {}).items()  # type: ignore[union-attr]
                },
            )
        instance._is_trained = True
        return instance
