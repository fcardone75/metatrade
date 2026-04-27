"""Model registry — stores and retrieves trained ML models.

The registry keeps an in-memory slot for the "active" model and optionally
persists model bytes to a configurable directory for durability.

Design:
    - In-memory primary store: fast access for the inference path
    - Optional disk persistence: bytes written atomically to avoid partial writes
    - Version tracking: each model snapshot is tagged with a version string
    - Thread-safe writes: model replacement is an atomic reference swap
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from metatrade.ml.artifacts import ArtifactStore
from metatrade.ml.calibration import ProbabilityCalibrator
from metatrade.ml.classifier import ClassifierMetrics, MLClassifier
from metatrade.ml.config import MLConfig
from metatrade.ml.contracts import MlFeatureSchema, MlModelArtifacts
from metatrade.ml.trainers import ExpectedValueTrainer


@dataclass(frozen=True)
class ModelSnapshot:
    """Metadata + reference for a trained model."""

    version: str                              # e.g. "v20240601_1430"
    classifier: MLClassifier
    metrics: ClassifierMetrics
    created_at: float                         # Unix timestamp
    symbol: str                               # Symbol the model was trained for
    tags: dict[str, Any] = field(default_factory=dict)
    feature_schema: MlFeatureSchema | None = None  # populated after fit()
    ev_trainer: ExpectedValueTrainer | None = None  # optional EV regressor


class ModelRegistry:
    """In-memory registry with optional disk persistence for trained models.

    One registry per symbol is recommended to keep model slots clean.

    Args:
        config: MLConfig (controls registry_dir path).
        persist: If True, serialize and write models to disk on registration.
    """

    def __init__(
        self,
        config: MLConfig | None = None,
        persist: bool = False,
    ) -> None:
        self._config = config or MLConfig()
        self._persist = persist
        self._snapshots: dict[str, ModelSnapshot] = {}  # version → snapshot
        self._active_version: str | None = None

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        classifier: MLClassifier,
        symbol: str,
        version: str | None = None,
        tags: dict[str, Any] | None = None,
        feature_schema: MlFeatureSchema | None = None,
        ev_trainer: ExpectedValueTrainer | None = None,
    ) -> ModelSnapshot:
        """Register a trained model in the registry.

        Args:
            classifier: Trained MLClassifier.
            symbol:     Symbol the model was trained for.
            version:    Optional version string (auto-generated if None).
            tags:       Optional free-form metadata dict.
            feature_schema: Optional override for the feature schema.
            ev_trainer: Optional fitted ``ExpectedValueTrainer`` to persist
                        alongside the classifier.

        Returns:
            The created ModelSnapshot.

        Raises:
            ValueError: If the classifier is not trained.
        """
        if not classifier.is_trained:
            raise ValueError("Cannot register an untrained MLClassifier.")
        if classifier.metrics is None:
            raise ValueError("Classifier has no metrics — call fit() first.")

        if version is None:
            ts = time.strftime("%Y%m%d_%H%M", time.localtime())
            version = f"v{ts}"

        # Resolve feature schema: caller-supplied > from classifier
        resolved_schema = feature_schema
        if resolved_schema is None:
            try:
                resolved_schema = classifier.feature_schema()
            except RuntimeError:
                pass

        snapshot = ModelSnapshot(
            version=version,
            classifier=classifier,
            metrics=classifier.metrics,
            created_at=time.time(),
            symbol=symbol,
            tags=tags or {},
            feature_schema=resolved_schema,
            ev_trainer=ev_trainer,
        )
        self._snapshots[version] = snapshot
        self._active_version = version

        if self._persist:
            self._write_to_disk(snapshot)

        return snapshot

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_active(self) -> ModelSnapshot | None:
        """Return the currently active model snapshot, or None if empty."""
        if self._active_version is None:
            return None
        return self._snapshots.get(self._active_version)

    def get(self, version: str) -> ModelSnapshot | None:
        """Return the snapshot for a specific version, or None if not found."""
        return self._snapshots.get(version)

    def list_versions(self) -> list[str]:
        """Return all registered version strings, newest first."""
        return sorted(self._snapshots.keys(), reverse=True)

    def promote(self, version: str) -> None:
        """Set a specific version as the active model.

        Args:
            version: Version string to promote.

        Raises:
            KeyError: If the version is not registered.
        """
        if version not in self._snapshots:
            raise KeyError(f"Version {version!r} not found in registry.")
        self._active_version = version

    def clear(self) -> None:
        """Remove all snapshots from the in-memory registry."""
        self._snapshots.clear()
        self._active_version = None

    @property
    def size(self) -> int:
        """Number of registered model snapshots."""
        return len(self._snapshots)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _write_to_disk(self, snapshot: ModelSnapshot) -> None:
        """Serialize and atomically write a snapshot to disk via ArtifactStore."""
        store = ArtifactStore(self._config.model_registry_dir)

        feature_schema = snapshot.feature_schema
        if feature_schema is None:
            try:
                feature_schema = snapshot.classifier.feature_schema()
            except RuntimeError:
                feature_schema = MlFeatureSchema(
                    feature_names=tuple(snapshot.classifier._feature_names),
                    feature_count=len(snapshot.classifier._feature_names),
                    min_bars=0,
                    vector_class="unknown",
                )

        metrics_dict: dict[str, object] = (
            snapshot.metrics.to_dict()
            if snapshot.metrics is not None
            else {}
        )

        ev_bytes: bytes | None = None
        if snapshot.ev_trainer is not None and snapshot.ev_trainer.is_trained:
            ev_bytes = snapshot.ev_trainer.serialize()

        artifacts = MlModelArtifacts(
            version=snapshot.version,
            symbol=snapshot.symbol,
            created_at=snapshot.created_at,
            feature_schema=feature_schema,
            model_bytes=snapshot.classifier.serialize(),
            metrics=metrics_dict,
            calibrator_bytes=snapshot.classifier.serialize_calibrator(),
            ev_model_bytes=ev_bytes,
            tags=dict(snapshot.tags),
        )
        store.save(artifacts)

    @classmethod
    def list_disk_snapshots(cls, registry_dir: str, symbol: str) -> list[dict[str, Any]]:
        """Return metadata for all persisted snapshots for a symbol, newest first.

        Each entry is the manifest dict (version, symbol, created_at, tags).
        Only manifests with a corresponding .pkl file are included.
        """
        d = Path(registry_dir)
        if not d.exists():
            return []
        results: list[dict[str, Any]] = []
        for meta_path in d.glob(f"{symbol}_*_meta.json"):
            pkl_path = meta_path.with_name(meta_path.name.replace("_meta.json", ".pkl"))
            if not pkl_path.exists():
                continue
            try:
                manifest = json.loads(meta_path.read_text(encoding="utf-8"))
                results.append(manifest)
            except (json.JSONDecodeError, OSError):
                continue
        results.sort(key=lambda m: m.get("created_at", 0.0), reverse=True)
        return results

    def load_from_disk(self, symbol: str, version: str) -> ModelSnapshot | None:
        """Load a previously persisted model from disk.

        Tries ArtifactStore first (new format with full metadata). If no
        ``_meta.json`` is found, falls back to loading the raw ``.pkl`` file
        directly (legacy format) with placeholder metrics.

        Args:
            symbol:  Symbol the model was trained for.
            version: Version string (e.g. "v20240601_1430").

        Returns:
            ModelSnapshot if found on disk, None otherwise.
        """
        registry_dir = self._config.model_registry_dir
        store = ArtifactStore(registry_dir)

        artifacts = store.load(symbol, version)
        if artifacts is not None:
            classifier = MLClassifier.deserialize(artifacts.model_bytes, self._config)
            if artifacts.calibrator_bytes is not None:
                classifier._calibrator = ProbabilityCalibrator.deserialize(
                    artifacts.calibrator_bytes
                )
            metrics = ClassifierMetrics(
                accuracy=float(artifacts.metrics.get("accuracy") or 0.0),
                n_samples=int(artifacts.metrics.get("n_samples") or 0),
                feature_importances={
                    str(k): float(v)
                    for k, v in (artifacts.metrics.get("feature_importances") or {}).items()
                },
                class_distribution={
                    int(k): int(v)
                    for k, v in (artifacts.metrics.get("class_distribution") or {}).items()
                },
            )
            ev_trainer: ExpectedValueTrainer | None = None
            if artifacts.ev_model_bytes is not None:
                ev_trainer = ExpectedValueTrainer.deserialize(
                    artifacts.ev_model_bytes, self._config
                )

            snapshot = ModelSnapshot(
                version=artifacts.version,
                classifier=classifier,
                metrics=metrics,
                created_at=artifacts.created_at,
                symbol=artifacts.symbol,
                tags=dict(artifacts.tags),
                feature_schema=artifacts.feature_schema,
                ev_trainer=ev_trainer,
            )
            self._snapshots[version] = snapshot
            self._active_version = version
            return snapshot

        # Legacy fallback: raw .pkl without _meta.json
        pkl_path = os.path.join(registry_dir, f"{symbol}_{version}.pkl")
        if not os.path.exists(pkl_path):
            return None

        with open(pkl_path, "rb") as f:
            data = f.read()

        classifier = MLClassifier.deserialize(data, self._config)
        placeholder_metrics = ClassifierMetrics(accuracy=0.0, n_samples=0)
        snapshot = ModelSnapshot(
            version=version,
            classifier=classifier,
            metrics=placeholder_metrics,
            created_at=0.0,
            symbol=symbol,
        )
        self._snapshots[version] = snapshot
        self._active_version = version
        return snapshot
