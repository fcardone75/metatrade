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

import os
import time
from dataclasses import dataclass, field
from typing import Any

from metatrade.ml.classifier import ClassifierMetrics, MLClassifier
from metatrade.ml.config import MLConfig


@dataclass(frozen=True)
class ModelSnapshot:
    """Metadata + reference for a trained model."""

    version: str                   # e.g. "v20240601_1430"
    classifier: MLClassifier
    metrics: ClassifierMetrics
    created_at: float              # Unix timestamp
    symbol: str                    # Symbol the model was trained for
    tags: dict[str, Any] = field(default_factory=dict)


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
    ) -> ModelSnapshot:
        """Register a trained model in the registry.

        Args:
            classifier: Trained MLClassifier.
            symbol:     Symbol the model was trained for.
            version:    Optional version string (auto-generated if None).
            tags:       Optional free-form metadata dict.

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

        snapshot = ModelSnapshot(
            version=version,
            classifier=classifier,
            metrics=classifier.metrics,
            created_at=time.time(),
            symbol=symbol,
            tags=tags or {},
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
        """Serialize and atomically write a snapshot to disk."""
        registry_dir = self._config.model_registry_dir
        os.makedirs(registry_dir, exist_ok=True)

        filename = f"{snapshot.symbol}_{snapshot.version}.pkl"
        target_path = os.path.join(registry_dir, filename)
        tmp_path = target_path + ".tmp"

        data = snapshot.classifier.serialize()
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, target_path)  # atomic on POSIX

    def load_from_disk(self, symbol: str, version: str) -> ModelSnapshot | None:
        """Load a previously persisted model from disk.

        Args:
            symbol:  Symbol the model was trained for.
            version: Version string (e.g. "v20240601_1430").

        Returns:
            ModelSnapshot if found on disk, None otherwise.
        """
        registry_dir = self._config.model_registry_dir
        filename = f"{symbol}_{version}.pkl"
        path = os.path.join(registry_dir, filename)

        if not os.path.exists(path):
            return None

        with open(path, "rb") as f:
            data = f.read()

        classifier = MLClassifier.deserialize(data, self._config)
        # Metrics are not persisted (model bytes only); create placeholder
        from metatrade.ml.classifier import ClassifierMetrics
        placeholder_metrics = ClassifierMetrics(
            accuracy=0.0,
            n_samples=0,
        )
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
