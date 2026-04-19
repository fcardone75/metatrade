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
        """Serialize and atomically write a snapshot + manifest to disk."""
        registry_dir = self._config.model_registry_dir
        os.makedirs(registry_dir, exist_ok=True)

        stem = f"{snapshot.symbol}_{snapshot.version}"
        pkl_path = os.path.join(registry_dir, f"{stem}.pkl")
        meta_path = os.path.join(registry_dir, f"{stem}_meta.json")

        # Write model bytes
        data = snapshot.classifier.serialize()
        tmp_pkl = pkl_path + ".tmp"
        with open(tmp_pkl, "wb") as f:
            f.write(data)
        os.replace(tmp_pkl, pkl_path)

        # Write metadata manifest (holdout_accuracy and tags readable without loading model)
        manifest = {
            "version": snapshot.version,
            "symbol": snapshot.symbol,
            "created_at": snapshot.created_at,
            "tags": snapshot.tags,
        }
        tmp_meta = meta_path + ".tmp"
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_meta, meta_path)

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
