"""Registry for trained SL/TP policy selector models.

Provides save/load/promote semantics identical in spirit to the existing
ModelRegistry (``metatrade.ml.registry``) but dedicated to SL/TP models.

Each model is saved as:
    {registry_dir}/{symbol}_sltp_{version}.pkl

Active version pointer:
    {registry_dir}/{symbol}_sltp_active.txt

Thread-safety: model replacement is an atomic in-memory reference swap.
Disk writes are not locked; use a single writer process in practice.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class SlTpModelSnapshot:
    """Metadata and trained classifier for one SL/TP policy model.

    Attributes:
        version:       Version string, e.g. ``"sltp_v20240601_1430"``.
        model:         Trained sklearn-compatible classifier.
        policy_ids:    Class labels (policy_id strings) the model knows.
        feature_names: Ordered feature column names for inference.
        symbol:        Symbol the model was trained for.
        mean_accuracy: Walk-forward mean accuracy across folds.
        n_samples:     Training dataset size.
        created_at:    Unix timestamp of model creation.
        tags:          Optional free-form metadata.
    """

    version: str
    model: Any
    policy_ids: list[str]
    feature_names: tuple[str, ...]
    symbol: str
    mean_accuracy: float
    n_samples: int
    created_at: float = field(default_factory=time.time)
    tags: dict[str, Any] = field(default_factory=dict)


class SlTpModelRegistry:
    """File-based registry for SL/TP policy selector models.

    One registry instance per symbol is recommended (each symbol may need
    its own model due to different volatility characteristics).

    Args:
        registry_dir: Directory for saving/loading model files.
        symbol:       Symbol identifier used in file names.
    """

    def __init__(self, registry_dir: str | Path, symbol: str = "EURUSD") -> None:
        self._dir = Path(registry_dir)
        self._symbol = symbol
        self._snapshots: dict[str, SlTpModelSnapshot] = {}
        self._active_version: str | None = None

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        model: Any,
        policy_ids: list[str],
        feature_names: tuple[str, ...],
        mean_accuracy: float,
        n_samples: int,
        version: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> SlTpModelSnapshot:
        """Store a trained model in the in-memory registry.

        Args:
            model:          Trained classifier.
            policy_ids:     Class labels.
            feature_names:  Ordered feature columns.
            mean_accuracy:  Walk-forward mean accuracy.
            n_samples:      Training dataset size.
            version:        Version string (auto-generated if None).
            tags:           Optional free-form metadata.

        Returns:
            The created SlTpModelSnapshot.
        """
        if version is None:
            from datetime import datetime
            version = datetime.now(UTC).strftime("sltp_v%Y%m%d_%H%M")

        snap = SlTpModelSnapshot(
            version=version,
            model=model,
            policy_ids=policy_ids,
            feature_names=feature_names,
            symbol=self._symbol,
            mean_accuracy=mean_accuracy,
            n_samples=n_samples,
            tags=tags or {},
        )
        self._snapshots[version] = snap
        return snap

    def promote(self, version: str) -> None:
        """Set a version as the active model for inference.

        Raises:
            KeyError: If version is not in the in-memory registry.
        """
        if version not in self._snapshots:
            raise KeyError(
                f"SlTpModelRegistry: version '{version}' not in registry"
            )
        self._active_version = version
        log.info("sltp_model_promoted", version=version, symbol=self._symbol)

    def get_active(self) -> SlTpModelSnapshot | None:
        """Return the active model snapshot, or None if none is set."""
        if self._active_version is None:
            return None
        return self._snapshots.get(self._active_version)

    def list_versions(self) -> list[str]:
        """Return all registered version strings."""
        return list(self._snapshots)

    # ── Disk persistence ──────────────────────────────────────────────────────

    def save_to_disk(self, version: str | None = None) -> Path | None:
        """Serialize a snapshot to disk.

        Args:
            version: Version to save (defaults to the active version).

        Returns:
            Path of the written file, or None on failure.
        """
        ver = version or self._active_version
        if ver is None:
            log.warning("sltp_save_no_version")
            return None
        snap = self._snapshots.get(ver)
        if snap is None:
            log.warning("sltp_save_version_not_found", version=ver)
            return None

        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{self._symbol}_sltp_{ver}.pkl"
        try:
            with open(path, "wb") as fh:
                pickle.dump(snap, fh, protocol=pickle.HIGHEST_PROTOCOL)
            # Update active pointer file
            if ver == self._active_version:
                (self._dir / f"{self._symbol}_sltp_active.txt").write_text(ver)
            log.info("sltp_model_saved", path=str(path), version=ver)
            return path
        except Exception as exc:
            log.error("sltp_model_save_failed", error=str(exc))
            return None

    def load_from_disk(self, version: str | None = None) -> SlTpModelSnapshot | None:
        """Deserialize a snapshot from disk.

        If version is None, reads the active pointer file to determine
        which version to load.

        Args:
            version: Version to load, or None to load the active version.

        Returns:
            SlTpModelSnapshot, or None if not found.
        """
        if version is None:
            ptr = self._dir / f"{self._symbol}_sltp_active.txt"
            if ptr.exists():
                version = ptr.read_text().strip()
            if not version:
                return None

        path = self._dir / f"{self._symbol}_sltp_{version}.pkl"
        if not path.exists():
            return None

        try:
            with open(path, "rb") as fh:
                snap: SlTpModelSnapshot = pickle.load(fh)
            self._snapshots[snap.version] = snap
            log.info("sltp_model_loaded", version=snap.version, symbol=self._symbol)
            return snap
        except Exception as exc:
            log.error("sltp_model_load_failed", path=str(path), error=str(exc))
            return None
