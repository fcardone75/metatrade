"""Artifact persistence — atomic save/load of MlModelArtifacts to disk.

Naming convention on disk:
    {symbol}_{version}.pkl       — raw model bytes
    {symbol}_{version}_meta.json — all metadata except model_bytes

Both files are written atomically via a .tmp → rename pattern, matching
the convention already used in the legacy ModelRegistry._write_to_disk().
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from metatrade.core.log import get_logger
from metatrade.ml.contracts import MlFeatureSchema, MlModelArtifacts

log = get_logger(__name__)


class ArtifactStore:
    """Save and load MlModelArtifacts to/from a local directory.

    One store per registry directory; multiple symbols are supported via
    the ``symbol`` field in ``MlModelArtifacts``.

    Args:
        registry_dir: Directory where artifact files are stored.
    """

    def __init__(self, registry_dir: str | Path) -> None:
        self._dir = Path(registry_dir)

    # ── Write ─────────────────────────────────────────────────────────────────

    def save(self, artifacts: MlModelArtifacts) -> Path:
        """Atomically write model bytes and metadata JSON to disk.

        Returns the path to the saved ``.pkl`` file.
        """
        os.makedirs(self._dir, exist_ok=True)
        stem = f"{artifacts.symbol}_{artifacts.version}"
        pkl_path = self._dir / f"{stem}.pkl"
        json_path = self._dir / f"{stem}_meta.json"

        # Model bytes — write to .tmp then atomically replace
        tmp_pkl = pkl_path.with_suffix(".pkl.tmp")
        tmp_pkl.write_bytes(artifacts.model_bytes)
        tmp_pkl.replace(pkl_path)

        # Calibrator bytes (optional) — write atomically if present
        cal_path = self._dir / f"{stem}_cal.pkl"
        if artifacts.calibrator_bytes is not None:
            tmp_cal = cal_path.with_suffix(".pkl.tmp")
            tmp_cal.write_bytes(artifacts.calibrator_bytes)
            tmp_cal.replace(cal_path)
        elif cal_path.exists():
            cal_path.unlink()

        # EV model bytes (optional) — write atomically if present
        ev_path = self._dir / f"{stem}_ev.pkl"
        if artifacts.ev_model_bytes is not None:
            tmp_ev = ev_path.with_suffix(".pkl.tmp")
            tmp_ev.write_bytes(artifacts.ev_model_bytes)
            tmp_ev.replace(ev_path)
        elif ev_path.exists():
            ev_path.unlink()

        # Metadata JSON — everything except binary blobs
        meta: dict[str, object] = {
            "version": artifacts.version,
            "symbol": artifacts.symbol,
            "created_at": artifacts.created_at,
            "feature_schema": {
                "feature_names": list(artifacts.feature_schema.feature_names),
                "feature_count": artifacts.feature_schema.feature_count,
                "min_bars": artifacts.feature_schema.min_bars,
                "vector_class": artifacts.feature_schema.vector_class,
            },
            "metrics": artifacts.metrics,
            "has_calibrator": artifacts.calibrator_bytes is not None,
            "has_ev_model": artifacts.ev_model_bytes is not None,
            "tags": artifacts.tags,
        }
        tmp_json = json_path.with_suffix(".json.tmp")
        tmp_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp_json.replace(json_path)

        log.info(
            "artifact_saved",
            symbol=artifacts.symbol,
            version=artifacts.version,
            pkl=str(pkl_path),
            has_calibrator=artifacts.calibrator_bytes is not None,
            has_ev_model=artifacts.ev_model_bytes is not None,
        )
        return pkl_path

    # ── Read ──────────────────────────────────────────────────────────────────

    def load(self, symbol: str, version: str) -> MlModelArtifacts | None:
        """Load artifacts from disk.

        Returns ``None`` if either the ``.pkl`` or ``_meta.json`` file is
        missing or cannot be parsed. Errors are logged, not raised.
        """
        stem = f"{symbol}_{version}"
        pkl_path = self._dir / f"{stem}.pkl"
        json_path = self._dir / f"{stem}_meta.json"

        if not pkl_path.exists() or not json_path.exists():
            return None

        try:
            model_bytes = pkl_path.read_bytes()
            meta: dict[str, object] = json.loads(
                json_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                "artifact_load_failed",
                symbol=symbol,
                version=version,
                error=str(exc),
            )
            return None

        fs_raw = meta.get("feature_schema")
        fs_raw = fs_raw if isinstance(fs_raw, dict) else {}
        feature_names_raw = fs_raw.get("feature_names")
        feature_names_raw = feature_names_raw if isinstance(feature_names_raw, list) else []
        feature_count = int(fs_raw.get("feature_count") or len(feature_names_raw))
        min_bars = int(fs_raw.get("min_bars") or 0)
        vector_class = str(fs_raw.get("vector_class") or "")

        feature_schema = MlFeatureSchema(
            feature_names=tuple(str(n) for n in feature_names_raw),
            feature_count=feature_count,
            min_bars=min_bars,
            vector_class=vector_class,
        )

        metrics_raw = meta.get("metrics")
        metrics_raw = metrics_raw if isinstance(metrics_raw, dict) else {}
        tags_raw = meta.get("tags")
        tags_raw = tags_raw if isinstance(tags_raw, dict) else {}

        # Load calibrator bytes if flagged in metadata and file exists
        calibrator_bytes: bytes | None = None
        if meta.get("has_calibrator"):
            cal_path = self._dir / f"{stem}_cal.pkl"
            if cal_path.exists():
                try:
                    calibrator_bytes = cal_path.read_bytes()
                except OSError as exc:
                    log.warning(
                        "calibrator_load_failed",
                        symbol=symbol,
                        version=version,
                        error=str(exc),
                    )

        # Load EV model bytes if flagged in metadata and file exists
        ev_model_bytes: bytes | None = None
        if meta.get("has_ev_model"):
            ev_path = self._dir / f"{stem}_ev.pkl"
            if ev_path.exists():
                try:
                    ev_model_bytes = ev_path.read_bytes()
                except OSError as exc:
                    log.warning(
                        "ev_model_load_failed",
                        symbol=symbol,
                        version=version,
                        error=str(exc),
                    )

        return MlModelArtifacts(
            version=str(meta.get("version") or version),
            symbol=str(meta.get("symbol") or symbol),
            created_at=float(meta.get("created_at") or 0.0),
            feature_schema=feature_schema,
            model_bytes=model_bytes,
            metrics=dict(metrics_raw),
            calibrator_bytes=calibrator_bytes,
            ev_model_bytes=ev_model_bytes,
            tags=dict(tags_raw),
        )

    def list_versions(self, symbol: str) -> list[str]:
        """Return all persisted versions for a symbol, newest first.

        Only versions with both a ``.pkl`` and a ``_meta.json`` file are
        included. Incomplete writes (e.g. only .tmp files) are ignored.
        """
        if not self._dir.exists():
            return []

        prefix = f"{symbol}_"
        versions: list[str] = []

        for pkl in sorted(self._dir.glob(f"{prefix}*.pkl")):
            if pkl.suffix == ".tmp" or pkl.name.endswith(".pkl.tmp"):
                continue
            stem = pkl.stem  # e.g. "EURUSD_v20240601_1430"
            json_path = self._dir / f"{stem}_meta.json"
            if json_path.exists():
                version = stem[len(prefix):]
                versions.append(version)

        return sorted(versions, reverse=True)
