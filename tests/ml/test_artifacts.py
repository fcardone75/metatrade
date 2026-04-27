"""Tests for ml/artifacts.py — ArtifactStore save/load/list."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from metatrade.ml.artifacts import ArtifactStore
from metatrade.ml.contracts import MlFeatureSchema, MlModelArtifacts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NAMES_35 = tuple(f"f{i}" for i in range(35))


def _schema() -> MlFeatureSchema:
    return MlFeatureSchema(
        feature_names=NAMES_35,
        feature_count=35,
        min_bars=70,
        vector_class="FeatureVector",
    )


def _artifacts(version: str = "v20240601_1200", symbol: str = "EURUSD") -> MlModelArtifacts:
    return MlModelArtifacts(
        version=version,
        symbol=symbol,
        created_at=1717228800.0,
        feature_schema=_schema(),
        model_bytes=b"fake_model_bytes_" + version.encode(),
        metrics={"accuracy": 0.65, "n_samples": 500},
        tags={"backend": "histgbm"},
    )


# ---------------------------------------------------------------------------
# save + load round-trip
# ---------------------------------------------------------------------------

class TestArtifactStoreRoundTrip:
    def test_save_load_all_fields(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        art = _artifacts()
        store.save(art)
        loaded = store.load("EURUSD", "v20240601_1200")

        assert loaded is not None
        assert loaded.version == art.version
        assert loaded.symbol == art.symbol
        assert loaded.created_at == art.created_at
        assert loaded.model_bytes == art.model_bytes
        assert loaded.feature_schema.feature_names == art.feature_schema.feature_names
        assert loaded.feature_schema.feature_count == art.feature_schema.feature_count
        assert loaded.feature_schema.min_bars == art.feature_schema.min_bars
        assert loaded.feature_schema.vector_class == art.feature_schema.vector_class
        assert loaded.metrics["accuracy"] == art.metrics["accuracy"]
        assert loaded.metrics["n_samples"] == art.metrics["n_samples"]
        assert loaded.tags["backend"] == "histgbm"

    def test_save_creates_both_files(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(_artifacts())
        assert (tmp_path / "EURUSD_v20240601_1200.pkl").exists()
        assert (tmp_path / "EURUSD_v20240601_1200_meta.json").exists()

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        result = store.load("EURUSD", "v_nonexistent")
        assert result is None

    def test_load_missing_pkl_returns_none(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(_artifacts())
        # Remove the pkl — only meta.json remains
        (tmp_path / "EURUSD_v20240601_1200.pkl").unlink()
        result = store.load("EURUSD", "v20240601_1200")
        assert result is None

    def test_load_missing_meta_returns_none(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(_artifacts())
        (tmp_path / "EURUSD_v20240601_1200_meta.json").unlink()
        result = store.load("EURUSD", "v20240601_1200")
        assert result is None

    def test_load_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(_artifacts())
        meta_path = tmp_path / "EURUSD_v20240601_1200_meta.json"
        meta_path.write_text("{ invalid json }", encoding="utf-8")
        result = store.load("EURUSD", "v20240601_1200")
        assert result is None


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

class TestArtifactStoreAtomicity:
    def test_tmp_file_not_visible_as_version(self, tmp_path: Path) -> None:
        """Leftover .tmp files must not appear as valid versions."""
        store = ArtifactStore(tmp_path)
        # Create a stale .tmp pkl (simulates interrupted write)
        (tmp_path / "EURUSD_v99.pkl.tmp").write_bytes(b"stale")
        versions = store.list_versions("EURUSD")
        assert "v99" not in versions

    def test_save_overwrites_previous(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        art1 = _artifacts("v20240601_1200")
        art2 = MlModelArtifacts(
            version="v20240601_1200",
            symbol="EURUSD",
            created_at=9999.0,
            feature_schema=_schema(),
            model_bytes=b"updated_bytes",
            metrics={"accuracy": 0.70, "n_samples": 600},
        )
        store.save(art1)
        store.save(art2)
        loaded = store.load("EURUSD", "v20240601_1200")
        assert loaded is not None
        assert loaded.model_bytes == b"updated_bytes"
        assert loaded.metrics["accuracy"] == 0.70


# ---------------------------------------------------------------------------
# list_versions
# ---------------------------------------------------------------------------

class TestArtifactStoreListVersions:
    def test_empty_dir(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        assert store.list_versions("EURUSD") == []

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "does_not_exist")
        assert store.list_versions("EURUSD") == []

    def test_multiple_versions_newest_first(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        for ver in ["v20240601_0900", "v20240601_1200", "v20240601_1500"]:
            store.save(_artifacts(ver))
        versions = store.list_versions("EURUSD")
        # sorted() reverses lexicographically — newest version string last → first
        assert versions[0] >= versions[-1]
        assert set(versions) == {"v20240601_0900", "v20240601_1200", "v20240601_1500"}

    def test_only_complete_pairs_listed(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(_artifacts("v20240601_1200"))
        # Remove meta of one, only pkl remains
        (tmp_path / "EURUSD_v20240601_1200_meta.json").unlink()
        assert store.list_versions("EURUSD") == []

    def test_different_symbols_not_mixed(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(_artifacts("v1", "EURUSD"))
        store.save(
            MlModelArtifacts(
                version="v1",
                symbol="GBPUSD",
                created_at=0.0,
                feature_schema=_schema(),
                model_bytes=b"gbp",
                metrics={},
            )
        )
        eurusd_versions = store.list_versions("EURUSD")
        gbpusd_versions = store.list_versions("GBPUSD")
        assert eurusd_versions == ["v1"]
        assert gbpusd_versions == ["v1"]

    def test_tmp_files_excluded(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        # Stale .tmp files should never appear as versions
        (tmp_path / "EURUSD_v_stale.pkl.tmp").write_bytes(b"x")
        (tmp_path / "EURUSD_v_stale_meta.json.tmp").write_text("{}", encoding="utf-8")
        assert store.list_versions("EURUSD") == []
