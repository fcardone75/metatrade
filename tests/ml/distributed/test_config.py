"""Tests for MongoTrainConfig."""

from __future__ import annotations

import pytest

from metatrade.ml.distributed.config import MongoTrainConfig


class TestMongoTrainConfig:
    def test_defaults(self):
        cfg = MongoTrainConfig()
        assert cfg.ml_train_mongo is False
        assert cfg.ml_mt5_master is True
        assert cfg.mongo_db == "metatrade"
        assert cfg.ml_worker_poll_sec == 30

    def test_is_enabled_false_by_default(self):
        cfg = MongoTrainConfig()
        assert cfg.is_enabled is False

    def test_is_master_true_by_default(self):
        cfg = MongoTrainConfig()
        assert cfg.is_master is True
        assert cfg.is_worker is False

    def test_worker_role(self, monkeypatch):
        monkeypatch.setenv("ML_MT5_MASTER", "false")
        cfg = MongoTrainConfig()
        assert cfg.is_worker is True
        assert cfg.is_master is False

    def test_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("ML_TRAIN_MONGO", "true")
        monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
        cfg = MongoTrainConfig()
        assert cfg.is_enabled is True
        assert cfg.mongo_uri == "mongodb://localhost:27017"
