"""Configuration for distributed ML training via MongoDB Atlas."""

from __future__ import annotations

from metatrade.core.config_base import BaseConfig


class MongoTrainConfig(BaseConfig):
    """Settings for the MongoDB-backed distributed training system.

    Env vars (no prefix — field names map directly):
      MONGO_URI, MONGO_DB
      ML_TRAIN_MONGO, ML_MT5_MASTER, ML_WORKER_POLL_SEC
    """

    mongo_uri: str = ""
    mongo_db: str = "metatrade"

    ml_train_mongo: bool = False
    ml_mt5_master: bool = True
    ml_worker_poll_sec: int = 30

    @property
    def is_enabled(self) -> bool:
        return self.ml_train_mongo

    @property
    def is_master(self) -> bool:
        return self.ml_mt5_master

    @property
    def is_worker(self) -> bool:
        return not self.ml_mt5_master
