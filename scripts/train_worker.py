"""Worker entry point — polls MongoDB Atlas for training jobs and runs them.

Usage:
    python scripts/train_worker.py

Required env vars:
    MONGO_URI       — MongoDB Atlas connection string
    MONGO_DB        — database name (default: metatrade)
    ML_TRAIN_MONGO  — must be "true"
    ML_MT5_MASTER   — must be "false" (this machine is the worker)

Optional:
    ML_WORKER_POLL_SEC — polling interval when idle (default: 30)
    LOG_LEVEL          — log level (default: INFO)
    LOG_DIR            — log directory (default: logs/)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metatrade.core.log import configure_logging, get_logger
from metatrade.ml.distributed.config import MongoTrainConfig
from metatrade.ml.distributed.worker_daemon import WorkerDaemon

log = get_logger(__name__)


def main() -> None:
    configure_logging()

    cfg = MongoTrainConfig()

    if not cfg.is_enabled:
        log.error("worker_disabled", reason="ML_TRAIN_MONGO is not true")
        sys.exit(1)

    if cfg.is_master:
        log.error("worker_is_master", reason="ML_MT5_MASTER=true — this machine is the master, not the worker")
        sys.exit(1)

    if not cfg.mongo_uri:
        log.error("worker_no_mongo_uri", reason="MONGO_URI is not set")
        sys.exit(1)

    daemon = WorkerDaemon(cfg)
    daemon.run()


if __name__ == "__main__":
    main()
