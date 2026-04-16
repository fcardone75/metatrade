"""Logging configuration via environment variables."""

from __future__ import annotations

from metatrade.core.config_base import BaseConfig


class LogConfig(BaseConfig):
    """Controls where and how logs are written.

    Fields map directly to env vars (no prefix — the field names already
    start with ``log_``):

    ================  ============  =============================================
    Env var           Default       Description
    ================  ============  =============================================
    LOG_LEVEL         INFO          Minimum level for both console and file.
    LOG_DIR           logs          Directory for rotating log files.
                                    Set to empty string to disable file output.
    LOG_MAX_MB        20            Max size per log file before rotation.
    LOG_BACKUP_DAYS   30            Days of daily log files to retain.
    ================  ============  =============================================
    """

    model_config = {
        "env_prefix": "",          # field names are already "log_*" → LOG_*
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "frozen": True,
    }

    log_level: str = "INFO"
    log_dir: str = "logs"
    log_max_mb: int = 20
    log_backup_days: int = 30
