"""Structured logging configuration — console + rotating JSON file.

Architecture
------------
structlog is used as the frontend (bound loggers, contextvars, per-call kwargs).
stdlib logging is used as the backend so that third-party libraries (pydantic,
fastapi, httpx, …) are captured automatically.

Two output channels are configured by ``configure_logging()``:

  Console  — human-readable, colored output via ConsoleRenderer.  Useful
             during development and live monitoring.

  File     — one JSON object per line written to ``<log_dir>/metatrade.log``.
             Rotated daily; up to ``backup_days`` old files are kept.
             Easy to grep, parse, and pipe into log-management tools.

JSON line format (errors shown as example)::

    {
      "timestamp": "2026-04-16T10:15:32.184Z",
      "level":     "error",
      "logger":    "metatrade.broker.mt5_adapter",
      "event":     "order_submission_failed",
      "error":     "MT5 order rejected: retcode=10016 comment=Invalid stops",
      "order_id":  "c9653f35-56e9-4641-8fed-ade30b7b5d97",
      "exc_info":  "Traceback …"          <- only when an exception is active
    }

Usage
-----
    from metatrade.core.log import configure_logging, get_logger

    # Call once at application start (before any other import that logs)
    configure_logging()          # reads LOG_* env vars automatically

    log = get_logger(__name__)
    log.info("session_started", symbol="EURUSD", mode="LIVE")
    log.error("broker_timeout", timeout_ms=5000, retry=False)
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import structlog


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_logging(
    level: str | None = None,
    *,
    log_dir: str | Path | None = None,
    log_filename: str = "metatrade.log",
    max_bytes: int | None = None,
    backup_count: int | None = None,
    console: bool = True,
) -> None:
    """Configure structlog + stdlib logging.  Call once at startup.

    Parameters are read from environment variables when not supplied:

    ================  ============  =============================================
    Env var           Default       Description
    ================  ============  =============================================
    LOG_LEVEL         INFO          Minimum level for both console and file.
    LOG_DIR           logs/         Directory for the rotating log file.
                                    Pass an empty string to disable file output.
    LOG_MAX_MB        20            Max size of each log file in megabytes.
    LOG_BACKUP_DAYS   30            Number of daily backup files to retain.
    ================  ============  =============================================

    The file log uses daily rotation (``TimedRotatingFileHandler``).
    """
    from metatrade.core.log_config import LogConfig  # local import avoids circular
    cfg = LogConfig()

    resolved_level   = (level        or cfg.log_level).upper()
    resolved_dir     = Path(log_dir  or cfg.log_dir) if (log_dir or cfg.log_dir) else None
    resolved_max     = max_bytes     or cfg.log_max_mb * 1024 * 1024
    resolved_backup  = backup_count  or cfg.log_backup_days

    int_level = getattr(logging, resolved_level, logging.INFO)

    # ── Shared structlog pre-processors ──────────────────────────────────────
    # These run for every log call, regardless of destination.
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    # ── structlog → stdlib bridge ─────────────────────────────────────────────
    # ProcessorFormatter lets stdlib handlers re-use structlog processors.
    # We prepare two formatters: one for the file (JSON) and one for console.

    file_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.ExceptionRenderer(),   # adds "exc_info" key
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.ExceptionRenderer(),
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        foreign_pre_chain=shared_processors,
    )

    # ── Build handlers ────────────────────────────────────────────────────────
    handlers: list[logging.Handler] = []

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(int_level)
        handlers.append(console_handler)

    if resolved_dir:
        resolved_dir.mkdir(parents=True, exist_ok=True)
        log_path = resolved_dir / log_filename

        # TimedRotatingFileHandler — rotates at midnight, keeps N backups.
        # Each backup is named  metatrade.log.YYYY-MM-DD
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(log_path),
            when="midnight",
            interval=1,
            backupCount=resolved_backup,
            encoding="utf-8",
            utc=True,
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(int_level)
        handlers.append(file_handler)

        # Dedicated error file — WARNING and above only, so problems stand out.
        error_path = resolved_dir / log_filename.replace(".log", "_errors.log")
        error_handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(error_path),
            when="midnight",
            interval=1,
            backupCount=resolved_backup,
            encoding="utf-8",
            utc=True,
        )
        error_handler.setFormatter(file_formatter)
        error_handler.setLevel(logging.WARNING)
        handlers.append(error_handler)

    # ── Configure stdlib root logger ──────────────────────────────────────────
    root = logging.getLogger()
    # Remove any handlers that were added before configure_logging() was called
    # (e.g. by basicConfig or a previous call).
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)
    root.setLevel(int_level)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "asyncio", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── Configure structlog ───────────────────────────────────────────────────
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(int_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Emit one startup entry so the log file always has a visible start marker.
    _startup_log = get_logger("metatrade.core.log")
    _startup_log.info(
        "logging_configured",
        level=resolved_level,
        console=console,
        log_dir=str(resolved_dir) if resolved_dir else None,
        log_file=str(log_path) if resolved_dir else None,
        error_file=str(error_path) if resolved_dir else None,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for the given module name."""
    return structlog.get_logger(name)
