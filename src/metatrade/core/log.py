"""Structured JSON logging configuration.

Uses structlog for structured, JSON-formatted log output.
Every log entry includes:
  - timestamp (UTC ISO-8601)
  - log level
  - logger name
  - message
  - any additional context bound via bind()

Usage:
    from metatrade.core.log import get_logger

    log = get_logger(__name__)
    log.info("order_submitted", order_id="abc-123", symbol="EURUSD", lots=0.10)
    log.error("broker_timeout", timeout_ms=5000, retry=False)

    # Bind context for the duration of a scope
    log = log.bind(run_mode="LIVE", symbol="EURUSD")
    log.info("signal_generated", direction="BUY", confidence=0.82)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO", *, as_json: bool = True) -> None:
    """Configure structlog globally. Call once at application startup."""

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    if as_json:
        final_processor: Any = structlog.processors.JSONRenderer()
    else:
        final_processor = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, final_processor],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Mirror to stdlib so third-party libraries (e.g. pydantic) also emit JSON
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for the given module name."""
    return structlog.get_logger(name)
