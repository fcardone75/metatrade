"""System-wide exception hierarchy.

Design rules:
- Every exception carries a machine-readable `code` for log filtering.
- Every exception carries optional `context` dict for structured logging.
- No silent failures: callers must explicitly handle or propagate.
- All exceptions inherit from MetaTradeError to allow broad catch at boundaries.
"""

from __future__ import annotations


class MetaTradeError(Exception):
    """Base exception for the entire MetaTrade system."""

    def __init__(
        self,
        message: str,
        code: str | None = None,
        context: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.context: dict = context or {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"code={self.code!r}, "
            f"message={self.message!r}, "
            f"context={self.context!r})"
        )


# ── Configuration ──────────────────────────────────────────────────────────────

class ConfigurationError(MetaTradeError):
    """Invalid or missing configuration."""


# ── Data ───────────────────────────────────────────────────────────────────────

class DataError(MetaTradeError):
    """Base class for data-related errors."""


class StaleDataError(DataError):
    """Data exists but is too old to be safely used."""


class MissingDataError(DataError):
    """Required data is not available."""


class CorruptDataError(DataError):
    """Data is present but structurally invalid or out of range."""


class GapDetectedError(DataError):
    """Unexpected gap found in a time series."""


class InsufficientDataError(DataError):
    """Not enough data points to compute a reliable result."""


# ── Execution ──────────────────────────────────────────────────────────────────

class ExecutionError(MetaTradeError):
    """Base class for order execution errors."""


class OrderRejectedError(ExecutionError):
    """Broker rejected the order."""


class DuplicateOrderError(ExecutionError):
    """Duplicate order detected — idempotency guard triggered."""


class BrokerConnectionError(ExecutionError):
    """Cannot connect to or communicate with the broker."""


class BrokerTimeoutError(ExecutionError):
    """Broker did not respond within the allowed timeout."""


# ── Risk ───────────────────────────────────────────────────────────────────────

class RiskVetoError(MetaTradeError):
    """Risk manager vetoed the proposed operation."""


class KillSwitchActiveError(MetaTradeError):
    """Kill switch is active; the operation is not permitted."""


class DrawdownLimitError(MetaTradeError):
    """Drawdown limit reached; trading halted."""


# ── Analysis Modules ───────────────────────────────────────────────────────────

class ModuleError(MetaTradeError):
    """Base class for analysis module errors."""


class ModuleNotReadyError(ModuleError):
    """Module has insufficient data to produce a signal (warmup period)."""


class ModuleDisabledError(ModuleError):
    """Module is explicitly disabled in configuration."""


# ── Versioning & Compatibility ─────────────────────────────────────────────────

class IncompatibleVersionError(MetaTradeError):
    """Version mismatch between modules or contract schemas."""


# ── Mode Guard ─────────────────────────────────────────────────────────────────

class ModeGuardError(MetaTradeError):
    """Attempted an operation not permitted in the current RunMode.

    Example: sending a real order while in BACKTEST mode.
    """
