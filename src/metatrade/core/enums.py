"""System-wide enumerations.

All enums use string values so they serialize cleanly to JSON and
are human-readable in logs and audit trails.
"""

from enum import Enum


class SignalDirection(str, Enum):
    """Output direction produced by every analysis module."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RunMode(str, Enum):
    """Operating mode of the system.

    BACKTEST — historical simulation, no real orders.
    PAPER    — live data, simulated orders.
    LIVE     — live data, real orders sent to broker.
    """

    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class PositionState(str, Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


class Timeframe(str, Enum):
    """Standard Forex timeframes, matching MT5 constants."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"

    @property
    def seconds(self) -> int:
        """Duration of one bar in seconds."""
        mapping = {
            "M1": 60,
            "M5": 300,
            "M15": 900,
            "M30": 1800,
            "H1": 3600,
            "H4": 14400,
            "D1": 86400,
            "W1": 604800,
            "MN1": 2592000,
        }
        return mapping[self.value]


class ConsensusMode(str, Enum):
    """Voting strategy for the consensus engine."""

    SIMPLE_VOTE = "SIMPLE_VOTE"
    WEIGHTED_VOTE = "WEIGHTED_VOTE"
    DYNAMIC_VOTE = "DYNAMIC_VOTE"


class KillSwitchLevel(int, Enum):
    """Kill switch severity levels.

    Higher level = broader impact.
    Levels are additive: EMERGENCY_HALT also blocks trades.
    """

    NONE = 0
    TRADE_GATE = 1      # Blocks the single trade only
    SESSION_GATE = 2    # Blocks new trades for the session
    EMERGENCY_HALT = 3  # Closes all positions, stops the system
    HARD_KILL = 4       # Manual override, same effect as EMERGENCY_HALT
