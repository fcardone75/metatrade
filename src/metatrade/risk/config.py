"""Risk management configuration."""

from __future__ import annotations

from decimal import Decimal

from pydantic import field_validator
from pydantic_settings import BaseSettings


class RiskConfig(BaseSettings):
    """All risk limits and position-sizing parameters.

    All fields are overridable via environment variables prefixed RISK_.

    Example .env:
        RISK_MAX_RISK_PCT=0.01
        RISK_MAX_OPEN_POSITIONS=3
        RISK_DAILY_LOSS_LIMIT_PCT=0.05
        RISK_MAX_SPREAD_PIPS=3.0
        RISK_RISK_REWARD_RATIO=2.0
    """

    model_config = {"env_prefix": "RISK_", "case_sensitive": False}

    # ── Position sizing ───────────────────────────────────────────────────────

    # Maximum fraction of balance to risk per trade (e.g. 0.01 = 1%).
    max_risk_pct: float = 0.01

    # Minimum lot size allowed by broker (e.g. 0.01 for standard MT5).
    min_lot_size: Decimal = Decimal("0.01")

    # Maximum lot size per trade.
    max_lot_size: Decimal = Decimal("10.0")

    # Lot step for rounding (e.g. 0.01 for most MT5 brokers).
    lot_step: Decimal = Decimal("0.01")

    # Risk:reward multiplier for take-profit calculation.
    # None = no take-profit order.
    risk_reward_ratio: float | None = 2.0

    # Pip value per standard lot in account currency (default: USD account on EURUSD).
    # This must be set correctly per pair. For EURUSD with USD account: 10.0.
    # For USDJPY with USD account: ~6.7 (depends on current price).
    pip_value_per_lot: Decimal = Decimal("10.0")

    # Number of decimal places that define 1 pip (4 for most pairs, 2 for JPY).
    pip_digits: int = 4

    # ── Trade gates ──────────────────────────────────────────────────────────

    # Maximum number of concurrently open positions (0 = unlimited).
    max_open_positions: int = 5

    # Maximum allowed spread in pips. Wider spreads → veto with SPREAD_TOO_WIDE.
    # 0.0 = no spread check.
    max_spread_pips: float = 5.0

    # Minimum free margin as a fraction of balance required before opening a trade.
    # E.g. 0.20 = must have at least 20% of balance as free margin.
    min_free_margin_pct: float = 0.20

    # ── Daily loss limit ─────────────────────────────────────────────────────

    # Maximum allowed daily loss as a fraction of starting balance.
    # 0.05 = 5%. Checked against account.equity vs account.balance.
    daily_loss_limit_pct: float = 0.05

    # ── Kill switch ───────────────────────────────────────────────────────────

    # Drawdown fraction of balance that auto-triggers SESSION_GATE.
    # 0.10 = 10% intraday drawdown triggers level-2 kill switch.
    auto_kill_drawdown_pct: float = 0.10

    # ── Volatility-scaled position sizing ─────────────────────────────────────
    # When enabled, lot size is scaled inversely to current ATR relative to a
    # target "normal" ATR level.  High volatility → smaller lots; low vol → normal.
    #
    #   vol_mult = clamp(target_atr_pips / current_atr_pips,
    #                    vol_scaling_min_mult, 1.0)
    #   effective_lots = standard_lots × vol_mult
    #
    # Disabled by default; enable for live trading to avoid over-sizing in spikes.
    vol_scaling_enabled:  bool  = False
    vol_target_atr_pips:  float = 20.0   # "normal" ATR in pips — calibrate per symbol
    vol_scaling_min_mult: float = 0.25   # never reduce below 25 % of standard size

    # ── Validators ───────────────────────────────────────────────────────────

    @field_validator("max_risk_pct", "daily_loss_limit_pct")
    @classmethod
    def _positive_fraction(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"Must be in (0.0, 1.0], got {v}")
        return v

    @field_validator("min_free_margin_pct", "auto_kill_drawdown_pct")
    @classmethod
    def _non_negative_fraction(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"Must be >= 0.0, got {v}")
        return v

    @field_validator("pip_digits")
    @classmethod
    def _pip_digits_valid(cls, v: int) -> int:
        if v not in (2, 3, 4, 5):
            raise ValueError(f"pip_digits must be 2, 3, 4, or 5, got {v}")
        return v
