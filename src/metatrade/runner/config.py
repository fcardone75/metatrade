"""Runner configuration — controls how the orchestrators behave."""

from __future__ import annotations

from pydantic import Field

from metatrade.core.config_base import BaseConfig


class RunnerConfig(BaseConfig):
    """Shared configuration for all runner modes (backtest, paper, live).

    Override with environment variables prefixed ``RUNNER_``.
    """

    # ── Symbol / timeframe ────────────────────────────────────────────────────
    symbol: str = Field(default="EURUSD")

    # ── Consensus ─────────────────────────────────────────────────────────────
    # Minimum consensus ratio to open a trade (0.6 = 60% of weighted votes)
    consensus_threshold: float = Field(default=0.6, ge=0.5, le=1.0)

    # Minimum number of module signals before consensus is attempted
    min_signals: int = Field(default=2, ge=1)

    # ── Risk ─────────────────────────────────────────────────────────────────
    max_risk_pct: float = Field(default=0.01, gt=0.0, le=0.05)
    daily_loss_limit_pct: float = Field(default=0.05, gt=0.0, le=1.0)
    auto_kill_drawdown_pct: float = Field(default=0.10, gt=0.0, le=1.0)
    max_open_positions: int = Field(default=3, ge=1)

    # ── Execution ─────────────────────────────────────────────────────────────
    # Maximum deviation in points for MT5 order execution
    max_deviation_points: int = Field(default=10, ge=0)

    # Magic number for MT5 orders (identifies this bot's orders)
    magic_number: int = Field(default=20240601, ge=0)

    # ── Spread filter ─────────────────────────────────────────────────────────
    # Maximum allowed spread in pips before blocking a new trade.
    # Set to 0 to disable the spread filter entirely.
    max_spread_pips: float = Field(default=3.0, ge=0.0)

    # ── Signal cooldown ───────────────────────────────────────────────────────
    # After any approved trade, block further entries for this many bars.
    # Set to 0 to disable cooldown.
    signal_cooldown_bars: int = Field(default=3, ge=0)

    # ── Drawdown recovery ─────────────────────────────────────────────────────
    # If peak-to-trough drawdown exceeds this fraction (e.g. 0.05 = 5 %),
    # all new trades use a reduced risk multiplier until equity recovers.
    drawdown_recovery_threshold_pct: float = Field(default=0.05, gt=0.0, le=1.0)
    # Lot-size multiplier applied while in drawdown-recovery mode (0 < x ≤ 1).
    drawdown_recovery_risk_mult: float = Field(default=0.5, gt=0.0, le=1.0)

    # ── Chandelier Exit (ATR-based stop-loss) ─────────────────────────────────
    # SL = entry ± chandelier_atr_mult × ATR(chandelier_atr_period)
    # Adapts to current volatility, far superior to a fixed-pip stop.
    chandelier_atr_period: int = Field(default=14, ge=1)
    chandelier_atr_mult: float = Field(default=2.0, gt=0.0)

    # ── Trailing stop ─────────────────────────────────────────────────────────
    # When enabled, the Chandelier SL is updated each bar while a position is
    # open. The stop only moves in the favourable direction (ratchets).
    # Requires BacktestRunner; LiveRunner update is done bar-by-bar in on_bar().
    trailing_stop_enabled: bool = Field(default=True)

    # ── Backtest slippage ─────────────────────────────────────────────────────
    # Uniform random slippage applied at fill time (always adverse to the trade).
    # 0.3 pips avg ≈ realistic for liquid EUR/USD during London/NY sessions.
    # Set to 0.0 to disable (deterministic fill at next-bar open).
    slippage_pips: float = Field(default=0.3, ge=0.0)

    # Seed for the backtest PRNG (slippage draws). 0 = random seed.
    random_seed: int = Field(default=42, ge=0)

    # ── Daily circuit breaker (absolute) ──────────────────────────────────────
    # Hard-stop when intraday equity loss exceeds this amount in account currency.
    # 0.0 = disabled. Works independently of daily_loss_limit_pct.
    daily_loss_limit_usd: float = Field(default=0.0, ge=0.0)

    # ── Backtest-specific ─────────────────────────────────────────────────────
    initial_balance: float = Field(default=10000.0, gt=0.0)

    # ── Paper-specific ────────────────────────────────────────────────────────
    # Starting balance for paper trading simulation
    paper_initial_balance: float = Field(default=10000.0, gt=0.0)

    model_config = {
        "env_prefix": "RUNNER_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "frozen": True,
    }
