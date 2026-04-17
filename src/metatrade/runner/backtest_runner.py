"""Backtest runner — replays historical OHLCV data through the signal pipeline.

The backtest runner feeds bars from a pre-loaded historical dataset bar-by-bar
through the analysis pipeline, tracks simulated PnL, and produces a
BacktestResult with statistics.

Design:
- No broker connection required
- Deterministic replay: same data always produces same results (fixed random_seed)
- Simulated fills at next-bar open price (no look-ahead)
- Realistic slippage: uniform random, always adverse to the trade direction
- Commission and spread modelled via config
- Trailing stop (Chandelier): SL ratchets in favourable direction each bar
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from decimal import Decimal

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.risk import RiskDecision
from metatrade.core.enums import OrderSide
from metatrade.runner.base import BaseRunner, RunStats
from metatrade.runner.config import RunnerConfig
from metatrade.technical_analysis.interface import ITechnicalModule

log = logging.getLogger(__name__)

_PIP_SIZE = Decimal("0.0001")   # 4-digit pip (EURUSD/GBPUSD/etc.)


@dataclass
class SimulatedTrade:
    """One completed simulated trade in the backtest."""

    bar_index: int
    side: OrderSide
    entry_price: Decimal
    exit_price: Decimal
    lot_size: Decimal
    pnl: Decimal                # Profit/loss in account currency
    exit_reason: str            # "sl", "tp", "trailing_sl", "end_of_data"
    slippage_pips: float = 0.0  # actual slippage applied at entry


@dataclass
class BacktestResult:
    """Aggregated backtest statistics."""

    stats: RunStats
    initial_balance: Decimal
    final_balance: Decimal
    trades: list[SimulatedTrade] = field(default_factory=list)

    @property
    def total_pnl(self) -> Decimal:
        return sum(t.pnl for t in self.trades)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def n_winning(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def n_losing(self) -> int:
        return sum(1 for t in self.trades if t.pnl < 0)

    @property
    def win_rate(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return self.n_winning / self.n_trades

    @property
    def return_pct(self) -> float:
        if self.initial_balance == 0:
            return 0.0
        return float(self.total_pnl / self.initial_balance)

    @property
    def avg_slippage_pips(self) -> float:
        """Mean slippage across all trades (for diagnostic output)."""
        if not self.trades:
            return 0.0
        return sum(t.slippage_pips for t in self.trades) / len(self.trades)


class BacktestRunner(BaseRunner):
    """Runs a backtest over a fixed list of historical bars.

    Args:
        config:             RunnerConfig (uses initial_balance, slippage_pips,
                            random_seed, trailing_stop_enabled).
        modules:            List of ITechnicalModule instances.
        bars:               Full historical bar series to replay.
        pip_value:          P&L per lot per pip (default 10.0 USD).
        commission_per_lot: Round-trip commission per standard lot (default 7.0).
    """

    def __init__(
        self,
        config: RunnerConfig,
        modules: list[ITechnicalModule],
        bars: list[Bar],
        pip_value: float = 10.0,
        commission_per_lot: float = 7.0,
    ) -> None:
        super().__init__(config, modules)
        self._bars = bars
        self._pip_value = Decimal(str(pip_value))
        self._commission = Decimal(str(commission_per_lot))

        # Seeded RNG for reproducible slippage simulation
        seed = config.random_seed if config.random_seed > 0 else None
        self._rng = random.Random(seed)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        """Replay bars through the pipeline and return aggregated statistics.

        Returns:
            BacktestResult with per-trade PnL and summary statistics.
        """
        initial_balance = Decimal(str(self._config.initial_balance))
        balance = initial_balance
        trades: list[SimulatedTrade] = []
        open_trade: SimulatedTrade | None = None
        current_decision: RiskDecision | None = None
        current_sl: Decimal = Decimal("0")   # tracks the live (possibly trailed) SL

        n = len(self._bars)
        for i in range(n):
            bar = self._bars[i]
            timestamp = bar.timestamp_utc
            window = self._bars[max(0, i - 200) : i + 1]

            # ── Step 1: update trailing SL and check exit ─────────────────────
            if open_trade is not None and current_decision is not None:
                # Ratchet trailing stop when enabled
                if self._config.trailing_stop_enabled:
                    new_sl = self._estimate_sl(window, open_trade.entry_price, open_trade.side)
                    current_sl = self._ratchet_sl(open_trade.side, current_sl, new_sl)

                closed, exit_price, exit_reason = self._check_exit_sl_tp(
                    trade=open_trade,
                    decision=current_decision,
                    current_sl=current_sl,
                    bar=bar,
                )
                if closed:
                    pnl = self._compute_pnl(
                        open_trade.side,
                        open_trade.entry_price,
                        exit_price,
                        open_trade.lot_size,
                    )
                    trades.append(SimulatedTrade(
                        bar_index=i,
                        side=open_trade.side,
                        entry_price=open_trade.entry_price,
                        exit_price=exit_price,
                        lot_size=open_trade.lot_size,
                        pnl=pnl,
                        exit_reason=exit_reason,
                        slippage_pips=open_trade.slippage_pips,
                    ))
                    balance += pnl
                    open_trade = None
                    current_decision = None
                    current_sl = Decimal("0")

            # ── Step 2: look for new entry ────────────────────────────────────
            if open_trade is None:
                result = self.process_bar(
                    bars=window,
                    account_balance=balance,
                    account_equity=balance,
                    free_margin=balance,
                    timestamp_utc=timestamp,
                    open_positions=0,
                )

                if result is not None and result.approved and result.position_size is not None:
                    # Fill at next bar's open (no look-ahead)
                    fill_bar_idx = i + 1
                    base_price = self._bars[fill_bar_idx].open if fill_bar_idx < n else bar.close

                    # Apply realistic slippage (always adverse)
                    slip_pips = self._rng.uniform(0.0, self._config.slippage_pips)
                    slip_price = Decimal(str(round(slip_pips * float(_PIP_SIZE), 6)))
                    if result.side == OrderSide.BUY:
                        fill_price = base_price + slip_price   # buyer pays more
                    else:
                        fill_price = base_price - slip_price   # seller receives less

                    lot_size = result.position_size.lot_size
                    balance -= self._commission * lot_size

                    # Initialise trailing SL at entry SL
                    current_sl = result.position_size.stop_loss_price

                    open_trade = SimulatedTrade(
                        bar_index=i,
                        side=result.side,
                        entry_price=fill_price,
                        exit_price=fill_price,      # placeholder until close
                        lot_size=lot_size,
                        pnl=Decimal("0"),
                        exit_reason="open",
                        slippage_pips=round(slip_pips, 3),
                    )
                    current_decision = result

        # ── Step 3: close remaining position at end of data ───────────────────
        if open_trade is not None:
            exit_price = self._bars[-1].close
            pnl = self._compute_pnl(
                open_trade.side,
                open_trade.entry_price,
                exit_price,
                open_trade.lot_size,
            )
            trades.append(SimulatedTrade(
                bar_index=n - 1,
                side=open_trade.side,
                entry_price=open_trade.entry_price,
                exit_price=exit_price,
                lot_size=open_trade.lot_size,
                pnl=pnl,
                exit_reason="end_of_data",
                slippage_pips=open_trade.slippage_pips,
            ))
            balance += pnl

        return BacktestResult(
            stats=self._stats,
            initial_balance=initial_balance,
            final_balance=balance,
            trades=trades,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ratchet_sl(
        side: OrderSide,
        current_sl: Decimal,
        new_sl: Decimal,
    ) -> Decimal:
        """Only update the trailing SL if it moves in the favourable direction."""
        if side == OrderSide.BUY:
            return new_sl if new_sl > current_sl else current_sl
        else:  # SELL
            return new_sl if new_sl < current_sl else current_sl

    def _check_exit_sl_tp(
        self,
        trade: SimulatedTrade,
        decision: RiskDecision,
        current_sl: Decimal,
        bar: Bar,
    ) -> tuple[bool, Decimal, str]:
        """Check if current bar triggers the live SL or original TP.

        The trailing stop uses ``current_sl`` (which may have been ratcheted);
        the take-profit always uses the original level from the decision.

        Returns:
            (should_close, exit_price, reason)
        """
        if decision.position_size is None:
            return False, Decimal("0"), ""

        tp = decision.position_size.take_profit_price

        if trade.side == OrderSide.BUY:
            if bar.low <= current_sl:
                reason = "trailing_sl" if self._config.trailing_stop_enabled else "sl"
                return True, current_sl, reason
            if tp is not None and bar.high >= tp:
                return True, tp, "tp"
        else:  # SELL
            if bar.high >= current_sl:
                reason = "trailing_sl" if self._config.trailing_stop_enabled else "sl"
                return True, current_sl, reason
            if tp is not None and bar.low <= tp:
                return True, tp, "tp"

        return False, Decimal("0"), ""

    def _compute_pnl(
        self,
        side: OrderSide,
        entry: Decimal,
        exit_price: Decimal,
        lot_size: Decimal,
    ) -> Decimal:
        """Compute PnL in account currency.

        PnL = pip_distance × lot_size × pip_value_per_lot
        Uses 4-digit pip (0.0001) — correct for EUR/USD, GBP/USD, etc.
        """
        pip_distance = (exit_price - entry) if side == OrderSide.BUY else (entry - exit_price)
        pip_count = pip_distance / _PIP_SIZE
        return pip_count * lot_size * self._pip_value
