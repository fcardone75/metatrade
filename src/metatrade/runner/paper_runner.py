"""Paper trading runner — live feed, simulated fills.

The paper runner connects to a real market data feed (MT5 or mock)
but executes trades against the in-memory PaperBroker.
It uses the same signal pipeline as the backtest runner but processes
bars in real-time as they arrive.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal

from metatrade.alerting.command_handler import CommandHandler
from metatrade.alerting.telegram_alerter import TelegramAlerter
from metatrade.broker.paper_broker import PaperBrokerAdapter as PaperBroker
from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.position import Position
from metatrade.core.enums import PositionSide, RunMode
from metatrade.core.contracts.risk import RiskDecision, RiskVeto
from metatrade.ml.live_tracker import LiveAccuracyTracker
from metatrade.ml.model_watcher import ModelWatcher
from metatrade.ml.retrain_scheduler import RetrainScheduler
from metatrade.observability.store import TelemetryStore
from metatrade.runner.base import BaseRunner
from metatrade.runner.config import RunnerConfig
from metatrade.technical_analysis.interface import ITechnicalModule

log = logging.getLogger(__name__)

_PIP_SIZE = Decimal("0.0001")


class PaperRunner(BaseRunner):
    """Paper trading orchestrator backed by PaperBroker.

    Processes bars one at a time (call `on_bar()`), executes approved
    decisions against the simulated broker, and tracks simulated account state.

    Args:
        config:   RunnerConfig (uses paper_initial_balance).
        modules:  List of analysis modules.
        broker:   PaperBroker instance. Created with paper_initial_balance if None.
    """

    def __init__(
        self,
        config: RunnerConfig,
        modules: list[ITechnicalModule],
        broker: PaperBroker | None = None,
        telemetry: TelemetryStore | None = None,
        session_id: str | None = None,
        timeframe: str | None = None,
        alerter: TelegramAlerter | None = None,
        live_tracker: LiveAccuracyTracker | None = None,
        model_watcher: ModelWatcher | None = None,
        retrain_scheduler: RetrainScheduler | None = None,
        command_handler: CommandHandler | None = None,
    ) -> None:
        super().__init__(
            config,
            modules,
            telemetry=telemetry,
            session_id=session_id,
            timeframe=timeframe,
            alerter=alerter,
            live_tracker=live_tracker,
            model_watcher=model_watcher,
            retrain_scheduler=retrain_scheduler,
            command_handler=command_handler,
            run_mode=RunMode.PAPER,
        )
        if broker is None:
            from metatrade.broker.paper_broker import PaperBrokerAdapter
            broker = PaperBrokerAdapter(initial_balance=Decimal(str(config.paper_initial_balance)))
        self._broker = broker
        self._broker.connect()  # paper broker connect is a no-op / state flag
        self._bar_buffer: list[Bar] = []
        self._pending_order_id: str | None = None

        # Daily stats — reset each session start; updated on position close
        self._daily_pnl: Decimal = Decimal("0")
        self._daily_wins: int = 0
        self._daily_losses: int = 0

    @property
    def broker(self) -> PaperBroker:
        return self._broker

    def on_bar(self, bar: Bar) -> RiskDecision | RiskVeto | None:
        """Process one bar tick through the pipeline.

        Appends the bar to the internal buffer, runs the signal pipeline,
        and submits any approved decisions to the paper broker.

        Args:
            bar: The newly closed bar.

        Returns:
            RiskDecision if a trade was opened,
            RiskVeto if blocked by risk,
            None if no signal.
        """
        self._bar_buffer.append(bar)
        # Keep buffer bounded to last 500 bars
        if len(self._bar_buffer) > 500:
            self._bar_buffer = self._bar_buffer[-500:]

        # Check SL/TP on existing positions before processing the new bar
        self._check_and_close_positions(bar)

        # Flush any pending fill
        if self._pending_order_id is not None:
            ts = bar.timestamp_utc
            try:
                fill = self._broker.fill_pending(self._pending_order_id, ts)
                if self._telemetry is not None:
                    self._telemetry.record_order_event(
                        session_id=self._session_id,
                        run_mode="PAPER",
                        event_type="paper_order_filled",
                        ts=fill.timestamp_utc,
                        symbol=fill.symbol,
                        side=fill.side.value,
                        order_id=fill.order_id,
                        broker_order_id=fill.broker_order_id,
                        status="FILLED",
                        lot_size=float(fill.filled_lots),
                        price=float(fill.filled_price),
                    )
            except Exception as exc:
                log.warning("Paper fill failed for %s: %s", self._pending_order_id, exc)
            self._pending_order_id = None

        account = self._broker.get_account()
        balance = account.balance
        equity = account.equity
        free_margin = account.free_margin

        result = self.process_bar(
            bars=self._bar_buffer,
            account_balance=balance,
            account_equity=equity,
            free_margin=free_margin,
            timestamp_utc=bar.timestamp_utc,
        )

        if result is not None and result.approved and result.position_size is not None:
            order_id = self._submit_to_paper(result, bar)
            if order_id:
                self._pending_order_id = order_id

        return result

    def _check_and_close_positions(self, bar: Bar) -> None:
        """Check all open positions for SL/TP hits on this bar and fire notifications."""
        closed = self._broker.check_sl_tp(
            bar_high=bar.high,
            bar_low=bar.low,
            bar_close=bar.close,
            now_utc=bar.timestamp_utc,
        )
        for closed_pos, exit_reason, pnl in closed:
            self._on_position_closed(closed_pos, exit_reason, pnl)

    def _on_position_closed(
        self,
        pos: Position,
        exit_reason: str,
        pnl: Decimal,
    ) -> None:
        """Update stats and fire Telegram close notification."""
        self._daily_pnl += pnl
        if pnl > 0:
            self._daily_wins += 1
        elif pnl < 0:
            self._daily_losses += 1

        if self._alerter is None:
            return

        pnl_pips: float | None = None
        if pos.close_price is not None:
            raw_diff = (
                (pos.close_price - pos.entry_price)
                if pos.side == PositionSide.LONG
                else (pos.entry_price - pos.close_price)
            )
            pnl_pips = float(raw_diff / _PIP_SIZE)

        duration_bars: int | None = None
        if pos.opened_at_utc and pos.closed_at_utc:
            delta = pos.closed_at_utc - pos.opened_at_utc
            duration_bars = max(1, int(delta.total_seconds() / 60))

        try:
            self._alerter.alert_trade_closed(
                symbol=pos.symbol,
                side=pos.side.value,
                pnl=float(pnl),
                exit_reason=exit_reason,
                entry=pos.entry_price,
                exit_price=pos.close_price,
                pnl_pips=pnl_pips,
                duration_bars=duration_bars,
                balance_after=self._broker._balance,
                session_pnl=float(self._daily_pnl),
            )
        except Exception as exc:
            log.warning("paper_runner_alert_trade_closed_failed: %s", exc)

    def _format_open_positions(self) -> str:
        """Return a detailed HTML string of all open positions for Telegram."""
        positions = self._broker.get_open_positions_list()
        if not positions:
            return "📋 <b>Posizioni aperte (0)</b>\n\nNessuna posizione aperta."

        lines = [f"📋 <b>Posizioni aperte ({len(positions)})</b>"]
        now = datetime.now(UTC)
        for pos in positions:
            side_icon = "🟢" if pos.side == PositionSide.LONG else "🔴"
            sl_str = f"{float(pos.stop_loss):.5f}" if pos.stop_loss else "—"
            tp_str = f"{float(pos.take_profit):.5f}" if pos.take_profit else "—"

            # Unrealized PnL using last known quote
            pnl_str = ""
            try:
                bid, ask = self._broker.get_current_price(pos.symbol)
                current = bid if pos.side == PositionSide.SHORT else ask
                raw_diff = (
                    (current - pos.entry_price)
                    if pos.side == PositionSide.LONG
                    else (pos.entry_price - current)
                )
                pnl_pips = float(raw_diff / _PIP_SIZE)
                sign = "+" if pnl_pips >= 0 else ""
                pnl_str = f"\n  PnL:    {sign}{pnl_pips:.1f} pip"
            except Exception:
                pass

            # Duration since open
            delta = now - pos.opened_at_utc
            hours, rem = divmod(int(delta.total_seconds()), 3600)
            mins = rem // 60
            dur_str = f"{hours}h {mins}m" if hours else f"{mins}m"

            lines.append(
                f"\n{side_icon} <b>{pos.symbol}</b>  {pos.side.value.upper()}"
                f"  {pos.lot_size} lot\n"
                f"  Entry:  {float(pos.entry_price):.5f}  ({dur_str} fa)\n"
                f"  SL:     {sl_str}  |  TP: {tp_str}"
                f"{pnl_str}"
            )
        return "\n".join(lines)

    def _get_open_positions(self) -> list:
        """Return open positions for intermarket exposure accounting."""
        return self._broker.get_open_positions_list()

    def _submit_to_paper(
        self,
        decision: RiskDecision,
        bar: Bar,
    ) -> str | None:
        """Submit a RiskDecision to the paper broker.

        Args:
            decision: Approved risk decision with lot_size.
            bar:      Current bar (used for bid/ask price).

        Returns:
            Order ID if submitted successfully, None on failure.
        """
        import uuid

        from metatrade.core.contracts.order import Order, OrderType
        from metatrade.core.enums import OrderStatus

        order_id = str(uuid.uuid4())
        ps = decision.position_size
        order = Order(
            order_id=order_id,
            symbol=self._config.symbol,
            side=decision.side,
            order_type=OrderType.MARKET,
            lot_size=ps.lot_size,
            price=None,
            stop_loss=ps.stop_loss_price,
            take_profit=ps.take_profit_price,
            status=OrderStatus.PENDING,
            timestamp_utc=bar.timestamp_utc,
        )

        # Set quote so broker knows current bid/ask
        half_spread = Decimal("0.00010")
        self._broker.set_quote(
            self._config.symbol,
            bid=bar.close - half_spread,
            ask=bar.close + half_spread,
        )

        try:
            self._broker.send_order(order)
            if self._telemetry is not None:
                self._telemetry.record_order_event(
                    session_id=self._session_id,
                    run_mode="PAPER",
                    event_type="paper_order_submitted",
                    ts=bar.timestamp_utc,
                    symbol=order.symbol,
                    side=order.side.value,
                    order_id=order.order_id,
                    status="SUBMITTED",
                    lot_size=float(order.lot_size),
                )
            return order_id
        except Exception as exc:
            log.error("Paper order submission failed: %s", exc)
            return None

    def reset(self) -> None:
        """Reset the paper runner state (clears bar buffer and pending orders)."""
        self._bar_buffer.clear()
        self._pending_order_id = None
