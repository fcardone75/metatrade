"""Paper trading runner — live feed, simulated fills.

The paper runner connects to a real market data feed (MT5 or mock)
but executes trades against the in-memory PaperBroker.
It uses the same signal pipeline as the backtest runner but processes
bars in real-time as they arrive.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from metatrade.broker.paper_broker import PaperBrokerAdapter as PaperBroker
from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.risk import RiskDecision, RiskVeto
from metatrade.observability.store import TelemetryStore
from metatrade.runner.base import BaseRunner
from metatrade.runner.config import RunnerConfig
from metatrade.technical_analysis.interface import ITechnicalModule

log = logging.getLogger(__name__)


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
    ) -> None:
        super().__init__(
            config,
            modules,
            telemetry=telemetry,
            session_id=session_id,
            timeframe=timeframe,
        )
        if broker is None:
            from metatrade.broker.paper_broker import PaperBrokerAdapter
            broker = PaperBrokerAdapter(initial_balance=Decimal(str(config.paper_initial_balance)))
        self._broker = broker
        self._broker.connect()  # paper broker connect is a no-op / state flag
        self._bar_buffer: list[Bar] = []
        self._pending_order_id: str | None = None

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
