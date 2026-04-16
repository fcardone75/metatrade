"""Live trading runner — real broker, real fills.

The live runner processes bars from a real market data feed and executes
orders via a real broker adapter (MT5 or any IBrokerAdapter implementation).

CRITICAL SAFETY NOTES:
- The kill switch must be operational before starting
- Always run with a tested risk config from staging first
- This runner never bypasses risk checks
- Session-level kills require force_reset() to recover

The live runner is intentionally thin — most logic lives in BaseRunner.
"""

from __future__ import annotations

from datetime import datetime

from metatrade.broker.interface import IBrokerAdapter
from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.risk import RiskDecision, RiskVeto
from metatrade.core.enums import RunMode
from metatrade.core.log import get_logger
from metatrade.execution.order_manager import OrderManager
from metatrade.observability.store import TelemetryStore
from metatrade.runner.base import BaseRunner
from metatrade.runner.config import RunnerConfig
from metatrade.technical_analysis.interface import ITechnicalModule


log = get_logger(__name__)


class LiveRunner(BaseRunner):
    """Live trading orchestrator.

    Connects to a real broker adapter, processes incoming bars,
    and submits orders through the OrderManager (which enforces
    idempotency and tracks order lifecycle).

    Args:
        config:        RunnerConfig.
        modules:       List of analysis modules.
        broker:        Concrete IBrokerAdapter (e.g. MT5Adapter).
        order_manager: Optional OrderManager; created internally if None.
    """

    def __init__(
        self,
        config: RunnerConfig,
        modules: list[ITechnicalModule],
        broker: IBrokerAdapter,
        order_manager: OrderManager | None = None,
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
        self._broker = broker
        self._order_manager = order_manager or OrderManager(
            broker,
            run_mode=RunMode.LIVE,
            telemetry=telemetry,
            session_id=session_id,
        )
        self._bar_buffer: list[Bar] = []
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """Connect to the broker. Raises BrokerConnectionError on failure."""
        self._broker.connect()
        self._connected = True
        log.info("LiveRunner connected to broker.")

    def disconnect(self) -> None:
        """Gracefully disconnect from the broker."""
        try:
            self._broker.disconnect()
        finally:
            self._connected = False
        log.info("LiveRunner disconnected from broker.")

    def on_bar(self, bar: Bar) -> RiskDecision | RiskVeto | None:
        """Process one live bar through the pipeline.

        Args:
            bar: The newly closed bar from the market data feed.

        Returns:
            RiskDecision if a trade was submitted,
            RiskVeto if risk blocked the trade,
            None if consensus was HOLD or no signals.

        Raises:
            RuntimeError: If on_bar() is called before connect().

        This method is designed to never propagate exceptions to the caller.
        Any error mid-bar is logged and the method returns None, keeping the
        runner alive so open positions continue to be managed.
        """
        if not self._connected:
            raise RuntimeError(
                "LiveRunner.on_bar() called without calling connect() first."
            )

        try:
            self._bar_buffer.append(bar)
            if len(self._bar_buffer) > 500:
                self._bar_buffer = self._bar_buffer[-500:]

            # Pull current account state from broker
            account = self._broker.get_account()
            balance = account.balance
            equity = account.equity
            free_margin = account.free_margin
            open_positions = account.open_positions_count

            result = self.process_bar(
                bars=self._bar_buffer,
                account_balance=balance,
                account_equity=equity,
                free_margin=free_margin,
                timestamp_utc=bar.timestamp_utc,
                open_positions=open_positions,
                run_mode=RunMode.LIVE,
            )

            if result is not None and result.approved and result.position_size is not None:
                self._submit_live_order(result, bar.timestamp_utc)

            return result

        except Exception as exc:
            # Never kill the runner over a single bar — open positions must
            # keep being managed. Log and return None to the caller.
            try:
                log.error(
                    "on_bar_unhandled_exception",
                    symbol=bar.symbol,
                    timestamp=bar.timestamp_utc.isoformat(),
                    error=str(exc),
                    exc_info=True,
                )
            except Exception:  # noqa: BLE001
                pass  # logging itself must not kill the runner
            return None

    def _submit_live_order(
        self,
        decision: RiskDecision,
        timestamp_utc: datetime,
    ) -> None:
        """Build and submit a live order through the OrderManager.

        Args:
            decision:      Approved risk decision.
            timestamp_utc: UTC timestamp for order creation.

        This method swallows all exceptions — a failed submission must never
        kill the runner while positions may already be open.
        """
        try:
            order = self._order_manager.build_order(decision, timestamp_utc)
            self._order_manager.submit(order)
            ps = decision.position_size
            log.info(
                "live_order_submitted",
                order_id=str(order.order_id),
                side=decision.side.value if decision.side else "?",
                symbol=decision.symbol,
                lots=float(ps.lot_size) if ps else 0.0,
                sl=str(ps.stop_loss_price) if ps else None,
                tp=str(ps.take_profit_price) if ps else None,
            )
        except Exception:
            # OrderManager already logs order_submission_failed with full context.
            # Swallow here so the runner stays alive for position management.
            pass

    def reset_buffer(self) -> None:
        """Clear the internal bar buffer (e.g. after a weekend gap)."""
        self._bar_buffer.clear()
