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

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from metatrade.alerting.command_handler import CommandHandler
from metatrade.alerting.telegram_alerter import TelegramAlerter
from metatrade.broker.interface import IBrokerAdapter
from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.risk import RiskDecision, RiskVeto
from metatrade.core.enums import RunMode
from metatrade.core.log import get_logger
from metatrade.execution.order_manager import OrderManager
from metatrade.ml.live_tracker import LiveAccuracyTracker
from metatrade.ml.model_watcher import ModelWatcher
from metatrade.ml.retrain_scheduler import RetrainScheduler
from metatrade.observability.store import TelemetryStore
from metatrade.runner.base import BaseRunner
from metatrade.runner.config import RunnerConfig
from metatrade.technical_analysis.interface import ITechnicalModule

if TYPE_CHECKING:
    from metatrade.exit_engine.engine import ExitEngine


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
        exit_profile_generator: object | None = None,
        exit_profile_selector: object | None = None,
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
            exit_profile_generator=exit_profile_generator,
            exit_profile_selector=exit_profile_selector,
            alerter=alerter,
            live_tracker=live_tracker,
            model_watcher=model_watcher,
            retrain_scheduler=retrain_scheduler,
            command_handler=command_handler,
            run_mode=RunMode.LIVE,
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
        # Last-known open positions by ticket — used to detect MT5 native closures
        # (SL/TP hit server-side without going through ExitEngine).
        self._tracked_positions: dict[int, dict] = {}

        # Register live-specific commands after super().__init__()
        if self._command_handler is not None:
            self._command_handler.register("/sync", self._cmd_sync)

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

            # Detect positions closed by MT5 natively (SL/TP server-side)
            self._detect_mt5_closes()

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

    def monitor_positions_once(
        self,
        exit_engine: ExitEngine,
        symbol: str | None = None,
    ) -> list[dict]:
        """Evaluate and act on all open positions in near real-time.

        Designed to be called between bar closes (e.g., every 5 seconds).
        For each open position the method:
          1. Runs ``ExitEngine.evaluate()`` against the current market price.
          2. Updates SL on MT5 if a tighter trailing stop is suggested.
          3. Closes the position if the engine votes CLOSE_FULL.

        Args:
            exit_engine: Configured ExitEngine instance.
            symbol:      Filter to a specific symbol (None = all positions).

        Returns:
            List of action dicts for logging/display:
            {"ticket", "symbol", "action", "new_sl", "reason"}.
        """
        if not self._connected:
            return []

        try:
            positions = self._broker.get_open_positions(symbol=symbol)
        except Exception as exc:
            log.warning("monitor_positions_get_failed", error=str(exc))
            return []

        if not positions:
            return []

        actions: list[dict] = []
        now_utc = datetime.now(UTC)

        for pos in positions:
            try:
                action = self._evaluate_and_act(pos, exit_engine, now_utc)
                if action:
                    actions.append(action)
            except Exception as exc:
                log.warning(
                    "monitor_positions_eval_failed",
                    ticket=pos.get("ticket"),
                    symbol=pos.get("symbol"),
                    error=str(exc),
                )

        return actions

    def _evaluate_and_act(
        self,
        pos: dict,
        exit_engine: ExitEngine,
        now_utc: datetime,
    ) -> dict | None:
        """Run ExitEngine for one position and execute the resulting decision."""
        from metatrade.exit_engine.contracts import ExitAction, PositionContext, PositionSide

        ticket = pos["ticket"]
        symbol = pos["symbol"]
        is_buy = pos["type"] == 0
        pos_side = PositionSide.LONG if is_buy else PositionSide.SHORT

        # Get current market price
        try:
            bid, ask = self._broker.get_current_price(symbol)
            current_price = ask if is_buy else bid
        except Exception:
            return None

        entry_price = Decimal(str(pos["price_open"]))
        lot_size    = Decimal(str(pos["volume"]))
        opened_at   = pos["time_utc"]

        # Peak favorable price: use entry as lower bound (the bar buffer may not
        # have a full history for this position yet).
        if is_buy:
            peak = max(entry_price, current_price)
        else:
            peak = min(entry_price, current_price)

        # Collect bars since position opened (from buffer)
        bars_since = [
            b for b in self._bar_buffer
            if b.timestamp_utc >= opened_at and b.symbol == symbol
        ] if self._bar_buffer else []

        current_bar = bars_since[-1] if bars_since else None
        if current_bar is None:
            # No closed bar available yet — skip this tick
            return None

        ctx = PositionContext(
            position_id=str(ticket),
            symbol=symbol,
            side=pos_side,
            entry_price=entry_price,
            lot_size=lot_size,
            opened_at_utc=opened_at,
            current_price=current_price,
            current_bar=current_bar,
            bars_since_entry=bars_since,
            peak_favorable_price=peak,
            stop_loss=Decimal(str(pos["sl"])) if pos.get("sl") else None,
            take_profit=Decimal(str(pos["tp"])) if pos.get("tp") else None,
        )

        decision = exit_engine.evaluate(ctx)

        # ── Act on the decision ───────────────────────────────────────────────
        if decision.action == ExitAction.CLOSE_FULL:
            mt5_profit = float(pos.get("profit", 0.0))
            closed = self._broker.close_position_by_ticket(
                ticket=ticket,
                symbol=symbol,
                volume=float(lot_size),
                is_buy=is_buy,
            )
            log.info(
                "monitor_positions_close_full",
                ticket=ticket,
                symbol=symbol,
                score=decision.aggregate_score,
                reason=decision.explanation,
                accepted=closed,
            )
            if closed:
                self._record_live_close(
                    pos=pos,
                    exit_price=current_price,
                    exit_reason=decision.explanation,
                    profit=mt5_profit,
                )
            return {
                "ticket": ticket, "symbol": symbol,
                "action": "CLOSE_FULL", "new_sl": None,
                "reason": decision.explanation,
            }

        if decision.action == ExitAction.CLOSE_PARTIAL:
            mt5_profit_partial = float(pos.get("profit", 0.0)) * decision.close_pct
            close_lots = float(lot_size) * decision.close_pct
            closed = self._broker.close_position_by_ticket(
                ticket=ticket,
                symbol=symbol,
                volume=close_lots,
                is_buy=is_buy,
            )
            log.info(
                "monitor_positions_close_partial",
                ticket=ticket,
                symbol=symbol,
                close_pct=decision.close_pct,
                score=decision.aggregate_score,
                accepted=closed,
            )
            if closed:
                self._record_live_close(
                    pos=pos,
                    exit_price=current_price,
                    exit_reason=f"partial ({decision.close_pct:.0%})",
                    profit=mt5_profit_partial,
                )
            return {
                "ticket": ticket, "symbol": symbol,
                "action": "CLOSE_PARTIAL", "new_sl": None,
                "reason": decision.explanation,
            }

        # HOLD — still update SL if trailing stop moved it
        if decision.new_stop_loss is not None:
            new_sl = float(decision.new_stop_loss)
            # Skip the broker call when SL hasn't actually changed — MT5 returns
            # retcode=10025 (NO_CHANGES) and we'd spam warnings every tick.
            current_sl = float(pos["sl"]) if pos.get("sl") else None
            if current_sl is not None and abs(new_sl - current_sl) < 1e-6:
                return None
            tp_val = float(pos["tp"]) if pos.get("tp") else None
            modified = self._broker.modify_sl_tp(
                ticket=ticket,
                symbol=symbol,
                new_sl=new_sl,
                new_tp=tp_val,
            )
            if modified:
                log.info(
                    "monitor_positions_sl_updated",
                    ticket=ticket,
                    symbol=symbol,
                    new_sl=new_sl,
                )
                return {
                    "ticket": ticket, "symbol": symbol,
                    "action": "SL_UPDATE", "new_sl": new_sl,
                    "reason": decision.explanation,
                }

        return None

    def _cmd_sync(self, _args: str) -> str:
        """Telegram command: rebuild daily stats from MT5 deal history."""
        result = self._sync_daily_from_history()
        if result is None:
            return "⚠️ Impossibile leggere la history MT5 — broker disconnesso?"
        wins, losses, pnl = result
        n = wins + losses
        wr = f"{wins / n:.1%}" if n > 0 else "n/a"
        sign = "+" if pnl >= 0 else ""
        return (
            f"🔄 <b>Sync completato — oggi</b>\n"
            f"  Chiusi:   {n}  (W:{wins} / L:{losses})\n"
            f"  Win rate: {wr}\n"
            f"  PnL:      {sign}{pnl:.2f} USD"
        )

    def _sync_daily_from_history(self) -> tuple[int, int, float] | None:
        """Read MT5 OUT deals from midnight UTC to now and reset daily counters.

        Returns (wins, losses, total_pnl) or None on broker error.
        This is idempotent — calling it multiple times just re-reads history.
        """
        try:
            from datetime import date
            today_start = datetime.combine(date.today(), datetime.min.time()).replace(tzinfo=UTC)
            deals = self._broker.get_deals_since(today_start, datetime.now(UTC))
        except Exception as exc:
            log.warning("sync_daily_history_failed", error=str(exc))
            return None

        wins = 0
        losses = 0
        total_pnl = 0.0
        for d in deals:
            if d.get("entry") != 1:  # only OUT (closing) deals
                continue
            pnl = d["profit"] + d["commission"] + d.get("swap", 0.0)
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1

        self._daily_pnl = Decimal(str(round(total_pnl, 2)))
        self._daily_wins = wins
        self._daily_losses = losses
        log.info(
            "daily_stats_synced_from_history",
            wins=wins, losses=losses, total_pnl=round(total_pnl, 2),
        )
        return (wins, losses, total_pnl)

    def _detect_mt5_closes(self) -> None:
        """Detect positions closed by MT5 (SL/TP server-side) and fire notifications.

        Compares the current MT5 position list against the previously tracked set.
        Any ticket that disappeared was closed outside of ExitEngine — query the
        deal history for the realized PnL and fire the standard close notification.
        """
        try:
            current = self._broker.get_open_positions()
        except Exception:
            return

        current_by_ticket = {p["ticket"]: p for p in current}

        for ticket, pos_data in list(self._tracked_positions.items()):
            if ticket not in current_by_ticket:
                # Position gone — find the OUT deal for PnL and exit price
                profit = 0.0
                exit_price: Decimal | None = None
                exit_reason = "sl/tp"
                try:
                    deals = self._broker.get_deals_for_position(ticket)
                    for d in deals:
                        if d.get("entry") == 1:  # DEAL_ENTRY_OUT
                            profit += d["profit"] + d["commission"] + d.get("swap", 0)
                            if d.get("price"):
                                exit_price = Decimal(str(d["price"]))
                except Exception as exc:
                    log.warning("live_detect_mt5_close_deal_failed", ticket=ticket, error=str(exc))

                if exit_price is None:
                    exit_price = Decimal(str(pos_data.get("price_open", 0)))

                self._record_live_close(
                    pos=pos_data,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    profit=profit,
                )

        self._tracked_positions = current_by_ticket

    def _record_live_close(
        self,
        pos: dict,
        exit_price: Decimal,
        exit_reason: str,
        profit: float,
    ) -> None:
        """Update daily stats and fire close notification after a live position closes."""
        self._daily_pnl += Decimal(str(round(profit, 2)))
        if profit > 0:
            self._daily_wins += 1
        elif profit < 0:
            self._daily_losses += 1

        if self._alerter is None:
            return

        is_buy = pos.get("type") == 0
        side = "BUY" if is_buy else "SELL"
        entry = Decimal(str(pos.get("price_open", 0)))
        opened_at = pos.get("time_utc")
        duration_bars: int | None = None
        if opened_at is not None:
            delta = datetime.now(UTC) - opened_at
            duration_bars = max(1, int(delta.total_seconds() / 60))

        try:
            self._alerter.alert_trade_closed(
                symbol=pos.get("symbol", self._symbol),
                side=side,
                pnl=profit,
                exit_reason=exit_reason,
                entry=entry,
                exit_price=exit_price,
                pnl_pips=None,
                duration_bars=duration_bars,
                session_pnl=float(self._daily_pnl),
            )
        except Exception as exc:
            log.warning("live_runner_alert_trade_closed_failed", error=str(exc))

    def _format_open_positions(self) -> str:
        """Return Telegram-ready HTML with live MT5 position details."""
        try:
            positions = self._broker.get_open_positions()
        except Exception:
            positions = []

        if not positions:
            return "📋 <b>Posizioni aperte (0)</b>\n\nNessuna posizione aperta."

        lines = [f"📋 <b>Posizioni aperte ({len(positions)})</b>"]
        now = datetime.now(UTC)
        for pos in positions:
            is_buy = pos.get("type") == 0
            side_icon = "🟢" if is_buy else "🔴"
            side_str = "BUY" if is_buy else "SELL"
            sl_val = pos.get("sl", 0.0)
            tp_val = pos.get("tp", 0.0)
            sl_str = f"{sl_val:.5f}" if sl_val else "—"
            tp_str = f"{tp_val:.5f}" if tp_val else "—"
            profit = pos.get("profit", 0.0)
            profit_sign = "+" if profit >= 0 else ""

            opened_at = pos.get("time_utc")
            dur_str = ""
            if opened_at is not None:
                delta = now - opened_at
                hours, rem = divmod(int(delta.total_seconds()), 3600)
                mins = rem // 60
                dur_str = f"  ({hours}h {mins}m fa)" if hours else f"  ({mins}m fa)"

            lines.append(
                f"\n{side_icon} <b>{pos.get('symbol', '?')}</b>  {side_str}"
                f"  {pos.get('volume', '?')} lot\n"
                f"  Entry:  {pos.get('price_open', 0):.5f}{dur_str}\n"
                f"  SL:     {sl_str}  |  TP: {tp_str}\n"
                f"  PnL:    {profit_sign}{profit:.2f} USD"
            )
        return "\n".join(lines)
