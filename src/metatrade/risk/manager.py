"""RiskManager — main entry point for all risk management decisions.

Coordinates:
    1. KillSwitchManager  — checks and manages kill switch state
    2. PreTradeChecker    — validates limits before position sizing
    3. PositionSizer      — calculates lot size if all checks pass

Usage:
    manager = RiskManager(config)
    decision = manager.evaluate(
        symbol="EURUSD", side=OrderSide.BUY,
        entry_price=Decimal("1.10000"), sl_price=Decimal("1.09800"),
        account=account_state, timestamp_utc=clock.now_utc(),
    )
    if decision.approved:
        # Send order with decision.position_size.lot_size
    else:
        log.info("trade_vetoed", reason=decision.veto.reason)
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from metatrade.core.contracts.account import AccountState
from metatrade.core.contracts.risk import RiskDecision
from metatrade.core.enums import KillSwitchLevel, OrderSide
from metatrade.core.log import get_logger
from metatrade.risk.config import RiskConfig
from metatrade.risk.kill_switch import KillSwitchManager
from metatrade.risk.position_sizer import PositionSizer
from metatrade.risk.pre_trade_checker import PreTradeChecker

log = get_logger(__name__)


class RiskManager:
    """Coordinates all risk management for a proposed trade.

    Args:
        config:      RiskConfig with all parameters.
        audit_store: Optional SQLiteAuditStore for kill switch logging.
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        audit_store: object | None = None,
    ) -> None:
        self._cfg = config or RiskConfig()
        self.kill_switch = KillSwitchManager(audit_store=audit_store)
        self._checker = PreTradeChecker(config=self._cfg, kill_switch=self.kill_switch)
        self._sizer = PositionSizer(config=self._cfg)

    def evaluate(
        self,
        symbol: str,
        side: OrderSide,
        entry_price: Decimal,
        sl_price: Decimal,
        account: AccountState,
        timestamp_utc: datetime,
        spread_pips: float | None = None,
        risk_pct: float | None = None,
        current_atr: Decimal | None = None,
        intermarket_risk_mult: Decimal | None = None,
    ) -> RiskDecision:
        """Evaluate whether a trade is permissible and size it.

        Args:
            symbol:                Forex pair (e.g. "EURUSD").
            side:                  BUY or SELL.
            entry_price:           Intended entry price.
            sl_price:              Stop-loss price.
            account:               Current account snapshot.
            timestamp_utc:         Evaluation timestamp.
            spread_pips:           Current spread in pips for spread check (optional).
            risk_pct:              Override config.max_risk_pct for this trade (optional).
            current_atr:           Current ATR in price units for vol-scaled sizing (optional).
            intermarket_risk_mult: Lot-size multiplier from IntermarketEngine (optional).
                                   Applied after normal sizing; values < 1.0 reduce the lot size.
                                   Must be > 0.  ``None`` or ``Decimal("1")`` = no effect.

        Returns:
            RiskDecision — approved with PositionSizeResult or vetoed with RiskVeto.
        """
        # 1. Pre-trade checks
        veto = self._checker.check(
            account=account,
            timestamp_utc=timestamp_utc,
            spread_pips=spread_pips,
        )
        if veto:
            return RiskDecision(
                approved=False,
                symbol=symbol,
                side=side,
                timestamp_utc=timestamp_utc,
                veto=veto,
            )

        # 2. Auto-trigger kill switch on large intraday drawdown
        self._maybe_auto_kill(account, timestamp_utc)

        # 3. Position sizing
        try:
            size = self._sizer.calculate(
                balance=account.balance,
                entry_price=entry_price,
                sl_price=sl_price,
                side=side,
                risk_pct=risk_pct,
                current_atr=current_atr,
            )
        except ValueError as exc:
            from metatrade.core.contracts.risk import RiskVeto
            return RiskDecision(
                approved=False,
                symbol=symbol,
                side=side,
                timestamp_utc=timestamp_utc,
                veto=RiskVeto(
                    reason=f"Position sizing failed: {exc}",
                    veto_code="SIZING_ERROR",
                    timestamp_utc=timestamp_utc,
                ),
            )

        # 4. Apply intermarket risk multiplier (lot-size scaling from IntermarketEngine)
        if intermarket_risk_mult is not None and intermarket_risk_mult != Decimal("1"):
            from dataclasses import replace as _replace
            scaled_lot = size.lot_size * intermarket_risk_mult
            if scaled_lot > Decimal("0"):
                size = _replace(size, lot_size=scaled_lot)
                log.info(
                    "intermarket_lot_scaled",
                    symbol=symbol,
                    multiplier=str(intermarket_risk_mult),
                    lot_size_final=str(size.lot_size),
                )

        log.info(
            "risk_approved",
            symbol=symbol,
            side=side.value,
            lot_size=str(size.lot_size),
            risk_pct=size.risk_pct,
            sl_pips=str(size.stop_loss_pips),
        )
        return RiskDecision(
            approved=True,
            symbol=symbol,
            side=side,
            timestamp_utc=timestamp_utc,
            position_size=size,
        )

    def _maybe_auto_kill(self, account: AccountState, ts: datetime) -> None:
        """Auto-activate SESSION_GATE if intraday drawdown exceeds threshold."""
        threshold_pct = self._cfg.auto_kill_drawdown_pct
        if threshold_pct <= 0 or self.kill_switch.level >= KillSwitchLevel.SESSION_GATE:
            return
        threshold_amount = account.balance * Decimal(str(threshold_pct))
        if account.drawdown_from_equity >= threshold_amount:
            self.kill_switch.activate(
                level=KillSwitchLevel.SESSION_GATE,
                reason=(
                    f"Auto-kill: intraday drawdown {account.drawdown_from_equity:.2f} "
                    f">= {threshold_amount:.2f} ({threshold_pct:.0%} of balance)"
                ),
                activated_by="risk_manager",
                ts=ts,
            )
