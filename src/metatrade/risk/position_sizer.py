"""Fixed-fractional position sizer with optional volatility scaling.

Base formula:
    lot_size = (balance * risk_pct) / (sl_pips * pip_value_per_lot)

Volatility scaling (optional, enabled via RiskConfig.vol_scaling_enabled):
    vol_mult = clamp(target_atr_pips / current_atr_pips,
                     vol_scaling_min_mult, 1.0)
    effective_lots = base_lots × vol_mult

During high-volatility regimes (ATR >> target), vol_mult < 1 → smaller positions.
During normal or low-volatility periods, vol_mult = 1.0 → standard sizing.
This prevents catastrophic over-sizing during news spikes.

Example (EURUSD, USD account, 1% risk):
    balance         = 10,000 USD
    risk_pct        = 0.01 (1%)
    sl_pips         = 20 pips
    pip_value       = 10 USD/lot

    lot_size = (10000 * 0.01) / (20 * 10) = 100 / 200 = 0.50 lots

The calculated lot is rounded down to the nearest lot_step and clamped to
[min_lot_size, max_lot_size].
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

from metatrade.core.contracts.risk import PositionSizeResult
from metatrade.core.enums import OrderSide
from metatrade.risk.config import RiskConfig


class PositionSizer:
    """Fixed-fractional position sizer.

    Args:
        config: RiskConfig — uses max_risk_pct, pip_value_per_lot, lot_step,
                             min/max_lot_size, risk_reward_ratio, pip_digits.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self._cfg = config or RiskConfig()

    def calculate(
        self,
        balance: Decimal,
        entry_price: Decimal,
        sl_price: Decimal,
        side: OrderSide,
        risk_pct: float | None = None,
        current_atr: Decimal | None = None,
    ) -> PositionSizeResult:
        """Calculate the lot size for a trade.

        Args:
            balance:     Account balance in account currency.
            entry_price: Intended entry price (market or limit).
            sl_price:    Stop-loss price.
            side:        BUY or SELL.
            risk_pct:    Override config.max_risk_pct for this trade.
            current_atr: Current ATR value in price units (optional).
                         When provided and vol_scaling_enabled=True, the lot
                         size is scaled down in high-volatility regimes.

        Returns:
            PositionSizeResult with lot_size, prices, and risk metrics.

        Raises:
            ValueError: If inputs are inconsistent (e.g. SL on wrong side of entry).
        """
        rpc = Decimal(str(risk_pct if risk_pct is not None else self._cfg.max_risk_pct))

        # Validate SL is on the correct side of entry
        if side == OrderSide.BUY and sl_price >= entry_price:
            raise ValueError(
                f"BUY stop-loss ({sl_price}) must be below entry ({entry_price})"
            )
        if side == OrderSide.SELL and sl_price <= entry_price:
            raise ValueError(
                f"SELL stop-loss ({sl_price}) must be above entry ({entry_price})"
            )

        # SL distance in pips
        pip_size = Decimal("10") ** -self._cfg.pip_digits
        sl_distance = abs(entry_price - sl_price)
        sl_pips = (sl_distance / pip_size).quantize(Decimal("0.1"))

        if sl_pips <= 0:
            raise ValueError(f"Computed SL pips = {sl_pips} — entry and SL cannot be equal")

        # Core sizing formula
        risk_amount = balance * rpc
        raw_lots = risk_amount / (sl_pips * self._cfg.pip_value_per_lot)

        # Volatility-scaled position sizing
        if (
            self._cfg.vol_scaling_enabled
            and current_atr is not None
            and current_atr > Decimal("0")
        ):
            pip_size = Decimal("10") ** -self._cfg.pip_digits
            current_atr_pips = float(current_atr / pip_size)
            if current_atr_pips > 0:
                vol_mult = min(1.0, self._cfg.vol_target_atr_pips / current_atr_pips)
                vol_mult = max(self._cfg.vol_scaling_min_mult, vol_mult)
                raw_lots = raw_lots * Decimal(str(round(vol_mult, 6)))

        # Round down to lot_step then clamp
        lot_size = (raw_lots / self._cfg.lot_step).to_integral_value(
            rounding=ROUND_DOWN
        ) * self._cfg.lot_step
        lot_size = max(self._cfg.min_lot_size, min(lot_size, self._cfg.max_lot_size))

        # Take-profit price
        tp_price: Decimal | None = None
        if self._cfg.risk_reward_ratio is not None:
            tp_distance = sl_distance * Decimal(str(self._cfg.risk_reward_ratio))
            if side == OrderSide.BUY:
                tp_price = entry_price + tp_distance
            else:
                tp_price = entry_price - tp_distance

        return PositionSizeResult(
            lot_size=lot_size,
            risk_amount=risk_amount.quantize(Decimal("0.01")),
            risk_pct=float(rpc),
            stop_loss_price=sl_price,
            stop_loss_pips=sl_pips,
            take_profit_price=tp_price,
            pip_value=self._cfg.pip_value_per_lot,
            method="fixed_fractional",
        )
