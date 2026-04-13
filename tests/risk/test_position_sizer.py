"""Tests for the fixed-fractional position sizer."""

from __future__ import annotations

from decimal import Decimal

import pytest

from metatrade.core.enums import OrderSide
from metatrade.risk.config import RiskConfig
from metatrade.risk.position_sizer import PositionSizer


def make_config(**kwargs) -> RiskConfig:
    defaults = dict(
        max_risk_pct=0.01,
        pip_value_per_lot=Decimal("10.0"),
        min_lot_size=Decimal("0.01"),
        max_lot_size=Decimal("10.0"),
        lot_step=Decimal("0.01"),
        risk_reward_ratio=2.0,
        pip_digits=4,
    )
    defaults.update(kwargs)
    return RiskConfig(**defaults)


class TestPositionSizer:
    def test_standard_buy_calculation(self) -> None:
        # balance=10000, risk=1%, sl=20 pips, pip_value=10 → lots=0.50
        cfg = make_config()
        sizer = PositionSizer(cfg)
        result = sizer.calculate(
            balance=Decimal("10000"),
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),  # 20 pips
            side=OrderSide.BUY,
        )
        assert result.lot_size == Decimal("0.50")
        assert result.stop_loss_pips == Decimal("20.0")
        assert result.risk_pct == pytest.approx(0.01)

    def test_sell_calculation(self) -> None:
        # SL above entry for SELL
        cfg = make_config()
        sizer = PositionSizer(cfg)
        result = sizer.calculate(
            balance=Decimal("10000"),
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.10200"),  # 20 pips above
            side=OrderSide.SELL,
        )
        assert result.lot_size == Decimal("0.50")
        assert result.stop_loss_pips == Decimal("20.0")

    def test_lot_size_rounded_down_to_step(self) -> None:
        # Exact calc gives 0.333... lots → rounded down to 0.33
        cfg = make_config(max_risk_pct=0.01)
        sizer = PositionSizer(cfg)
        result = sizer.calculate(
            balance=Decimal("10000"),
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09700"),  # 30 pips
            side=OrderSide.BUY,
        )
        # 10000 * 0.01 / (30 * 10) = 100/300 = 0.3333... → 0.33
        assert result.lot_size == Decimal("0.33")

    def test_lot_size_clamped_to_min(self) -> None:
        cfg = make_config(max_risk_pct=0.001, min_lot_size=Decimal("0.01"))
        sizer = PositionSizer(cfg)
        result = sizer.calculate(
            balance=Decimal("1000"),
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09000"),  # 100 pips
            side=OrderSide.BUY,
        )
        # 1000 * 0.001 / (100 * 10) = 1/1000 = 0.001 → clamped to 0.01
        assert result.lot_size == Decimal("0.01")

    def test_lot_size_clamped_to_max(self) -> None:
        cfg = make_config(max_risk_pct=0.10, max_lot_size=Decimal("2.0"))
        sizer = PositionSizer(cfg)
        result = sizer.calculate(
            balance=Decimal("100000"),
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09900"),  # 10 pips
            side=OrderSide.BUY,
        )
        # 100000 * 0.10 / (10 * 10) = 10000/100 = 100 lots → clamped to 2.0
        assert result.lot_size == Decimal("2.0")

    def test_take_profit_buy(self) -> None:
        cfg = make_config(risk_reward_ratio=2.0)
        sizer = PositionSizer(cfg)
        result = sizer.calculate(
            balance=Decimal("10000"),
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),  # 20 pips SL
            side=OrderSide.BUY,
        )
        # TP = entry + 2 * sl_distance = 1.10000 + 2 * 0.00200 = 1.10400
        assert result.take_profit_price == Decimal("1.10400")

    def test_take_profit_sell(self) -> None:
        cfg = make_config(risk_reward_ratio=2.0)
        sizer = PositionSizer(cfg)
        result = sizer.calculate(
            balance=Decimal("10000"),
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.10200"),  # 20 pips SL
            side=OrderSide.SELL,
        )
        # TP = entry - 2 * sl_distance = 1.10000 - 0.00400 = 1.09600
        assert result.take_profit_price == Decimal("1.09600")

    def test_no_take_profit_when_ratio_none(self) -> None:
        cfg = make_config(risk_reward_ratio=None)
        sizer = PositionSizer(cfg)
        result = sizer.calculate(
            balance=Decimal("10000"),
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),
            side=OrderSide.BUY,
        )
        assert result.take_profit_price is None

    def test_buy_sl_above_entry_raises(self) -> None:
        cfg = make_config()
        sizer = PositionSizer(cfg)
        with pytest.raises(ValueError, match="below entry"):
            sizer.calculate(
                balance=Decimal("10000"),
                entry_price=Decimal("1.10000"),
                sl_price=Decimal("1.10200"),  # above entry — invalid for BUY
                side=OrderSide.BUY,
            )

    def test_sell_sl_below_entry_raises(self) -> None:
        cfg = make_config()
        sizer = PositionSizer(cfg)
        with pytest.raises(ValueError, match="above entry"):
            sizer.calculate(
                balance=Decimal("10000"),
                entry_price=Decimal("1.10000"),
                sl_price=Decimal("1.09800"),  # below entry — invalid for SELL
                side=OrderSide.SELL,
            )

    def test_risk_amount_correct(self) -> None:
        cfg = make_config()
        sizer = PositionSizer(cfg)
        result = sizer.calculate(
            balance=Decimal("10000"),
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),
            side=OrderSide.BUY,
        )
        assert result.risk_amount == Decimal("100.00")  # 10000 * 0.01

    def test_method_is_fixed_fractional(self) -> None:
        sizer = PositionSizer()
        result = sizer.calculate(
            balance=Decimal("10000"),
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),
            side=OrderSide.BUY,
        )
        assert result.method == "fixed_fractional"

    def test_custom_risk_pct_override(self) -> None:
        cfg = make_config(max_risk_pct=0.01)
        sizer = PositionSizer(cfg)
        result = sizer.calculate(
            balance=Decimal("10000"),
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),
            side=OrderSide.BUY,
            risk_pct=0.02,  # override to 2%
        )
        assert result.risk_pct == pytest.approx(0.02)
        assert result.lot_size == Decimal("1.00")  # double the 1% case


class TestVolScaledPositionSizing:
    """Tests for volatility-scaled lot sizing."""

    def _base_cfg(self, **extra) -> RiskConfig:
        return make_config(
            vol_scaling_enabled=True,
            vol_target_atr_pips=20.0,
            vol_scaling_min_mult=0.25,
            **extra,
        )

    def _calculate(self, cfg: RiskConfig, current_atr: Decimal | None = None) -> object:
        sizer = PositionSizer(cfg)
        return sizer.calculate(
            balance=Decimal("10000"),
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),  # 20 pip SL → 0.50 lots base
            side=OrderSide.BUY,
            current_atr=current_atr,
        )

    def test_vol_scaling_disabled_ignores_atr(self) -> None:
        cfg = make_config(vol_scaling_enabled=False)
        base_result = PositionSizer(cfg).calculate(
            balance=Decimal("10000"),
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),
            side=OrderSide.BUY,
        )
        # Even with a very high ATR, disabled scaling should give the same lot
        high_atr = Decimal("0.0100")  # 100 pips — would halve lots if enabled
        scaled_result = PositionSizer(cfg).calculate(
            balance=Decimal("10000"),
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),
            side=OrderSide.BUY,
            current_atr=high_atr,
        )
        assert base_result.lot_size == scaled_result.lot_size

    def test_vol_scaling_at_target_atr_gives_full_lots(self) -> None:
        """When current ATR == target, vol_mult=1.0 → no reduction."""
        cfg = self._base_cfg()
        # target=20 pips, current=20 pips → mult=1.0
        current_atr = Decimal("0.0020")  # 20 pips
        result = self._calculate(cfg, current_atr)
        # base lots for 20-pip SL = 0.50 → with mult=1.0 still 0.50
        assert result.lot_size == Decimal("0.50")

    def test_vol_scaling_reduces_lots_in_high_vol(self) -> None:
        """ATR > target → vol_mult < 1.0 → fewer lots."""
        cfg = self._base_cfg()
        # current ATR = 40 pips (2× target) → mult=0.50
        current_atr = Decimal("0.0040")  # 40 pips
        result = self._calculate(cfg, current_atr)
        # base = 0.50, mult=0.50 → 0.25 lots
        assert result.lot_size == Decimal("0.25")

    def test_vol_scaling_clamped_to_min_mult(self) -> None:
        """Extremely high ATR should not reduce below min_mult (25%)."""
        cfg = self._base_cfg()  # min_mult=0.25 is the default in _base_cfg
        # current ATR = 200 pips → raw mult=0.10, clamped to 0.25
        current_atr = Decimal("0.0200")  # 200 pips
        result = self._calculate(cfg, current_atr)
        # base = 0.50, min_mult=0.25 → min 0.12 lots (rounds to min_lot=0.12)
        assert result.lot_size >= Decimal("0.12")
        # Must also not exceed unclamped result
        assert result.lot_size <= Decimal("0.50")

    def test_vol_scaling_no_atr_uses_base_lots(self) -> None:
        """When current_atr=None, no scaling should be applied."""
        cfg = self._base_cfg()
        result = self._calculate(cfg, current_atr=None)
        assert result.lot_size == Decimal("0.50")  # unscaled

    def test_vol_scaling_zero_atr_uses_base_lots(self) -> None:
        """When current_atr=0, skip scaling to avoid division by zero."""
        cfg = self._base_cfg()
        result = self._calculate(cfg, current_atr=Decimal("0"))
        assert result.lot_size == Decimal("0.50")

    def test_vol_scaling_low_vol_caps_at_one(self) -> None:
        """ATR below target should not increase lot size (capped at 1.0)."""
        cfg = self._base_cfg()
        # current ATR = 5 pips (4× below target) → raw mult=4.0, capped to 1.0
        current_atr = Decimal("0.0005")  # 5 pips
        result = self._calculate(cfg, current_atr)
        assert result.lot_size == Decimal("0.50")  # no increase allowed

    def test_riskconfig_vol_scaling_defaults(self) -> None:
        cfg = RiskConfig()
        assert cfg.vol_scaling_enabled is False
        assert cfg.vol_target_atr_pips == pytest.approx(20.0)
        assert cfg.vol_scaling_min_mult == pytest.approx(0.25)
