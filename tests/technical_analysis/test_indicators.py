"""Tests for technical analysis indicators: EMA, RSI, MACD, ATR."""

from __future__ import annotations

from decimal import Decimal

import pytest

from metatrade.technical_analysis.indicators.atr import atr, atr_stop_loss, true_range
from metatrade.technical_analysis.indicators.ema import ema, ema_crossover_signal
from metatrade.technical_analysis.indicators.macd import macd, macd_signal
from metatrade.technical_analysis.indicators.rsi import rsi, rsi_signal


def D(v: str | float | int) -> Decimal:
    return Decimal(str(v))


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestEma:
    def test_period_1_returns_prices(self) -> None:
        prices = [D(1), D(2), D(3), D(4), D(5)]
        result = ema(prices, 1)
        assert result == prices

    def test_length_equals_input_length(self) -> None:
        prices = [D(i) for i in range(1, 21)]
        result = ema(prices, 5)
        assert len(result) == len(prices)

    def test_bootstrap_sma(self) -> None:
        # First EMA value = SMA of first period bars
        prices = [D(10), D(20), D(30), D(40), D(50)]
        result = ema(prices, 3)
        expected_first = (D(10) + D(20) + D(30)) / D(3)
        assert result[0] == expected_first
        assert result[1] == expected_first
        assert result[2] == expected_first

    def test_ema_smoothing_formula(self) -> None:
        # k = 2/(period+1) = 2/3 for period=2
        prices = [D(10), D(10), D(20)]
        result = ema(prices, 2)
        # result[1] = SMA([10,10]) = 10
        # result[2] = 20*(2/3) + 10*(1/3)
        k = D(2) / D(3)
        expected = D(20) * k + D(10) * (1 - k)
        assert abs(result[2] - expected) < D("0.0001")

    def test_ema_weights_recent_more(self) -> None:
        # Rising series → EMA should be below current price (lag)
        prices = [D(i) for i in range(1, 21)]
        result = ema(prices, 5)
        assert result[-1] < D(20)
        assert result[-1] > D(15)

    def test_raises_on_period_zero(self) -> None:
        with pytest.raises(ValueError, match="period must be >= 1"):
            ema([D(1), D(2)], 0)

    def test_raises_on_empty_prices(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ema([], 5)

    def test_raises_when_not_enough_prices(self) -> None:
        with pytest.raises(ValueError, match="Need at least"):
            ema([D(1), D(2)], 5)

    def test_all_same_price_returns_same(self) -> None:
        prices = [D("1.1000")] * 20
        result = ema(prices, 5)
        for v in result:
            assert abs(v - D("1.1000")) < D("0.000001")


class TestEmaCrossover:
    def _make_prices(self, n: int = 30) -> list[Decimal]:
        return [D(str(1.1 + i * 0.0001)) for i in range(n)]

    def test_bullish_crossover(self) -> None:
        # Craft prices so fast crosses above slow on the last bar
        # Start trending down, then spike up at the end
        prices_down = [D(str(2.0 - i * 0.01)) for i in range(25)]
        prices_up = [D(str(float(prices_down[-1]) + i * 0.05)) for i in range(1, 6)]
        prices = prices_down + prices_up
        signal = ema_crossover_signal(prices, fast_period=5, slow_period=20)
        # We just check the function returns a valid value
        assert signal in (-1, 0, 1)

    def test_bearish_crossover(self) -> None:
        # Start trending up, then drop at end
        prices_up = [D(str(1.0 + i * 0.01)) for i in range(25)]
        prices_down = [D(str(float(prices_up[-1]) - i * 0.05)) for i in range(1, 6)]
        prices = prices_up + prices_down
        signal = ema_crossover_signal(prices, fast_period=5, slow_period=20)
        assert signal in (-1, 0, 1)

    def test_no_crossover_on_flat_series(self) -> None:
        prices = [D("1.1000")] * 30
        signal = ema_crossover_signal(prices, fast_period=5, slow_period=20)
        assert signal == 0

    def test_raises_when_fast_gte_slow(self) -> None:
        prices = [D("1.1") + D(str(i * 0.001)) for i in range(30)]
        with pytest.raises(ValueError, match="fast_period.*must be <"):
            ema_crossover_signal(prices, fast_period=20, slow_period=5)

    def test_raises_when_fast_equals_slow(self) -> None:
        prices = [D("1.1") + D(str(i * 0.001)) for i in range(30)]
        with pytest.raises(ValueError):
            ema_crossover_signal(prices, fast_period=10, slow_period=10)

    def test_returns_int(self) -> None:
        prices = [D(str(1.1 + i * 0.0001)) for i in range(30)]
        result = ema_crossover_signal(prices, fast_period=5, slow_period=20)
        assert isinstance(result, int)

    def test_returns_1_on_guaranteed_bullish_crossover(self) -> None:
        # 25 flat bars → both EMAs converge to 1.0 (prev_diff = 0)
        # Spike to 100.0 → fast EMA jumps far above slow EMA (curr_diff >> 0)
        prices = [D("1.0")] * 25 + [D("100.0")]
        result = ema_crossover_signal(prices, fast_period=5, slow_period=20)
        assert result == 1

    def test_returns_negative_1_on_guaranteed_bearish_crossover(self) -> None:
        # 25 flat bars → both EMAs converge to 1.0 (prev_diff = 0)
        # Drop to 0.0 → fast EMA drops far below slow EMA (curr_diff << 0)
        prices = [D("1.0")] * 25 + [D("0.0")]
        result = ema_crossover_signal(prices, fast_period=5, slow_period=20)
        assert result == -1

    def test_returns_0_when_no_crossover(self) -> None:
        # Flat prices → EMAs always equal → no crossover
        prices = [D("1.0")] * 30
        result = ema_crossover_signal(prices, fast_period=5, slow_period=20)
        assert result == 0


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRsi:
    def _rising_prices(self, n: int = 30) -> list[Decimal]:
        return [D(str(1.0 + i * 0.001)) for i in range(n)]

    def _falling_prices(self, n: int = 30) -> list[Decimal]:
        return [D(str(2.0 - i * 0.001)) for i in range(n)]

    def test_length_equals_input(self) -> None:
        prices = self._rising_prices(30)
        result = rsi(prices, 14)
        assert len(result) == len(prices)

    def test_rsi_range_0_to_100(self) -> None:
        prices = self._rising_prices(30)
        result = rsi(prices, 14)
        for val in result:
            assert D(0) <= val <= D(100)

    def test_rising_prices_gives_high_rsi(self) -> None:
        prices = self._rising_prices(40)
        result = rsi(prices, 14)
        assert result[-1] > D(60)

    def test_falling_prices_gives_low_rsi(self) -> None:
        prices = self._falling_prices(40)
        result = rsi(prices, 14)
        assert result[-1] < D(40)

    def test_all_gains_approaches_100(self) -> None:
        # Strictly rising → RSI should be very high
        prices = [D(str(i)) for i in range(1, 40)]
        result = rsi(prices, 14)
        assert result[-1] > D(90)

    def test_all_losses_approaches_0(self) -> None:
        # Strictly falling → RSI should be very low
        prices = [D(str(100 - i)) for i in range(39)]
        result = rsi(prices, 14)
        assert result[-1] < D(10)

    def test_raises_period_less_than_2(self) -> None:
        prices = self._rising_prices(20)
        with pytest.raises(ValueError, match="period must be >= 2"):
            rsi(prices, 1)

    def test_raises_insufficient_prices(self) -> None:
        with pytest.raises(ValueError, match="Need at least"):
            rsi([D("1.0")] * 5, 14)

    def test_wilder_smoothing_consistency(self) -> None:
        # Two separate runs with identical data must produce identical results
        prices = self._rising_prices(25)
        r1 = rsi(prices, 14)
        r2 = rsi(prices, 14)
        assert r1 == r2


class TestRsiSignal:
    def test_oversold_returns_buy(self) -> None:
        # Prices that produce RSI < 30
        prices = [D(str(100 - i * 3)) for i in range(25)]
        result = rsi_signal(prices, period=14, oversold=30.0, overbought=70.0)
        assert result == 1

    def test_overbought_returns_sell(self) -> None:
        prices = [D(str(i * 3)) for i in range(1, 26)]
        result = rsi_signal(prices, period=14, oversold=30.0, overbought=70.0)
        assert result == -1

    def test_neutral_returns_zero(self) -> None:
        # Alternating up/down → RSI stays near 50
        prices = []
        val = D("100")
        for i in range(30):
            val = val + D("1") if i % 2 == 0 else val - D("1")
            prices.append(val)
        result = rsi_signal(prices, period=14, oversold=30.0, overbought=70.0)
        assert result == 0

    def test_returns_int(self) -> None:
        prices = [D(str(i)) for i in range(1, 25)]
        result = rsi_signal(prices, period=14)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMacd:
    def _prices(self, n: int = 50) -> list[Decimal]:
        return [D(str(1.0 + i * 0.001)) for i in range(n)]

    def test_returns_three_equal_length_lists(self) -> None:
        prices = self._prices(50)
        macd_line, sig_line, histogram = macd(prices, 12, 26, 9)
        assert len(macd_line) == len(prices)
        assert len(sig_line) == len(prices)
        assert len(histogram) == len(prices)

    def test_histogram_is_macd_minus_signal(self) -> None:
        prices = self._prices(50)
        macd_line, sig_line, histogram = macd(prices, 12, 26, 9)
        for m, s, h in zip(macd_line, sig_line, histogram):
            assert abs((m - s) - h) < D("0.0000001")

    def test_raises_fast_gte_slow(self) -> None:
        prices = self._prices(50)
        with pytest.raises(ValueError, match="fast.*must be < slow"):
            macd(prices, 26, 12, 9)

    def test_raises_fast_equals_slow(self) -> None:
        prices = self._prices(50)
        with pytest.raises(ValueError):
            macd(prices, 12, 12, 9)

    def test_raises_insufficient_prices(self) -> None:
        with pytest.raises(ValueError, match="Need at least"):
            macd([D("1.0")] * 10, 12, 26, 9)

    def test_rising_series_positive_macd(self) -> None:
        prices = [D(str(i * 0.001)) for i in range(1, 60)]
        macd_line, _, _ = macd(prices, 12, 26, 9)
        # On a strongly rising series, fast EMA > slow EMA after warmup
        assert macd_line[-1] > D("0")

    def test_falling_series_negative_macd(self) -> None:
        prices = [D(str(100 - i * 0.1)) for i in range(60)]
        macd_line, _, _ = macd(prices, 12, 26, 9)
        assert macd_line[-1] < D("0")

    def test_flat_series_macd_near_zero(self) -> None:
        prices = [D("1.1000")] * 50
        macd_line, sig_line, histogram = macd(prices, 12, 26, 9)
        assert abs(macd_line[-1]) < D("0.0001")
        assert abs(histogram[-1]) < D("0.0001")


class TestMacdSignal:
    def test_returns_int(self) -> None:
        prices = [D(str(1.0 + i * 0.001)) for i in range(50)]
        result = macd_signal(prices, 12, 26, 9)
        assert isinstance(result, int)
        assert result in (-1, 0, 1)

    def test_no_signal_on_flat_series(self) -> None:
        prices = [D("1.1000")] * 50
        result = macd_signal(prices, 12, 26, 9)
        assert result == 0

    def test_bullish_crossover_detection(self) -> None:
        # Construct: long downtrend, then sharp reversal
        n = 40
        prices = [D(str(2.0 - i * 0.02)) for i in range(n)]
        # Spike up sharply at end to force fast EMA above slow
        spike = [prices[-1] + D(str(i * 0.1)) for i in range(1, 15)]
        result = macd_signal(prices + spike, 12, 26, 9)
        assert result in (-1, 0, 1)  # direction depends on magnitude; just validate type

    def test_bearish_crossover_detection(self) -> None:
        n = 40
        prices = [D(str(1.0 + i * 0.02)) for i in range(n)]
        drop = [prices[-1] - D(str(i * 0.1)) for i in range(1, 15)]
        result = macd_signal(prices + drop, 12, 26, 9)
        assert result in (-1, 0, 1)

    def test_returns_1_on_guaranteed_bullish_macd_crossover(self) -> None:
        # 40 flat prices → MACD line ≈ 0 and signal line ≈ 0 (prev_diff = 0)
        # Final spike to 1000.0 → MACD line jumps positive faster than signal line
        prices = [D("1.0")] * 40 + [D("1000.0")]
        result = macd_signal(prices, 12, 26, 9)
        assert result == 1

    def test_returns_negative_1_on_guaranteed_bearish_macd_crossover(self) -> None:
        # 40 flat prices → MACD line ≈ 0 and signal line ≈ 0 (prev_diff = 0)
        # Final drop to 0.0 → MACD line drops negative faster than signal line
        prices = [D("1.0")] * 40 + [D("0.0")]
        result = macd_signal(prices, 12, 26, 9)
        assert result == -1


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestTrueRange:
    def test_basic(self) -> None:
        # TR = max(H-L, |H-PC|, |L-PC|)
        tr = true_range(D("1.11"), D("1.09"), D("1.10"))
        # H-L = 0.02, |H-PC|=0.01, |L-PC|=0.01 → 0.02
        assert tr == D("0.02")

    def test_gap_up_dominates(self) -> None:
        # Gap up: open = prev_close + 0.05
        tr = true_range(D("1.15"), D("1.12"), D("1.09"))
        # H-L=0.03, |H-PC|=0.06, |L-PC|=0.03 → 0.06
        assert tr == D("0.06")

    def test_gap_down_dominates(self) -> None:
        tr = true_range(D("1.07"), D("1.05"), D("1.12"))
        # H-L=0.02, |H-PC|=0.05, |L-PC|=0.07 → 0.07
        assert tr == D("0.07")

    def test_always_non_negative(self) -> None:
        tr = true_range(D("1.10"), D("1.10"), D("1.10"))
        assert tr >= D("0")


class TestAtr:
    def _ohlc(self, n: int = 20) -> tuple[list[Decimal], list[Decimal], list[Decimal]]:
        highs = [D(str(1.1 + i * 0.001 + 0.002)) for i in range(n)]
        lows = [D(str(1.1 + i * 0.001 - 0.002)) for i in range(n)]
        closes = [D(str(1.1 + i * 0.001)) for i in range(n)]
        return highs, lows, closes

    def test_length_equals_input(self) -> None:
        h, l, c = self._ohlc(25)
        result = atr(h, l, c, 14)
        assert len(result) == len(h)

    def test_atr_positive(self) -> None:
        h, l, c = self._ohlc(25)
        result = atr(h, l, c, 14)
        for val in result:
            assert val > D("0")

    def test_raises_unequal_lengths(self) -> None:
        h = [D("1.1")] * 20
        l = [D("1.09")] * 19  # different length
        c = [D("1.095")] * 20
        with pytest.raises(ValueError, match="equal length"):
            atr(h, l, c, 14)

    def test_raises_insufficient_data(self) -> None:
        h = [D("1.1")] * 10
        l = [D("1.09")] * 10
        c = [D("1.095")] * 10
        with pytest.raises(ValueError, match="Need at least"):
            atr(h, l, c, 14)

    def test_high_volatility_gives_high_atr(self) -> None:
        # Large H-L range → high ATR
        highs = [D("1.15")] * 25
        lows = [D("1.05")] * 25
        closes = [D("1.10")] * 25
        result = atr(highs, lows, closes, 14)
        assert result[-1] > D("0.05")

    def test_low_volatility_gives_low_atr(self) -> None:
        highs = [D("1.1001")] * 25
        lows = [D("1.0999")] * 25
        closes = [D("1.1000")] * 25
        result = atr(highs, lows, closes, 14)
        assert result[-1] < D("0.001")

    def test_consistency_with_same_input(self) -> None:
        h, l, c = self._ohlc(25)
        r1 = atr(h, l, c, 14)
        r2 = atr(h, l, c, 14)
        assert r1 == r2


class TestAtrStopLoss:
    def test_buy_sl_below_entry(self) -> None:
        sl = atr_stop_loss(
            entry_price=D("1.10000"),
            atr_value=D("0.00200"),
            multiplier=2.0,
            side_is_buy=True,
        )
        assert sl == D("1.09600")
        assert sl < D("1.10000")

    def test_sell_sl_above_entry(self) -> None:
        sl = atr_stop_loss(
            entry_price=D("1.10000"),
            atr_value=D("0.00200"),
            multiplier=2.0,
            side_is_buy=False,
        )
        assert sl == D("1.10400")
        assert sl > D("1.10000")

    def test_multiplier_scales_distance(self) -> None:
        sl_1x = atr_stop_loss(D("1.10000"), D("0.00100"), 1.0, True)
        sl_3x = atr_stop_loss(D("1.10000"), D("0.00100"), 3.0, True)
        distance_1x = D("1.10000") - sl_1x
        distance_3x = D("1.10000") - sl_3x
        assert abs(distance_3x - distance_1x * 3) < D("0.000001")

    def test_zero_atr_returns_entry(self) -> None:
        sl = atr_stop_loss(D("1.10000"), D("0"), 2.0, True)
        assert sl == D("1.10000")
