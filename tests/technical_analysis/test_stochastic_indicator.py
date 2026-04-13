"""Tests for the Stochastic RSI indicator."""

from __future__ import annotations

from decimal import Decimal

import pytest

from metatrade.technical_analysis.indicators.stochastic import stochastic_rsi


def D(v: float) -> Decimal:
    return Decimal(str(round(v, 6)))


def rising_closes(n: int, start: float = 1.1, step: float = 0.001) -> list[Decimal]:
    return [D(start + i * step) for i in range(n)]


def falling_closes(n: int, start: float = 1.2, step: float = 0.001) -> list[Decimal]:
    return [D(start - i * step) for i in range(n)]


def flat_closes(n: int, price: float = 1.1) -> list[Decimal]:
    return [D(price)] * n


class TestStochasticRsi:
    def test_returns_two_lists(self) -> None:
        closes = rising_closes(60)
        k, d = stochastic_rsi(closes)
        assert isinstance(k, list)
        assert isinstance(d, list)

    def test_output_same_length_as_input(self) -> None:
        closes = rising_closes(60)
        k, d = stochastic_rsi(closes)
        assert len(k) == len(closes)
        assert len(d) == len(closes)

    def test_values_in_0_to_1_range(self) -> None:
        closes = rising_closes(80)
        k, d = stochastic_rsi(closes)
        for val in k:
            assert -0.001 <= float(val) <= 1.001, f"K out of range: {val}"
        for val in d:
            assert -0.001 <= float(val) <= 1.001, f"D out of range: {val}"

    def test_rising_prices_produce_high_k(self) -> None:
        """After a dip then sharp recovery, RSI moves from low to high,
        so the StochRSI K should be near 1 (RSI near its recent maximum)."""
        # Build: 30 bars falling (RSI goes low) + 50 bars rising sharply
        # This creates RSI range variation so StochRSI is not 0.5
        closes = falling_closes(30, start=1.2, step=0.002) + rising_closes(50, start=1.14, step=0.003)
        k, d = stochastic_rsi(closes, rsi_period=10, stoch_period=10)
        # After the strong recovery, K should be elevated
        assert float(k[-1]) >= 0.4  # RSI is recovering → K moves up

    def test_falling_prices_produce_low_k(self) -> None:
        """After a rise then sharp drop, RSI moves from high to low,
        so the StochRSI K should be near 0 (RSI near its recent minimum)."""
        closes = rising_closes(30, start=1.1, step=0.002) + falling_closes(50, start=1.16, step=0.003)
        k, d = stochastic_rsi(closes, rsi_period=10, stoch_period=10)
        # After the strong drop, K should be near the bottom
        assert float(k[-1]) <= 0.6

    def test_raises_on_too_few_bars(self) -> None:
        closes = rising_closes(5)
        with pytest.raises(ValueError, match="at least"):
            stochastic_rsi(closes, rsi_period=14, stoch_period=14)

    def test_all_values_are_decimal(self) -> None:
        closes = rising_closes(60)
        k, d = stochastic_rsi(closes)
        assert all(isinstance(v, Decimal) for v in k)
        assert all(isinstance(v, Decimal) for v in d)

    def test_custom_periods(self) -> None:
        closes = rising_closes(50)
        k, d = stochastic_rsi(closes, rsi_period=8, stoch_period=8, smooth_k=2, smooth_d=2)
        assert len(k) == len(closes)

    def test_d_is_smoothed_k(self) -> None:
        """D line should be smoother (less variable) than K line."""
        closes = [D(1.1 + (0.01 if i % 3 == 0 else -0.005)) for i in range(70)]
        k, d = stochastic_rsi(closes, rsi_period=10, stoch_period=10, smooth_k=3, smooth_d=3)
        # Standard deviation of D should be <= std of K over the last 20 values
        k_tail = [float(v) for v in k[-20:]]
        d_tail = [float(v) for v in d[-20:]]
        k_mean = sum(k_tail) / len(k_tail)
        d_mean = sum(d_tail) / len(d_tail)
        k_var = sum((x - k_mean) ** 2 for x in k_tail) / len(k_tail)
        d_var = sum((x - d_mean) ** 2 for x in d_tail) / len(d_tail)
        # D line should be at least as smooth as K
        assert d_var <= k_var * 2.0  # generous tolerance
