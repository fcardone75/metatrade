"""Tests for CurrencyExposureService and parse_symbol.

Coverage targets
----------------
- Symbol parsing: standard 6-char, slash-separated, with suffix, unknown.
- Long/short exposure delta for all major pairs.
- Net exposure with multiple positions (accumulation and netting).
- Gross exposure (always positive per currency).
- net_exposure_with_candidate correctly adds candidate delta.
- Closed positions are excluded.
- Unknown/exotic symbols are silently skipped.
- to_currency_exposure_list produces correct CurrencyExposure objects.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from metatrade.core.contracts.position import Position
from metatrade.core.enums import OrderSide, PositionSide, PositionState
from metatrade.intermarket.exposure import CurrencyExposureService, parse_symbol

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pos(
    symbol: str,
    side: PositionSide,
    lot_size: float = 1.0,
    state: PositionState = PositionState.OPEN,
) -> Position:
    return Position(
        position_id=f"test_{symbol}_{side.value}",
        symbol=symbol,
        side=side,
        lot_size=Decimal(str(lot_size)),
        entry_price=Decimal("1.10000"),
        opened_at_utc=_NOW,
        state=state,
        closed_at_utc=_NOW if state == PositionState.CLOSED else None,
        close_price=Decimal("1.10000") if state == PositionState.CLOSED else None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# parse_symbol
# ══════════════════════════════════════════════════════════════════════════════

class TestParseSymbol:
    def test_standard_eurusd(self) -> None:
        assert parse_symbol("EURUSD") == ("EUR", "USD")

    def test_standard_usdjpy(self) -> None:
        assert parse_symbol("USDJPY") == ("USD", "JPY")

    def test_standard_gbpjpy(self) -> None:
        assert parse_symbol("GBPJPY") == ("GBP", "JPY")

    def test_slash_separated(self) -> None:
        assert parse_symbol("EUR/USD") == ("EUR", "USD")

    def test_lowercase_input(self) -> None:
        assert parse_symbol("eurusd") == ("EUR", "USD")

    def test_with_broker_suffix(self) -> None:
        # Some brokers append suffixes like .c
        assert parse_symbol("EURUSD.c") == ("EUR", "USD")

    def test_unknown_base_currency_returns_none(self) -> None:
        assert parse_symbol("BTCUSD") is None

    def test_unknown_quote_currency_returns_none(self) -> None:
        assert parse_symbol("EURZYX") is None

    def test_too_short_returns_none(self) -> None:
        assert parse_symbol("EUR") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_symbol("") is None

    def test_all_major_pairs(self) -> None:
        pairs = [
            ("EURUSD", ("EUR", "USD")),
            ("GBPUSD", ("GBP", "USD")),
            ("USDCHF", ("USD", "CHF")),
            ("USDJPY", ("USD", "JPY")),
            ("AUDUSD", ("AUD", "USD")),
            ("NZDUSD", ("NZD", "USD")),
            ("USDCAD", ("USD", "CAD")),
            ("EURGBP", ("EUR", "GBP")),
            ("EURJPY", ("EUR", "JPY")),
            ("GBPJPY", ("GBP", "JPY")),
        ]
        for symbol, expected in pairs:
            assert parse_symbol(symbol) == expected, f"Failed for {symbol}"


# ══════════════════════════════════════════════════════════════════════════════
# CurrencyExposureService.position_delta
# ══════════════════════════════════════════════════════════════════════════════

class TestPositionDelta:
    def setup_method(self) -> None:
        self.svc = CurrencyExposureService()

    def test_long_eurusd(self) -> None:
        delta = self.svc.position_delta("EURUSD", PositionSide.LONG, Decimal("1.0"))
        assert delta["EUR"] == Decimal("1.0")
        assert delta["USD"] == Decimal("-1.0")

    def test_short_eurusd(self) -> None:
        delta = self.svc.position_delta("EURUSD", PositionSide.SHORT, Decimal("1.0"))
        assert delta["EUR"] == Decimal("-1.0")
        assert delta["USD"] == Decimal("1.0")

    def test_buy_order_side_same_as_long(self) -> None:
        delta = self.svc.position_delta("EURUSD", OrderSide.BUY, Decimal("1.0"))
        assert delta["EUR"] == Decimal("1.0")
        assert delta["USD"] == Decimal("-1.0")

    def test_sell_order_side_same_as_short(self) -> None:
        delta = self.svc.position_delta("EURUSD", OrderSide.SELL, Decimal("1.0"))
        assert delta["EUR"] == Decimal("-1.0")
        assert delta["USD"] == Decimal("1.0")

    def test_short_gbpjpy(self) -> None:
        delta = self.svc.position_delta("GBPJPY", PositionSide.SHORT, Decimal("2.0"))
        assert delta["GBP"] == Decimal("-2.0")
        assert delta["JPY"] == Decimal("2.0")

    def test_long_usdchf(self) -> None:
        # long USD/CHF: +USD, -CHF
        delta = self.svc.position_delta("USDCHF", PositionSide.LONG, Decimal("1.0"))
        assert delta["USD"] == Decimal("1.0")
        assert delta["CHF"] == Decimal("-1.0")

    def test_unknown_symbol_returns_empty(self) -> None:
        delta = self.svc.position_delta("BTCUSD", PositionSide.LONG, Decimal("1.0"))
        assert delta == {}

    def test_lot_size_scaling(self) -> None:
        delta = self.svc.position_delta("EURUSD", PositionSide.LONG, Decimal("0.1"))
        assert delta["EUR"] == Decimal("0.1")
        assert delta["USD"] == Decimal("-0.1")


# ══════════════════════════════════════════════════════════════════════════════
# CurrencyExposureService.compute_net_exposure
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeNetExposure:
    def setup_method(self) -> None:
        self.svc = CurrencyExposureService()

    def test_single_long_eurusd(self) -> None:
        positions = [_pos("EURUSD", PositionSide.LONG, 1.0)]
        net = self.svc.compute_net_exposure(positions)
        assert net["EUR"] == Decimal("1.0")
        assert net["USD"] == Decimal("-1.0")

    def test_two_longs_same_currency(self) -> None:
        positions = [
            _pos("EURUSD", PositionSide.LONG, 1.0),
            _pos("EURGBP", PositionSide.LONG, 0.5),
        ]
        net = self.svc.compute_net_exposure(positions)
        # EUR appears in both: +1.0 + 0.5 = +1.5
        assert net["EUR"] == Decimal("1.5")

    def test_net_exposure_long_and_short_same_pair_cancels(self) -> None:
        positions = [
            _pos("EURUSD", PositionSide.LONG, 1.0),
            _pos("EURUSD", PositionSide.SHORT, 1.0),
        ]
        net = self.svc.compute_net_exposure(positions)
        assert net.get("EUR", Decimal("0")) == Decimal("0")
        assert net.get("USD", Decimal("0")) == Decimal("0")

    def test_closed_positions_excluded(self) -> None:
        positions = [
            _pos("EURUSD", PositionSide.LONG, 1.0, state=PositionState.CLOSED),
        ]
        net = self.svc.compute_net_exposure(positions)
        assert net.get("EUR", Decimal("0")) == Decimal("0")

    def test_empty_positions(self) -> None:
        net = self.svc.compute_net_exposure([])
        assert net == {}

    def test_multiple_pairs_all_usd_short(self) -> None:
        # Long EURUSD + Long GBPUSD → net USD = -2.0
        positions = [
            _pos("EURUSD", PositionSide.LONG, 1.0),
            _pos("GBPUSD", PositionSide.LONG, 1.0),
        ]
        net = self.svc.compute_net_exposure(positions)
        assert net["USD"] == Decimal("-2.0")

    def test_unknown_symbol_skipped(self) -> None:
        positions = [_pos("EURUSD", PositionSide.LONG, 1.0)]
        # Manually create a position with an exotic symbol
        exotic = Position(
            position_id="exotic",
            symbol="BTCUSD",
            side=PositionSide.LONG,
            lot_size=Decimal("1.0"),
            entry_price=Decimal("50000"),
            opened_at_utc=_NOW,
        )
        net = self.svc.compute_net_exposure([positions[0], exotic])
        assert "EUR" in net  # EURUSD still present
        # BTC not in result (not a known major)
        assert "BTC" not in net


# ══════════════════════════════════════════════════════════════════════════════
# CurrencyExposureService.compute_gross_exposure
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeGrossExposure:
    def setup_method(self) -> None:
        self.svc = CurrencyExposureService()

    def test_gross_always_positive(self) -> None:
        positions = [_pos("EURUSD", PositionSide.SHORT, 2.0)]
        gross = self.svc.compute_gross_exposure(positions)
        assert gross["EUR"] == Decimal("2.0")
        assert gross["USD"] == Decimal("2.0")

    def test_gross_sums_absolute_values(self) -> None:
        # Long 1 EURUSD: EUR +1, USD -1
        # Short 1 GBPUSD: GBP -1, USD +1
        # USD gross = |−1| + |+1| = 2
        positions = [
            _pos("EURUSD", PositionSide.LONG, 1.0),
            _pos("GBPUSD", PositionSide.SHORT, 1.0),
        ]
        gross = self.svc.compute_gross_exposure(positions)
        assert gross["USD"] == Decimal("2.0")

    def test_gross_ne_net_when_mixed_directions(self) -> None:
        positions = [
            _pos("EURUSD", PositionSide.LONG, 1.0),
            _pos("EURUSD", PositionSide.SHORT, 1.0),
        ]
        net = self.svc.compute_net_exposure(positions)
        gross = self.svc.compute_gross_exposure(positions)
        # Net EUR = 0; gross EUR = 2
        assert net.get("EUR", Decimal("0")) == Decimal("0")
        assert gross["EUR"] == Decimal("2.0")


# ══════════════════════════════════════════════════════════════════════════════
# CurrencyExposureService.exposure_delta / net_exposure_with_candidate
# ══════════════════════════════════════════════════════════════════════════════

class TestExposureDelta:
    def setup_method(self) -> None:
        self.svc = CurrencyExposureService()

    def test_exposure_delta_buy(self) -> None:
        delta = self.svc.exposure_delta("EURUSD", OrderSide.BUY, Decimal("1.0"))
        assert delta["EUR"] == Decimal("1.0")
        assert delta["USD"] == Decimal("-1.0")

    def test_net_exposure_with_candidate_adds_delta(self) -> None:
        open_pos = [_pos("EURUSD", PositionSide.LONG, 1.0)]
        projected = self.svc.net_exposure_with_candidate(
            open_pos, "EURUSD", OrderSide.BUY, Decimal("0.5")
        )
        assert projected["EUR"] == Decimal("1.5")
        assert projected["USD"] == Decimal("-1.5")

    def test_net_exposure_with_candidate_reduces_net_on_opposite_direction(self) -> None:
        open_pos = [_pos("EURUSD", PositionSide.LONG, 1.0)]
        projected = self.svc.net_exposure_with_candidate(
            open_pos, "EURUSD", OrderSide.SELL, Decimal("1.0")
        )
        assert projected["EUR"] == Decimal("0")
        assert projected["USD"] == Decimal("0")


# ══════════════════════════════════════════════════════════════════════════════
# CurrencyExposureService.to_currency_exposure_list
# ══════════════════════════════════════════════════════════════════════════════

class TestToCurrencyExposureList:
    def setup_method(self) -> None:
        self.svc = CurrencyExposureService()

    def test_produces_currency_exposure_objects(self) -> None:
        from metatrade.intermarket.contracts import CurrencyExposure
        net = {"EUR": Decimal("1.0"), "USD": Decimal("-1.0")}
        gross = {"EUR": Decimal("1.0"), "USD": Decimal("1.0")}
        result = self.svc.to_currency_exposure_list(net, gross)
        assert all(isinstance(e, CurrencyExposure) for e in result)

    def test_correct_values(self) -> None:
        net = {"EUR": Decimal("1.5"), "USD": Decimal("-1.0")}
        gross = {"EUR": Decimal("2.0"), "USD": Decimal("1.0")}
        result = self.svc.to_currency_exposure_list(net, gross)
        by_ccy = {e.currency: e for e in result}
        assert by_ccy["EUR"].net_lots == Decimal("1.5")
        assert by_ccy["EUR"].gross_lots == Decimal("2.0")
        assert by_ccy["USD"].net_lots == Decimal("-1.0")

    def test_currencies_from_both_dicts_included(self) -> None:
        net = {"EUR": Decimal("1.0")}
        gross = {"EUR": Decimal("1.0"), "CHF": Decimal("0.5")}
        result = self.svc.to_currency_exposure_list(net, gross)
        currencies = {e.currency for e in result}
        assert "EUR" in currencies
        assert "CHF" in currencies
