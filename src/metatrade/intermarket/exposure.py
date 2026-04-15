"""Currency exposure accounting for the intermarket portfolio-risk layer.

Translates positions and candidate trades into per-currency net/gross
lot exposure.  This is purely arithmetic — no market data required.

Design notes
------------
- Exposure is measured in **lots** (not notional currency units), which is
  the natural quantity available from Position objects without requiring
  current prices or pip values.
- Long EURUSD = +EUR exposure, −USD exposure.
- Short GBPJPY = −GBP exposure, +JPY exposure.
- Unknown symbols (non-standard 6-char pairs) are silently ignored so
  that the engine degrades gracefully on exotic instruments.
"""

from __future__ import annotations

from decimal import Decimal

from metatrade.core.contracts.position import Position
from metatrade.core.enums import OrderSide, PositionSide
from metatrade.intermarket.contracts import CurrencyCode, CurrencyExposure

# Recognised major-currency codes.  Used for symbol parsing validation.
_KNOWN_CURRENCIES: frozenset[str] = frozenset(
    {"EUR", "USD", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"}
)


# ── Symbol parsing ─────────────────────────────────────────────────────────────

def parse_symbol(symbol: str) -> tuple[CurrencyCode, CurrencyCode] | None:
    """Extract (base, quote) currency codes from a forex symbol string.

    Handles:
    - Standard 6-char symbols: ``"EURUSD"`` → ``("EUR", "USD")``
    - Slash-separated:         ``"EUR/USD"`` → ``("EUR", "USD")``
    - Broker suffixes:         ``"EURUSD.c"`` → ``("EUR", "USD")``

    Returns ``None`` if either extracted code is not a recognised major
    currency, preventing misidentification of CFDs, indices, etc.

    Args:
        symbol: Instrument identifier.

    Returns:
        ``(base, quote)`` tuple or ``None`` if unparseable.
    """
    clean = (
        symbol.upper()
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
    )
    # Strip known broker suffixes (e.g. ".c", "m", "+")
    if len(clean) > 6:
        clean = clean[:6]

    if len(clean) < 6:
        return None

    base = clean[:3]
    quote = clean[3:6]

    if base not in _KNOWN_CURRENCIES or quote not in _KNOWN_CURRENCIES:
        return None

    return base, quote


# ── Service ────────────────────────────────────────────────────────────────────

class CurrencyExposureService:
    """Computes net and gross currency exposure from positions and candidate trades.

    All exposure values are in lots (the natural unit of Position objects).
    The service is stateless and deterministic — it does not cache anything.
    """

    # ── Core exposure accounting ───────────────────────────────────────────────

    def position_delta(
        self,
        symbol: str,
        side: PositionSide | OrderSide,
        lot_size: Decimal,
    ) -> dict[CurrencyCode, Decimal]:
        """Return the currency exposure delta for a single position or trade.

        Convention:
        - Long / BUY  EURUSD:  ``{"EUR": +lot_size, "USD": -lot_size}``
        - Short / SELL GBPJPY: ``{"GBP": -lot_size, "JPY": +lot_size}``

        Returns an empty dict for unrecognised symbols.

        Args:
            symbol:   Instrument identifier.
            side:     ``PositionSide`` (open position) or ``OrderSide`` (candidate).
            lot_size: Lot size (must be > 0).
        """
        parsed = parse_symbol(symbol)
        if parsed is None:
            return {}

        base, quote = parsed
        is_long = side in (PositionSide.LONG, OrderSide.BUY)

        if is_long:
            return {base: lot_size, quote: -lot_size}
        return {base: -lot_size, quote: lot_size}

    def compute_net_exposure(
        self,
        positions: list[Position],
    ) -> dict[CurrencyCode, Decimal]:
        """Return signed net lot exposure per currency for all open positions.

        Args:
            positions: All positions (open and closed); closed ones are skipped.

        Returns:
            Dict of ``currency → net_lots`` (positive = net long).
        """
        exposure: dict[str, Decimal] = {}
        for pos in positions:
            if not pos.is_open:
                continue
            delta = self.position_delta(pos.symbol, pos.side, pos.lot_size)
            for ccy, lots in delta.items():
                exposure[ccy] = exposure.get(ccy, Decimal("0")) + lots
        return exposure

    def compute_gross_exposure(
        self,
        positions: list[Position],
    ) -> dict[CurrencyCode, Decimal]:
        """Return gross (absolute) lot exposure per currency for all open positions.

        Gross exposure sums the absolute values of each position's lot
        contribution, so a long 1-lot EURUSD and a short 1-lot GBPUSD
        both contribute +1 to USD gross exposure.

        Args:
            positions: All positions; closed ones are skipped.
        """
        gross: dict[str, Decimal] = {}
        for pos in positions:
            if not pos.is_open:
                continue
            delta = self.position_delta(pos.symbol, pos.side, pos.lot_size)
            for ccy, lots in delta.items():
                gross[ccy] = gross.get(ccy, Decimal("0")) + abs(lots)
        return gross

    def exposure_delta(
        self,
        symbol: str,
        side: OrderSide,
        lot_size: Decimal,
    ) -> dict[CurrencyCode, Decimal]:
        """Return the currency exposure change if a candidate trade is opened.

        Equivalent to ``position_delta`` but takes ``OrderSide`` explicitly,
        making the intent clear at call sites.

        Args:
            symbol:   Instrument to trade.
            side:     BUY or SELL.
            lot_size: Proposed lot size.
        """
        return self.position_delta(symbol, side, lot_size)

    def net_exposure_with_candidate(
        self,
        positions: list[Position],
        symbol: str,
        side: OrderSide,
        lot_size: Decimal,
    ) -> dict[CurrencyCode, Decimal]:
        """Return projected net exposure after adding a candidate trade.

        Args:
            positions: Current open positions.
            symbol:    Candidate symbol.
            side:      Candidate side.
            lot_size:  Candidate lot size.
        """
        current = self.compute_net_exposure(positions)
        delta = self.exposure_delta(symbol, side, lot_size)
        combined = dict(current)
        for ccy, lots in delta.items():
            combined[ccy] = combined.get(ccy, Decimal("0")) + lots
        return combined

    # ── Structured output ──────────────────────────────────────────────────────

    def to_currency_exposure_list(
        self,
        net: dict[CurrencyCode, Decimal],
        gross: dict[CurrencyCode, Decimal],
    ) -> tuple[CurrencyExposure, ...]:
        """Convert flat net/gross dicts into a tuple of ``CurrencyExposure`` objects.

        Args:
            net:   Net lot exposure per currency.
            gross: Gross lot exposure per currency.
        """
        currencies = frozenset(net) | frozenset(gross)
        return tuple(
            CurrencyExposure(
                currency=ccy,
                net_lots=net.get(ccy, Decimal("0")),
                gross_lots=gross.get(ccy, Decimal("0")),
            )
            for ccy in sorted(currencies)
        )
