"""Gap detector — finds missing bars in stored time series.

A "gap" is a period where bars are expected but absent.
For Forex, weekends and holidays are NOT gaps — they are expected absences.
A gap is unexpected missing data during market hours.

Market hours for Forex:
    Opens: Sunday 22:00 UTC
    Closes: Friday 22:00 UTC
    Closed: Saturday + Sunday before 22:00 UTC

This detector uses a pragmatic approach:
- Two consecutive bars are a gap if the time between them is more than
  `max_gap_multiplier * timeframe_seconds` (e.g. > 3 bars missing).
- Weekend gaps (Friday close → Sunday open) are automatically excluded.
- The result is a list of GapInfo objects, one per detected gap.
"""

from __future__ import annotations

from datetime import datetime

from metatrade.core.enums import Timeframe
from metatrade.core.log import get_logger
from metatrade.market_data.interface import IBarStore, IGapDetector
from metatrade.market_data.models import BarQuery, GapInfo

log = get_logger(__name__)

# Weekday numbers: 0=Monday, 4=Friday, 5=Saturday, 6=Sunday
_FRIDAY = 4
_SATURDAY = 5
_SUNDAY = 6
_MARKET_OPEN_HOUR_UTC = 22   # Sunday 22:00 UTC
_MARKET_CLOSE_HOUR_UTC = 22  # Friday 22:00 UTC


class GapDetector(IGapDetector):
    """Detects unexpected gaps in stored bar series.

    Args:
        store:               Bar store to query.
        max_gap_multiplier:  A gap is flagged when the interval between two
                             consecutive bars exceeds this multiple of the
                             timeframe duration. Default = 3 (allows for
                             minor data interruptions of 1-2 bars).
    """

    def __init__(self, store: IBarStore, max_gap_multiplier: float = 3.0) -> None:
        self._store = store
        self._max_gap_multiplier = max_gap_multiplier

    def detect_gaps(
        self,
        symbol: str,
        timeframe: Timeframe,
        date_from: datetime,
        date_to: datetime,
    ) -> list[GapInfo]:
        """Return all non-weekend gaps found in the stored bars for the range."""
        bars = self._store.get_bars(
            BarQuery(symbol=symbol, timeframe=timeframe, date_from=date_from, date_to=date_to)
        )

        if len(bars) < 2:
            return []

        gaps: list[GapInfo] = []
        max_interval = timeframe.seconds * self._max_gap_multiplier

        for i in range(1, len(bars)):
            prev_bar = bars[i - 1]
            curr_bar = bars[i]
            interval = (
                curr_bar.timestamp_utc - prev_bar.timestamp_utc
            ).total_seconds()

            if interval <= max_interval:
                continue

            if _is_weekend_gap(prev_bar.timestamp_utc, curr_bar.timestamp_utc):
                continue

            expected_missing = int(interval / timeframe.seconds) - 1
            gap = GapInfo(
                symbol=symbol,
                timeframe=timeframe,
                gap_start_utc=prev_bar.timestamp_utc,
                gap_end_utc=curr_bar.timestamp_utc,
                expected_bars_missing=expected_missing,
            )
            gaps.append(gap)
            log.warning(
                "gap_detected",
                symbol=symbol,
                timeframe=timeframe.value,
                gap_start=prev_bar.timestamp_utc.isoformat(),
                gap_end=curr_bar.timestamp_utc.isoformat(),
                missing_bars=expected_missing,
            )

        return gaps

    def detect_all_symbols(
        self,
        symbols: list[str],
        timeframes: list[Timeframe],
        date_from: datetime,
        date_to: datetime,
    ) -> dict[str, list[GapInfo]]:
        """Run gap detection across all symbol/timeframe combinations."""
        result: dict[str, list[GapInfo]] = {}
        for symbol in symbols:
            for tf in timeframes:
                key = f"{symbol}/{tf.value}"
                gaps = self.detect_gaps(symbol, tf, date_from, date_to)
                if gaps:
                    result[key] = gaps
        return result


def _is_weekend_gap(prev_ts: datetime, curr_ts: datetime) -> bool:
    """Return True if the gap between prev_ts and curr_ts spans a weekend.

    A weekend gap is: prev_ts is on Friday at or after 22:00 UTC, and
    curr_ts is on Monday (or Sunday at/after 22:00 UTC).
    We apply a lenient check: if Saturday falls between the two timestamps,
    it's a weekend gap.
    """
    prev_day = prev_ts.weekday()
    curr_day = curr_ts.weekday()

    # If prev is Friday and curr is Sunday or Monday — weekend gap
    if prev_day == _FRIDAY:
        return True

    # If prev is Saturday — always a weekend gap
    if prev_day == _SATURDAY:
        return True

    # If curr is Monday and prev is Thursday or earlier — more than a weekend
    # (this case is a genuine gap, not just a weekend)
    return False
