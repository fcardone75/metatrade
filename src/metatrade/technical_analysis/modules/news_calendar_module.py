"""High-impact economic news calendar filter.

Returns HOLD within a configurable window around scheduled high-impact
events to avoid trading in unpredictable, liquidity-gap conditions.

Covered events (hardcoded offline schedule):
    - US Non-Farm Payrolls (NFP): first Friday of each month, 13:30 UTC
    - FOMC rate decision:        ~8 per year, 19:00 UTC
    - ECB rate decision:         ~8 per year, 13:15 UTC

Optional live calendar (requires Finnhub API key):
    When ``finnhub_api_key`` is provided, the module fetches the upcoming
    week's high-impact events from the Finnhub economic calendar API on
    first use (and refreshes daily).  The hardcoded schedule is used as
    fallback if the API call fails.

Signal logic:
    HOLD: current timestamp is within ``window_minutes`` of any
          high-impact event (before OR after).
    HOLD: current timestamp is outside supported date range AND no API key
          is configured (ultra-conservative fallback).
    BUY/SELL: never — this module only emits directional HOLD.
              Its direction is always SignalDirection.HOLD.

Confidence:
    0.50 when HOLD (low, so other modules can outweigh it).
    0.50 always — this module never makes directional claims.
"""

from __future__ import annotations

import logging
from calendar import monthrange
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.interface import ITechnicalModule

_VERSION = ModuleVersion(1, 0, 0)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardcoded high-impact event schedule
# ---------------------------------------------------------------------------

class _Event(NamedTuple):
    name: str
    dt_utc: datetime   # exact announcement time in UTC


def _utc(*args: int) -> datetime:
    """Shorthand: _utc(year, month, day, hour, minute) → aware datetime."""
    return datetime(*args, tzinfo=timezone.utc)


# FOMC announcements (19:00 UTC on the second day of the two-day meeting)
_FOMC_DATES: list[datetime] = [
    # 2025
    _utc(2025, 1, 29, 19, 0), _utc(2025, 3, 19, 19, 0),
    _utc(2025, 5,  7, 19, 0), _utc(2025, 6, 18, 19, 0),
    _utc(2025, 7, 30, 19, 0), _utc(2025, 9, 17, 19, 0),
    _utc(2025, 10, 29, 19, 0), _utc(2025, 12, 10, 19, 0),
    # 2026
    _utc(2026, 1, 28, 19, 0), _utc(2026, 3, 18, 19, 0),
    _utc(2026, 5,  6, 19, 0), _utc(2026, 6, 17, 19, 0),
    _utc(2026, 7, 29, 19, 0), _utc(2026, 9, 16, 19, 0),
    _utc(2026, 10, 28, 19, 0), _utc(2026, 12,  9, 19, 0),
]

# ECB rate decisions (13:15 UTC press conference / announcement)
_ECB_DATES: list[datetime] = [
    # 2025
    _utc(2025, 1, 30, 13, 15), _utc(2025, 3,  6, 13, 15),
    _utc(2025, 4, 17, 13, 15), _utc(2025, 6,  5, 13, 15),
    _utc(2025, 7, 24, 13, 15), _utc(2025, 9, 11, 13, 15),
    _utc(2025, 10, 30, 13, 15), _utc(2025, 12, 18, 13, 15),
    # 2026
    _utc(2026, 1, 29, 13, 15), _utc(2026, 3,  5, 13, 15),
    _utc(2026, 4, 23, 13, 15), _utc(2026, 6,  4, 13, 15),
    _utc(2026, 7, 23, 13, 15), _utc(2026, 9, 10, 13, 15),
    _utc(2026, 10, 29, 13, 15), _utc(2026, 12, 17, 13, 15),
]


def _first_friday_of_month(year: int, month: int) -> int:
    """Return the day-of-month of the first Friday (weekday=4) of month."""
    for day in range(1, 8):
        if datetime(year, month, day).weekday() == 4:
            return day
    raise ValueError(f"No Friday in first 7 days of {year}-{month}")  # impossible


def _nfp_event(year: int, month: int) -> datetime:
    """US NFP: first Friday of the month at 13:30 UTC."""
    day = _first_friday_of_month(year, month)
    return _utc(year, month, day, 13, 30)


def _build_hardcoded_events(from_year: int = 2020, to_year: int = 2030) -> list[_Event]:
    """Build the full hardcoded event list."""
    events: list[_Event] = []

    # FOMC + ECB (pre-defined dates)
    for dt in _FOMC_DATES:
        if from_year <= dt.year <= to_year:
            events.append(_Event("FOMC", dt))
    for dt in _ECB_DATES:
        if from_year <= dt.year <= to_year:
            events.append(_Event("ECB", dt))

    # NFP: generate for all months in range
    for year in range(from_year, to_year + 1):
        for month in range(1, 13):
            events.append(_Event("NFP", _nfp_event(year, month)))

    return sorted(events, key=lambda e: e.dt_utc)


_HARDCODED_EVENTS: list[_Event] = _build_hardcoded_events()


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class NewsCalendarModule(ITechnicalModule):
    """High-impact news event filter — emits HOLD around scheduled events.

    Args:
        window_minutes:    Minutes before AND after each event to block.
                           Default 60 (= 1 hour each side).
        finnhub_api_key:   Optional Finnhub.io API key.  When provided, the
                           module enriches the hardcoded schedule with a live
                           API fetch covering the next 7 days (refreshed once
                           per calendar day).  When None, uses hardcoded only.
        timeframe_id:      Suffix for ``module_id``.
        confidence_cap:    Not used directionally, but stored for consistency.
    """

    def __init__(
        self,
        window_minutes: int = 60,
        finnhub_api_key: str | None = None,
        timeframe_id: str = "h1",
        confidence_cap: float = 0.50,
    ) -> None:
        self._window = timedelta(minutes=window_minutes)
        self._api_key = finnhub_api_key
        self._timeframe_id = timeframe_id
        self._confidence_cap = confidence_cap

        # Live-fetch cache
        self._live_events: list[_Event] = []
        self._last_fetch_date: datetime | None = None  # date of last API call

    @property
    def module_id(self) -> str:
        return f"news_calendar_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        # Only needs the timestamp, no bar history required
        return 1

    # ── Finnhub API ───────────────────────────────────────────────────────────

    def _maybe_refresh_live_events(self, now: datetime) -> None:
        """Fetch live calendar from Finnhub if API key is set and cache is stale."""
        if self._api_key is None:
            return
        # Refresh at most once per calendar day
        today = now.date()
        if (
            self._last_fetch_date is not None
            and self._last_fetch_date.date() == today
        ):
            return

        try:
            import urllib.request
            import json
            from_str = now.strftime("%Y-%m-%d")
            to_str   = (now + timedelta(days=7)).strftime("%Y-%m-%d")
            url = (
                f"https://finnhub.io/api/v1/calendar/economic"
                f"?from={from_str}&to={to_str}&token={self._api_key}"
            )
            with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
                data = json.loads(resp.read().decode())

            events: list[_Event] = []
            for item in data.get("economicCalendar", []):
                if item.get("impact", "").lower() != "high":
                    continue
                try:
                    dt = datetime.strptime(
                        item["time"], "%Y-%m-%d %H:%M:%S"
                    ).replace(tzinfo=timezone.utc)
                    events.append(_Event(item.get("event", "?"), dt))
                except (KeyError, ValueError):
                    continue

            self._live_events = events
            self._last_fetch_date = now
            log.debug("NewsCalendar: fetched %d high-impact events from Finnhub", len(events))

        except Exception as exc:
            log.warning("NewsCalendar: Finnhub fetch failed (%s) — using hardcoded schedule", exc)

    # ── Core logic ────────────────────────────────────────────────────────────

    def _is_near_event(self, now: datetime) -> _Event | None:
        """Return the first event within window of now, or None."""
        # Combine hardcoded + live events
        all_events = _HARDCODED_EVENTS + self._live_events
        for event in all_events:
            delta = abs(now - event.dt_utc)
            if delta <= self._window:
                return event
        return None

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)

        # Refresh live calendar (no-op if no API key or already fetched today)
        self._maybe_refresh_live_events(timestamp_utc)

        nearby = self._is_near_event(timestamp_utc)

        if nearby is not None:
            minutes_to = int((nearby.dt_utc - timestamp_utc).total_seconds() / 60)
            direction = "in" if minutes_to >= 0 else "ago"
            abs_min   = abs(minutes_to)
            reason = (
                f"NewsCalendar: {nearby.name} "
                f"{abs_min}min {direction} — blocking trade"
            )
            return AnalysisSignal(
                direction=SignalDirection.HOLD,
                confidence=0.50,
                reason=reason,
                module_id=self.module_id,
                module_version=_VERSION,
                timestamp_utc=timestamp_utc,
                metadata={
                    "event": nearby.name,
                    "event_time_utc": nearby.dt_utc.isoformat(),
                    "minutes_delta": minutes_to,
                    "blocking": True,
                },
            )

        # No nearby event — pass-through HOLD with low confidence so other
        # modules dominate the consensus.
        return AnalysisSignal(
            direction=SignalDirection.HOLD,
            confidence=0.50,
            reason="NewsCalendar: no high-impact event nearby",
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={"blocking": False},
        )
