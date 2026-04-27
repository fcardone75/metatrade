"""Gap fill and CSV serialization for Massive aggregate rows."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from metatrade.market_data.providers.massive.timeframes import timeframe_minutes


def fill_micro_gaps(
    rows: list[tuple[int, float, float, float, float, float]],
    timeframe: str,
    gap_fill_max_bars: int,
) -> list[tuple[int, float, float, float, float, float]]:
    """Insert flat candles for short gaps; leave macro gaps untouched."""
    if gap_fill_max_bars <= 0 or len(rows) < 2:
        return rows

    step_ms = timeframe_minutes(timeframe) * 60_000
    filled: list[tuple[int, float, float, float, float, float]] = []
    for i, row in enumerate(rows):
        t_ms, o, h, low, c, v = row
        if i == 0:
            filled.append(row)
            continue
        prev_t, _, _, _, prev_c, _ = filled[-1]
        delta = (t_ms - prev_t) // step_ms
        if delta <= 1:
            filled.append(row)
            continue
        missing = int(delta - 1)
        if missing <= gap_fill_max_bars:
            for k in range(1, missing + 1):
                syn_t = prev_t + k * step_ms
                filled.append((syn_t, prev_c, prev_c, prev_c, prev_c, 0.0))
        filled.append(row)
    return filled


def write_ohlcv_csv(
    path: Path,
    rows: list[tuple[int, float, float, float, float, float]],
) -> None:
    """Write datetime,open,high,low,close,volume (UTC) for CsvCollector default map."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["datetime", "open", "high", "low", "close", "volume"])
        for t_ms, o, h, low, c, v in rows:
            dt = datetime.fromtimestamp(t_ms / 1000.0, tz=UTC)
            w.writerow(
                [
                    dt.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{o:.5f}",
                    f"{h:.5f}",
                    f"{low:.5f}",
                    f"{c:.5f}",
                    f"{v:.2f}",
                ]
            )
    tmp.replace(path)
