"""Massive.com REST settings from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class MassiveSettings:
    """Configuration for Massive market data API (Polygon-compatible REST)."""

    api_key: str
    base_url: str = "https://api.massive.com"
    timeout_sec: float = 60.0
    min_interval_sec: float = 13.0  # Currencies Basic: 5 req/min → ~12s + margin
    max_aggs_limit: int = 50_000
    gap_fill_max_bars: int = 5
    max_history_days: int = 730  # Currencies Basic tier

    @classmethod
    def from_env(cls) -> MassiveSettings:
        root = Path(__file__).resolve().parents[5]
        load_dotenv(root / ".env", override=False)

        key = (os.environ.get("MASSIVE_API_KEY") or "").strip()
        if not key:
            msg = "MASSIVE_API_KEY non impostata (richiesta per il download Massive)"
            raise OSError(msg)
        base = (os.environ.get("MASSIVE_BASE_URL") or "https://api.massive.com").rstrip("/")
        timeout = float(os.environ.get("MASSIVE_HTTP_TIMEOUT_SEC") or "60")
        interval = float(os.environ.get("MASSIVE_MIN_REQUEST_INTERVAL_SEC") or "13")
        limit = int(os.environ.get("MASSIVE_AGGS_LIMIT") or "50000")
        gap = int(os.environ.get("MASSIVE_GAP_FILL_MAX_BARS") or "5")
        hist = int(os.environ.get("MASSIVE_MAX_HISTORY_DAYS") or "730")
        return cls(
            api_key=key,
            base_url=base,
            timeout_sec=timeout,
            min_interval_sec=interval,
            max_aggs_limit=min(50_000, max(1000, limit)),
            gap_fill_max_bars=max(0, gap),
            max_history_days=max(1, hist),
        )
