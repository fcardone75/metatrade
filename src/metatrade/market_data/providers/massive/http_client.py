"""HTTP helpers with throttling for Massive REST API."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from metatrade.market_data.providers.massive.settings import MassiveSettings


class MassiveHttpError(RuntimeError):
    """Non-success HTTP response from Massive API."""

    def __init__(self, status: int, body: str, url: str) -> None:
        self.status = status
        self.body = body
        self.url = url
        super().__init__(f"Massive HTTP {status} for {url}: {body[:500]}")


def append_query(url: str, params: dict[str, str]) -> str:
    """Merge params into URL query string."""
    parts = urllib.parse.urlparse(url)
    q = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
    for k, v in params.items():
        q[k] = [v]
    flat = [(k, vv) for k, vals in q.items() for vv in vals]
    new_query = urllib.parse.urlencode(flat)
    return urllib.parse.urlunparse(parts._replace(query=new_query))


class ThrottledMassiveClient:
    """GET JSON with minimum spacing between requests (rate limit tier)."""

    def __init__(
        self,
        settings: MassiveSettings,
        *,
        urlopen: Callable[..., Any] | None = None,
    ) -> None:
        self._s = settings
        self._urlopen = urlopen or urllib.request.urlopen
        self._last_request_mono: float = 0.0

    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self._s.min_interval_sec - (now - self._last_request_mono)
        if wait > 0:
            time.sleep(wait)
        self._last_request_mono = time.monotonic()

    def get_json(
        self, path_or_url: str, extra_params: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """GET and parse JSON. path_or_url may be absolute (next_url) or relative path."""
        url = path_or_url
        if not url.startswith("http"):
            url = f"{self._s.base_url}{path_or_url}"
        url = append_query(url, {"apiKey": self._s.api_key})
        if extra_params:
            url = append_query(url, extra_params)

        self._throttle()
        req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
        try:
            with self._urlopen(req, timeout=self._s.timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                status = getattr(resp, "status", None) or resp.getcode()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise MassiveHttpError(e.code, body, url) from e

        if status and status >= 400:
            raise MassiveHttpError(int(status), raw, url)

        return json.loads(raw)
