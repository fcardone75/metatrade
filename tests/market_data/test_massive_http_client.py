"""HTTP client tests with mocked urlopen."""

from __future__ import annotations

import json
from io import BytesIO
import pytest

from metatrade.market_data.providers.massive.http_client import MassiveHttpError, ThrottledMassiveClient
from metatrade.market_data.providers.massive.settings import MassiveSettings


def test_get_json_success() -> None:
    settings = MassiveSettings(api_key="secret", min_interval_sec=0.0)
    payload = {"status": "OK", "results": []}

    class Resp:
        def __init__(self) -> None:
            self._raw = json.dumps(payload).encode()

        def read(self) -> bytes:
            return self._raw

        def getcode(self) -> int:
            return 200

        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

    def fake_urlopen(req: object, timeout: float = 0) -> Resp:
        return Resp()

    client = ThrottledMassiveClient(settings, urlopen=fake_urlopen)
    out = client.get_json("/v3/reference/tickers", {"limit": "1"})
    assert out["status"] == "OK"


def test_get_json_http_error() -> None:
    import urllib.error

    settings = MassiveSettings(api_key="secret", min_interval_sec=0.0)

    def boom(req: object, timeout: float = 0) -> object:
        raise urllib.error.HTTPError("url", 401, "Unauthorized", hdrs=None, fp=BytesIO(b'{"error":"nope"}'))

    client = ThrottledMassiveClient(settings, urlopen=boom)
    with pytest.raises(MassiveHttpError) as ei:
        client.get_json("/v1/test")
    assert ei.value.status == 401
