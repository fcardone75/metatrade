"""Run the MetaTrade dashboard API."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metatrade.core.log import configure_logging
from metatrade.observability.api import create_app
from metatrade.observability.config import ObservabilityConfig


def main() -> None:
    configure_logging()
    cfg = ObservabilityConfig.load()
    uvicorn.run(create_app(), host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
