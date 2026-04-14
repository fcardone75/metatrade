"""Configuration for the observability API/dashboard."""

from __future__ import annotations

from metatrade.core.config_base import BaseConfig


class ObservabilityConfig(BaseConfig):
    host: str = "0.0.0.0"
    port: int = 8080
    title: str = "MetaTrade Control Room"
    refresh_seconds: int = 10

    model_config = {
        "env_prefix": "OBSERVABILITY_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "frozen": True,
    }
