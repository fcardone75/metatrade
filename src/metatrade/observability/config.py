"""Configuration for the observability API/dashboard."""

from __future__ import annotations

from pathlib import Path

from metatrade.core.config_base import BaseConfig


class ObservabilityConfig(BaseConfig):
    host: str = "0.0.0.0"
    port: int = 8080
    title: str = "MetaTrade Control Room"
    refresh_seconds: int = 10

    # HTTP Basic Auth — set both to enable password protection.
    # Leave either empty to disable auth (useful for local development).
    # Set via OBSERVABILITY_DASHBOARD_USER / OBSERVABILITY_DASHBOARD_PASSWORD.
    dashboard_user: str = ""
    dashboard_password: str = ""

    model_config = {
        "env_prefix": "OBSERVABILITY_",
        "env_file": str(Path(__file__).parent.parent.parent.parent / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "frozen": True,
    }
