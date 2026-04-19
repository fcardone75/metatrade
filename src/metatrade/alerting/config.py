"""Alert configuration — loaded from environment variables with ALERT_ prefix."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from metatrade.core.config_base import BaseConfig


class AlertConfig(BaseConfig):
    """Configuration for Telegram alerts and command receiver.

    Override via environment variables with the ``ALERT_`` prefix.
    Example: ``ALERT_BOT_TOKEN=123:ABC``, ``ALERT_CHAT_ID=-100123456``.
    """

    # ── Credentials ───────────────────────────────────────────────────────────
    bot_token: str = Field(default="")
    chat_id: str = Field(default="")
    enabled: bool = Field(default=True)

    # ── Outbound settings ─────────────────────────────────────────────────────
    # Only fire trade-closed alerts when |pnl| >= this value (0 = always).
    min_pnl_alert: float = Field(default=0.0, ge=0.0)

    # HTTP request timeout for Telegram API calls (seconds).
    timeout_secs: int = Field(default=5, ge=1)

    # Throttle: skip sending the same alert type more than once per N seconds.
    # Prevents floods if a condition fires every bar (e.g. spread blocked).
    throttle_sec: int = Field(default=60, ge=0)

    # UTC hour (0–23) to send the automatic daily summary.  None = disabled.
    daily_summary_hour_utc: int | None = Field(default=21, ge=0, lt=24)

    # ── Per-category toggles ─────────────────────────────────────────────────
    notify_trade_opened: bool = Field(default=True)
    notify_trade_closed: bool = Field(default=True)
    notify_trade_sl_moved: bool = Field(default=False)   # opt-in (noisy)
    notify_session_start: bool = Field(default=True)
    notify_session_end: bool = Field(default=True)
    notify_daily_summary: bool = Field(default=True)
    notify_model_events: bool = Field(default=True)
    notify_accuracy_checkpoint: bool = Field(default=False)  # opt-in
    notify_spread_blocked: bool = Field(default=False)       # opt-in (noisy)
    notify_system_errors: bool = Field(default=True)
    notify_retrain_events: bool = Field(default=True)

    # ── Command receiver ─────────────────────────────────────────────────────
    # Whether to start the background polling thread for inbound commands.
    commands_enabled: bool = Field(default=True)

    # How often to poll Telegram for new messages (seconds).
    poll_interval_sec: int = Field(default=3, ge=1)

    model_config = {
        "env_prefix": "ALERT_",
        "env_file": str(Path(__file__).parent.parent.parent.parent / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "frozen": True,
    }
