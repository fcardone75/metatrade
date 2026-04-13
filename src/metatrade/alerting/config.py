"""Alert configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AlertConfig:
    """Configuration for the Telegram alerter.

    Attributes:
        bot_token:     Telegram Bot API token (from @BotFather).
                       None or empty → alerter is silently disabled.
        chat_id:       Telegram chat/channel ID to send messages to.
                       None or empty → alerter is silently disabled.
        enabled:       Master switch.  False → all sends are silently dropped.
        min_pnl_alert: Only fire trade-closed alerts when |pnl| >= this value.
                       0.0 = alert on every closed trade.
        timeout_secs:  HTTP request timeout for Telegram API calls.
    """

    bot_token: str | None = None
    chat_id: str | None = None
    enabled: bool = True
    min_pnl_alert: float = 0.0
    timeout_secs: int = 5
