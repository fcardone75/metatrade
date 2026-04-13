"""Telegram alerter — fire-and-forget operational notifications.

Sends messages to a Telegram bot using the Bot API.  All network errors are
logged (not raised), so the trading system is never blocked by an alert failure.

No external dependencies — uses only Python stdlib ``urllib.request``.

Usage:
    cfg = AlertConfig(bot_token="123:ABC", chat_id="-1001234567890")
    alerter = TelegramAlerter(cfg)
    alerter.alert_trade_opened("EURUSD", "BUY", 0.1, 1.1050, 1.1020, 1.1110)
    alerter.alert_trade_closed("EURUSD", "BUY", pnl=30.0, exit_reason="tp")
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from decimal import Decimal

from metatrade.alerting.config import AlertConfig

log = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramAlerter:
    """Send operational alerts to a Telegram channel.

    Args:
        config: AlertConfig with bot_token, chat_id, and options.
    """

    def __init__(self, config: AlertConfig | None = None) -> None:
        self._cfg = config or AlertConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def send(self, message: str) -> None:
        """Send a raw text message.  Silently drops if not configured/enabled."""
        if not self._is_active():
            return
        self._post(message)

    def alert_trade_opened(
        self,
        symbol: str,
        side: str,
        lot_size: float | Decimal,
        entry: float | Decimal,
        sl: float | Decimal,
        tp: float | Decimal | None = None,
    ) -> None:
        """Alert when a new trade is opened."""
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        tp_str = f"{float(tp):.5f}" if tp is not None else "—"
        msg = (
            f"{emoji} Trade opened: {symbol} {side.upper()}\n"
            f"  Lots: {float(lot_size):.2f}\n"
            f"  Entry: {float(entry):.5f}\n"
            f"  SL:    {float(sl):.5f}\n"
            f"  TP:    {tp_str}"
        )
        self.send(msg)

    def alert_trade_closed(
        self,
        symbol: str,
        side: str,
        pnl: float | Decimal,
        exit_reason: str,
    ) -> None:
        """Alert when a trade is closed (subject to min_pnl_alert threshold)."""
        if abs(float(pnl)) < self._cfg.min_pnl_alert:
            return
        pnl_f = float(pnl)
        emoji = "✅" if pnl_f >= 0 else "❌"
        sign = "+" if pnl_f >= 0 else ""
        msg = (
            f"{emoji} Trade closed: {symbol} {side.upper()}\n"
            f"  PnL:    {sign}{pnl_f:.2f} USD\n"
            f"  Reason: {exit_reason}"
        )
        self.send(msg)

    def alert_drawdown(
        self,
        equity: float | Decimal,
        peak: float | Decimal,
        dd_pct: float,
    ) -> None:
        """Alert when drawdown exceeds a threshold."""
        msg = (
            f"⚠️ Drawdown alert\n"
            f"  Equity: {float(equity):.2f} USD\n"
            f"  Peak:   {float(peak):.2f} USD\n"
            f"  DD:     {dd_pct * 100:.1f}%"
        )
        self.send(msg)

    def alert_kill_switch(self, level: str, reason: str) -> None:
        """Alert when the kill switch is activated."""
        msg = f"🚨 Kill switch activated — level {level}\n  Reason: {reason}"
        self.send(msg)

    def alert_error(self, source: str, error: str) -> None:
        """Alert on an unexpected system error."""
        msg = f"⛔ Error in {source}:\n  {error}"
        self.send(msg)

    def send_daily_summary(
        self,
        n_trades: int,
        total_pnl: float | Decimal,
        win_rate: float,
        balance: float | Decimal,
    ) -> None:
        """Send end-of-day summary."""
        pnl_f = float(total_pnl)
        sign = "+" if pnl_f >= 0 else ""
        msg = (
            f"📊 Daily summary\n"
            f"  Trades:   {n_trades}\n"
            f"  Win rate: {win_rate * 100:.1f}%\n"
            f"  PnL:      {sign}{pnl_f:.2f} USD\n"
            f"  Balance:  {float(balance):.2f} USD"
        )
        self.send(msg)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _is_active(self) -> bool:
        """Return True only when fully configured and enabled."""
        return bool(
            self._cfg.enabled
            and self._cfg.bot_token
            and self._cfg.chat_id
        )

    def _post(self, text: str) -> None:
        """HTTP POST to Telegram sendMessage — fire-and-forget."""
        url = _TELEGRAM_API.format(token=self._cfg.bot_token)
        payload = json.dumps(
            {"chat_id": self._cfg.chat_id, "text": text, "parse_mode": "HTML"}
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.timeout_secs):
                pass
        except urllib.error.URLError as exc:
            log.warning("Telegram alert failed (network): %s", exc)
        except Exception as exc:
            log.warning("Telegram alert failed: %s", exc)
