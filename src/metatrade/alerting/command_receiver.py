"""TelegramCommandReceiver — background daemon thread for inbound commands.

Polls Telegram getUpdates every ``poll_interval_sec`` seconds.  Filters by
chat_id — only messages from the configured owner chat are processed.
Everything else is silently ignored.

Threading model: runs in a daemon thread (auto-killed when the main process
exits) so it never blocks the runner loop.

Security guarantee: chat_id filtering happens before any dispatch, so a
third-party Telegram user cannot trigger runner callbacks.
"""

from __future__ import annotations

import json
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request

# Skip SSL verification for Telegram API — required on Windows hosts where a
# corporate proxy or AV product inserts a self-signed cert into the TLS chain.
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

from metatrade.alerting.command_handler import CommandHandler
from metatrade.alerting.config import AlertConfig
from metatrade.core.log import get_logger

log = get_logger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramCommandReceiver:
    """Background Telegram long-poll receiver.

    Args:
        config:  AlertConfig (bot_token, chat_id, commands_enabled,
                 poll_interval_sec, timeout_secs).
        handler: CommandHandler that routes commands to callbacks.
    """

    def __init__(
        self,
        config: AlertConfig | None = None,
        handler: CommandHandler | None = None,
    ) -> None:
        self._cfg = config or AlertConfig()
        self._handler = handler or CommandHandler()
        self._offset: int = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background polling thread (idempotent)."""
        if not self._cfg.commands_enabled:
            log.info("telegram_command_receiver_disabled")
            return
        if not self._cfg.bot_token or not self._cfg.chat_id:
            log.info("telegram_command_receiver_no_credentials")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="TelegramCommandReceiver",
        )
        self._thread.start()
        log.info(
            "telegram_command_receiver_started",
            poll_interval_sec=self._cfg.poll_interval_sec,
        )

    def stop(self) -> None:
        """Signal the polling thread to exit at the next poll interval."""
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        """True if the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ── Polling loop ──────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception as exc:
                log.warning("telegram_command_receiver_poll_error", error=str(exc))
            self._stop_event.wait(timeout=self._cfg.poll_interval_sec)

    def _poll_once(self) -> None:
        data = self._get_updates()
        if data is None or not data.get("ok"):
            return
        for update in data.get("result", []):
            update_id = update.get("update_id", 0)
            self._offset = max(self._offset, update_id + 1)
            try:
                self._process_update(update)
            except Exception as exc:
                log.warning(
                    "telegram_command_receiver_update_error",
                    update_id=update_id,
                    error=str(exc),
                )

    def _process_update(self, update: dict) -> None:
        message = update.get("message") or update.get("edited_message")
        if not message:
            return

        chat_id = str(message.get("chat", {}).get("id", ""))
        if chat_id != str(self._cfg.chat_id):
            log.debug(
                "telegram_command_receiver_unauthorized",
                chat_id=chat_id,
            )
            return

        text = (message.get("text") or "").strip()
        if not text.startswith("/"):
            return

        # "/pause 5 min" → command="/pause", args="5 min"
        parts = text.split(None, 1)
        command = parts[0].lower()
        # Strip bot suffix: /status@MyBotName → /status
        if "@" in command:
            command = command.split("@")[0]
        args = parts[1] if len(parts) > 1 else ""

        log.info(
            "telegram_command_received",
            command=command,
            chat_id=chat_id,
        )

        reply = self._handler.dispatch(command, args)
        if reply:
            self._send_reply(chat_id, reply)

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _get_updates(self) -> dict | None:
        params = {
            "offset": self._offset,
            "timeout": 0,
            "limit": 100,
        }
        url = (
            _TELEGRAM_API.format(token=self._cfg.bot_token, method="getUpdates")
            + "?"
            + urllib.parse.urlencode(params)
        )
        try:
            with urllib.request.urlopen(url, timeout=self._cfg.timeout_secs, context=_SSL_CTX) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            log.warning("telegram_get_updates_failed", error=str(exc))
            return None

    def _send_reply(self, chat_id: str, text: str) -> None:
        url = _TELEGRAM_API.format(token=self._cfg.bot_token, method="sendMessage")
        payload = json.dumps({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.timeout_secs, context=_SSL_CTX):
                pass
        except Exception as exc:
            log.warning(
                "telegram_send_reply_failed",
                chat_id=chat_id,
                error=str(exc),
            )
