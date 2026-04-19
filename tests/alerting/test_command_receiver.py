"""Tests for TelegramCommandReceiver: chat_id filtering, command parsing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from metatrade.alerting.command_handler import CommandHandler
from metatrade.alerting.command_receiver import TelegramCommandReceiver
from metatrade.alerting.config import AlertConfig


def _cfg(**kwargs) -> AlertConfig:
    defaults = {
        "bot_token": "123:TEST",
        "chat_id": "-1001234",
        "commands_enabled": True,
        "poll_interval_sec": 1,
        "timeout_secs": 5,
    }
    defaults.update(kwargs)
    return AlertConfig.model_construct(**defaults)


def _update(text: str, chat_id: str = "-1001234", update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "chat": {"id": int(chat_id)},
            "text": text,
        },
    }


class TestChatIdFiltering:
    def test_authorized_chat_dispatches_command(self):
        handler = MagicMock()
        handler.dispatch.return_value = "ok"
        cfg = _cfg()
        receiver = TelegramCommandReceiver(config=cfg, handler=handler)

        with patch.object(receiver, "_send_reply") as mock_reply:
            receiver._process_update(_update("/status", chat_id="-1001234"))

        handler.dispatch.assert_called_once_with("/status", "")
        mock_reply.assert_called_once()

    def test_unauthorized_chat_is_ignored(self):
        handler = MagicMock()
        cfg = _cfg()
        receiver = TelegramCommandReceiver(config=cfg, handler=handler)

        receiver._process_update(_update("/status", chat_id="-9999999"))
        handler.dispatch.assert_not_called()

    def test_no_text_message_is_skipped(self):
        handler = MagicMock()
        receiver = TelegramCommandReceiver(config=_cfg(), handler=handler)
        receiver._process_update({"update_id": 1, "message": {"chat": {"id": -1001234}}})
        handler.dispatch.assert_not_called()

    def test_non_command_message_is_skipped(self):
        handler = MagicMock()
        receiver = TelegramCommandReceiver(config=_cfg(), handler=handler)
        receiver._process_update(_update("hello world"))
        handler.dispatch.assert_not_called()


class TestCommandParsing:
    def test_command_with_args(self):
        handler = MagicMock()
        handler.dispatch.return_value = ""
        receiver = TelegramCommandReceiver(config=_cfg(), handler=handler)

        with patch.object(receiver, "_send_reply"):
            receiver._process_update(_update("/pause 10 minutes"))

        handler.dispatch.assert_called_once_with("/pause", "10 minutes")

    def test_bot_suffix_stripped(self):
        handler = MagicMock()
        handler.dispatch.return_value = ""
        receiver = TelegramCommandReceiver(config=_cfg(), handler=handler)

        with patch.object(receiver, "_send_reply"):
            receiver._process_update(_update("/status@MyBotName"))

        handler.dispatch.assert_called_once_with("/status", "")

    def test_stop_confirm_parsed_as_args(self):
        handler = MagicMock()
        handler.dispatch.return_value = "stopped"
        receiver = TelegramCommandReceiver(config=_cfg(), handler=handler)

        with patch.object(receiver, "_send_reply"):
            receiver._process_update(_update("/stop confirm"))

        handler.dispatch.assert_called_once_with("/stop", "confirm")


class TestOffsetAdvancement:
    def test_offset_advances_after_update(self):
        handler = MagicMock()
        handler.dispatch.return_value = ""
        receiver = TelegramCommandReceiver(config=_cfg(), handler=handler)

        data = {
            "ok": True,
            "result": [_update("/status", update_id=42)],
        }
        with patch.object(receiver, "_get_updates", return_value=data):
            with patch.object(receiver, "_send_reply"):
                receiver._poll_once()

        assert receiver._offset == 43

    def test_empty_result_does_not_crash(self):
        receiver = TelegramCommandReceiver(config=_cfg())
        data = {"ok": True, "result": []}
        with patch.object(receiver, "_get_updates", return_value=data):
            receiver._poll_once()


class TestStartStop:
    def test_start_disabled_when_commands_enabled_false(self):
        cfg = _cfg(commands_enabled=False)
        receiver = TelegramCommandReceiver(config=cfg)
        receiver.start()
        assert not receiver.is_running

    def test_start_disabled_when_no_token(self):
        cfg = _cfg(bot_token="")
        receiver = TelegramCommandReceiver(config=cfg)
        receiver.start()
        assert not receiver.is_running

    def test_no_reply_when_callback_returns_empty(self):
        handler = MagicMock()
        handler.dispatch.return_value = ""
        receiver = TelegramCommandReceiver(config=_cfg(), handler=handler)

        with patch.object(receiver, "_send_reply") as mock_reply:
            receiver._process_update(_update("/status"))

        mock_reply.assert_not_called()
