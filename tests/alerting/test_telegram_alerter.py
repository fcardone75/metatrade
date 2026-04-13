"""Tests for TelegramAlerter and AlertConfig."""

from __future__ import annotations

import json
import urllib.error
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from metatrade.alerting.config import AlertConfig
from metatrade.alerting.telegram_alerter import TelegramAlerter


# ── AlertConfig tests ─────────────────────────────────────────────────────────


class TestAlertConfig:
    def test_defaults(self) -> None:
        cfg = AlertConfig()
        assert cfg.bot_token is None
        assert cfg.chat_id is None
        assert cfg.enabled is True
        assert cfg.min_pnl_alert == pytest.approx(0.0)
        assert cfg.timeout_secs == 5

    def test_custom_values(self) -> None:
        cfg = AlertConfig(
            bot_token="123:ABC",
            chat_id="-100123",
            enabled=False,
            min_pnl_alert=10.0,
            timeout_secs=3,
        )
        assert cfg.bot_token == "123:ABC"
        assert cfg.chat_id == "-100123"
        assert cfg.enabled is False
        assert cfg.min_pnl_alert == pytest.approx(10.0)
        assert cfg.timeout_secs == 3


# ── TelegramAlerter: _is_active ───────────────────────────────────────────────


class TestIsActive:
    def test_inactive_without_token(self) -> None:
        a = TelegramAlerter(AlertConfig(bot_token=None, chat_id="123"))
        assert not a._is_active()

    def test_inactive_without_chat_id(self) -> None:
        a = TelegramAlerter(AlertConfig(bot_token="tok", chat_id=None))
        assert not a._is_active()

    def test_inactive_when_disabled(self) -> None:
        a = TelegramAlerter(AlertConfig(bot_token="tok", chat_id="123", enabled=False))
        assert not a._is_active()

    def test_active_when_configured(self) -> None:
        a = TelegramAlerter(AlertConfig(bot_token="tok", chat_id="123", enabled=True))
        assert a._is_active()

    def test_default_config_is_inactive(self) -> None:
        a = TelegramAlerter()
        assert not a._is_active()


# ── TelegramAlerter: send (fire-and-forget) ───────────────────────────────────


class TestSend:
    def _active_alerter(self) -> TelegramAlerter:
        return TelegramAlerter(AlertConfig(bot_token="123:TESTTOKEN", chat_id="-100999"))

    def test_send_no_op_when_inactive(self) -> None:
        """Should not call urlopen when not configured."""
        alerter = TelegramAlerter()
        with patch("urllib.request.urlopen") as mock_open:
            alerter.send("hello")
            mock_open.assert_not_called()

    def test_send_calls_urlopen_when_active(self) -> None:
        alerter = self._active_alerter()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_ctx) as mock_open:
            alerter.send("Test message")
            mock_open.assert_called_once()

    def test_send_posts_correct_payload(self) -> None:
        alerter = self._active_alerter()
        captured_args: list = []

        def fake_urlopen(req, timeout=None):
            captured_args.append((req, timeout))
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            alerter.send("my message")

        assert len(captured_args) == 1
        req = captured_args[0][0]
        payload = json.loads(req.data.decode())
        assert payload["text"] == "my message"
        assert payload["chat_id"] == "-100999"

    def test_network_error_does_not_raise(self) -> None:
        """URLError should be silently logged, not raised."""
        alerter = self._active_alerter()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            alerter.send("will fail silently")  # must not raise

    def test_unexpected_error_does_not_raise(self) -> None:
        alerter = self._active_alerter()
        with patch("urllib.request.urlopen", side_effect=RuntimeError("unexpected")):
            alerter.send("safe")  # must not raise


# ── TelegramAlerter: alert helpers ────────────────────────────────────────────


class TestAlertHelpers:
    def _alerter(self, **cfg_kwargs) -> TelegramAlerter:
        return TelegramAlerter(AlertConfig(bot_token="tok", chat_id="cid", **cfg_kwargs))

    def _capture_message(self, alerter: TelegramAlerter, method, *args, **kwargs) -> str:
        """Call method on alerter, capturing the message text sent."""
        texts: list[str] = []

        def fake_post(text: str) -> None:
            texts.append(text)

        alerter._post = fake_post  # type: ignore[assignment]
        method(*args, **kwargs)
        return texts[-1] if texts else ""

    def test_alert_trade_opened_buy(self) -> None:
        alerter = self._alerter()
        msg = self._capture_message(
            alerter,
            alerter.alert_trade_opened,
            "EURUSD", "BUY", 0.10, 1.10500, 1.10200, 1.11100,
        )
        assert "🟢" in msg
        assert "BUY" in msg
        assert "EURUSD" in msg
        assert "1.10500" in msg

    def test_alert_trade_opened_sell(self) -> None:
        alerter = self._alerter()
        msg = self._capture_message(
            alerter,
            alerter.alert_trade_opened,
            "GBPUSD", "SELL", 0.05, 1.28000, 1.28500, None,
        )
        assert "🔴" in msg
        assert "SELL" in msg
        assert "—" in msg  # no TP

    def test_alert_trade_closed_profit(self) -> None:
        alerter = self._alerter()
        msg = self._capture_message(
            alerter, alerter.alert_trade_closed, "EURUSD", "BUY", 50.0, "tp"
        )
        assert "✅" in msg
        assert "+50.00" in msg
        assert "tp" in msg

    def test_alert_trade_closed_loss(self) -> None:
        alerter = self._alerter()
        msg = self._capture_message(
            alerter, alerter.alert_trade_closed, "EURUSD", "BUY", -20.0, "sl"
        )
        assert "❌" in msg
        assert "-20.00" in msg

    def test_alert_trade_closed_respects_min_pnl(self) -> None:
        """Trades below min_pnl_alert threshold should not fire."""
        alerter = self._alerter(min_pnl_alert=50.0)
        texts: list[str] = []
        alerter._post = lambda t: texts.append(t)  # type: ignore[assignment]
        alerter.alert_trade_closed("EURUSD", "BUY", 10.0, "tp")  # pnl=10 < 50
        assert len(texts) == 0

    def test_alert_drawdown(self) -> None:
        alerter = self._alerter()
        msg = self._capture_message(
            alerter, alerter.alert_drawdown, 9500.0, 10000.0, 0.05
        )
        assert "⚠️" in msg
        assert "5.0%" in msg

    def test_alert_kill_switch(self) -> None:
        alerter = self._alerter()
        msg = self._capture_message(
            alerter, alerter.alert_kill_switch, "SESSION_GATE", "Drawdown exceeded"
        )
        assert "🚨" in msg
        assert "SESSION_GATE" in msg

    def test_alert_error(self) -> None:
        alerter = self._alerter()
        msg = self._capture_message(
            alerter, alerter.alert_error, "BacktestRunner", "ATR compute failed"
        )
        assert "⛔" in msg
        assert "BacktestRunner" in msg

    def test_daily_summary(self) -> None:
        alerter = self._alerter()
        msg = self._capture_message(
            alerter, alerter.send_daily_summary, 15, 250.0, 0.60, 10250.0
        )
        assert "📊" in msg
        assert "15" in msg
        assert "60.0%" in msg
        assert "+250.00" in msg

    def test_decimal_amounts_accepted(self) -> None:
        """All float/Decimal parameters should be accepted without error."""
        alerter = self._alerter()
        alerter._post = lambda t: None  # type: ignore[assignment]
        alerter.alert_trade_opened(
            "EURUSD", "BUY",
            Decimal("0.10"), Decimal("1.10500"), Decimal("1.10200"), Decimal("1.11100"),
        )
        alerter.alert_trade_closed("EURUSD", "BUY", Decimal("50.00"), "tp")
        alerter.alert_drawdown(Decimal("9500"), Decimal("10000"), 0.05)
        alerter.send_daily_summary(5, Decimal("100.00"), 0.6, Decimal("10100.00"))
