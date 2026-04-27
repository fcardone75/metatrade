"""Telegram alerter — fire-and-forget operational notifications.

Sends messages to a Telegram bot using the Bot API.  All network errors are
logged (not raised), so the trading system is never blocked by an alert failure.

Throttle: the same alert type is sent at most once every ``throttle_sec``
seconds (configurable).  This prevents floods when a condition fires every bar.

No external dependencies — uses only Python stdlib ``urllib.request``.
"""

from __future__ import annotations

import logging
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from decimal import Decimal

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

from metatrade.alerting.config import AlertConfig

log = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


def _sign(v: float) -> str:
    return "+" if v >= 0 else ""


def _pnl_emoji(v: float) -> str:
    return "✅" if v >= 0 else "❌"


class TelegramAlerter:
    """Send operational alerts to a Telegram channel.

    Args:
        config: AlertConfig with bot_token, chat_id, and options.
    """

    def __init__(self, config: AlertConfig | None = None) -> None:
        self._cfg = config or AlertConfig()
        # throttle: last send time per alert type key
        self._last_sent: dict[str, float] = {}

    # ── Raw send ──────────────────────────────────────────────────────────────

    def send(self, message: str, *, throttle_key: str | None = None) -> None:
        """Send a raw text message.  Silently drops if not configured/enabled.

        Args:
            message:      Text to send (HTML allowed).
            throttle_key: If set, the message is skipped if the same key was
                          sent within the last ``throttle_sec`` seconds.
        """
        if not self._is_active():
            return
        if throttle_key is not None and self._cfg.throttle_sec > 0:
            last = self._last_sent.get(throttle_key, 0.0)
            if time.monotonic() - last < self._cfg.throttle_sec:
                return
            self._last_sent[throttle_key] = time.monotonic()
        self._post(message)

    # ── Trade lifecycle ───────────────────────────────────────────────────────

    def alert_trade_opened(
        self,
        symbol: str,
        side: str,
        lot_size: float | Decimal,
        entry: float | Decimal,
        sl: float | Decimal,
        tp: float | Decimal | None = None,
        atr: float | None = None,
        model_version: str | None = None,
        model_holdout: float | None = None,
    ) -> None:
        """Alert when a new trade is opened."""
        if not self._cfg.notify_trade_opened:
            return
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        entry_f = float(entry)
        sl_f = float(sl)
        sl_pips = abs(entry_f - sl_f) * 10000
        tp_str = "—"
        rr_str = ""
        if tp is not None:
            tp_f = float(tp)
            tp_pips = abs(tp_f - entry_f) * 10000
            tp_str = f"{tp_f:.5f}  (+{tp_pips:.1f} pip)"
            if sl_pips > 0:
                rr_str = f"  R:R {tp_pips/sl_pips:.1f}"
        atr_str = f"\n  ATR(14): {atr:.5f}" if atr is not None else ""
        model_str = ""
        if model_version:
            h_str = f"  holdout {model_holdout:.1%}" if model_holdout else ""
            model_str = f"\n  Modello: {model_version}{h_str}"
        msg = (
            f"{emoji} <b>{side.upper()} aperto — {symbol}</b>\n"
            f"  Lots:    {float(lot_size):.2f}\n"
            f"  Entry:   {entry_f:.5f}\n"
            f"  SL:      {sl_f:.5f}  (−{sl_pips:.1f} pip)\n"
            f"  TP:      {tp_str}{rr_str}{atr_str}{model_str}"
        )
        self.send(msg)

    def alert_trade_closed(
        self,
        symbol: str,
        side: str,
        pnl: float | Decimal,
        exit_reason: str,
        entry: float | Decimal | None = None,
        exit_price: float | Decimal | None = None,
        pnl_pips: float | None = None,
        duration_bars: int | None = None,
        balance_after: float | Decimal | None = None,
        session_pnl: float | Decimal | None = None,
    ) -> None:
        """Alert when a trade is closed."""
        if not self._cfg.notify_trade_closed:
            return
        pnl_f = float(pnl)
        if abs(pnl_f) < self._cfg.min_pnl_alert:
            return
        emoji = _pnl_emoji(pnl_f)
        sign = _sign(pnl_f)
        prices_str = ""
        if entry is not None and exit_price is not None:
            prices_str = f"\n  Entry:   {float(entry):.5f}  →  Exit: {float(exit_price):.5f}"
        pips_str = f"  ({sign}{pnl_pips:.1f} pip)" if pnl_pips is not None else ""
        dur_str = f"\n  Durata:  {duration_bars} barre" if duration_bars is not None else ""
        bal_str = ""
        if balance_after is not None:
            session_str = (
                f"  (sessione: {_sign(float(session_pnl))}{float(session_pnl):.2f})"
                if session_pnl is not None else ""
            )
            bal_str = f"\n  Balance: ${float(balance_after):.2f}{session_str}"
        msg = (
            f"{emoji} <b>{side.upper()} chiuso — {symbol}</b>  "
            f"{sign}{pnl_f:.2f} USD{pips_str}{prices_str}{dur_str}\n"
            f"  Motivo:  {exit_reason}{bal_str}"
        )
        self.send(msg)

    def alert_sl_moved(
        self,
        symbol: str,
        side: str,
        new_sl: float | Decimal,
        protected_pips: float,
        reason: str = "trailing_stop",
    ) -> None:
        """Alert when SL is moved (breakeven or trailing)."""
        if not self._cfg.notify_trade_sl_moved:
            return
        msg = (
            f"↕️ SL aggiornato — {symbol} {side.upper()}\n"
            f"  Nuovo SL: {float(new_sl):.5f}\n"
            f"  Profitto protetto: {protected_pips:+.1f} pip\n"
            f"  Motivo: {reason}"
        )
        self.send(msg, throttle_key=f"sl_moved_{symbol}")

    # ── Sessione ─────────────────────────────────────────────────────────────

    def alert_session_start(
        self,
        mode: str,
        symbol: str,
        timeframe: str,
        balance: float | Decimal,
        model_version: str | None = None,
        session_filter: str | None = None,
    ) -> None:
        """Alert when a runner session starts."""
        if not self._cfg.notify_session_start:
            return
        mode_emoji = {"live": "💰", "paper": "📋", "backtest": "📊"}.get(mode.lower(), "🤖")
        model_str = f"\n  Modello:  {model_version}" if model_version else ""
        sess_str = f"\n  Sessione: {session_filter}" if session_filter else ""
        msg = (
            f"{mode_emoji} <b>MetaTrade avviato</b>  [{mode.upper()}]\n"
            f"  Symbol:   {symbol} {timeframe}\n"
            f"  Balance:  ${float(balance):.2f}{model_str}{sess_str}"
        )
        self.send(msg)

    def alert_session_end(
        self,
        mode: str,
        symbol: str,
        n_trades: int,
        total_pnl: float | Decimal,
        win_rate: float | None,
        balance: float | Decimal,
        duration_min: int | None = None,
    ) -> None:
        """Alert when a runner session ends."""
        if not self._cfg.notify_session_end:
            return
        pnl_f = float(total_pnl)
        wr_str = f"  Win rate: {win_rate:.1%}\n" if win_rate is not None else ""
        dur_str = f"  Durata:   {duration_min} min\n" if duration_min is not None else ""
        msg = (
            f"🏁 <b>Sessione terminata</b>  [{mode.upper()}]  {symbol}\n"
            f"  Trade:    {n_trades}\n"
            f"{wr_str}"
            f"  PnL:      {_sign(pnl_f)}{pnl_f:.2f} USD\n"
            f"{dur_str}"
            f"  Balance:  ${float(balance):.2f}"
        )
        self.send(msg)

    def alert_daily_summary(
        self,
        symbol: str,
        timeframe: str,
        n_trades: int,
        total_pnl: float | Decimal,
        pnl_pips: float | None,
        win_rate: float,
        max_dd: float | Decimal,
        balance: float | Decimal,
        model_version: str | None = None,
        live_accuracy: float | None = None,
        live_samples: int | None = None,
    ) -> None:
        """Send end-of-day trading summary."""
        if not self._cfg.notify_daily_summary:
            return
        pnl_f = float(total_pnl)
        pips_str = f"  PnL pip:  {_sign(pnl_pips or 0)}{pnl_pips:.1f} pip\n" if pnl_pips is not None else ""
        model_str = ""
        if model_version:
            acc_str = f"  live={live_accuracy:.1%} ({live_samples} camp.)" if live_accuracy is not None else ""
            model_str = f"\n  Modello:  {model_version}{acc_str}"
        msg = (
            f"📊 <b>Riepilogo giornaliero</b> — {symbol} {timeframe}\n"
            f"  Trade:    {n_trades}  (win rate {win_rate:.1%})\n"
            f"  PnL:      {_sign(pnl_f)}{pnl_f:.2f} USD\n"
            f"{pips_str}"
            f"  Max DD:   −${float(max_dd):.2f}\n"
            f"  Balance:  ${float(balance):.2f}{model_str}"
        )
        self.send(msg)

    # ── Risk & sistema ────────────────────────────────────────────────────────

    def alert_drawdown(
        self,
        equity: float | Decimal,
        peak: float | Decimal,
        dd_pct: float,
        recovery_mode: bool = False,
    ) -> None:
        """Alert when drawdown exceeds a threshold."""
        recovery_str = "\n  ⚡ Modalità recovery attiva (lot ridotti)" if recovery_mode else ""
        msg = (
            f"⚠️ <b>Drawdown alert</b>\n"
            f"  Equity: ${float(equity):.2f}\n"
            f"  Peak:   ${float(peak):.2f}\n"
            f"  DD:     {dd_pct * 100:.1f}%{recovery_str}"
        )
        self.send(msg, throttle_key="drawdown")

    def alert_kill_switch(self, level: str, reason: str, activated_by: str = "system") -> None:
        """Alert when the kill switch is activated."""
        msg = (
            f"🚨 <b>Kill switch attivato — livello {level}</b>\n"
            f"  Motivo:  {reason}\n"
            f"  Da:      {activated_by}"
        )
        self.send(msg)

    def alert_daily_loss_limit(
        self,
        loss: float | Decimal,
        limit: float | Decimal,
        balance: float | Decimal,
    ) -> None:
        """Alert when the daily loss limit is reached."""
        msg = (
            f"🛑 <b>Limite perdita giornaliero raggiunto</b>\n"
            f"  Perdita: −${float(loss):.2f}\n"
            f"  Soglia:  −${float(limit):.2f}\n"
            f"  Balance: ${float(balance):.2f}\n"
            f"  Nuove entry bloccate fino a domani."
        )
        self.send(msg)

    def alert_error(self, source: str, error: str) -> None:
        """Alert on an unexpected system error."""
        if not self._cfg.notify_system_errors:
            return
        truncated = error[:400] + "…" if len(error) > 400 else error
        msg = f"⛔ <b>Errore in {source}</b>\n<code>{truncated}</code>"
        self.send(msg, throttle_key=f"error_{source}")

    # ── Modello ML ────────────────────────────────────────────────────────────

    def alert_model_switched(
        self,
        symbol: str,
        old_version: str,
        new_version: str,
        old_holdout: float | None,
        new_holdout: float,
        live_accuracy: float | None,
        reason: str,
        n_positions_waited: int = 0,
    ) -> None:
        """Alert when the active model is replaced by a better candidate."""
        if not self._cfg.notify_model_events:
            return
        live_str = f"{live_accuracy:.1%}" if live_accuracy is not None else "n/a"
        old_h_str = f"holdout {old_holdout:.1%}" if old_holdout is not None else "holdout n/a"
        wait_str = f"\n  Atteso:   chiusura {n_positions_waited} pos." if n_positions_waited else ""
        msg = (
            f"🔄 <b>Modello aggiornato</b> — {symbol}\n"
            f"  Precedente: {old_version}  ({old_h_str}, live {live_str})\n"
            f"  Nuovo:      {new_version}  (holdout {new_holdout:.1%})\n"
            f"  Trigger:    {reason}{wait_str}"
        )
        self.send(msg)

    def alert_model_degraded(
        self,
        symbol: str,
        model_version: str,
        live_accuracy: float,
        threshold: float,
        n_evaluated: int,
    ) -> None:
        """Alert when live accuracy drops below the warning threshold."""
        if not self._cfg.notify_model_events:
            return
        msg = (
            f"⚠️ <b>Accuracy modello degradata</b> — {symbol}\n"
            f"  Modello:  {model_version}\n"
            f"  Live acc: {live_accuracy:.1%}  (soglia: {threshold:.1%})\n"
            f"  Campioni: {n_evaluated}  ✅ affidabile"
        )
        self.send(msg, throttle_key=f"model_degraded_{symbol}")

    def alert_accuracy_checkpoint(
        self,
        symbol: str,
        model_version: str,
        n_evaluated: int,
        accuracy: float,
        is_reliable: bool,
    ) -> None:
        """Periodic accuracy checkpoint (opt-in)."""
        if not self._cfg.notify_accuracy_checkpoint:
            return
        reliable_str = "✅ affidabile" if is_reliable else "⏳ in accumulo"
        msg = (
            f"🧠 Accuracy checkpoint — {symbol}\n"
            f"  Modello:  {model_version}\n"
            f"  Campioni: {n_evaluated}  {reliable_str}\n"
            f"  Accuracy: {accuracy:.1%}"
        )
        self.send(msg, throttle_key=f"acc_checkpoint_{symbol}_{n_evaluated // 50}")

    def alert_retrain_started(
        self,
        symbol: str,
        timeframe: str,
        trigger: str,
        bars_or_hours: int,
    ) -> None:
        """Alert when background retraining begins."""
        if not self._cfg.notify_retrain_events:
            return
        unit = "barre" if trigger == "bars" else "ore"
        msg = (
            f"🔁 <b>Retraining avviato</b> — {symbol} {timeframe}\n"
            f"  Trigger: ogni {bars_or_hours} {unit}"
        )
        self.send(msg, throttle_key=f"retrain_started_{symbol}")

    def alert_retrain_complete(
        self,
        symbol: str,
        timeframe: str,
        new_version: str,
        holdout_accuracy: float,
        is_candidate: bool,
        reason: str,
    ) -> None:
        """Alert when a background retraining run finishes."""
        if not self._cfg.notify_retrain_events:
            return
        candidate_str = "✅ Candidato per switch" if is_candidate else "ℹ️ Non candidato"
        msg = (
            f"🤖 <b>Retraining completato</b> — {symbol} {timeframe}\n"
            f"  Versione:  {new_version}\n"
            f"  Holdout:   {holdout_accuracy:.1%}\n"
            f"  {candidate_str}\n"
            f"  Motivo: {reason}"
        )
        self.send(msg)

    def alert_retrain_failed(
        self,
        symbol: str,
        timeframe: str,
        exit_code: int,
    ) -> None:
        """Alert when the retraining subprocess exits with an error."""
        if not self._cfg.notify_retrain_events:
            return
        msg = (
            f"❌ <b>Retraining fallito</b> — {symbol} {timeframe}\n"
            f"  Exit code: {exit_code}\n"
            f"  Modello precedente ancora attivo."
        )
        self.send(msg, throttle_key=f"retrain_failed_{symbol}")

    def alert_training_autotune_improved(
        self,
        symbol: str,
        timeframe: str,
        trial: int | str,
        max_trials: int | str,
        holdout: float,
        training_backend: str | None = None,
        previous_model_acc: float | None = None,
        previous_model_backend: str | None = None,
    ) -> None:
        """Alert when auto-tune finds a new best holdout during a single adaptive attempt."""
        if not self._cfg.notify_retrain_events:
            return
        backend_str = f" [{training_backend}]" if training_backend else ""
        if previous_model_acc is not None:
            prev_backend_str = f" ({previous_model_backend})" if previous_model_backend else ""
            prev_str = f"  Modello precedente: {previous_model_acc:.1%}{prev_backend_str}\n"
        else:
            prev_str = ""
        msg = (
            f"🔍 <b>Auto-tune migliorato</b>{backend_str} — {symbol} {timeframe}\n"
            f"  Trial: {trial}/{max_trials}\n"
            f"{prev_str}"
            f"  Miglior holdout: {holdout:.1%}"
        )
        self.send(msg, throttle_key=f"autotune_improved_{symbol}_{timeframe}")

    def alert_training_precision_improved(
        self,
        symbol: str,
        timeframe: str,
        attempt: int,
        precision_buy: float | None,
        precision_sell: float | None,
        target_buy: float,
        target_sell: float,
        holdout: float | None,
    ) -> None:
        """Alert when training attempt improves BUY or SELL precision over the previous best."""
        if not self._cfg.notify_retrain_events:
            return
        buy_str = f"{precision_buy:.1%}" if precision_buy is not None else "n/a"
        sell_str = f"{precision_sell:.1%}" if precision_sell is not None else "n/a"
        target_buy_str = f"{target_buy:.1%}"
        target_sell_str = f"{target_sell:.1%}"
        holdout_str = f"{holdout:.1%}" if holdout is not None else "n/a"
        buy_icon = "✅" if precision_buy is not None and precision_buy >= target_buy else "🔄"
        sell_icon = "✅" if precision_sell is not None and precision_sell >= target_sell else "🔄"
        msg = (
            f"📈 <b>Training migliorato</b> — {symbol} {timeframe} (tentativo {attempt})\n"
            f"  {buy_icon} BUY precision:  {buy_str} (target {target_buy_str})\n"
            f"  {sell_icon} SELL precision: {sell_str} (target {target_sell_str})\n"
            f"  Holdout: {holdout_str}"
        )
        self.send(msg, throttle_key=f"training_precision_{symbol}_{timeframe}")

    def alert_position_pnl(
        self,
        symbol: str,
        side: str,
        unrealized_pnl: float | Decimal,
        entry: float | Decimal,
        current: float | Decimal,
        duration_bars: int,
    ) -> None:
        """Periodic unrealized PnL update for an open position."""
        pnl_f = float(unrealized_pnl)
        emoji = "📈" if pnl_f >= 0 else "📉"
        msg = (
            f"{emoji} Posizione aperta: {symbol} {side.upper()}\n"
            f"  Entry:   {float(entry):.5f}\n"
            f"  Attuale: {float(current):.5f}\n"
            f"  PnL:     {_sign(pnl_f)}{pnl_f:.2f} USD\n"
            f"  Barre:   {duration_bars}"
        )
        self.send(msg, throttle_key=f"pos_pnl_{symbol}")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _is_active(self) -> bool:
        return bool(self._cfg.enabled and self._cfg.bot_token and self._cfg.chat_id)

    # ── Reply keyboard ────────────────────────────────────────────────────────

    # Bottoni rapidi mostrati sotto il campo messaggio del bot.
    # Gli emoji sono decorativi; il testo del bottone viene inviato come
    # messaggio quando premuto, quindi deve coincidere con un comando valido
    # registrato in ``CommandHandler``.
    QUICK_KEYBOARD: tuple[tuple[str, ...], ...] = (
        ("📊 positions", "🔄 sync", "📅 daily"),
        ("🔁 retrain", "🎯 training", "🏆 trained"),
        ("🗑 removetraining", "⏹ stoptraining", "📥 retrain_refresh"),
    )

    def send_quick_keyboard(
        self,
        message: str = "⌨️ Comandi rapidi attivi",
        *,
        persistent: bool = True,
    ) -> None:
        """Invia (o aggiorna) la reply keyboard persistente con i comandi rapidi.

        Args:
            message:    Testo del messaggio a cui è allegata la keyboard.
            persistent: Se True, la keyboard rimane visibile finché non viene
                        rimossa esplicitamente.
        """
        if not self._is_active():
            return
        keyboard = [
            [{"text": label} for label in row]
            for row in self.QUICK_KEYBOARD
        ]
        reply_markup = {
            "keyboard": keyboard,
            "resize_keyboard": True,
            "is_persistent": persistent,
            "one_time_keyboard": False,
        }
        self._post(message, reply_markup=reply_markup)

    def remove_quick_keyboard(self, message: str = "⌨️ Tastiera rimossa") -> None:
        """Rimuove la reply keyboard persistente."""
        if not self._is_active():
            return
        self._post(message, reply_markup={"remove_keyboard": True})

    def _post(self, text: str, *, reply_markup: dict | None = None) -> None:
        """HTTP POST to Telegram sendMessage — fire-and-forget."""
        import json as _json
        url = _TELEGRAM_API.format(token=self._cfg.bot_token, method="sendMessage")
        body: dict = {
            "chat_id": self._cfg.chat_id,
            "text": text,
            "parse_mode": "HTML",
        }
        if reply_markup is not None:
            body["reply_markup"] = reply_markup
        payload = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.timeout_secs, context=_SSL_CTX):
                pass
        except urllib.error.URLError as exc:
            log.warning("Telegram alert failed (network): %s", exc)
        except Exception as exc:
            log.warning("Telegram alert failed: %s", exc)

    def _get(self, method: str, params: dict) -> dict | None:
        """HTTP GET to Telegram API — returns parsed JSON or None on error."""
        import json as _json
        url = (
            _TELEGRAM_API.format(token=self._cfg.bot_token, method=method)
            + "?" + urllib.parse.urlencode(params)
        )
        try:
            with urllib.request.urlopen(url, timeout=self._cfg.timeout_secs, context=_SSL_CTX) as resp:
                return _json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            log.warning("Telegram GET %s failed: %s", method, exc)
            return None
