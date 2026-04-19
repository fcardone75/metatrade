"""CommandHandler — registry mapping Telegram commands to runner callbacks.

Inbound commands arrive from TelegramCommandReceiver and are dispatched here.
Each command string (e.g. "/status") maps to a callable registered via
register().

Two-step /stop flow:
  1. /stop → prompts for confirmation, sets pending flag
  2. /stop confirm → calls the /stop callback and clears the flag
  Any other command clears the pending flag.
"""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)

CommandCallback = Callable[[str], str]  # callback(args: str) -> reply text


class CommandHandler:
    """Routes parsed Telegram commands to registered Python callbacks.

    Usage::

        handler = CommandHandler()
        handler.register("/status", lambda args: "Running!")
        reply = handler.dispatch("/status", "")
    """

    def __init__(self) -> None:
        self._registry: dict[str, CommandCallback] = {}
        self._pending_stop: bool = False

    def register(self, command: str, callback: CommandCallback) -> None:
        """Register a callback for a command (e.g. "/status").

        Later calls with the same command overwrite earlier ones.
        """
        self._registry[command.lower().strip()] = callback

    def dispatch(self, command: str, args: str) -> str:
        """Dispatch a parsed command to its callback.

        Handles the /stop two-step confirmation flow internally.

        Returns the reply string to send back to the user.
        """
        cmd = command.lower().strip()

        # Two-step /stop confirmation
        if cmd == "/stop":
            if args.strip().lower() == "confirm":
                self._pending_stop = False
                callback = self._registry.get("/stop")
                if callback is None:
                    return "⚠️ Comando /stop non registrato."
                try:
                    return callback(args)
                except Exception as exc:
                    log.warning("CommandHandler /stop callback error: %s", exc)
                    return f"Errore nell'esecuzione di /stop: {exc}"
            else:
                self._pending_stop = True
                return (
                    "⚠️ Sei sicuro di voler fermare il bot?\n"
                    "Invia <code>/stop confirm</code> per confermare.\n"
                    "Qualsiasi altro comando annulla."
                )

        # Any other command clears pending stop
        self._pending_stop = False

        if cmd == "/help":
            return self._help_text()

        callback = self._registry.get(cmd)
        if callback is None:
            available = ", ".join(sorted(self._registry)) if self._registry else "nessuno"
            return (
                f"❓ Comando non riconosciuto: <code>{command}</code>\n"
                f"Comandi disponibili: {available}\n"
                f"Usa /help per i dettagli."
            )
        try:
            return callback(args)
        except Exception as exc:
            log.warning("CommandHandler dispatch error for %s: %s", command, exc)
            return f"⛔ Errore nell'esecuzione di <code>{command}</code>: {exc}"

    @property
    def pending_stop(self) -> bool:
        """True if the user issued /stop and is waiting for /stop confirm."""
        return self._pending_stop

    def _help_text(self) -> str:
        commands = {
            "/status": "Stato generale del sistema",
            "/positions": "Posizioni aperte in dettaglio",
            "/balance": "Snapshot account (balance, equity, margin)",
            "/model": "Info modello ML attivo",
            "/accuracy": "Accuracy live dettagliata",
            "/daily": "Riepilogo giornata on-demand",
            "/retrain": "Avvia retraining manuale",
            "/training": "Stato training adattivo (tentativi, holdout, target)",
            "/stoptraining": "Interrompe training in corso",
            "/pause": "Pausa nuove entry (exit continuano)",
            "/resume": "Riprende entry dopo /pause",
            "/stop": "Stop completo (richiede conferma)",
            "/help": "Questa lista",
        }
        lines = ["📋 <b>Comandi disponibili</b>\n"]
        for cmd, desc in commands.items():
            registered = "✅" if cmd in self._registry or cmd == "/help" else "⚠️"
            lines.append(f"{registered} <code>{cmd}</code> — {desc}")
        return "\n".join(lines)
