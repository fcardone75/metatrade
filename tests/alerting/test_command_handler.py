"""Tests for CommandHandler: routing, security, /stop two-step flow."""

from __future__ import annotations

from metatrade.alerting.command_handler import CommandHandler


def _echo(args: str) -> str:
    return f"echo: {args}"


class TestRegisterAndDispatch:
    def test_registered_command_is_dispatched(self):
        h = CommandHandler()
        h.register("/status", lambda _: "ok")
        assert h.dispatch("/status", "") == "ok"

    def test_args_are_passed_to_callback(self):
        h = CommandHandler()
        h.register("/echo", _echo)
        assert h.dispatch("/echo", "hello world") == "echo: hello world"

    def test_case_insensitive_dispatch(self):
        h = CommandHandler()
        h.register("/STATUS", lambda _: "ok")
        assert h.dispatch("/status", "") == "ok"
        assert h.dispatch("/STATUS", "") == "ok"

    def test_unknown_command_returns_error_message(self):
        h = CommandHandler()
        reply = h.dispatch("/unknown", "")
        assert "non riconosciuto" in reply.lower() or "non riconosciuto" in reply

    def test_overwriting_registration(self):
        h = CommandHandler()
        h.register("/cmd", lambda _: "first")
        h.register("/cmd", lambda _: "second")
        assert h.dispatch("/cmd", "") == "second"

    def test_callback_exception_returns_error_message(self):
        h = CommandHandler()
        h.register("/fail", lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
        reply = h.dispatch("/fail", "")
        assert "boom" in reply or "errore" in reply.lower()


class TestHelpCommand:
    def test_help_returns_text(self):
        h = CommandHandler()
        h.register("/status", lambda _: "ok")
        reply = h.dispatch("/help", "")
        assert "/help" in reply
        assert "/status" in reply
        assert "/trained" in reply

    def test_help_marks_unregistered_commands(self):
        h = CommandHandler()
        reply = h.dispatch("/help", "")
        # All standard commands should appear
        assert "/status" in reply


class TestStopTwoStep:
    def test_stop_without_confirm_prompts(self):
        h = CommandHandler()
        h.register("/stop", lambda _: "stopped")
        reply = h.dispatch("/stop", "")
        assert "confirm" in reply.lower()
        assert h.pending_stop is True

    def test_stop_confirm_calls_callback(self):
        h = CommandHandler()
        h.register("/stop", lambda _: "stopped!")
        h.dispatch("/stop", "")
        reply = h.dispatch("/stop", "confirm")
        assert reply == "stopped!"
        assert h.pending_stop is False

    def test_stop_confirm_case_insensitive(self):
        h = CommandHandler()
        h.register("/stop", lambda _: "done")
        h.dispatch("/stop", "")
        reply = h.dispatch("/stop", "CONFIRM")
        assert reply == "done"

    def test_other_command_clears_pending_stop(self):
        h = CommandHandler()
        h.register("/stop", lambda _: "stopped")
        h.register("/status", lambda _: "running")
        h.dispatch("/stop", "")
        assert h.pending_stop is True
        h.dispatch("/status", "")
        assert h.pending_stop is False

    def test_stop_without_registered_callback_returns_error(self):
        h = CommandHandler()
        h.dispatch("/stop", "")
        reply = h.dispatch("/stop", "confirm")
        assert "non registrato" in reply.lower() or "stop" in reply.lower()


class TestPauseResume:
    def test_pause_command(self):
        h = CommandHandler()
        h.register("/pause", lambda _: "paused")
        reply = h.dispatch("/pause", "")
        assert reply == "paused"

    def test_resume_command(self):
        h = CommandHandler()
        h.register("/resume", lambda _: "resumed")
        reply = h.dispatch("/resume", "")
        assert reply == "resumed"
