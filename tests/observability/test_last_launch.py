"""Tests for last runner launch persistence."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_save_and_load_last_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from metatrade.observability import last_launch as ll

    monkeypatch.setattr(ll, "DEFAULT_REL_PATH", tmp_path / "last_runner_launch.json")
    fake_script = tmp_path / "run_live.py"
    fake_script.write_text("#", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(fake_script), "--symbol", "EURUSD", "--confirm"])

    ll.save_last_launch(mode="live", script_path=fake_script)
    data = ll.load_last_launch()
    assert data is not None
    assert data["mode"] == "live"
    assert data["argv"] == ["--symbol", "EURUSD", "--confirm"]
    assert Path(data["script"]).resolve() == fake_script.resolve()
