"""Persist last run_live / run_paper command line for dashboard restart."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_REL_PATH = Path("data/last_runner_launch.json")


def last_launch_path() -> Path:
    return DEFAULT_REL_PATH


def save_last_launch(*, mode: str, script_path: Path) -> None:
    """Write JSON with executable + argv so the dashboard can respawn the same process."""
    payload: dict[str, Any] = {
        "mode": mode,
        "executable": sys.executable,
        "script": str(script_path.resolve()),
        "argv": list(sys.argv[1:]),
        "cwd": str(Path.cwd().resolve()),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    path = last_launch_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_last_launch() -> dict[str, Any] | None:
    path = last_launch_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if not data.get("executable") or not data.get("script"):
        return None
    return data
