"""CLI helper to activate or reset the kill switch via the shared TelemetryStore.

Works cross-platform (Windows / macOS / Linux) because it writes to the SQLite
database that the runner polls once per bar.

Usage:
    python scripts/kill_switch_cli.py activate [--level 1-4] [--reason "..."]
    python scripts/kill_switch_cli.py reset

Levels:
    1 = TRADE_GATE      Block next single trade only.
    2 = SESSION_GATE    Block all new trades until reset.   (default)
    3 = EMERGENCY_HALT  Close all positions and stop the system.
    4 = HARD_KILL       Manual override — same behaviour as EMERGENCY_HALT.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metatrade.observability.store import TelemetryStore

_LEVEL_NAMES = {1: "TRADE_GATE", 2: "SESSION_GATE", 3: "EMERGENCY_HALT", 4: "HARD_KILL"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Activate or reset the MetaTrade kill switch."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    act = sub.add_parser("activate", help="Activate the kill switch")
    act.add_argument(
        "--level",
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help="Kill switch level (default: 2 = SESSION_GATE)",
    )
    act.add_argument("--reason", default="", help="Human-readable reason")

    sub.add_parser("reset", help="Reset the kill switch — re-enables trading")

    args = parser.parse_args()
    store = TelemetryStore.from_env()

    try:
        if args.cmd == "activate":
            reason = args.reason or f"{_LEVEL_NAMES[args.level]} activated via CLI"
            store.write_kill_command(level=args.level, reason=reason, activated_by="operator")
            print(
                f"Kill switch activated: {_LEVEL_NAMES[args.level]} (level {args.level})."
                "\nThe runner will pick this up on the next bar."
            )
        else:
            store.write_kill_command(level=0, reason="Reset via CLI", activated_by="operator")
            print("Kill switch reset. Trading will resume on the next bar.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
