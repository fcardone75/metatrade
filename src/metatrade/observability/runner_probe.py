"""Detect run_live / run_paper OS processes (same machine as dashboard)."""

from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class TraderProcessInfo:
    """Whether a MetaTrade trader subprocess appears to be running."""

    running: bool
    pids: tuple[int, ...]


def probe_trader_processes() -> TraderProcessInfo:
    """Return PIDs of python processes running run_live or run_paper (not run_dashboard)."""
    pids: list[int] = []
    for proc in psutil.process_iter():
        try:
            parts = proc.cmdline()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if not parts:
            continue
        flat = " ".join(parts)
        if "run_dashboard" in flat:
            continue
        if "run_live" in flat or "run_paper" in flat:
            pids.append(int(proc.pid))
    pids.sort()
    return TraderProcessInfo(running=len(pids) > 0, pids=tuple(pids))
