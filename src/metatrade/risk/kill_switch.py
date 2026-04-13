"""Kill switch manager.

Maintains the current kill switch level and provides activation/reset methods.
Level can only increase (escalate) or reset to NONE — never decrease to an
intermediate level, preventing race conditions where a higher severity is
accidentally downgraded.

Levels (from core/enums.py):
    NONE          = 0  No restriction.
    TRADE_GATE    = 1  Block the next single trade only.
    SESSION_GATE  = 2  Block all new trades until reset.
    EMERGENCY_HALT= 3  Close all positions and stop the system.
    HARD_KILL     = 4  Manual override (same behaviour as EMERGENCY_HALT).

An optional SQLiteAuditStore logs every activation and reset for audit trail.
"""

from __future__ import annotations

from datetime import datetime, timezone

from metatrade.core.contracts.risk import KillSwitchState
from metatrade.core.enums import KillSwitchLevel
from metatrade.core.errors import KillSwitchActiveError
from metatrade.core.log import get_logger

log = get_logger(__name__)


class KillSwitchManager:
    """Thread-safe (asyncio-safe) in-memory kill switch.

    Args:
        audit_store: Optional SQLiteAuditStore for persistent logging.
                     If None, events are only written to the structured log.
    """

    def __init__(self, audit_store: object | None = None) -> None:
        self._state = KillSwitchState(
            level=KillSwitchLevel.NONE,
            reason="System started",
            activated_at_utc=datetime.now(timezone.utc),
            activated_by="system",
        )
        self._audit_store = audit_store  # duck-typed: expects .log_kill_switch()

    # ── Read ──────────────────────────────────────────────────────────────────

    @property
    def state(self) -> KillSwitchState:
        """Current kill switch state (immutable snapshot)."""
        return self._state

    @property
    def level(self) -> KillSwitchLevel:
        return self._state.level

    def is_active(self) -> bool:
        return self._state.is_active

    def blocks_new_trades(self) -> bool:
        return self._state.blocks_new_trades

    def blocks_all_activity(self) -> bool:
        return self._state.blocks_all_activity

    # ── Write ─────────────────────────────────────────────────────────────────

    def activate(
        self,
        level: KillSwitchLevel,
        reason: str,
        activated_by: str = "system",
        ts: datetime | None = None,
    ) -> KillSwitchState:
        """Activate or escalate the kill switch.

        If the requested level is lower than the current level, the call is
        ignored (kill switch can only escalate, never downgrade).

        Args:
            level:        Target KillSwitchLevel.
            reason:       Human-readable explanation.
            activated_by: Component or user that triggered the activation.
            ts:           Override UTC timestamp (defaults to now).

        Returns:
            The new KillSwitchState after activation.
        """
        if level == KillSwitchLevel.NONE:
            raise ValueError("Use reset() to deactivate the kill switch, not activate(NONE)")

        if level <= self._state.level:
            log.debug(
                "kill_switch_activate_ignored",
                requested=level.name,
                current=self._state.level.name,
                reason="level not higher than current",
            )
            return self._state

        now = ts or datetime.now(timezone.utc)
        self._state = KillSwitchState(
            level=level,
            reason=reason,
            activated_at_utc=now,
            activated_by=activated_by,
        )
        log.warning(
            "kill_switch_activated",
            level=level.name,
            reason=reason,
            activated_by=activated_by,
        )
        self._write_audit(level, reason, activated_by, now)
        return self._state

    def reset(
        self,
        activated_by: str = "manual",
        ts: datetime | None = None,
    ) -> KillSwitchState:
        """Reset kill switch to NONE.

        Only allowed from levels below HARD_KILL. A HARD_KILL requires
        explicit operator intervention (call force_reset()).

        Returns:
            The new KillSwitchState (level=NONE).

        Raises:
            KillSwitchActiveError: If current level is HARD_KILL.
        """
        if self._state.level == KillSwitchLevel.HARD_KILL:
            raise KillSwitchActiveError(
                message="HARD_KILL cannot be reset via reset() — use force_reset()",
                context={"level": KillSwitchLevel.HARD_KILL.name},
            )
        now = ts or datetime.now(timezone.utc)
        self._state = KillSwitchState(
            level=KillSwitchLevel.NONE,
            reason=f"Reset by {activated_by}",
            activated_at_utc=now,
            activated_by=activated_by,
        )
        log.info("kill_switch_reset", activated_by=activated_by)
        self._write_audit(KillSwitchLevel.NONE, f"Reset by {activated_by}", activated_by, now)
        return self._state

    def force_reset(
        self,
        activated_by: str = "operator",
        ts: datetime | None = None,
    ) -> KillSwitchState:
        """Force-reset from any level including HARD_KILL.

        Requires explicit call — prevents accidental reset of the highest
        severity level.
        """
        now = ts or datetime.now(timezone.utc)
        self._state = KillSwitchState(
            level=KillSwitchLevel.NONE,
            reason=f"Force-reset by {activated_by}",
            activated_at_utc=now,
            activated_by=activated_by,
        )
        log.warning("kill_switch_force_reset", activated_by=activated_by)
        self._write_audit(KillSwitchLevel.NONE, f"Force-reset by {activated_by}", activated_by, now)
        return self._state

    def assert_trading_allowed(self) -> None:
        """Raise KillSwitchActiveError if the kill switch blocks new trades."""
        if self._state.blocks_new_trades:
            raise KillSwitchActiveError(
                message=(
                    f"Kill switch level {self._state.level.name} blocks new trades: "
                    f"{self._state.reason}"
                ),
                context={"level": self._state.level.name},
            )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write_audit(
        self,
        level: KillSwitchLevel,
        reason: str,
        activated_by: str,
        ts: datetime,
    ) -> None:
        if self._audit_store is None:
            return
        try:
            self._audit_store.log_kill_switch(  # type: ignore[union-attr]
                level=level.value,
                reason=reason,
                activated_by=activated_by,
                ts=ts,
            )
        except Exception as exc:
            log.error("kill_switch_audit_write_failed", error=str(exc))
