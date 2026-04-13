"""Idempotency guard — prevents duplicate orders.

The guard maintains a set of recently-seen order IDs. Before submitting an
order, call check(order). If the order_id has already been seen, a
DuplicateOrderError is raised.

The guard uses a time-bounded window (TTL in seconds) so that order IDs
eventually expire from memory. This prevents unbounded memory growth during
long-running sessions while still catching duplicates within the window.

Design:
    - In-memory only (no persistence). Restart clears history.
    - TTL default: 3600 seconds (1 hour). Sufficient for typical session length.
    - Thread/asyncio-safe for single-threaded usage (no locks needed).
"""

from __future__ import annotations

from datetime import datetime, timezone

from metatrade.core.errors import DuplicateOrderError
from metatrade.core.log import get_logger

log = get_logger(__name__)


class IdempotencyGuard:
    """Prevents duplicate order submissions within a TTL window.

    Args:
        ttl_seconds: How long (in seconds) an order_id is remembered.
                     After TTL expires, the same order_id could be resubmitted
                     (this is intentional — old IDs should never recur).
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._ttl = ttl_seconds
        # {order_id: submitted_at_unix_ts}
        self._seen: dict[str, float] = {}

    def check(self, order_id: str) -> None:
        """Assert that order_id has not been recently submitted.

        Raises:
            DuplicateOrderError: If the order_id was already seen within the TTL.
        """
        self._expire_old_entries()
        if order_id in self._seen:
            raise DuplicateOrderError(
                message=f"Duplicate order detected: {order_id}",
                code="DUPLICATE_ORDER",
                context={"order_id": order_id},
            )

    def register(self, order_id: str) -> None:
        """Record that order_id has been submitted.

        Call this after a successful broker submission, not before.
        """
        self._expire_old_entries()
        self._seen[order_id] = datetime.now(timezone.utc).timestamp()
        log.debug("order_registered", order_id=order_id)

    def check_and_register(self, order_id: str) -> None:
        """Atomically check for duplicates and register on success.

        Raises:
            DuplicateOrderError: If the order_id has already been registered.
        """
        self.check(order_id)
        self.register(order_id)

    def is_known(self, order_id: str) -> bool:
        """Return True if order_id is currently registered (within TTL)."""
        self._expire_old_entries()
        return order_id in self._seen

    def clear(self) -> None:
        """Clear all registered order IDs (e.g. on session reset)."""
        self._seen.clear()

    @property
    def known_count(self) -> int:
        """Number of currently tracked order IDs."""
        self._expire_old_entries()
        return len(self._seen)

    def _expire_old_entries(self) -> None:
        now = datetime.now(timezone.utc).timestamp()
        cutoff = now - self._ttl
        expired = [oid for oid, ts in self._seen.items() if ts < cutoff]
        for oid in expired:
            del self._seen[oid]
