"""Internal async event bus (pub/sub).

Used for decoupled communication between system components.
Example events: NEW_BAR, SIGNAL_GENERATED, ORDER_FILLED, KILL_SWITCH_ACTIVATED.

Design choices:
- Async handlers (asyncio): the live runner is inherently async (I/O-bound).
- Errors in handlers are caught and logged — one bad handler never blocks others.
- Event history is kept for debugging and audit (bounded ring-buffer).
- All event timestamps are UTC.

Usage:
    bus = EventBus()

    async def on_signal(event: Event) -> None:
        print(event.payload)

    bus.subscribe("SIGNAL_GENERATED", on_signal)

    await bus.publish(Event(
        event_type="SIGNAL_GENERATED",
        payload=signal,
        source="trend_following_module",
        timestamp_utc=clock.now_utc(),
    ))
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable

from metatrade.core.log import get_logger

log = get_logger(__name__)

# A handler is an async callable that receives an Event and returns nothing.
EventHandler = Callable[["Event"], Awaitable[None]]


@dataclass
class Event:
    """A published event payload."""

    event_type: str
    payload: Any
    timestamp_utc: datetime
    source: str = ""

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("Event.timestamp_utc must be timezone-aware")
        if not self.event_type:
            raise ValueError("Event.event_type cannot be empty")


class EventBus:
    """Async publish/subscribe event bus.

    Thread-safety: Not thread-safe by design. The system uses a single asyncio
    event loop. If multi-threading is introduced later, add explicit locking.
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: list[Event] = []
        self._max_history = max_history

    # ── Subscription ──────────────────────────────────────────────────────────

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register `handler` to be called when `event_type` is published."""
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a previously registered handler. No-op if not registered."""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    # ── Publishing ────────────────────────────────────────────────────────────

    async def publish(self, event: Event) -> None:
        """Publish an event to all registered handlers.

        Handlers run concurrently. An exception in one handler is logged and
        does NOT prevent other handlers from running.
        """
        self._append_history(event)

        handlers = list(self._handlers.get(event.event_type, []))
        if not handlers:
            return

        results = await asyncio.gather(
            *[handler(event) for handler in handlers],
            return_exceptions=True,
        )

        for handler, result in zip(handlers, results):
            if isinstance(result, Exception):
                log.error(
                    "event_handler_failed",
                    event_type=event.event_type,
                    handler=getattr(handler, "__name__", repr(handler)),
                    error=str(result),
                )

    # ── History ───────────────────────────────────────────────────────────────

    def get_history(
        self,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """Return recent events, optionally filtered by type."""
        events = self._history if event_type is None else [
            e for e in self._history if e.event_type == event_type
        ]
        return events[-limit:]

    def clear_history(self) -> None:
        self._history.clear()

    def clear_handlers(self) -> None:
        """Remove all subscriptions. Useful between tests."""
        self._handlers.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _append_history(self, event: Event) -> None:
        if len(self._history) >= self._max_history:
            self._history.pop(0)
        self._history.append(event)


# ── Well-known event type constants ───────────────────────────────────────────
# Using constants avoids typos in event_type strings across the codebase.

class EventType:
    NEW_BAR = "NEW_BAR"
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    CONSENSUS_READY = "CONSENSUS_READY"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    RISK_VETO = "RISK_VETO"
    KILL_SWITCH_ACTIVATED = "KILL_SWITCH_ACTIVATED"
    KILL_SWITCH_RESET = "KILL_SWITCH_RESET"
    BROKER_DISCONNECTED = "BROKER_DISCONNECTED"
    BROKER_RECONNECTED = "BROKER_RECONNECTED"
