"""Tests for core/event_bus.py."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from metatrade.core.event_bus import Event, EventBus, EventType

UTC = UTC
T0 = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)


def make_event(event_type: str = "TEST", payload: object = None) -> Event:
    return Event(event_type=event_type, payload=payload or {}, timestamp_utc=T0, source="test")


class TestEvent:
    def test_valid_construction(self) -> None:
        e = make_event("FOO", {"key": "value"})
        assert e.event_type == "FOO"
        assert e.source == "test"

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            Event(event_type="X", payload={}, timestamp_utc=datetime(2024, 1, 1))

    def test_empty_event_type_raises(self) -> None:
        with pytest.raises(ValueError, match="event_type cannot be empty"):
            Event(event_type="", payload={}, timestamp_utc=T0)


class TestEventBus:
    @pytest.mark.asyncio
    async def test_subscribe_and_receive(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("TEST", handler)
        event = make_event("TEST", "hello")
        await bus.publish(event)

        assert len(received) == 1
        assert received[0].payload == "hello"

    @pytest.mark.asyncio
    async def test_multiple_handlers_same_event(self) -> None:
        bus = EventBus()
        log: list[str] = []

        async def h1(e: Event) -> None:
            log.append("h1")

        async def h2(e: Event) -> None:
            log.append("h2")

        bus.subscribe("TEST", h1)
        bus.subscribe("TEST", h2)
        await bus.publish(make_event("TEST"))

        assert "h1" in log
        assert "h2" in log

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        bus = EventBus()
        called: list[bool] = []

        async def handler(event: Event) -> None:
            called.append(True)

        bus.subscribe("TEST", handler)
        bus.unsubscribe("TEST", handler)
        await bus.publish(make_event("TEST"))

        assert called == []

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_is_noop(self) -> None:
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        bus.unsubscribe("NONEXISTENT", handler)  # should not raise

    @pytest.mark.asyncio
    async def test_no_handlers_no_error(self) -> None:
        bus = EventBus()
        await bus.publish(make_event("UNHANDLED"))  # should not raise

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_block_others(self) -> None:
        """A crashing handler must not prevent other handlers from running."""
        bus = EventBus()
        log: list[str] = []

        async def bad_handler(event: Event) -> None:
            raise RuntimeError("intentional crash")

        async def good_handler(event: Event) -> None:
            log.append("good")

        bus.subscribe("TEST", bad_handler)
        bus.subscribe("TEST", good_handler)
        await bus.publish(make_event("TEST"))

        assert log == ["good"]

    @pytest.mark.asyncio
    async def test_duplicate_subscribe_ignored(self) -> None:
        """Registering the same handler twice should only call it once."""
        bus = EventBus()
        count: list[int] = [0]

        async def handler(event: Event) -> None:
            count[0] += 1

        bus.subscribe("TEST", handler)
        bus.subscribe("TEST", handler)  # duplicate
        await bus.publish(make_event("TEST"))

        assert count[0] == 1

    @pytest.mark.asyncio
    async def test_history_recorded(self) -> None:
        bus = EventBus()

        async def noop(e: Event) -> None:
            pass

        bus.subscribe("A", noop)
        e1 = make_event("A", 1)
        e2 = make_event("A", 2)
        await bus.publish(e1)
        await bus.publish(e2)

        history = bus.get_history("A")
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_history_limit(self) -> None:
        bus = EventBus()
        for i in range(5):
            await bus.publish(make_event("X", i))
        history = bus.get_history(limit=3)
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_history_max_size_respected(self) -> None:
        bus = EventBus(max_history=5)
        for i in range(10):
            await bus.publish(make_event("X", i))
        assert len(bus.get_history()) == 5

    @pytest.mark.asyncio
    async def test_clear_history(self) -> None:
        bus = EventBus()
        await bus.publish(make_event("X"))
        bus.clear_history()
        assert bus.get_history() == []

    @pytest.mark.asyncio
    async def test_clear_handlers(self) -> None:
        bus = EventBus()
        called: list[bool] = []

        async def handler(event: Event) -> None:
            called.append(True)

        bus.subscribe("TEST", handler)
        bus.clear_handlers()
        await bus.publish(make_event("TEST"))
        assert called == []

    @pytest.mark.asyncio
    async def test_filter_by_event_type(self) -> None:
        bus = EventBus()
        await bus.publish(make_event("A"))
        await bus.publish(make_event("B"))
        await bus.publish(make_event("A"))

        a_events = bus.get_history("A")
        b_events = bus.get_history("B")
        assert len(a_events) == 2
        assert len(b_events) == 1

    def test_event_type_constants_exist(self) -> None:
        assert EventType.NEW_BAR
        assert EventType.SIGNAL_GENERATED
        assert EventType.ORDER_FILLED
        assert EventType.KILL_SWITCH_ACTIVATED
