"""Tests for IdempotencyGuard."""

from __future__ import annotations

import time

import pytest

from metatrade.core.errors import DuplicateOrderError
from metatrade.execution.idempotency_guard import IdempotencyGuard


class TestIdempotencyGuard:
    def test_new_order_id_passes(self) -> None:
        guard = IdempotencyGuard()
        guard.check("order-001")  # must not raise

    def test_duplicate_raises(self) -> None:
        guard = IdempotencyGuard()
        guard.register("order-001")
        with pytest.raises(DuplicateOrderError):
            guard.check("order-001")

    def test_check_and_register_sequential(self) -> None:
        guard = IdempotencyGuard()
        guard.check_and_register("order-A")
        with pytest.raises(DuplicateOrderError):
            guard.check_and_register("order-A")

    def test_different_ids_no_conflict(self) -> None:
        guard = IdempotencyGuard()
        guard.register("order-001")
        guard.check("order-002")  # different ID — must not raise

    def test_is_known_returns_true_after_register(self) -> None:
        guard = IdempotencyGuard()
        guard.register("order-X")
        assert guard.is_known("order-X")

    def test_is_known_returns_false_before_register(self) -> None:
        guard = IdempotencyGuard()
        assert not guard.is_known("order-X")

    def test_known_count_increments(self) -> None:
        guard = IdempotencyGuard()
        assert guard.known_count == 0
        guard.register("a")
        guard.register("b")
        assert guard.known_count == 2

    def test_clear_removes_all_ids(self) -> None:
        guard = IdempotencyGuard()
        guard.register("a")
        guard.register("b")
        guard.clear()
        assert guard.known_count == 0
        guard.check("a")  # must not raise after clear

    def test_expired_id_no_longer_blocks(self) -> None:
        guard = IdempotencyGuard(ttl_seconds=0)  # expire immediately
        guard.register("order-exp")
        time.sleep(0.01)  # tiny sleep so timestamp difference registers
        guard.check("order-exp")  # should not raise — TTL expired

    def test_error_code_is_duplicate_order(self) -> None:
        guard = IdempotencyGuard()
        guard.register("order-dup")
        with pytest.raises(DuplicateOrderError) as exc_info:
            guard.check("order-dup")
        assert exc_info.value.code == "DUPLICATE_ORDER"
