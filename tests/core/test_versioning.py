"""Tests for core/versioning.py."""

from __future__ import annotations

import pytest

from metatrade.core.versioning import ModuleVersion, SchemaVersion


class TestModuleVersion:
    def test_parse_valid(self) -> None:
        v = ModuleVersion.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_parse_zeros(self) -> None:
        v = ModuleVersion.parse("0.0.0")
        assert v == ModuleVersion(0, 0, 0)

    def test_parse_large_numbers(self) -> None:
        v = ModuleVersion.parse("10.20.300")
        assert v.major == 10
        assert v.minor == 20
        assert v.patch == 300

    def test_parse_with_whitespace(self) -> None:
        v = ModuleVersion.parse("  2.0.1  ")
        assert v == ModuleVersion(2, 0, 1)

    def test_parse_invalid_format(self) -> None:
        for bad in ("1.2", "1.2.3.4", "v1.2.3", "1.2.x", "", "abc"):
            with pytest.raises(ValueError, match="Invalid semantic version"):
                ModuleVersion.parse(bad)

    def test_str(self) -> None:
        assert str(ModuleVersion(1, 2, 3)) == "1.2.3"

    def test_equality(self) -> None:
        assert ModuleVersion(1, 0, 0) == ModuleVersion(1, 0, 0)
        assert ModuleVersion(1, 0, 0) != ModuleVersion(2, 0, 0)

    def test_less_than(self) -> None:
        assert ModuleVersion(1, 0, 0) < ModuleVersion(2, 0, 0)
        assert ModuleVersion(1, 0, 0) < ModuleVersion(1, 1, 0)
        assert ModuleVersion(1, 0, 0) < ModuleVersion(1, 0, 1)

    def test_less_than_or_equal(self) -> None:
        assert ModuleVersion(1, 0, 0) <= ModuleVersion(1, 0, 0)
        assert ModuleVersion(1, 0, 0) <= ModuleVersion(2, 0, 0)

    def test_is_compatible_same_major(self) -> None:
        assert ModuleVersion(1, 0, 0).is_compatible_with(ModuleVersion(1, 5, 3))

    def test_is_not_compatible_different_major(self) -> None:
        assert not ModuleVersion(1, 0, 0).is_compatible_with(ModuleVersion(2, 0, 0))

    def test_negative_values_raise(self) -> None:
        with pytest.raises(ValueError):
            ModuleVersion(-1, 0, 0)
        with pytest.raises(ValueError):
            ModuleVersion(0, -1, 0)
        with pytest.raises(ValueError):
            ModuleVersion(0, 0, -1)

    def test_frozen(self) -> None:
        v = ModuleVersion(1, 0, 0)
        with pytest.raises(Exception):  # FrozenInstanceError
            v.major = 2  # type: ignore[misc]

    def test_hashable(self) -> None:
        versions = {ModuleVersion(1, 0, 0), ModuleVersion(1, 0, 0), ModuleVersion(2, 0, 0)}
        assert len(versions) == 2


class TestSchemaVersion:
    def test_valid_construction(self) -> None:
        v = SchemaVersion(1)
        assert v.version == 1

    def test_str(self) -> None:
        assert str(SchemaVersion(3)) == "3"

    def test_int(self) -> None:
        assert int(SchemaVersion(5)) == 5

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            SchemaVersion(0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            SchemaVersion(-1)

    def test_compatible_same_version(self) -> None:
        assert SchemaVersion(1).is_compatible_with(SchemaVersion(1))

    def test_not_compatible_different_version(self) -> None:
        assert not SchemaVersion(1).is_compatible_with(SchemaVersion(2))

    def test_next(self) -> None:
        assert SchemaVersion(1).next() == SchemaVersion(2)
        assert SchemaVersion(9).next() == SchemaVersion(10)

    def test_frozen(self) -> None:
        v = SchemaVersion(1)
        with pytest.raises(Exception):
            v.version = 2  # type: ignore[misc]

    def test_equality(self) -> None:
        assert SchemaVersion(1) == SchemaVersion(1)
        assert SchemaVersion(1) != SchemaVersion(2)

    def test_hashable(self) -> None:
        s = {SchemaVersion(1), SchemaVersion(1), SchemaVersion(2)}
        assert len(s) == 2
