"""Tests for BaseConfig — the shared base class for all module configs."""

from __future__ import annotations

import pytest
from pydantic import Field

from metatrade.core.config_base import BaseConfig
from metatrade.core.errors import ConfigurationError


class _SimpleConfig(BaseConfig):
    """Minimal concrete subclass for testing."""
    value: int = Field(default=42, ge=0)
    name: str = Field(default="test")

    model_config = {
        "env_prefix": "TEST_CFG_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "frozen": True,
    }


class _StrictConfig(BaseConfig):
    """Config with a required field (no default) for error testing."""
    required_field: int = Field(gt=0)

    model_config = {
        "env_prefix": "TEST_STRICT_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "frozen": True,
    }


class TestBaseConfig:
    def test_instantiation_with_defaults(self) -> None:
        cfg = _SimpleConfig()
        assert cfg.value == 42
        assert cfg.name == "test"

    def test_override_field(self) -> None:
        cfg = _SimpleConfig(value=100)
        assert cfg.value == 100

    def test_config_is_frozen(self) -> None:
        cfg = _SimpleConfig()
        with pytest.raises(Exception):
            cfg.value = 99  # type: ignore[misc]

    def test_load_factory_returns_instance(self) -> None:
        cfg = _SimpleConfig.load()
        assert isinstance(cfg, _SimpleConfig)
        assert cfg.value == 42

    def test_load_wraps_validation_error_as_configuration_error(self) -> None:
        # Provide an invalid value via direct construction that triggers a
        # validation error inside load()
        class _InvalidConfig(BaseConfig):
            must_be_positive: int = Field(gt=0)
            model_config = {
                "env_prefix": "TEST_INV_",
                "env_file": ".env",
                "env_file_encoding": "utf-8",
                "case_sensitive": False,
                "extra": "ignore",
                "frozen": True,
            }

        # Without a valid env var set, the field has no default → ValidationError
        with pytest.raises(ConfigurationError) as exc_info:
            _InvalidConfig.load()
        assert exc_info.value.code == "INVALID_CONFIG"

    def test_extra_fields_ignored(self) -> None:
        # extra="ignore" means unknown fields are silently dropped
        cfg = _SimpleConfig(value=5, unknown_field="ignored")  # type: ignore[call-arg]
        assert cfg.value == 5
        assert not hasattr(cfg, "unknown_field")
