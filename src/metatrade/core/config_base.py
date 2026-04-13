"""Base configuration class for all MetaTrade modules.

All module configs inherit from BaseConfig.
Values are loaded from environment variables and .env files.
No hardcoded defaults for sensitive values (broker credentials, etc.).

Validation is performed by Pydantic on instantiation.
A failed config raises ConfigurationError immediately — never a silent wrong default.

Example:
    class RiskConfig(BaseConfig):
        model_config = ConfigDict(env_prefix="RISK_")

        risk_pct_per_trade: float = Field(gt=0.0, le=0.05)
        max_open_positions: int = Field(ge=1, le=20)

    cfg = RiskConfig()   # reads RISK_RISK_PCT_PER_TRADE etc. from env
"""

from __future__ import annotations

from pydantic import ConfigDict, ValidationError
from pydantic_settings import BaseSettings

from metatrade.core.errors import ConfigurationError


class BaseConfig(BaseSettings):
    """Base for all configuration classes.

    Reads from:
    1. Environment variables
    2. .env file in the current working directory
    3. Field defaults

    In that priority order (env vars win over .env, .env wins over defaults).
    """

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,  # Configs are immutable after construction
    )

    @classmethod
    def load(cls) -> "BaseConfig":
        """Factory method with structured error wrapping.

        Converts Pydantic's ValidationError into ConfigurationError so callers
        only need to handle one exception type at the boundary.
        """
        try:
            return cls()
        except ValidationError as exc:
            raise ConfigurationError(
                message=f"Invalid configuration for {cls.__name__}: {exc}",
                code="INVALID_CONFIG",
                context={"errors": exc.errors()},
            ) from exc
