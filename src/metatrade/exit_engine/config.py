"""Exit engine configuration.

Loadable from environment variables (prefix EXIT_) or a YAML file via
ExitEngineConfig.from_yaml().

All rule sub-configs are nested dataclasses so they can be overridden
individually without touching unrelated settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Per-rule config dataclasses ───────────────────────────────────────────────

@dataclass
class TrailingStopConfig:
    enabled:      bool  = True
    mode:         str   = "atr"    # "atr" | "fixed_pct" | "fixed_pips"
    atr_period:   int   = 14
    atr_mult:     float = 2.0
    fixed_pct:    float | None = None   # e.g. 0.005 = 0.5%
    fixed_pips:   float | None = None   # e.g. 15.0 pips


@dataclass
class BreakEvenConfig:
    enabled:      bool  = True
    trigger_pips: float = 15.0   # move SL to break-even after 15 pip profit
    buffer_pips:  float = 2.0    # SL at entry + 2 pip (not exactly 0)


@dataclass
class TimeExitConfig:
    enabled:              bool  = True
    max_holding_bars:     int   = 48      # barre al timeframe corrente
    close_on_session_end: bool  = True
    session_end_hour_utc: int   = 21      # New York session close


@dataclass
class SetupInvalidationConfig:
    enabled:                bool         = True
    ma_cross_fast:          int          = 9
    ma_cross_slow:          int          = 21
    invalidate_on_ma_cross: bool         = True
    price_level:            float | None = None  # hard level override


@dataclass
class VolatilityExitConfig:
    enabled:          bool  = True
    atr_period:       int   = 14
    atr_expansion_mult: float = 2.0  # exit if current ATR > entry_atr * mult


@dataclass
class GiveBackConfig:
    enabled:        bool  = True
    give_back_pct:  float = 0.40   # exit if floating profit drops 40% from peak


@dataclass
class PartialExitLevel:
    pips:      float
    close_pct: float   # fraction to close at this level (e.g. 0.33)


@dataclass
class PartialExitConfig:
    enabled: bool = True
    levels:  list[PartialExitLevel] = field(default_factory=lambda: [
        PartialExitLevel(pips=20.0, close_pct=0.33),
        PartialExitLevel(pips=40.0, close_pct=0.33),
    ])


# ── Reputation config ──────────────────────────────────────────────────────────

@dataclass
class ReputationConfig:
    learning_rate:      float = 0.15   # EMA alpha for weight updates
    min_weight:         float = 5.0
    max_weight:         float = 95.0
    initial_weight:     float = 50.0   # cold-start neutral
    cold_start_trades:  int   = 10     # blend with prior until this many trades
    decay_days:         int   = 7      # days of inactivity before weight decays toward 50
    decay_rate:         float = 0.05   # fraction of gap-to-50 removed per inactive day
    per_symbol:         bool  = False  # separate reputation per symbol


# ── Top-level config ───────────────────────────────────────────────────────────

@dataclass
class ExitEngineConfig:
    """Top-level configuration for the ExitEngine.

    Usage:
        cfg = ExitEngineConfig()                  # all defaults
        cfg = ExitEngineConfig.from_yaml(path)    # override from file
        cfg = ExitEngineConfig(close_full_threshold=70)  # inline override
    """

    enabled:                  bool  = True
    close_full_threshold:     float = 60.0   # aggregate score >= this → CLOSE_FULL
    close_partial_threshold:  float = 35.0   # aggregate score >= this → CLOSE_PARTIAL

    reputation:         ReputationConfig       = field(default_factory=ReputationConfig)
    trailing_stop:      TrailingStopConfig      = field(default_factory=TrailingStopConfig)
    break_even:         BreakEvenConfig         = field(default_factory=BreakEvenConfig)
    time_exit:          TimeExitConfig          = field(default_factory=TimeExitConfig)
    setup_invalidation: SetupInvalidationConfig = field(default_factory=SetupInvalidationConfig)
    volatility_exit:    VolatilityExitConfig    = field(default_factory=VolatilityExitConfig)
    give_back:          GiveBackConfig          = field(default_factory=GiveBackConfig)
    partial_exit:       PartialExitConfig       = field(default_factory=PartialExitConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExitEngineConfig:
        """Load config from a YAML file.  Requires PyYAML (optional dependency)."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("PyYAML is required to load ExitEngineConfig from YAML.") from exc

        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        root = data.get("exit_engine", data)
        return cls._from_dict(root)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> ExitEngineConfig:
        """Build from a plain dict (used by from_yaml)."""
        def _sub(klass, key: str):
            sub = d.get(key, {})
            if isinstance(sub, dict):
                # Only pass fields that the dataclass knows about
                import dataclasses
                known = {f.name for f in dataclasses.fields(klass)}
                return klass(**{k: v for k, v in sub.items() if k in known})
            return klass()

        levels_raw = d.get("partial_exit", {}).get("levels", None)
        partial_cfg = _sub(PartialExitConfig, "partial_exit")
        if levels_raw:
            partial_cfg.levels = [
                PartialExitLevel(**lv) if isinstance(lv, dict) else lv
                for lv in levels_raw
            ]

        return cls(
            enabled=d.get("enabled", True),
            close_full_threshold=float(d.get("close_full_threshold", 60.0)),
            close_partial_threshold=float(d.get("close_partial_threshold", 35.0)),
            reputation=_sub(ReputationConfig, "reputation"),
            trailing_stop=_sub(TrailingStopConfig, "trailing_stop"),
            break_even=_sub(BreakEvenConfig, "break_even"),
            time_exit=_sub(TimeExitConfig, "time_exit"),
            setup_invalidation=_sub(SetupInvalidationConfig, "setup_invalidation"),
            volatility_exit=_sub(VolatilityExitConfig, "volatility_exit"),
            give_back=_sub(GiveBackConfig, "give_back"),
            partial_exit=partial_cfg,
        )
