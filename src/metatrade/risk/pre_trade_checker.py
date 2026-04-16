"""Pre-trade risk checker.

Validates a proposed trade against all configured risk limits before the
position sizer runs. Returns a RiskVeto if any check fails.

Checks performed (in order):
    1. Kill switch — blocks if level >= TRADE_GATE.
    2. Max open positions — blocks if already at or above limit.
    3. Daily loss limit — blocks if drawdown exceeds configured fraction.
    4. Spread too wide — blocks if current spread exceeds max_spread_pips.
    5. Free margin — blocks if insufficient margin available.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from metatrade.core.contracts.account import AccountState
from metatrade.core.contracts.risk import RiskVeto
from metatrade.core.enums import KillSwitchLevel
from metatrade.core.log import get_logger
from metatrade.risk.config import RiskConfig
from metatrade.risk.kill_switch import KillSwitchManager

log = get_logger(__name__)


class PreTradeChecker:
    """Validates a proposed trade against all risk limits.

    Args:
        config:       RiskConfig with all limit thresholds.
        kill_switch:  KillSwitchManager to query current state.
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        kill_switch: KillSwitchManager | None = None,
    ) -> None:
        self._cfg = config or RiskConfig()
        self._ks = kill_switch or KillSwitchManager()

    def check(
        self,
        account: AccountState,
        timestamp_utc: datetime,
        spread_pips: float | None = None,
        max_positions_override: int | None = None,
    ) -> RiskVeto | None:
        """Run all pre-trade checks.

        Args:
            account:                Current account snapshot.
            timestamp_utc:          Evaluation timestamp for the veto record.
            spread_pips:            Current spread in pips (None = skip spread check).
            max_positions_override: Effective position cap computed dynamically by
                                    the runner (e.g. based on margin health and
                                    system confidence).  ``None`` = use config
                                    default.  ``0`` = block all new positions.

        Returns:
            RiskVeto if the trade should be blocked, None if approved.
        """
        veto = (
            self._check_kill_switch(timestamp_utc)
            or self._check_max_positions(account, timestamp_utc, max_positions_override)
            or self._check_daily_loss(account, timestamp_utc)
            or self._check_spread(spread_pips, timestamp_utc)
            or self._check_free_margin(account, timestamp_utc)
        )
        if veto:
            log.warning(
                "pre_trade_veto",
                code=veto.veto_code,
                reason=veto.reason,
            )
        return veto

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_kill_switch(self, ts: datetime) -> RiskVeto | None:
        state = self._ks.state
        if state.blocks_new_trades:
            return RiskVeto(
                reason=f"Kill switch {state.level.name} active: {state.reason}",
                veto_code="KILL_SWITCH_ACTIVE",
                timestamp_utc=ts,
                context={"level": state.level.name},
            )
        return None

    def _check_max_positions(
        self,
        account: AccountState,
        ts: datetime,
        max_positions_override: int | None = None,
    ) -> RiskVeto | None:
        if max_positions_override is not None and max_positions_override == 0:
            # Dynamic cap of 0 signals a critical condition (margin/drawdown).
            return RiskVeto(
                reason="Dynamic position cap: 0 — critical margin or drawdown, no new entries",
                veto_code="DYNAMIC_CAP_ZERO",
                timestamp_utc=ts,
                context={"dynamic_cap": 0},
            )
        limit = (
            max_positions_override
            if max_positions_override is not None
            else self._cfg.max_open_positions
        )
        if limit > 0 and account.open_positions_count >= limit:
            return RiskVeto(
                reason=(
                    f"Max open positions reached: "
                    f"{account.open_positions_count}/{limit}"
                ),
                veto_code="MAX_POSITIONS",
                timestamp_utc=ts,
                context={"open": account.open_positions_count, "limit": limit},
            )
        return None

    def _check_daily_loss(self, account: AccountState, ts: datetime) -> RiskVeto | None:
        if self._cfg.daily_loss_limit_pct <= 0:
            return None
        loss_limit = account.balance * Decimal(str(self._cfg.daily_loss_limit_pct))
        drawdown = account.drawdown_from_equity
        if drawdown >= loss_limit:
            return RiskVeto(
                reason=(
                    f"Daily loss limit breached: "
                    f"drawdown={drawdown:.2f} >= limit={loss_limit:.2f}"
                ),
                veto_code="DAILY_LOSS_LIMIT",
                timestamp_utc=ts,
                context={
                    "drawdown": str(drawdown),
                    "limit": str(loss_limit),
                    "pct": self._cfg.daily_loss_limit_pct,
                },
            )
        return None

    def _check_spread(self, spread_pips: float | None, ts: datetime) -> RiskVeto | None:
        if spread_pips is None or self._cfg.max_spread_pips <= 0:
            return None
        if spread_pips > self._cfg.max_spread_pips:
            return RiskVeto(
                reason=(
                    f"Spread too wide: {spread_pips:.1f} pips "
                    f"> max {self._cfg.max_spread_pips:.1f} pips"
                ),
                veto_code="SPREAD_TOO_WIDE",
                timestamp_utc=ts,
                context={"spread": spread_pips, "max": self._cfg.max_spread_pips},
            )
        return None

    def _check_free_margin(self, account: AccountState, ts: datetime) -> RiskVeto | None:
        if self._cfg.min_free_margin_pct <= 0:
            return None
        # Use equity (balance + credit + floating PnL) as the reference capital.
        # Raw balance excludes broker credit, which is the primary capital on most
        # demo accounts. Equity represents actual purchasing power.
        reference = max(account.equity, account.balance)
        required = reference * Decimal(str(self._cfg.min_free_margin_pct))
        if account.free_margin < required:
            return RiskVeto(
                reason=(
                    f"Insufficient free margin: "
                    f"{account.free_margin:.2f} < required {required:.2f} "
                    f"({self._cfg.min_free_margin_pct:.0%} of equity {reference:.2f})"
                ),
                veto_code="INSUFFICIENT_MARGIN",
                timestamp_utc=ts,
                context={
                    "free_margin": str(account.free_margin),
                    "required": str(required),
                    "equity": str(account.equity),
                },
            )
        return None
