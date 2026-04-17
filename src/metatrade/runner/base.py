"""Base runner — shared logic across backtest, paper, and live modes.

The BaseRunner implements the core signal→decision→execution pipeline:
1. Receive bar data (via process_bar())
2. Run each technical/ML module to produce AnalysisSignals
3. Feed signals to the consensus engine
4. Evaluate risk (kill switch, limits, position sizing)
5. Return the RiskDecision (approved or vetoed) to the caller

Each runner subclass calls process_bar() with appropriate account data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal

from metatrade.alerting.telegram_alerter import TelegramAlerter
from metatrade.consensus.adaptive_threshold import AdaptiveThresholdManager
from metatrade.consensus.config import ConsensusConfig
from metatrade.consensus.engine import ConsensusEngine
from metatrade.consensus.market_accuracy_tracker import MarketAccuracyTracker
from metatrade.core.contracts.account import AccountState
from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.risk import RiskDecision, RiskVeto
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import ConsensusMode, OrderSide, RunMode, SignalDirection
from metatrade.core.log import get_logger
from metatrade.observability.store import TelemetryStore
from metatrade.risk.config import RiskConfig
from metatrade.risk.manager import RiskManager
from metatrade.runner.config import RunnerConfig
from metatrade.technical_analysis.interface import ITechnicalModule

log = get_logger(__name__)


@dataclass
class RunStats:
    """Accumulated statistics for a runner session."""

    bars_processed: int = 0
    signals_generated: int = 0
    trades_attempted: int = 0
    trades_executed: int = 0
    trades_vetoed: int = 0
    kill_switch_activations: int = 0
    weight_updates: int = 0           # signals evaluated against market


class BaseRunner:
    """Core orchestration logic shared by all runner modes.

    Subclasses call `process_bar()` to run the full pipeline for one tick.

    Args:
        config:  RunnerConfig for this session.
        modules: List of ITechnicalModule instances to query each bar.
    """

    def __init__(
        self,
        config: RunnerConfig,
        modules: list[ITechnicalModule],
        auto_weight: bool = True,
        weight_forward_bars: int = 5,
        alerter: TelegramAlerter | None = None,
        telemetry: TelemetryStore | None = None,
        session_id: str | None = None,
        timeframe: str | None = None,
        intermarket_engine: object | None = None,  # IntermarketEngine | None
        exit_profile_generator: object | None = None,  # ExitProfileCandidateGenerator | None
        exit_profile_selector: object | None = None,   # ExitProfileSelector | None
    ) -> None:
        """
        Args:
            config:                 RunnerConfig for this session.
            modules:                List of ITechnicalModule instances.
            auto_weight:            If True (default), use DYNAMIC_VOTE consensus
                                    and automatically update module weights based
                                    on how well each module predicted market
                                    direction over the next ``weight_forward_bars``
                                    bars.  Weights are proportional to accuracy:
                                    a large error reduces the weight more than a
                                    small one.
            weight_forward_bars:    Bars ahead used to evaluate signal accuracy.
                                    Must match the ML labelling horizon (default 5).
            alerter:                Optional TelegramAlerter for operational alerts.
                                    None (default) = silent, no alerts sent.
            intermarket_engine:     Optional ``IntermarketEngine`` for cross-pair
                                    correlation and exposure control.  When set,
                                    it is evaluated between the consensus step and
                                    the risk manager.  ``None`` = disabled
                                    (default, fully backward-compatible).
            exit_profile_generator: Optional ``ExitProfileCandidateGenerator``.
                                    When set together with ``exit_profile_selector``,
                                    steps 6-7 of the pipeline run after risk
                                    approval and override the default Chandelier
                                    SL/TP with the selected exit profile.
                                    ``None`` = disabled (default, fully
                                    backward-compatible).
            exit_profile_selector:  Optional ``ExitProfileSelector``.  Required
                                    together with ``exit_profile_generator`` to
                                    enable exit profile selection.
        """
        self._config = config
        self._modules = modules

        consensus_cfg = ConsensusConfig(
            mode=ConsensusMode.DYNAMIC_VOTE if auto_weight else ConsensusMode.SIMPLE_VOTE,
            threshold=config.consensus_threshold,
            min_signals=config.min_signals,
        )
        self._consensus = ConsensusEngine(consensus_cfg)

        # Adaptive threshold manager — per-module confidence thresholds that
        # self-adjust based on market accuracy (lower when right, raise when wrong).
        # Persisted to the TelemetryStore so thresholds survive restarts.
        self._threshold_mgr = AdaptiveThresholdManager(
            store=telemetry,
        )

        # Market accuracy tracker — evaluates each signal N bars later and
        # (1) updates the dynamic weight of the module in the consensus engine,
        # (2) updates the adaptive threshold for the module.
        def _on_eval(module_id: str, score: float) -> None:
            self._consensus.update_performance_scored(module_id, score)
            self._threshold_mgr.on_eval(module_id, score)

        if auto_weight:
            self._tracker: MarketAccuracyTracker | None = MarketAccuracyTracker(
                update_callback=_on_eval,
                forward_bars=weight_forward_bars,
            )
        else:
            self._tracker = None

        risk_cfg = RiskConfig(
            max_risk_pct=config.max_risk_pct,
            daily_loss_limit_pct=config.daily_loss_limit_pct,
            auto_kill_drawdown_pct=config.auto_kill_drawdown_pct,
            max_open_positions=config.max_open_positions,
        )
        self._risk = RiskManager(risk_cfg)
        self._stats = RunStats()

        # ── Cooldown counter ──────────────────────────────────────────────────
        # Counts down the remaining bars to skip after an approved trade.
        self._cooldown_remaining: int = 0

        # ── Drawdown recovery ─────────────────────────────────────────────────
        # Tracks the highest equity seen so far to measure peak-to-trough DD.
        self._peak_equity: Decimal = Decimal("0")

        # ── Daily circuit breaker ─────────────────────────────────────────────
        # Resets each day; blocks new entries once intraday loss >= limit.
        self._today_date: date | None = None
        self._session_start_equity: Decimal = Decimal("0")

        # ── Alerter ───────────────────────────────────────────────────────────
        self._alerter: TelegramAlerter | None = alerter
        self._telemetry = telemetry
        self._session_id = session_id
        self._timeframe = timeframe

        # ── Intermarket engine (optional) ──────────────────────────────────────
        # Type hint kept as object to avoid hard import at module level.
        self._intermarket = intermarket_engine

        # ── Exit profile selection (optional, Steps 6-7) ───────────────────────
        # Both generator and selector must be set together.  When only one is
        # provided the feature is effectively disabled (no hard error).
        self._exit_profile_generator = exit_profile_generator
        self._exit_profile_selector = exit_profile_selector

    @property
    def stats(self) -> RunStats:
        return self._stats

    @property
    def risk_manager(self) -> RiskManager:
        return self._risk

    @property
    def accuracy_tracker(self) -> MarketAccuracyTracker | None:
        """The market accuracy tracker, or None if auto_weight=False."""
        return self._tracker

    @property
    def threshold_manager(self) -> AdaptiveThresholdManager:
        """The adaptive threshold manager for per-module confidence thresholds."""
        return self._threshold_mgr

    def get_module_weight(self, module_id: str) -> float | None:
        """Current dynamic weight for a module (None if auto_weight=False)."""
        return self._consensus.get_module_weight(module_id)

    def system_confidence(self) -> float:
        """Aggregate system confidence in [0.0, 1.0].

        Derived from the average dynamic module weight, normalized from the
        [5, 95] weight range to [0, 1].  Returns 0.5 (neutral) when no
        modules are loaded or when auto_weight is disabled (weights are None).
        """
        weights = [
            w
            for m in self._modules
            if (w := self.get_module_weight(m.module_id)) is not None
        ]
        if not weights:
            return 0.5
        avg = sum(weights) / len(weights)
        # 5 → 0.0, 50 → 0.5, 95 → 1.0
        return max(0.0, min(1.0, (avg - 5.0) / 90.0))

    def _compute_dynamic_position_cap(self, account: AccountState) -> int:
        """Compute the effective max-open-positions cap for this bar.

        Combines three signals to produce a value in [0, config.max_open_positions]:

        1. **Margin health** — ``free_margin / equity`` ratio.
           Critical  (< 5 %)  → 0 (block).
           Tight     (< 20 %) → ×0.33.
           Moderate  (< 50 %) → ×0.67.
           Healthy   (≥ 50 %) → ×1.0.

        2. **Drawdown** — peak-to-trough vs ``drawdown_recovery_threshold_pct``.
           Below threshold          → ×1.0.
           threshold … 2×threshold  → ×0.5.
           ≥ 2×threshold            → 0 (block).

        3. **System confidence** — average module weight normalised to [0, 1].
           < 0.30 (declining)  → ×0.33.
           < 0.45 (neutral)    → ×0.67.
           ≥ 0.45 (confident)  → ×1.0.

        Returns:
            0  → block all new entries (margin critical or severe drawdown).
            1…max_open_positions → dynamic effective limit.
        """
        base = self._config.max_open_positions

        # ── 1. Margin health ─────────────────────────────────────────────────
        if account.equity > Decimal("0"):
            margin_ratio = float(account.free_margin / account.equity)
        else:
            margin_ratio = 0.0

        if margin_ratio < 0.05:
            log.warning(
                "dynamic_cap_margin_critical",
                margin_ratio=round(margin_ratio, 4),
                free_margin=float(account.free_margin),
                equity=float(account.equity),
            )
            return 0
        elif margin_ratio < 0.20:
            margin_factor = 0.33
        elif margin_ratio < 0.50:
            margin_factor = 0.67
        else:
            margin_factor = 1.0

        # ── 2. Drawdown ──────────────────────────────────────────────────────
        threshold = self._config.drawdown_recovery_threshold_pct
        if self._peak_equity > Decimal("0"):
            dd_pct = float(
                (self._peak_equity - account.equity) / self._peak_equity
            )
        else:
            dd_pct = 0.0

        if dd_pct >= 2 * threshold:
            log.warning(
                "dynamic_cap_severe_drawdown",
                dd_pct=round(dd_pct * 100, 2),
                threshold_pct=round(threshold * 100, 2),
            )
            return 0
        elif dd_pct >= threshold:
            dd_factor = 0.5
        else:
            dd_factor = 1.0

        # ── 3. System confidence ─────────────────────────────────────────────
        conf = self.system_confidence()
        if conf < 0.30:
            conf_factor = 0.33
        elif conf < 0.45:
            conf_factor = 0.67
        else:
            conf_factor = 1.0

        effective = max(1, int(base * margin_factor * dd_factor * conf_factor))
        cap = min(effective, base)

        if cap < base:
            log.debug(
                "dynamic_cap_reduced",
                base=base,
                cap=cap,
                margin_factor=round(margin_factor, 2),
                dd_factor=round(dd_factor, 2),
                conf_factor=round(conf_factor, 2),
                confidence=round(conf, 3),
            )

        return cap

    # ── Kill switch DB sync ───────────────────────────────────────────────────

    def _sync_kill_switch_from_db(self) -> None:
        """Read the shared kill_switch_command from SQLite and sync in-memory state.

        Called once per bar so that a dashboard (or CLI) signal is picked up
        within one candle.  If level > 0, escalates the in-memory kill switch;
        if level == 0 and the switch is active, resets it.
        """
        if self._telemetry is None:
            return
        try:
            cmd = self._telemetry.read_kill_command()
        except Exception as exc:
            log.warning("kill_switch_db_sync_failed", error=str(exc))
            return

        level = int(cmd.get("level", 0))
        reason = str(cmd.get("reason", "") or "Kill switch activated via dashboard")
        activated_by = str(cmd.get("activated_by", "") or "dashboard")

        from metatrade.core.enums import KillSwitchLevel
        if level > 0:
            try:
                ks_level = KillSwitchLevel(level)
            except ValueError:
                return
            if ks_level > self._risk.kill_switch.level:
                self._risk.kill_switch.activate(
                    ks_level,
                    reason=reason,
                    activated_by=activated_by,
                )
        elif self._risk.kill_switch.is_active():
            try:
                self._risk.kill_switch.reset(activated_by=activated_by)
            except Exception:
                # HARD_KILL requires force_reset; that must be done deliberately
                pass

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def process_bar(
        self,
        bars: list[Bar],
        account_balance: Decimal,
        account_equity: Decimal,
        free_margin: Decimal,
        timestamp_utc: datetime | None = None,
        open_positions: int = 0,
        current_spread_pips: float | None = None,
        run_mode: RunMode = RunMode.PAPER,
    ) -> RiskDecision | None:
        """Run the full pipeline for one bar tick.

        Args:
            bars:                 Recent bars (oldest first, newest last).
            account_balance:      Current account balance.
            account_equity:       Balance + unrealized PnL.
            free_margin:          Capital available for new positions.
            timestamp_utc:        UTC timestamp; defaults to now.
            open_positions:       Number of currently open positions.
            current_spread_pips:  Current spread in pips (for spread check).
            run_mode:             Execution context (PAPER, LIVE, BACKTEST).

        Returns:
            RiskDecision (approved=True or approved=False with veto) if consensus
            produced an actionable signal. None if consensus is HOLD.
        """
        if timestamp_utc is None:
            timestamp_utc = datetime.now(UTC)

        # Sync kill switch from shared DB (cross-process signal from dashboard).
        self._sync_kill_switch_from_db()

        self._stats.bars_processed += 1

        used_margin = (
            account_balance - free_margin
            if account_balance >= free_margin
            else Decimal("0")
        )
        account = AccountState(
            balance=account_balance,
            equity=account_equity,
            free_margin=free_margin,
            used_margin=used_margin,
            timestamp_utc=timestamp_utc,
            open_positions_count=open_positions,
            run_mode=run_mode,
        )
        self._record_account_snapshot(account)

        # Compute the dynamic position cap once per bar so the risk manager
        # can use margin health, drawdown, and system confidence together.
        dynamic_cap = self._compute_dynamic_position_cap(account)

        # ── Guard: spread filter ──────────────────────────────────────────────
        # Block new entries when the broker spread is too wide.
        max_spread = self._config.max_spread_pips
        if (
            max_spread > 0
            and current_spread_pips is not None
            and current_spread_pips > max_spread
        ):
            log.debug(
                "spread_filter: %.2f pips > max %.2f — skipping bar",
                current_spread_pips,
                max_spread,
            )
            return None

        # ── Guard: daily circuit breaker (absolute USD) ───────────────────────
        # Resets the daily reference equity at the start of each new calendar day.
        # Once intraday losses exceed daily_loss_limit_usd, all new entries are blocked.
        bar_date = timestamp_utc.date()
        if self._today_date != bar_date:
            self._today_date = bar_date
            self._session_start_equity = account_equity
        daily_limit = Decimal(str(self._config.daily_loss_limit_usd))
        if daily_limit > Decimal("0"):
            daily_loss = self._session_start_equity - account_equity
            if daily_loss >= daily_limit:
                log.debug(
                    "daily_circuit_breaker: loss=%.2f >= limit=%.2f — skipping bar",
                    float(daily_loss),
                    float(daily_limit),
                )
                return None

        # ── Guard: signal cooldown ────────────────────────────────────────────
        # After any approved trade, stay out of the market for N bars.
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            log.debug(
                "signal_cooldown: %d bars remaining — skipping bar",
                self._cooldown_remaining + 1,
            )
            return None

        current_close = bars[-1].close

        # ── Step 1a: Evaluate past signals against current price ──────────────
        # Must run BEFORE collecting new signals so that weights are already
        # updated when the consensus engine votes on the current bar.
        if self._tracker is not None:
            evaluations = self._tracker.on_bar(current_close, timestamp_utc)
            self._stats.weight_updates += len(evaluations)

        # ── Step 1b: Collect signals ──────────────────────────────────────────
        signals: list[AnalysisSignal] = []
        for module in self._modules:
            if len(bars) < module.min_bars:
                continue
            try:
                sig = module.analyse(bars, timestamp_utc)
                signals.append(sig)
                self._stats.signals_generated += 1
            except Exception as exc:
                log.warning(
                    "Module %s raised during analyse(): %s",
                    module.module_id,
                    exc,
                )

        if not signals:
            return None

        # ── Step 1c: Record ALL signals for future threshold/weight evaluation ─
        # We record signals BEFORE filtering so that even excluded signals are
        # evaluated — this is what drives the threshold to recover when a module
        # that was filtered out would have been right.
        if self._tracker is not None:
            self._tracker.record_signals(signals, current_close, timestamp_utc)

        # ── Step 1d: Apply per-module adaptive confidence threshold filter ────
        # Signals whose confidence is below their module's current threshold are
        # excluded from the consensus vote (but still evaluated in Step 1c above).
        signals = self._threshold_mgr.filter_signals(signals)

        if not signals:
            return None

        # ── Step 2: Consensus ─────────────────────────────────────────────────
        consensus_result = self._consensus.evaluate(
            symbol=self._config.symbol,
            signals=signals,
            timestamp_utc=timestamp_utc,
        )

        if not consensus_result.is_actionable:
            self._record_decision(signals, consensus_result, None, run_mode)
            return None

        direction = consensus_result.final_direction
        if direction == SignalDirection.BUY:
            side = OrderSide.BUY
        elif direction == SignalDirection.SELL:
            side = OrderSide.SELL
        else:
            return None

        # ── Step 3b: Intermarket evaluation ──────────────────────────────────
        # Inserted between consensus and risk.  When no engine is configured
        # (the common single-symbol case), this block is a no-op.
        im_risk_mult: Decimal | None = None
        if self._intermarket is not None:
            try:
                from metatrade.core.contracts.position import Position as _Position
                from metatrade.intermarket.engine import IntermarketEngine
                if isinstance(self._intermarket, IntermarketEngine):
                    # Collect open positions from the broker/paper state if
                    # available.  BaseRunner does not own positions directly,
                    # so we pass an empty list; subclasses may override via
                    # _get_open_positions() if they need richer integration.
                    open_positions: list[_Position] = self._get_open_positions()
                    # Estimate a nominal lot size for exposure-delta computation.
                    # We use 0.01 lots as a placeholder — actual sizing happens
                    # in the risk manager; the intermarket engine only needs the
                    # direction and a representative magnitude.
                    _nominal_lot = Decimal("0.01")
                    im_decision = self._intermarket.evaluate(
                        symbol=self._config.symbol,
                        side=side,
                        lot_size=_nominal_lot,
                        open_positions=open_positions,
                        timestamp_utc=timestamp_utc,
                        session_id=self._session_id,
                    )
                    if not im_decision.approved:
                        self._stats.trades_vetoed += 1
                        log.info(
                            "intermarket_veto",
                            symbol=self._config.symbol,
                            side=side.value,
                            reason=im_decision.reason,
                        )
                        vetoed = RiskDecision(
                            approved=False,
                            symbol=self._config.symbol,
                            side=side,
                            timestamp_utc=timestamp_utc,
                            veto=RiskVeto(
                                reason=im_decision.reason,
                                veto_code="INTERMARKET_VETO",
                                timestamp_utc=timestamp_utc,
                                context={
                                    "correlated_positions": list(
                                        im_decision.correlated_positions
                                    ),
                                    "warnings": list(im_decision.warnings),
                                },
                            ),
                        )
                        self._record_decision(signals, consensus_result, vetoed, run_mode)
                        return vetoed
                    im_risk_mult = im_decision.risk_multiplier
            except Exception as exc:
                # Never let intermarket evaluation break the trading pipeline.
                log.warning("intermarket_eval_error", error=str(exc))

        # ── Step 3c: Drawdown-recovery tracking ───────────────────────────────
        # Keep a running peak-equity and detect when we're in recovery mode.
        if account_equity > self._peak_equity:
            self._peak_equity = account_equity

        in_recovery = False
        if self._peak_equity > Decimal("0"):
            dd_pct = float(
                (self._peak_equity - account_equity) / self._peak_equity
            )
            if dd_pct >= self._config.drawdown_recovery_threshold_pct:
                in_recovery = True
                log.debug(
                    "drawdown_recovery: equity=%.2f peak=%.2f dd=%.2f%% — "
                    "scaling risk by %.2f",
                    float(account_equity),
                    float(self._peak_equity),
                    dd_pct * 100,
                    self._config.drawdown_recovery_risk_mult,
                )

        # ── Step 4: Risk evaluation ───────────────────────────────────────────
        self._stats.trades_attempted += 1

        entry_price = bars[-1].close
        sl_price = self._estimate_sl(bars, entry_price, side)
        current_atr = self._compute_current_atr(bars)

        decision = self._risk.evaluate(
            symbol=self._config.symbol,
            side=side,
            entry_price=entry_price,
            sl_price=sl_price,
            account=account,
            timestamp_utc=timestamp_utc,
            spread_pips=current_spread_pips,
            current_atr=current_atr,
            intermarket_risk_mult=im_risk_mult,
            max_positions_override=dynamic_cap,
        )

        # ── Step 4b: Apply drawdown-recovery lot-size reduction ──────────────
        if decision.approved and in_recovery and decision.position_size is not None:
            mult = Decimal(str(self._config.drawdown_recovery_risk_mult))
            scaled_ps = replace(
                decision.position_size,
                lot_size=decision.position_size.lot_size * mult,
            )
            decision = replace(decision, position_size=scaled_ps)

        # ── Steps 6-7: Exit profile selection ────────────────────────────────
        # Runs only when both generator and selector are configured.
        # Overrides the default Chandelier SL/TP computed in Step 4 with the
        # exit profile chosen by the selector (ML or deterministic fallback).
        # The lot size is recomputed to preserve the configured risk_pct given
        # the new SL distance.
        if decision.approved and self._exit_profile_generator is not None and self._exit_profile_selector is not None:
            decision = self._apply_exit_profile(
                decision=decision,
                bars=bars,
                account=account,
                timestamp_utc=timestamp_utc,
                current_spread_pips=current_spread_pips,
            )

        # ── Step 4c: Update stats, alerter, and cooldown ─────────────────────
        if decision.approved:
            self._stats.trades_executed += 1
            # Fire operational alert if alerter is configured
            if self._alerter is not None and decision.position_size is not None:
                ps = decision.position_size
                self._alerter.alert_trade_opened(
                    symbol=self._config.symbol,
                    side=side.value,
                    lot_size=float(ps.lot_size),
                    entry=float(entry_price),
                    sl=float(ps.stop_loss_price),
                    tp=float(ps.take_profit_price) if ps.take_profit_price else None,
                )
            # Activate cooldown — prevents re-entry for N bars after a trade.
            if self._config.signal_cooldown_bars > 0:
                self._cooldown_remaining = self._config.signal_cooldown_bars
        else:
            self._stats.trades_vetoed += 1
            if (
                decision.veto is not None
                and "KILL_SWITCH" in decision.veto.veto_code
            ):
                self._stats.kill_switch_activations += 1

        self._record_decision(signals, consensus_result, decision, run_mode)
        return decision

    def _record_account_snapshot(self, account: AccountState) -> None:
        if self._telemetry is None or self._session_id is None:
            return
        try:
            self._telemetry.record_account_snapshot(
                session_id=self._session_id,
                symbol=self._config.symbol,
                timeframe=self._timeframe,
                account=account,
                details={"bars_processed": self._stats.bars_processed},
            )
        except Exception as exc:
            log.warning("account_snapshot_telemetry_failed: %s", exc)

    def _record_decision(
        self,
        signals: list[AnalysisSignal],
        consensus_result,
        decision: RiskDecision | None,
        run_mode: RunMode,
    ) -> None:
        if self._telemetry is None or self._session_id is None:
            return
        try:
            self._telemetry.record_decision(
                session_id=self._session_id,
                symbol=self._config.symbol,
                timeframe=self._timeframe,
                run_mode=run_mode.value,
                signals=signals,
                consensus_result=consensus_result,
                decision=decision,
            )
        except Exception as exc:
            log.warning("decision_telemetry_failed: %s", exc)

    def _get_open_positions(self) -> list:
        """Return currently open positions for intermarket exposure accounting.

        The base implementation returns an empty list, which means the
        intermarket engine operates with no existing-position context —
        only the current candidate trade's currency delta is evaluated.

        Subclasses that maintain a position registry (e.g. PaperRunner)
        can override this to return their actual open-position list.
        This is a deliberate extension point, not a required override.
        """
        return []

    def _estimate_sl(
        self,
        bars: list[Bar],
        entry_price: Decimal,
        side: OrderSide,
    ) -> Decimal:
        """Estimate a stop-loss price using the Chandelier Exit method.

        SL = entry ± ATR(chandelier_atr_period) × chandelier_atr_mult

        The Chandelier Exit uses actual volatility to set the stop distance,
        adapting to current market conditions rather than a fixed pip value.

        Falls back to a 20-pip default (0.0020) when there are insufficient
        bars to compute ATR.
        """
        period = self._config.chandelier_atr_period
        mult = Decimal(str(self._config.chandelier_atr_mult))
        fallback_distance = Decimal("0.0020")

        if len(bars) >= period + 1:
            try:
                from metatrade.technical_analysis.indicators.atr import atr as compute_atr
                highs  = [b.high  for b in bars]
                lows   = [b.low   for b in bars]
                closes = [b.close for b in bars]
                atr_vals = compute_atr(highs, lows, closes, period)
                curr_atr = atr_vals[-1]
                if curr_atr > Decimal("0"):
                    distance = curr_atr * mult
                    if side == OrderSide.BUY:
                        return entry_price - distance
                    return entry_price + distance
            except Exception as exc:
                log.debug("chandelier_sl fallback: %s", exc)

        # Fallback: fixed 20-pip distance
        if side == OrderSide.BUY:
            return entry_price - fallback_distance
        return entry_price + fallback_distance

    def _apply_exit_profile(
        self,
        decision: RiskDecision,
        bars: list[Bar],
        account: AccountState,
        timestamp_utc: datetime,
        current_spread_pips: float | None,
    ) -> RiskDecision:
        """Apply exit profile selection (Steps 6-7) to an approved decision.

        Generates exit profile candidates for the current context, selects the
        best one via the configured selector, then recomputes the lot size to
        preserve the configured risk fraction with the new SL distance.

        If no valid candidate can be selected, the original decision is returned
        unchanged — the Chandelier-derived SL/TP from Step 4 remains in effect.

        Args:
            decision:            Approved RiskDecision from Step 4/5.
            bars:                Recent bar history.
            account:             Current AccountState.
            timestamp_utc:       Bar timestamp.
            current_spread_pips: Spread in pips (may be None).

        Returns:
            Updated RiskDecision with the selected exit profile's SL/TP and
            recomputed lot size, or the original decision when selection fails.
        """
        try:
            from metatrade.ml.exit_profile_candidate_generator import (
                ExitProfileCandidateGenerator,
            )
            from metatrade.ml.exit_profile_contracts import ExitProfileContext
            from metatrade.ml.exit_profile_selector import ExitProfileSelector

            if not isinstance(self._exit_profile_generator, ExitProfileCandidateGenerator):
                return decision
            if not isinstance(self._exit_profile_selector, ExitProfileSelector):
                return decision

            assert decision.position_size is not None  # guarded by decision.approved

            entry_price = bars[-1].close
            current_atr = self._compute_current_atr(bars)
            if current_atr is None:
                return decision  # not enough bars to compute ATR

            # Determine pip_digits from symbol name (JPY pairs = 2, others = 4)
            symbol = self._config.symbol
            pip_digits = 2 if symbol.upper().endswith("JPY") else 4

            start = max(0, len(bars) - 51)
            recent_highs = tuple(b.high for b in bars[start:])
            recent_lows  = tuple(b.low  for b in bars[start:])

            ctx = ExitProfileContext(
                symbol=symbol,
                side=decision.side,
                entry_price=entry_price,
                atr=current_atr,
                spread_pips=float(current_spread_pips or 0.0),
                account_balance=account.balance,
                timestamp_utc=timestamp_utc,
                pip_digits=pip_digits,
                recent_highs=recent_highs,
                recent_lows=recent_lows,
            )

            candidates = self._exit_profile_generator.generate(ctx)
            if not candidates:
                log.debug("exit_profile_no_candidates", symbol=symbol)
                return decision

            selected = self._exit_profile_selector.select(candidates, ctx)
            if selected is None:
                log.debug("exit_profile_selector_returned_none", symbol=symbol)
                return decision

            profile = selected.candidate

            # ── Enforce minimum SL pip distance ──────────────────────────────
            # Exit-profile candidates can produce very tight SLs (e.g. 1-3 pip
            # swing levels on M1).  A 2-pip SL at 1% risk yields 2+ lots on a
            # $4k account, which then gets clamped by the broker and creates
            # volume / margin errors.  Widening the SL here prevents over-sizing
            # BEFORE the lot calculation, which is the only safe place to fix it.
            min_sl_pips_val = Decimal(str(self._config.min_sl_pips))
            if min_sl_pips_val > 0 and profile.sl_pips < min_sl_pips_val:
                pip_size = Decimal("10") ** -pip_digits
                new_sl_dist = min_sl_pips_val * pip_size
                if decision.side == OrderSide.BUY:
                    adjusted_sl_price = entry_price - new_sl_dist
                else:
                    adjusted_sl_price = entry_price + new_sl_dist
                # Rescale TP proportionally if one was set
                if profile.tp_price is not None and profile.tp_pips is not None:
                    rr = profile.risk_reward
                    new_tp_dist = new_sl_dist * rr
                    if decision.side == OrderSide.BUY:
                        adjusted_tp_price = entry_price + new_tp_dist
                    else:
                        adjusted_tp_price = entry_price - new_tp_dist
                    new_tp_pips = min_sl_pips_val * rr
                else:
                    adjusted_tp_price = profile.tp_price
                    new_tp_pips = profile.tp_pips
                log.warning(
                    "exit_profile_sl_widened_to_min",
                    symbol=symbol,
                    profile_id=profile.profile_id,
                    original_sl_pips=str(profile.sl_pips),
                    min_sl_pips=str(min_sl_pips_val),
                    adjusted_sl=str(adjusted_sl_price),
                )
                profile = replace(
                    profile,
                    sl_price=adjusted_sl_price,
                    sl_pips=min_sl_pips_val,
                    tp_price=adjusted_tp_price,
                    tp_pips=new_tp_pips,
                )

            # Recompute lot size so the configured risk_pct is preserved
            # with the newly selected SL distance.
            sizing_capital = max(account.equity, account.balance)
            try:
                new_ps = self._risk.recompute_sizing(
                    balance=sizing_capital,
                    entry_price=entry_price,
                    sl_price=profile.sl_price,
                    side=decision.side,
                )
            except ValueError as exc:
                log.warning(
                    "exit_profile_recompute_sizing_failed",
                    error=str(exc),
                    profile_id=profile.profile_id,
                )
                return decision

            # Attach the selected TP.
            # Trailing profiles have tp_price=None (ExitEngine manages exit
            # dynamically).  Fall back to the standard TP from the original
            # PositionSizer result so that MT5 holds a safety-net order: if the
            # ExitEngine ever stops or crashes, the broker will still close the
            # position at the configured risk:reward level.
            fallback_tp = decision.position_size.take_profit_price if decision.position_size else None
            effective_tp = profile.tp_price if profile.tp_price is not None else fallback_tp
            new_ps = replace(new_ps, take_profit_price=effective_tp)

            log.info(
                "exit_profile_selected",
                symbol=symbol,
                profile_id=profile.profile_id,
                exit_mode=profile.exit_mode,
                sl_pips=str(profile.sl_pips),
                tp_pips=str(profile.tp_pips) if profile.tp_pips else "none (fallback to standard)",
                rr=str(profile.risk_reward),
                confidence=selected.confidence,
                method=selected.method,
                lot_size=str(new_ps.lot_size),
                tp_is_fallback=profile.tp_price is None,
            )

            return replace(decision, position_size=new_ps)

        except Exception as exc:
            # Never let exit profile selection break the trading pipeline.
            log.warning("exit_profile_apply_failed", error=str(exc))
            return decision

    def _compute_current_atr(self, bars: list[Bar]) -> Decimal | None:
        """Return the most recent ATR value, or None if insufficient data.

        Used to supply vol-scaled position sizing with the current market
        volatility without recomputing ATR a second time.
        """
        period = self._config.chandelier_atr_period
        if len(bars) < period + 1:
            return None
        try:
            from metatrade.technical_analysis.indicators.atr import atr as compute_atr
            highs  = [b.high  for b in bars]
            lows   = [b.low   for b in bars]
            closes = [b.close for b in bars]
            atr_vals = compute_atr(highs, lows, closes, period)
            val = atr_vals[-1]
            return val if val > Decimal("0") else None
        except Exception as exc:
            log.debug("_compute_current_atr failed: %s", exc)
            return None
