"""Dataset builder for exit profile selector training.

For each historical signal, all candidate exit profiles are simulated on the
bars that follow the signal bar.  The simulation determines which profile would
have produced the best risk-adjusted outcome.  The result is a list of
(features, label) pairs suitable for supervised ML training.

Simulation methodology
----------------------
Three simulation modes depending on ``exit_mode``:

fixed_tp
    Walk forward bar-by-bar (up to max_holding_bars).
    Each bar: check if bar_low <= sl → SL hit, or bar_high >= tp → TP hit.
    Time-exit at bar close if neither triggered within max_holding_bars.

trailing_stop
    No fixed TP.  Walk forward with a ratcheting trailing stop:
    BUY:  trailing_sl = max(initial_sl,
                            bar_high − trailing_atr_mult × entry_atr)
    SELL: trailing_sl = min(initial_sl,
                            bar_low  + trailing_atr_mult × entry_atr)
    Exit when bar_low <= trailing_sl (BUY) or bar_high >= trailing_sl (SELL),
    or time-exit at max_holding_bars.

hybrid
    Same as fixed_tp — the TP cap is the binding exit constraint for
    labelling purposes; the trailing component is an ExitEngine detail.

Label derivation
----------------
Score = pnl_pips − 0.1 × mae_pips

The profile with the highest score is the label for this signal.
Signals where the best score ≤ 0 are excluded (no edge in that sample).

No look-ahead bias: simulation uses only bars strictly after the signal bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import OrderSide
from metatrade.ml.exit_profile_candidate_generator import ExitProfileCandidateGenerator
from metatrade.ml.exit_profile_contracts import ExitProfileCandidate, ExitProfileContext
from metatrade.ml.exit_profile_feature_builder import ExitProfileFeatureBuilder


@dataclass
class TradeSignalRecord:
    """A historical signal entry for dataset construction.

    Attributes:
        bar_index:     Index into the bar list where the signal fired.
        side:          BUY or SELL.
        entry_price:   Bar close price at the signal (the intended entry).
        symbol:        Forex symbol.
        atr:           ATR(14) value at the signal bar.
        spread_pips:   Spread in pips at the signal bar.
        timestamp_utc: UTC timestamp of the signal bar.
    """

    bar_index: int
    side: OrderSide
    entry_price: Decimal
    symbol: str
    atr: Decimal
    spread_pips: float
    timestamp_utc: datetime


@dataclass
class CandidateSimResult:
    """Simulation outcome for one exit profile candidate on one historical signal."""

    profile_id: str
    exit_mode: str
    hit_tp: bool         # only True for fixed_tp / hybrid profiles
    hit_sl: bool
    pnl_pips: float      # positive = profit, negative = loss
    mae_pips: float      # max adverse excursion (always >= 0)
    mfe_pips: float      # max favorable excursion (always >= 0)
    bars_held: int


@dataclass
class DatasetRow:
    """One labeled row in the training dataset."""

    features: dict[str, float]
    label: str             # optimal profile_id
    signal_index: int      # bar_index of the signal
    symbol: str
    timestamp_utc: datetime


class ExitProfileDatasetBuilder:
    """Builds a labeled training dataset from historical signals and bars.

    Args:
        candidate_generator: Generates exit profile candidates.
        feature_builder:     Extracts features for each context.
        max_holding_bars:    Maximum bars to simulate per trade (default 48).
        pip_digits:          Pip decimal places (default 4 for EURUSD).
    """

    def __init__(
        self,
        candidate_generator: ExitProfileCandidateGenerator | None = None,
        feature_builder: ExitProfileFeatureBuilder | None = None,
        max_holding_bars: int = 48,
        pip_digits: int = 4,
    ) -> None:
        self._generator = candidate_generator or ExitProfileCandidateGenerator()
        self._features = feature_builder or ExitProfileFeatureBuilder()
        self._max_bars = max_holding_bars
        self._pip_digits = pip_digits

    @property
    def pip_size(self) -> Decimal:
        return Decimal("10") ** -self._pip_digits

    # ── Public ────────────────────────────────────────────────────────────────

    def build_dataset(
        self,
        signals: list[TradeSignalRecord],
        bars: list[Bar],
        min_forward_bars: int = 3,
    ) -> list[DatasetRow]:
        """Build training rows for all signals.

        Args:
            signals:          Historical signal records.
            bars:             Complete bar list (oldest first).
            min_forward_bars: Minimum forward bars needed for simulation.

        Returns:
            List of DatasetRow objects.  Signals without a positive-edge
            label are excluded.
        """
        rows: list[DatasetRow] = []

        for sig in signals:
            forward = bars[sig.bar_index + 1 :]
            if len(forward) < min_forward_bars:
                continue

            # Recent bar context for candidate generation and features
            start = max(0, sig.bar_index - 50)
            recent_highs = tuple(b.high for b in bars[start : sig.bar_index + 1])
            recent_lows  = tuple(b.low  for b in bars[start : sig.bar_index + 1])

            ctx = ExitProfileContext(
                symbol=sig.symbol,
                side=sig.side,
                entry_price=sig.entry_price,
                atr=sig.atr,
                spread_pips=sig.spread_pips,
                account_balance=Decimal("10000"),   # normalised reference balance
                timestamp_utc=sig.timestamp_utc,
                pip_digits=self._pip_digits,
                recent_highs=recent_highs,
                recent_lows=recent_lows,
            )

            candidates = self._generator.generate(ctx)
            if not candidates:
                continue

            sim_results = [
                self._simulate(cand, sig, forward)
                for cand in candidates
            ]

            label = self._derive_label(sim_results)
            if label is None:
                continue

            rows.append(DatasetRow(
                features=self._features.build(ctx),
                label=label,
                signal_index=sig.bar_index,
                symbol=sig.symbol,
                timestamp_utc=sig.timestamp_utc,
            ))

        return rows

    # ── Simulation ────────────────────────────────────────────────────────────

    def _simulate(
        self,
        candidate: ExitProfileCandidate,
        sig: TradeSignalRecord,
        forward: list[Bar],
    ) -> CandidateSimResult:
        """Dispatch to the correct simulation based on exit_mode."""
        if candidate.exit_mode == "trailing_stop":
            return self._simulate_trailing(candidate, sig, forward)
        # fixed_tp and hybrid both use TP-based simulation
        return self._simulate_fixed_tp(candidate, sig, forward)

    def _simulate_fixed_tp(
        self,
        candidate: ExitProfileCandidate,
        sig: TradeSignalRecord,
        forward: list[Bar],
    ) -> CandidateSimResult:
        """Simulate a fixed-TP or hybrid profile bar-by-bar.

        Exits when: SL hit, TP hit, or max_holding_bars reached.
        """
        sl = float(candidate.sl_price)
        # tp_price is guaranteed non-None for fixed_tp / hybrid; use a far
        # sentinel value if somehow missing (should not happen in practice).
        tp = float(candidate.tp_price) if candidate.tp_price is not None else (
            float(sig.entry_price) * 100.0
        )
        entry = float(sig.entry_price)
        is_buy = sig.side == OrderSide.BUY
        pip = float(self.pip_size)

        mae = 0.0
        mfe = 0.0
        pnl = 0.0
        hit_tp = False
        hit_sl = False

        for i, bar in enumerate(forward[: self._max_bars]):
            lo = float(bar.low)
            hi = float(bar.high)

            if is_buy:
                if lo <= sl:
                    hit_sl = True
                    pnl = (sl - entry) / pip
                    break
                if hi >= tp:
                    hit_tp = True
                    pnl = (tp - entry) / pip
                    break
                adv = (lo - entry) / pip
                fav = (hi - entry) / pip
            else:
                if hi >= sl:
                    hit_sl = True
                    pnl = (entry - sl) / pip
                    break
                if lo <= tp:
                    hit_tp = True
                    pnl = (entry - tp) / pip
                    break
                adv = (entry - hi) / pip
                fav = (entry - lo) / pip

            mae = min(mae, adv)
            mfe = max(mfe, fav)

            # Last bar: time-exit at close
            if i == self._max_bars - 1 or i == len(forward) - 1:
                close = float(bar.close)
                pnl = ((close - entry) / pip) if is_buy else ((entry - close) / pip)

        return CandidateSimResult(
            profile_id=candidate.profile_id,
            exit_mode=candidate.exit_mode,
            hit_tp=hit_tp,
            hit_sl=hit_sl,
            pnl_pips=pnl,
            mae_pips=abs(mae),
            mfe_pips=max(mfe, 0.0),
            bars_held=min(len(forward), self._max_bars),
        )

    def _simulate_trailing(
        self,
        candidate: ExitProfileCandidate,
        sig: TradeSignalRecord,
        forward: list[Bar],
    ) -> CandidateSimResult:
        """Simulate a trailing-stop profile with a ratcheting stop.

        The trailing stop ratchets in the favorable direction each bar:
            BUY:  trailing_sl = max(trailing_sl,
                                    bar_high − trailing_atr_mult × entry_atr)
            SELL: trailing_sl = min(trailing_sl,
                                    bar_low  + trailing_atr_mult × entry_atr)

        Exits when the trailing stop is hit or max_holding_bars is reached.
        """
        entry = float(sig.entry_price)
        is_buy = sig.side == OrderSide.BUY
        pip = float(self.pip_size)

        trailing_mult = float(
            candidate.trailing_atr_mult
            if candidate.trailing_atr_mult is not None
            else Decimal("2.0")
        )
        atr_dist = float(sig.atr) * trailing_mult

        # Initialise trailing stop at the initial SL
        trailing_sl = float(candidate.sl_price)

        mae = 0.0
        mfe = 0.0
        pnl = 0.0
        hit_sl = False

        for i, bar in enumerate(forward[: self._max_bars]):
            lo = float(bar.low)
            hi = float(bar.high)

            if is_buy:
                # Ratchet: trail upward as price makes new highs
                new_trail = hi - atr_dist
                if new_trail > trailing_sl:
                    trailing_sl = new_trail
                # Check if stop hit (use low vs trailing stop)
                if lo <= trailing_sl:
                    hit_sl = True
                    pnl = (trailing_sl - entry) / pip
                    break
                adv = (lo - entry) / pip
                fav = (hi - entry) / pip
            else:
                # Ratchet: trail downward as price makes new lows
                new_trail = lo + atr_dist
                if new_trail < trailing_sl:
                    trailing_sl = new_trail
                # Check if stop hit
                if hi >= trailing_sl:
                    hit_sl = True
                    pnl = (entry - trailing_sl) / pip
                    break
                adv = (entry - hi) / pip
                fav = (entry - lo) / pip

            mae = min(mae, adv)
            mfe = max(mfe, fav)

            # Last bar: time-exit at close
            if i == self._max_bars - 1 or i == len(forward) - 1:
                close = float(bar.close)
                pnl = ((close - entry) / pip) if is_buy else ((entry - close) / pip)

        return CandidateSimResult(
            profile_id=candidate.profile_id,
            exit_mode=candidate.exit_mode,
            hit_tp=False,   # trailing profiles never hit a TP
            hit_sl=hit_sl,
            pnl_pips=pnl,
            mae_pips=abs(mae),
            mfe_pips=max(mfe, 0.0),
            bars_held=min(len(forward), self._max_bars),
        )

    def _derive_label(self, results: list[CandidateSimResult]) -> str | None:
        """Return the profile_id with the best risk-adjusted score.

        Score = pnl_pips − 0.1 × mae_pips

        Returns None if the best profile has a non-positive expected score
        (no edge in this sample → exclude from training).
        """
        if not results:
            return None

        def score(r: CandidateSimResult) -> float:
            return r.pnl_pips - 0.1 * r.mae_pips

        best = max(results, key=score)
        return best.profile_id if score(best) > 0 else None
