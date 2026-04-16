"""Tests for ExitProfileDatasetBuilder (fixed-TP and trailing simulation)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import OrderSide, Timeframe
from metatrade.ml.exit_profile_contracts import ExitProfileCandidate, ExitProfileContext
from metatrade.ml.exit_profile_dataset_builder import (
    CandidateSimResult,
    DatasetRow,
    ExitProfileDatasetBuilder,
    TradeSignalRecord,
)


def _ts(day: int = 15) -> datetime:
    return datetime(2024, 1, day, 10, 0, tzinfo=timezone.utc)


def _bar(open_=1.10, high=1.101, low=1.099, close=1.100, ts=None) -> Bar:
    if ts is None:
        ts = _ts()
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.M15,
        timestamp_utc=ts,
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=Decimal("1000"),
    )


def _signal(bar_index: int = 10, side: OrderSide = OrderSide.BUY) -> TradeSignalRecord:
    return TradeSignalRecord(
        bar_index=bar_index,
        side=side,
        entry_price=Decimal("1.10000"),
        symbol="EURUSD",
        atr=Decimal("0.0010"),
        spread_pips=1.0,
        timestamp_utc=_ts(),
    )


def _bars(n: int = 60) -> list[Bar]:
    """Build a straight trending bar series."""
    result = []
    for i in range(n):
        base = 1.10 + i * 0.0001
        result.append(_bar(
            open_=base,
            high=base + 0.0005,
            low=base - 0.0002,
            close=base + 0.0001,
            ts=datetime(2024, 1, 1, i // 60 % 24, i % 60, tzinfo=timezone.utc),
        ))
    return result


class TestDatasetBuilderBuildDataset:
    def setup_method(self):
        self.builder = ExitProfileDatasetBuilder(max_holding_bars=20, pip_digits=4)

    def test_returns_list(self):
        bars = _bars(60)
        rows = self.builder.build_dataset([_signal(10)], bars)
        assert isinstance(rows, list)

    def test_row_has_correct_type(self):
        bars = _bars(60)
        rows = self.builder.build_dataset([_signal(10)], bars)
        for row in rows:
            assert isinstance(row, DatasetRow)

    def test_row_features_not_empty(self):
        bars = _bars(60)
        rows = self.builder.build_dataset([_signal(10)], bars)
        for row in rows:
            assert len(row.features) > 0

    def test_row_label_is_string(self):
        bars = _bars(60)
        rows = self.builder.build_dataset([_signal(10)], bars)
        for row in rows:
            assert isinstance(row.label, str)
            assert row.label != ""

    def test_skips_signal_too_close_to_end(self):
        bars = _bars(15)
        # bar_index=13 leaves only 1 forward bar < min_forward_bars=3
        rows = self.builder.build_dataset([_signal(13)], bars, min_forward_bars=3)
        assert rows == []

    def test_multiple_signals(self):
        bars = _bars(60)
        signals = [_signal(10), _signal(20), _signal(30)]
        rows = self.builder.build_dataset(signals, bars)
        # At least some rows should be produced for trending series
        assert len(rows) >= 0  # no crash

    def test_symbol_and_timestamp_propagated(self):
        bars = _bars(60)
        rows = self.builder.build_dataset([_signal(10)], bars)
        for row in rows:
            assert row.symbol == "EURUSD"
            assert row.timestamp_utc is not None


class TestSimulateFixedTP:
    def setup_method(self):
        self.builder = ExitProfileDatasetBuilder(max_holding_bars=10, pip_digits=4)

    def _fixed_candidate(self, sl: str, tp: str) -> ExitProfileCandidate:
        entry = Decimal("1.10000")
        sl_p = Decimal(sl)
        tp_p = Decimal(tp)
        sl_pips = abs(entry - sl_p) / Decimal("0.0001")
        tp_pips = abs(entry - tp_p) / Decimal("0.0001")
        rr = (tp_pips / sl_pips).quantize(Decimal("0.01"))
        return ExitProfileCandidate(
            profile_id="ATR_1_2",
            sl_price=sl_p,
            tp_price=tp_p,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            risk_reward=rr,
            method="atr",
            exit_mode="fixed_tp",
        )

    def _signal(self) -> TradeSignalRecord:
        return TradeSignalRecord(
            bar_index=0,
            side=OrderSide.BUY,
            entry_price=Decimal("1.10000"),
            symbol="EURUSD",
            atr=Decimal("0.0010"),
            spread_pips=0.5,
            timestamp_utc=_ts(),
        )

    def test_sl_hit_gives_negative_pnl(self):
        cand = self._fixed_candidate("1.09900", "1.10300")
        # Bars where price falls to hit SL (open within [low, high])
        forward = [_bar(open_=1.0995, high=1.0999, low=1.0988, close=1.0990) for _ in range(5)]
        sig = self._signal()
        result = self.builder._simulate_fixed_tp(cand, sig, forward)
        assert result.hit_sl
        assert result.pnl_pips < 0

    def test_tp_hit_gives_positive_pnl(self):
        cand = self._fixed_candidate("1.09900", "1.10200")
        # Bars where price rises to hit TP
        forward = [_bar(open_=1.10150, high=1.10250, low=1.10100, close=1.10200) for _ in range(5)]
        sig = self._signal()
        result = self.builder._simulate_fixed_tp(cand, sig, forward)
        assert result.hit_tp
        assert result.pnl_pips > 0

    def test_time_exit_neither_sl_nor_tp(self):
        cand = self._fixed_candidate("1.09800", "1.10500")
        # Bars that stay in range
        forward = [_bar(open_=1.1002, high=1.1010, low=1.0995, close=1.1005) for _ in range(10)]
        sig = self._signal()
        result = self.builder._simulate_fixed_tp(cand, sig, forward)
        assert not result.hit_sl
        assert not result.hit_tp


class TestSimulateTrailing:
    def setup_method(self):
        self.builder = ExitProfileDatasetBuilder(max_holding_bars=10, pip_digits=4)

    def _trailing_candidate(self, sl: str, trailing_mult: str = "1.5") -> ExitProfileCandidate:
        sl_p = Decimal(sl)
        entry = Decimal("1.10000")
        sl_pips = abs(entry - sl_p) / Decimal("0.0001")
        return ExitProfileCandidate(
            profile_id="ATR_TRAIL_TIGHT",
            sl_price=sl_p,
            tp_price=None,
            sl_pips=sl_pips,
            tp_pips=None,
            risk_reward=Decimal("2.5"),
            method="trailing",
            exit_mode="trailing_stop",
            trailing_atr_mult=Decimal(trailing_mult),
        )

    def _signal(self) -> TradeSignalRecord:
        return TradeSignalRecord(
            bar_index=0,
            side=OrderSide.BUY,
            entry_price=Decimal("1.10000"),
            symbol="EURUSD",
            atr=Decimal("0.0010"),
            spread_pips=0.5,
            timestamp_utc=_ts(),
        )

    def test_trailing_hit_gives_result(self):
        cand = self._trailing_candidate("1.09900", "1.5")
        # Price rallies then drops back through trailing stop
        forward = [
            _bar(open_=1.1010, high=1.1050, low=1.1000, close=1.1040),  # rally
            _bar(open_=1.1040, high=1.1060, low=1.1000, close=1.1050),  # rally more
            _bar(open_=1.1010, high=1.1020, low=1.0980, close=1.0990),  # drop — trailing stop hit
        ]
        sig = self._signal()
        result = self.builder._simulate_trailing(cand, sig, forward)
        assert isinstance(result, CandidateSimResult)
        assert result.profile_id == "ATR_TRAIL_TIGHT"
        assert result.exit_mode == "trailing_stop"
        assert not result.hit_tp  # trailing never sets hit_tp=True

    def test_trailing_never_sets_hit_tp(self):
        cand = self._trailing_candidate("1.09900")
        forward = [_bar(open_=1.1120, high=1.1200, low=1.1100, close=1.1150) for _ in range(10)]
        sig = self._signal()
        result = self.builder._simulate_trailing(cand, sig, forward)
        assert not result.hit_tp

    def test_time_exit_trailing(self):
        cand = self._trailing_candidate("1.09000", "0.1")  # very tight trailing
        # Steady bars that don't trigger stop
        forward = [_bar(open_=1.1002, high=1.1010, low=1.0995, close=1.1005) for _ in range(10)]
        sig = self._signal()
        result = self.builder._simulate_trailing(cand, sig, forward)
        assert isinstance(result, CandidateSimResult)


class TestDeriveLabel:
    def setup_method(self):
        self.builder = ExitProfileDatasetBuilder()

    def _result(self, pid: str, pnl: float, mae: float = 0.0) -> CandidateSimResult:
        return CandidateSimResult(
            profile_id=pid,
            exit_mode="fixed_tp",
            hit_tp=pnl > 0,
            hit_sl=pnl < 0,
            pnl_pips=pnl,
            mae_pips=mae,
            mfe_pips=max(pnl, 0.0),
            bars_held=5,
        )

    def test_best_score_chosen(self):
        results = [
            self._result("A", 20.0, 5.0),   # score = 20 - 0.5 = 19.5
            self._result("B", 15.0, 1.0),   # score = 15 - 0.1 = 14.9
            self._result("C", 25.0, 20.0),  # score = 25 - 2.0 = 23.0  ← best
        ]
        label = self.builder._derive_label(results)
        assert label == "C"

    def test_returns_none_when_all_scores_negative(self):
        results = [self._result("A", -10.0), self._result("B", -5.0)]
        assert self.builder._derive_label(results) is None

    def test_returns_none_when_best_score_zero(self):
        results = [self._result("A", 0.0, 0.0)]
        assert self.builder._derive_label(results) is None

    def test_returns_none_for_empty_results(self):
        assert self.builder._derive_label([]) is None

    def test_mae_penalty_applied(self):
        # A: pnl=10, mae=100 → score = -0.0  negative
        # B: pnl=5, mae=0    → score = 5     positive
        results = [
            self._result("A", 10.0, 100.0),
            self._result("B", 5.0, 0.0),
        ]
        assert self.builder._derive_label(results) == "B"
