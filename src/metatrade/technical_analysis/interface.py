"""Abstract interface for technical analysis modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.errors import InsufficientDataError, ModuleNotReadyError


class ITechnicalModule(ABC):
    """Contract for all technical analysis modules.

    A technical module receives a list of Bars (most recent last) and
    produces a single AnalysisSignal representing its current view.

    Modules must track their own warmup state — calling analyse() before
    the minimum required bars are available raises ModuleNotReadyError.
    """

    @property
    @abstractmethod
    def module_id(self) -> str:
        """Unique stable identifier for this module."""

    @property
    @abstractmethod
    def min_bars(self) -> int:
        """Minimum number of bars required to produce a signal."""

    @abstractmethod
    def analyse(
        self,
        bars: list[Bar],
        timestamp_utc: datetime,
    ) -> AnalysisSignal:
        """Analyse the bar series and produce a signal.

        Args:
            bars:          Historical bars, oldest first, newest last.
                           Must contain at least `min_bars` entries.
            timestamp_utc: UTC time of the analysis (from the system clock).

        Returns:
            AnalysisSignal with direction and confidence.

        Raises:
            ModuleNotReadyError:   If len(bars) < min_bars.
            InsufficientDataError: If data is present but unusable.
        """

    def _require_min_bars(self, bars: list[Bar]) -> None:
        """Raise ModuleNotReadyError if not enough bars are available."""
        if len(bars) < self.min_bars:
            raise ModuleNotReadyError(
                message=(
                    f"{self.module_id}: need {self.min_bars} bars, "
                    f"got {len(bars)}"
                ),
                code="MODULE_NOT_READY",
                context={
                    "module_id": self.module_id,
                    "required": self.min_bars,
                    "provided": len(bars),
                },
            )
