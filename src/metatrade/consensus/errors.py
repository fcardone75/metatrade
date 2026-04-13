"""Consensus module errors."""

from __future__ import annotations

from metatrade.core.errors import MetaTradeError


class ConsensusError(MetaTradeError):
    """Base error for the consensus engine."""
