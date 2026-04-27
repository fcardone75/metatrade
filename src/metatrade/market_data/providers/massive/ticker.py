"""Forex ticker normalization for Massive aggregates path."""


def normalize_forex_ticker(symbol: str) -> str:
    """Return API ticker e.g. C:EURUSD from user symbol EURUSD."""
    s = symbol.strip().upper()
    if ":" in s:
        return s
    if len(s) == 6 and s.isalpha():
        return f"C:{s}"
    return f"C:{s}"


def strip_massive_prefix(ticker: str) -> str:
    """C:EURUSD -> EURUSD for filenames."""
    t = ticker.strip().upper()
    if t.startswith("C:"):
        return t[2:]
    return t
