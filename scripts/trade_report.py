"""Trade journal report — query and display closed trades from the local SQLite journal.

Usage:
    # Report odierno (default)
    python scripts/trade_report.py

    # Ultimi 7 giorni
    python scripts/trade_report.py --period weekly

    # Report mensile
    python scripts/trade_report.py --period monthly

    # Lista trade raw con date personalizzate
    python scripts/trade_report.py --from 2026-04-01 --to 2026-04-15 --trades

    # Filtra per simbolo e run mode
    python scripts/trade_report.py --period monthly --symbol EURUSD --mode LIVE

    # Importa i trade reali da MT5 (richiede MT5 connesso)
    python scripts/trade_report.py --sync-mt5

    # Tutti i periodi disponibili (massimo storico)
    python scripts/trade_report.py --period monthly --limit 120
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metatrade.observability.store import TelemetryStore  # noqa: E402


# ── Formatting helpers ────────────────────────────────────────────────────────

_GREEN = "\033[92m"
_RED   = "\033[91m"
_CYAN  = "\033[96m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_RESET = "\033[0m"

_NO_COLOR = not sys.stdout.isatty()


def _color(text: str, code: str) -> str:
    if _NO_COLOR:
        return text
    return f"{code}{text}{_RESET}"


def _pnl_color(value: float | None) -> str:
    if value is None:
        return "  —"
    sign = "+" if value >= 0 else ""
    formatted = f"{sign}{value:,.2f}"
    if value > 0:
        return _color(formatted, _GREEN)
    if value < 0:
        return _color(formatted, _RED)
    return formatted


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _fmt_dt(unix_ts: int | None) -> str:
    if unix_ts is None:
        return "—"
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _divider(width: int = 80) -> str:
    return _color("─" * width, _DIM)


# ── Report sections ───────────────────────────────────────────────────────────

def _print_stats_block(stats: dict) -> None:
    total = stats.get("total_trades") or 0
    wins = stats.get("winning_trades") or 0
    losses = stats.get("losing_trades") or 0
    win_rate = stats.get("win_rate")
    net_pnl = stats.get("total_net_pnl")
    gross_pnl = stats.get("total_gross_pnl")
    fees = (stats.get("total_commission") or 0) + (stats.get("total_swap") or 0)
    avg_win = stats.get("avg_win")
    avg_loss = stats.get("avg_loss")
    pf = stats.get("profit_factor")
    best = stats.get("best_trade")
    worst = stats.get("worst_trade")
    total_pips = stats.get("total_pips")
    avg_pips = stats.get("avg_pips")

    print(f"\n  Totale trade : {_color(str(total), _BOLD)}")
    print(f"  Vincenti     : {wins}  ({_pct(win_rate)})")
    print(f"  Perdenti     : {losses}")
    print(f"  Net P&L      : {_pnl_color(net_pnl)}")
    if gross_pnl is not None and fees:
        print(f"  Gross P&L    : {_pnl_color(gross_pnl)}   fees: {_pnl_color(-fees)}")
    if total_pips is not None:
        print(f"  Pips totali  : {_pnl_color(total_pips)}   media: {_pnl_color(avg_pips)}")
    if avg_win is not None:
        print(f"  Media win    : {_pnl_color(avg_win)}")
    if avg_loss is not None:
        print(f"  Media loss   : {_pnl_color(avg_loss)}")
    if pf is not None:
        print(f"  Profit Factor: {pf:.2f}")
    if best is not None:
        print(f"  Best trade   : {_pnl_color(best)}")
    if worst is not None:
        print(f"  Worst trade  : {_pnl_color(worst)}")


def _print_period_table(rows: list[dict], period_label: str) -> None:
    if not rows:
        print("  Nessun dato disponibile.")
        return
    col_period = max(len(period_label), max(len(str(r.get(period_label, ""))) for r in rows)) + 2
    header = (
        f"  {period_label.upper():<{col_period}}"
        f"{'TRADE':>6}  {'WIN%':>6}  {'NET P&L':>10}  {'PIPS':>8}  {'FEES':>8}"
    )
    print(_color(header, _BOLD))
    print("  " + _divider(len(header) - 2))
    for row in rows:
        period_val = str(row.get(period_label, ""))
        trades = row.get("trades") or 0
        win_rate = row.get("win_rate")
        net_pnl = row.get("net_pnl")
        pips = row.get("pips")
        fees = row.get("fees") or 0.0

        pnl_str = _pnl_color(net_pnl)
        pips_str = _pnl_color(pips)
        win_str = _pct(win_rate)
        fees_str = _color(f"{fees:,.2f}", _DIM)

        # plain widths for alignment (strip ANSI when computing padding)
        print(
            f"  {period_val:<{col_period}}"
            f"{trades:>6}  {win_str:>6}  {pnl_str:>10}  {pips_str:>8}  {fees_str:>8}"
        )


def _print_trades_table(trades: list[dict]) -> None:
    if not trades:
        print("  Nessun trade trovato per il periodo selezionato.")
        return
    header = (
        f"  {'#':>4}  {'SYMBOL':<8}  {'DIR':<5}  {'LOTS':>5}  "
        f"{'ENTRY TIME':<17}  {'ENTRY':>8}  {'EXIT TIME':<17}  {'EXIT':>8}  "
        f"{'PIPS':>7}  {'NET P&L':>10}  {'REASON':<18}"
    )
    print(_color(header, _BOLD))
    print("  " + _divider(len(header) - 2))
    for i, t in enumerate(trades, 1):
        net = t.get("net_pnl")
        pips = t.get("pnl_pips")
        symbol = t.get("symbol", "")
        direction = t.get("direction", "")
        lots = t.get("lot_size", 0)
        entry_ts = t.get("entry_time_utc")
        exit_ts = t.get("exit_time_utc")
        entry_px = t.get("entry_price") or 0
        exit_px = t.get("exit_price") or 0
        reason = (t.get("exit_reason") or "")[:18]

        print(
            f"  {i:>4}  {symbol:<8}  {direction:<5}  {lots:>5.2f}  "
            f"{_fmt_dt(entry_ts):<17}  {entry_px:>8.5f}  "
            f"{_fmt_dt(exit_ts):<17}  {exit_px:>8.5f}  "
            f"{_pnl_color(pips):>7}  {_pnl_color(net):>10}  {reason:<18}"
        )


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trade journal report per MetaTrade.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--period",
        choices=["daily", "weekly", "monthly"],
        default=None,
        help="Aggregazione per periodo (default: mostra solo i dati del giorno corrente)",
    )
    parser.add_argument(
        "--from",
        dest="from_date",
        metavar="YYYY-MM-DD",
        help="Data inizio (UTC)",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        metavar="YYYY-MM-DD",
        help="Data fine (UTC, default: oggi)",
    )
    parser.add_argument("--symbol", help="Filtra per simbolo (es. EURUSD)")
    parser.add_argument(
        "--mode",
        dest="run_mode",
        help="Filtra per run mode: LIVE | PAPER | BACKTEST",
    )
    parser.add_argument(
        "--trades",
        action="store_true",
        help="Mostra la lista dei singoli trade (oltre al riepilogo)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=90,
        help="Numero massimo di periodi/trade da mostrare (default: 90)",
    )
    parser.add_argument(
        "--sync-mt5",
        action="store_true",
        help="Importa i trade chiusi da MT5 nel journal locale (richiede MT5 connesso)",
    )
    parser.add_argument(
        "--sync-days",
        type=int,
        default=90,
        metavar="N",
        help="Quanti giorni di storia MT5 importare (default: 90)",
    )
    return parser.parse_args()


def _ts_from_date(date_str: str, *, end_of_day: bool) -> int:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if end_of_day and fmt == "%Y-%m-%d":
                dt = dt.replace(hour=23, minute=59, second=59)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    print(f"[ERRORE] Data non riconosciuta: {date_str!r}. Usa il formato YYYY-MM-DD.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    args = _parse_args()
    store = TelemetryStore.from_env()

    try:
        # ── Sync from MT5 ─────────────────────────────────────────────────────
        if args.sync_mt5:
            _sync_from_mt5(store, days_back=args.sync_days)
            print()

        # ── Determine date range ──────────────────────────────────────────────
        now_utc = datetime.now(timezone.utc)
        from_ts: int | None = None
        to_ts: int | None = None

        if args.from_date:
            from_ts = _ts_from_date(args.from_date, end_of_day=False)
        if args.to_date:
            to_ts = _ts_from_date(args.to_date, end_of_day=True)

        # Default: se nessun periodo/date → report giornaliero di oggi
        if args.period is None and not args.from_date and not args.to_date:
            today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            from_ts = int(today_start.timestamp())
            to_ts = int(now_utc.timestamp())
            period_label = f"Oggi  ({today_start.strftime('%Y-%m-%d')})"
        elif args.period:
            period_label = f"Riepilogo {args.period}"
        else:
            period_label = "Periodo personalizzato"

        # ── Header ────────────────────────────────────────────────────────────
        print()
        print(_color("━" * 80, _CYAN))
        title = "MetaTrade — Trade Journal"
        if args.symbol:
            title += f"  [{args.symbol}]"
        if args.run_mode:
            title += f"  [{args.run_mode}]"
        print(_color(f"  {title}", _BOLD))
        print(_color("━" * 80, _CYAN))

        # ── Overall stats ─────────────────────────────────────────────────────
        print(_color(f"\n  {period_label} — statistiche aggregate\n", _BOLD))
        stats = store.get_trade_summary(
            from_ts=from_ts,
            to_ts=to_ts,
            symbol=args.symbol,
            run_mode=args.run_mode,
        )
        _print_stats_block(stats)

        # ── Periodic breakdown ────────────────────────────────────────────────
        if args.period:
            print()
            print(_color(f"\n  Dettaglio {args.period}\n", _BOLD))
            rows = store.get_periodic_summary(
                period=args.period,
                symbol=args.symbol,
                run_mode=args.run_mode,
                limit=args.limit,
            )
            period_key = {"daily": "date", "weekly": "week", "monthly": "month"}[args.period]
            _print_period_table(rows, period_key)

        # ── Trade list ────────────────────────────────────────────────────────
        if args.trades or (args.period is None and not args.sync_mt5):
            print()
            print(_color("\n  Singoli trade\n", _BOLD))
            trades = store.list_trades(
                from_ts=from_ts,
                to_ts=to_ts,
                symbol=args.symbol,
                run_mode=args.run_mode,
                closed_only=True,
                limit=args.limit,
            )
            _print_trades_table(trades)

        print()
        print(_color("━" * 80, _CYAN))
        print()

    finally:
        store.close()


def _sync_from_mt5(store: TelemetryStore, days_back: int) -> None:
    """Import closed deals from MT5 runtime reader into the local journal."""
    print(_color("  Importazione trade da MT5...", _BOLD))
    try:
        from metatrade.observability.mt5_runtime import MT5RuntimeReader
        mt5 = MT5RuntimeReader()
        deals = mt5.get_closed_deals(limit=5000)
    except Exception as exc:
        print(f"  [ERRORE] Impossibile leggere la storia MT5: {exc}", file=sys.stderr)
        return

    if not deals:
        print("  Nessun deal trovato in MT5.")
        return

    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
    imported = 0
    skipped = 0
    for deal in deals:
        deal_time = deal.get("time") or deal.get("time_utc") or 0
        if deal_time < cutoff_ts:
            skipped += 1
            continue
        entry = deal.get("entry", "")
        if entry not in ("out", "in/out", 1, 2):
            skipped += 1
            continue
        broker_id = str(deal.get("deal") or deal.get("ticket") or "")
        symbol = deal.get("symbol") or "UNKNOWN"
        direction = "LONG" if deal.get("type") in (0, "DEAL_TYPE_BUY") else "SHORT"
        lot_size = float(deal.get("volume") or 0)
        price_close = float(deal.get("price") or 0)
        profit = float(deal.get("profit") or 0)
        commission = float(deal.get("commission") or 0)
        swap = float(deal.get("swap") or 0)
        close_dt = datetime.fromtimestamp(float(deal_time), tz=timezone.utc)
        trade_id = f"mt5-{broker_id}" if broker_id else f"mt5-{deal_time}"
        store.record_closed_trade(
            trade_id=trade_id,
            session_id=None,
            symbol=symbol,
            run_mode="LIVE",
            direction=direction,
            lot_size=lot_size,
            entry_price=float(deal.get("price_open") or price_close),
            entry_time_utc=datetime.fromtimestamp(
                float(deal.get("time_open") or deal_time), tz=timezone.utc
            ),
            exit_price=price_close,
            exit_time_utc=close_dt,
            pnl_currency=profit,
            commission=commission,
            swap=swap,
            exit_reason=str(deal.get("reason") or "mt5"),
            broker_order_id=broker_id,
            details={k: deal.get(k) for k in ("comment", "magic", "position_id") if deal.get(k)},
        )
        imported += 1

    print(f"  Importati: {imported}  |  Saltati: {skipped}")


if __name__ == "__main__":
    main()
