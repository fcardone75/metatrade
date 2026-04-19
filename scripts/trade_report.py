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

from metatrade.core.log import configure_logging, get_logger  # noqa: E402
from metatrade.observability.store import TelemetryStore  # noqa: E402

log = get_logger(__name__)


# ── Formatting helpers ────────────────────────────────────────────────────────

def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _fmt_dt(unix_ts: int | None) -> str:
    if unix_ts is None:
        return "n/a"
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _pnl_str(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:,.2f}"


# ── Report sections ───────────────────────────────────────────────────────────

def _log_stats_block(stats: dict) -> None:
    log.info(
        "trade_summary",
        total_trades=stats.get("total_trades") or 0,
        winning=stats.get("winning_trades") or 0,
        losing=stats.get("losing_trades") or 0,
        win_rate=_pct(stats.get("win_rate")),
        net_pnl=_pnl_str(stats.get("total_net_pnl")),
        gross_pnl=_pnl_str(stats.get("total_gross_pnl")),
        fees=_pnl_str(
            (stats.get("total_commission") or 0) + (stats.get("total_swap") or 0)
        ),
        avg_win=_pnl_str(stats.get("avg_win")),
        avg_loss=_pnl_str(stats.get("avg_loss")),
        profit_factor=f"{stats['profit_factor']:.2f}" if stats.get("profit_factor") else "n/a",
        best_trade=_pnl_str(stats.get("best_trade")),
        worst_trade=_pnl_str(stats.get("worst_trade")),
        total_pips=_pnl_str(stats.get("total_pips")),
        avg_pips=_pnl_str(stats.get("avg_pips")),
    )


def _log_period_table(rows: list[dict], period_label: str) -> None:
    if not rows:
        log.info("period_table_empty")
        return
    for row in rows:
        log.info(
            "period_row",
            period=str(row.get(period_label, "")),
            trades=row.get("trades") or 0,
            win_rate=_pct(row.get("win_rate")),
            net_pnl=_pnl_str(row.get("net_pnl")),
            pips=_pnl_str(row.get("pips")),
            fees=f"{row.get('fees') or 0.0:,.2f}",
        )


def _log_trades_table(trades: list[dict]) -> None:
    if not trades:
        log.info("trades_table_empty")
        return
    for i, t in enumerate(trades, 1):
        log.info(
            "trade_row",
            n=i,
            symbol=t.get("symbol", ""),
            direction=t.get("direction", ""),
            lot_size=t.get("lot_size", 0),
            entry_time=_fmt_dt(t.get("entry_time_utc")),
            entry_price=t.get("entry_price") or 0,
            exit_time=_fmt_dt(t.get("exit_time_utc")),
            exit_price=t.get("exit_price") or 0,
            pips=_pnl_str(t.get("pnl_pips")),
            net_pnl=_pnl_str(t.get("net_pnl")),
            exit_reason=(t.get("exit_reason") or "")[:40],
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
    log.error("invalid_date_format", value=date_str, expected="YYYY-MM-DD")
    sys.exit(1)


def main() -> None:
    configure_logging()
    args = _parse_args()
    store = TelemetryStore.from_env()

    try:
        # ── Sync from MT5 ─────────────────────────────────────────────────────
        if args.sync_mt5:
            _sync_from_mt5(store, days_back=args.sync_days)

        # ── Determine date range ──────────────────────────────────────────────
        now_utc = datetime.now(timezone.utc)
        from_ts: int | None = None
        to_ts: int | None = None

        if args.from_date:
            from_ts = _ts_from_date(args.from_date, end_of_day=False)
        if args.to_date:
            to_ts = _ts_from_date(args.to_date, end_of_day=True)

        if args.period is None and not args.from_date and not args.to_date:
            today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            from_ts = int(today_start.timestamp())
            to_ts = int(now_utc.timestamp())
            period_label = f"today ({today_start.strftime('%Y-%m-%d')})"
        elif args.period:
            period_label = args.period
        else:
            period_label = "custom"

        log.info(
            "report_start",
            period_label=period_label,
            symbol=args.symbol,
            run_mode=args.run_mode,
        )

        # ── Overall stats ─────────────────────────────────────────────────────
        stats = store.get_trade_summary(
            from_ts=from_ts,
            to_ts=to_ts,
            symbol=args.symbol,
            run_mode=args.run_mode,
        )
        _log_stats_block(stats)

        # ── Periodic breakdown ────────────────────────────────────────────────
        if args.period:
            rows = store.get_periodic_summary(
                period=args.period,
                symbol=args.symbol,
                run_mode=args.run_mode,
                limit=args.limit,
            )
            period_key = {"daily": "date", "weekly": "week", "monthly": "month"}[args.period]
            _log_period_table(rows, period_key)

        # ── Trade list ────────────────────────────────────────────────────────
        if args.trades or (args.period is None and not args.sync_mt5):
            trades = store.list_trades(
                from_ts=from_ts,
                to_ts=to_ts,
                symbol=args.symbol,
                run_mode=args.run_mode,
                closed_only=True,
                limit=args.limit,
            )
            _log_trades_table(trades)

    finally:
        store.close()


def _sync_from_mt5(store: TelemetryStore, days_back: int) -> None:
    """Import closed deals from MT5 runtime reader into the local journal."""
    log.info("mt5_sync_start", days_back=days_back)
    try:
        from metatrade.observability.mt5_runtime import MT5RuntimeReader
        mt5 = MT5RuntimeReader()
        deals = mt5.get_closed_deals(limit=5000)
    except Exception as exc:
        log.error("mt5_sync_failed", error=str(exc))
        return

    if not deals:
        log.warning("mt5_no_deals_found")
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

    log.info("mt5_sync_complete", imported=imported, skipped=skipped)


if __name__ == "__main__":
    main()
