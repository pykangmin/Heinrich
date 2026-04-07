"""DB·run_scoring 연동 백테스트 루프."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import Engine, bindparam, text

from backtest.simulator import calc_period_return, equal_weight_holdings
from data_pipeline.db_manager import DBManager
from scoring.engine import run_scoring

BENCH = "^GSPC"


@contextmanager
def _suppress_scoring_logs():
    root = logging.getLogger()
    prev = root.level
    root.setLevel(logging.WARNING)
    try:
        yield
    finally:
        root.setLevel(prev)


def _fetch_trading_calendar(engine: Engine, start: date, end: date) -> list[date]:
    q = text(
        """
        SELECT DISTINCT date
        FROM daily_prices
        WHERE symbol = :sym
          AND date >= :start
          AND date <= :end
        ORDER BY date
        """
    )
    df = pd.read_sql(q, engine, params={"sym": BENCH, "start": start, "end": end})
    if df.empty or "date" not in df.columns:
        return []
    dcol = df["date"]
    if hasattr(dcol.iloc[0], "date"):
        return [r.date() if hasattr(r, "date") else r for r in dcol.tolist()]
    return [date.fromisoformat(str(r)[:10]) if not isinstance(r, date) else r for r in dcol.tolist()]


def _first_trading_day_of_each_isoweek(calendar: list[date]) -> list[date]:
    seen: dict[tuple[int, int], date] = {}
    for d in sorted(calendar):
        iso = d.isocalendar()
        key = (iso.year, iso.week)
        if key not in seen:
            seen[key] = d
    return sorted(seen.values())


def _fetch_prices_for_dates(engine: Engine, dates: list[date]) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame(columns=["symbol", "date", "close"])
    stmt = text(
        """
        SELECT symbol, date, close
        FROM daily_prices
        WHERE date IN :ds
        """
    ).bindparams(bindparam("ds", expanding=True))
    df = pd.read_sql(stmt, engine, params={"ds": dates})
    return df


def run_backtest(
    start: date,
    end: date,
    cost_pct: float = 0.001,
    top_n: int = 20,
) -> dict[str, Any]:
    top_n = int(os.getenv("BACKTEST_TOP_N", str(top_n)))

    db = DBManager()
    cal = _fetch_trading_calendar(db.engine, start, end)
    rebal_dates = _first_trading_day_of_each_isoweek(cal)
    if len(rebal_dates) < 2:
        return {
            "portfolio": pd.Series(dtype=float),
            "benchmark": pd.Series(dtype=float),
            "rebal_log": [],
        }

    price_df = _fetch_prices_for_dates(db.engine, rebal_dates)
    if price_df.empty:
        return {
            "portfolio": pd.Series(dtype=float),
            "benchmark": pd.Series(dtype=float),
            "rebal_log": [],
        }

    price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
    close_px = price_df.pivot(index="date", columns="symbol", values="close")
    if BENCH not in close_px.columns:
        return {
            "portfolio": pd.Series(dtype=float),
            "benchmark": pd.Series(dtype=float),
            "rebal_log": [],
        }

    portfolio_vals: dict[date, float] = {}
    benchmark_vals: dict[date, float] = {}
    rebal_log: list[dict[str, Any]] = []
    prev_holdings: dict[str, float] = {}

    t0_first = rebal_dates[0]
    portfolio_vals[t0_first] = 1.0
    benchmark_vals[t0_first] = 1.0

    for i in range(len(rebal_dates) - 1):
        t0, t1 = rebal_dates[i], rebal_dates[i + 1]
        row0 = close_px.loc[t0] if t0 in close_px.index else None
        row1 = close_px.loc[t1] if t1 in close_px.index else None
        if row0 is None or row1 is None:
            rebal_log.append(
                {"t0": t0, "t1": t1, "skipped": True, "reason": "missing_benchmark_row"}
            )
            portfolio_vals[t1] = portfolio_vals.get(t0, 1.0)
            benchmark_vals[t1] = benchmark_vals.get(t0, 1.0)
            continue

        g0, g1 = float(row0[BENCH]), float(row1[BENCH])
        bench_ret = g1 / g0 - 1.0 if g0 else 0.0

        with _suppress_scoring_logs():
            candidates_df = run_scoring(t0)

        holdings = equal_weight_holdings(candidates_df, top_n=top_n)
        net_ret = calc_period_return(
            holdings,
            prev_holdings,
            row0,
            row1,
            cost_pct,
        )
        pv0 = portfolio_vals.get(t0, 1.0)
        bv0 = benchmark_vals.get(t0, 1.0)
        portfolio_vals[t1] = pv0 * (1.0 + net_ret)
        benchmark_vals[t1] = bv0 * (1.0 + bench_ret)
        prev_holdings = holdings
        rebal_log.append(
            {
                "t0": t0,
                "t1": t1,
                "net_ret": net_ret,
                "bench_ret": bench_ret,
                "n_holdings": len(holdings),
                "skipped": False,
            }
        )

    port_series = pd.Series(portfolio_vals, name="portfolio").sort_index()
    bench_series = pd.Series(benchmark_vals, name="benchmark").sort_index()
    return {
        "portfolio": port_series,
        "benchmark": bench_series,
        "rebal_log": rebal_log,
    }
