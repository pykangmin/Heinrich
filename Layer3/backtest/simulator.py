"""백테스트 구간 수익·턴오버 계산 (DB/IO 없음)."""

from __future__ import annotations

import pandas as pd


def equal_weight_holdings(candidates_df: pd.DataFrame, top_n: int = 20) -> dict[str, float]:
    """`total_heinrich` 상위 top_n → 동일 비중 {symbol: 1/n}."""
    if candidates_df is None or candidates_df.empty:
        return {}
    if "symbol" not in candidates_df.columns or "total_heinrich" not in candidates_df.columns:
        return {}
    df = candidates_df.dropna(subset=["symbol", "total_heinrich"]).copy()
    if df.empty:
        return {}
    df = df.sort_values("total_heinrich", ascending=False).head(int(top_n))
    if df.empty:
        return {}
    n = len(df)
    w = 1.0 / n
    return {str(r["symbol"]): w for _, r in df.iterrows()}


def _price_ok(series: pd.Series, sym: str) -> bool:
    return sym in series.index and pd.notna(series.loc[sym])


def calc_period_return(
    holdings: dict[str, float],
    prev_holdings: dict[str, float],
    prices_t0: pd.Series,
    prices_t1: pd.Series,
    cost_pct: float,
) -> float:
    """유효 가격 종목만 gross 수익률·비용 반영. 턴오버는 prev∪holdings 전 범위에서 L1/2."""
    holdings = holdings or {}
    prev_holdings = prev_holdings or {}

    syms = [
        s
        for s in holdings
        if _price_ok(prices_t0, s) and _price_ok(prices_t1, s)
    ]
    raw_w = {s: holdings[s] for s in syms}
    tot = sum(raw_w.values())
    if tot > 0:
        w_new_real = {s: raw_w[s] / tot for s in syms}
    else:
        w_new_real = {}

    gross_ret = 0.0
    if w_new_real:
        gross_ret = sum(
            w_new_real[s] * (float(prices_t1.loc[s]) / float(prices_t0.loc[s]) - 1.0)
            for s in syms
        )

    all_syms = set(prev_holdings.keys()) | set(holdings.keys())
    if not all_syms:
        return 0.0

    w_new_full = {s: 0.0 for s in all_syms}
    for s in syms:
        w_new_full[s] = w_new_real.get(s, 0.0)
    w_old_full = {s: float(prev_holdings.get(s, 0.0)) for s in all_syms}
    turnover = 0.5 * sum(abs(w_new_full[s] - w_old_full[s]) for s in all_syms)

    return gross_ret - cost_pct * turnover
