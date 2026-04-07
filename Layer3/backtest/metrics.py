"""성과 지표 (순수 계산, DB 없음)."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def calc_metrics(
    portfolio_curve: pd.Series,
    benchmark_curve: pd.Series,
    rf: float = 0.05,
) -> dict[str, Any]:
    """주간 리밸런싱 구간 수익률 기준 지표.

    cagr: (end/start)^(52/n_periods) - 1
    sharpe: (cagr - rf) / (weekly_std * sqrt(52))
    mdd: min(port / port.cummax() - 1) (음수)
    beta: cov(p, b) / var(b)
    alpha: cagr - (rf + beta * (bench_cagr - rf))
    """
    out: dict[str, Any] = {}
    pc = portfolio_curve.dropna().sort_index()
    bc = benchmark_curve.dropna().sort_index()
    idx = pc.index.intersection(bc.index)
    if len(idx) < 2:
        keys = (
            "cagr",
            "sharpe",
            "mdd",
            "beta",
            "alpha",
            "win_rate",
            "total_return",
            "bench_cagr",
            "bench_total_return",
        )
        return {k: float("nan") for k in keys}

    pc = pc.loc[idx]
    bc = bc.loc[idx]
    n_periods = len(pc) - 1
    if n_periods <= 0 or pc.iloc[0] == 0:
        return {k: float("nan") for k in (
            "alpha", "beta", "bench_cagr", "bench_total_return", "cagr", "mdd", "sharpe",
            "total_return", "win_rate",
        )}

    total_return = float(pc.iloc[-1] / pc.iloc[0] - 1.0)
    bench_total_return = float(bc.iloc[-1] / bc.iloc[0] - 1.0)
    cagr = float((pc.iloc[-1] / pc.iloc[0]) ** (52.0 / n_periods) - 1.0)
    bench_cagr = float((bc.iloc[-1] / bc.iloc[0]) ** (52.0 / n_periods) - 1.0)

    p_ret = pc.pct_change().dropna()
    b_ret = bc.pct_change().dropna()
    aligned = pd.concat([p_ret.rename("p"), b_ret.rename("b")], axis=1, join="inner").dropna()
    weekly_std = float(p_ret.std(ddof=1)) if len(p_ret) > 1 else float("nan")
    if weekly_std > 0 and not math.isnan(weekly_std):
        sharpe = (cagr - rf) / (weekly_std * math.sqrt(52.0))
    else:
        sharpe = float("nan")

    dd = pc / pc.cummax() - 1.0
    mdd = float(dd.min())

    win_rate = float((p_ret > 0).mean()) if len(p_ret) else float("nan")

    if len(aligned) >= 2 and float(aligned["b"].var(ddof=1)) > 0:
        beta = float(
            aligned["p"].cov(aligned["b"]) / aligned["b"].var(ddof=1)
        )
    else:
        beta = float("nan")

    if not math.isnan(beta):
        alpha = float(cagr - (rf + beta * (bench_cagr - rf)))
    else:
        alpha = float("nan")

    out["cagr"] = cagr
    out["sharpe"] = sharpe
    out["mdd"] = mdd
    out["beta"] = beta
    out["alpha"] = alpha
    out["win_rate"] = win_rate
    out["total_return"] = total_return
    out["bench_cagr"] = bench_cagr
    out["bench_total_return"] = bench_total_return
    return out
