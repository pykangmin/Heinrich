"""
Mixed Regime 시 현금 비중 방어 및 동일 비중 포트폴리오 구성.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Any

import pandas as pd


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def build_portfolio(
    candidates: set[str],
    scores_df: pd.DataFrame,
    regime: str,
    score_date: date,
    cash_pct: float | None = None,
) -> dict[str, Any]:
    """
    Parameters
    ----------
    candidates
        `apply_risk_filter` 반환 심볼 집합.
    scores_df
        `symbol`, `total_heinrich` 컬럼 기대.
    regime
        `get_regime()` 값. ``regime.lower() == MIXED_REGIME_KEY.lower()`` 이면 ``MIXED_CASH_PCT``.
    score_date
        스코어·포트폴리오 기준일(출력 dict ``date``).
    cash_pct
        지정 시 Mixed 판별 대신 사용 후 ``max(0, min(1, cash_pct))`` 클램프.
    """
    mixed_key = os.getenv("MIXED_REGIME_KEY", "mixed").strip()
    regime_norm = regime.strip()
    is_mixed = regime_norm.lower() == mixed_key.lower()

    if cash_pct is not None:
        cp = float(cash_pct)
    else:
        cp = _env_float("MIXED_CASH_PCT", 0.30) if is_mixed else 0.0

    cp = max(0.0, min(1.0, cp))
    equity_pct = 1.0 - cp

    syms = {str(s) for s in candidates}
    n = len(syms)
    per_weight = (equity_pct / n) if n else 0.0

    scores_map: dict[str, float] = {}
    ordered: list[str] = []
    if scores_df is not None and not scores_df.empty and syms:
        sc = scores_df[scores_df["symbol"].astype(str).isin(syms)].copy()
        sc["symbol"] = sc["symbol"].astype(str)
        sc["total_heinrich"] = pd.to_numeric(sc["total_heinrich"], errors="coerce")
        sc = sc.sort_values("total_heinrich", ascending=False)
        ordered = sc["symbol"].tolist()
        for _, row in sc.iterrows():
            sym = str(row["symbol"])
            v = row["total_heinrich"]
            if pd.notna(v):
                scores_map[sym] = float(v)
    missing = sorted(syms - set(ordered))
    ordered.extend(missing)

    holdings: list[dict[str, Any]] = []
    for sym in ordered:
        holdings.append(
            {
                "symbol": sym,
                "weight": per_weight,
                "score": scores_map.get(sym),
            }
        )

    return {
        "date": score_date,
        "regime": regime_norm,
        "cash_pct": cp,
        "equity_pct": equity_pct,
        "holdings": holdings,
    }
