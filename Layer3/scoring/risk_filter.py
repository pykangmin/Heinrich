"""
SNR·Divergence·Beta·상관 리스크 필터.

파이프라인: SNR → Divergence → Beta → Top-K → Greedy correlation.

당일 `total_heinrich`는 아직 DB에 없을 수 있어 `apply_risk_filter(..., scores_df=)` 로 전달한다.
"""
from __future__ import annotations

import logging
import os
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import Engine, bindparam, text

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def _load_universe(engine: Engine) -> list[str]:
    df = pd.read_sql(text("SELECT symbol FROM tickers ORDER BY symbol"), engine)
    if df.empty:
        return []
    return df["symbol"].astype(str).tolist()


def _snr_passing_symbols(
    engine: Engine,
    score_date: date,
    universe: list[str],
    lookback: int,
    min_ratio: float,
) -> set[str]:
    """`total_heinrich` 이력 SNR = mean/std (표본 std). 이력 < 3행이면 패스."""
    if not universe:
        return set()
    q = text(
        """
        SELECT symbol, total_heinrich
        FROM daily_scores
        WHERE date <= :score_date
        ORDER BY symbol, date DESC
        """
    )
    hist = pd.read_sql(q, engine, params={"score_date": score_date})
    if hist.empty:
        return set(universe)

    hist = hist[hist["symbol"].astype(str).isin(universe)]
    hist = hist.groupby("symbol", sort=False).head(lookback)
    passing: set[str] = set()
    seen: set[str] = set()

    for sym, grp in hist.groupby("symbol", sort=False):
        sym = str(sym)
        seen.add(sym)
        vals = (
            pd.to_numeric(grp["total_heinrich"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if vals.size < 3:
            passing.add(sym)
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        if not np.isfinite(s) or s <= 1e-12:
            passing.add(sym)
            continue
        snr = m / s
        if np.isfinite(snr) and snr >= min_ratio:
            passing.add(sym)

    for s in universe:
        if s not in seen:
            passing.add(s)
    return passing


def _divergence_passing_symbols(
    engine: Engine,
    score_date: date,
    universe: list[str],
    scores_df: pd.DataFrame | None,
    score_min: float,
    return_days: int,
    return_min: float,
) -> set[str]:
    """
    종합점수 >= score_min 인데 최근 return_days 거래일 수익률 < return_min 이면 제외.
    가격 부족·당일 점수 없음 → 패스.
    """
    if not universe:
        return set()

    if scores_df is None or scores_df.empty:
        logger.warning("risk_filter divergence: scores_df 없음 → 다이버전스 필터 전체 패스")
        return set(universe)

    sc = scores_df[["symbol", "total_heinrich"]].copy()
    sc["symbol"] = sc["symbol"].astype(str)
    sc["total_heinrich"] = pd.to_numeric(sc["total_heinrich"], errors="coerce")

    need_price = set(
        sc.loc[sc["total_heinrich"] >= score_min, "symbol"].astype(str).tolist()
    )
    if not need_price:
        return set(universe)

    pq = text(
        """
        SELECT symbol, date, close
        FROM daily_prices
        WHERE date <= :score_date
        ORDER BY symbol, date DESC
        """
    )
    prices = pd.read_sql(pq, engine, params={"score_date": score_date})

    ret_by_sym: dict[str, float | None] = {}
    if not prices.empty:
        prices = prices[prices["symbol"].astype(str).isin(need_price)]
        nwin = return_days + 1
        for sym, g in prices.groupby("symbol", sort=False):
            grp = g.sort_values("date", ascending=True)
            grp_tail = grp.tail(nwin)
            if grp_tail.shape[0] < nwin:
                ret_by_sym[str(sym)] = None
                continue
            p_start = float(grp_tail.iloc[0]["close"])
            p_end = float(grp_tail.iloc[-1]["close"])
            if (
                p_start == 0
                or not np.isfinite(p_start)
                or not np.isfinite(p_end)
            ):
                ret_by_sym[str(sym)] = None
            else:
                ret_by_sym[str(sym)] = (p_end / p_start) - 1.0

    passing: set[str] = set()
    score_map = sc.set_index("symbol")["total_heinrich"]

    for sym in universe:
        if sym not in score_map.index:
            passing.add(sym)
            continue
        th = score_map.loc[sym]
        if th is None or (isinstance(th, float) and pd.isna(th)):
            passing.add(sym)
            continue
        th = float(th)
        if th < score_min:
            passing.add(sym)
            continue
        r = ret_by_sym.get(sym)
        if r is None:
            passing.add(sym)
            continue
        if r < return_min:
            logger.debug("risk_filter divergence 제외 %s score=%s ret=%.4f", sym, th, r)
            continue
        passing.add(sym)

    return passing


GSPC_SYMBOL = "^GSPC"


def _fetch_price_returns(
    engine: Engine,
    score_date: date,
    symbols: list[str],
    lookback: int,
) -> pd.DataFrame:
    """`score_date` 이전까지 최근 (lookback+1)일 종가 → 일간 수익률 wide (index=date, columns=symbol)."""
    if not symbols:
        return pd.DataFrame()
    syms = list(dict.fromkeys(str(s) for s in symbols))
    stmt = text(
        """
        SELECT symbol, date, close
        FROM daily_prices
        WHERE date <= :score_date AND symbol IN :syms
        ORDER BY date ASC, symbol ASC
        """
    ).bindparams(bindparam("syms", expanding=True))
    df = pd.read_sql(stmt, engine, params={"score_date": score_date, "syms": syms})
    if df.empty:
        return pd.DataFrame()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    wide = df.pivot(index="date", columns="symbol", values="close").sort_index()
    tail_n = lookback + 1
    if wide.shape[0] > tail_n:
        wide = wide.iloc[-tail_n:]
    rets = wide.pct_change()
    return rets.iloc[1:]


def _beta_passing_symbols(
    engine: Engine,
    score_date: date,
    universe: list[str],
    lookback: int,
    beta_max: float,
    min_rows: int,
) -> set[str]:
    """시장(^GSPC) 대비 rolling beta 상한 필터. ^GSPC 부재 시 universe 전체 통과."""
    if not universe:
        return set()
    syms_for_fetch = list(dict.fromkeys([GSPC_SYMBOL] + [str(s) for s in universe]))
    rets = _fetch_price_returns(engine, score_date, syms_for_fetch, lookback)
    if rets.empty or GSPC_SYMBOL not in rets.columns:
        logger.warning(
            "risk_filter beta: ^GSPC 수익률 없음 → beta 필터 생략 (전체 통과)"
        )
        return set(universe)
    mkt_all = pd.to_numeric(rets[GSPC_SYMBOL], errors="coerce")
    v_mkt_full = np.nanvar(mkt_all.to_numpy(dtype=float), ddof=1)
    if mkt_all.notna().sum() < 1 or (not np.isfinite(v_mkt_full) or v_mkt_full <= 1e-18):
        logger.warning("risk_filter beta: 시장 수익률 분산 무시 가능 → beta 필터 생략")
        return set(universe)

    passing: set[str] = set()
    for sym in universe:
        sym = str(sym)
        if sym not in rets.columns:
            passing.add(sym)
            continue
        stock = pd.to_numeric(rets[sym], errors="coerce")
        pair = pd.DataFrame({"s": stock, "m": mkt_all}).dropna()
        n = int(pair.shape[0])
        if n < min_rows:
            passing.add(sym)
            continue
        m_arr = pair["m"].to_numpy(dtype=float)
        v_mkt = float(np.var(m_arr, ddof=1))
        if not np.isfinite(v_mkt) or v_mkt <= 1e-18:
            passing.add(sym)
            continue
        c = np.cov(pair["s"].to_numpy(dtype=float), m_arr, ddof=1)
        beta = float(c[0, 1]) / v_mkt
        if not np.isfinite(beta) or beta > beta_max:
            logger.debug("risk_filter beta 제외 %s beta=%.4f", sym, beta)
            continue
        passing.add(sym)
    return passing


def _corr_filter_symbols(
    candidates_sorted: list[str],
    returns_df: pd.DataFrame,
    corr_max: float,
    min_rows: int,
) -> set[str]:
    """total_heinrich 내림차순 리스트에 대해 Greedy 저상관 서브셋."""
    selected: list[str] = []
    for sym in candidates_sorted:
        sym = str(sym)
        if sym not in returns_df.columns:
            selected.append(sym)
            continue
        col = returns_df[sym]
        if int(col.count()) < min_rows:
            selected.append(sym)
            continue
        if not selected:
            selected.append(sym)
            continue
        selected_in_df = [s for s in selected if s in returns_df.columns]
        if not selected_in_df:
            selected.append(sym)
            continue
        corr_vals = returns_df[selected_in_df + [sym]].dropna().corr()[sym].drop(sym)
        mx = float(corr_vals.abs().max())
        if not np.isfinite(mx):
            continue
        if mx < corr_max:
            selected.append(sym)
    return set(selected)


def apply_risk_filter(
    engine: Engine,
    score_date: date,
    scores_df: pd.DataFrame | None = None,
) -> set[str]:
    """
    SNR·Divergence·(Beta)·Top-K·Greedy correlation 적용 후 최종 후보 심볼 집합 반환.

    Parameters
    ----------
    scores_df
        당일 산출 `symbol`, `total_heinrich` (DB 적재 전). 없으면 다이버전스는 전체 패스, Top-K는 생략.
    """
    lookback = _env_int("SNR_LOOKBACK_DAYS", 10)
    min_ratio = _env_float("SNR_MIN_RATIO", 2.0)
    score_min = _env_float("DIVERGENCE_SCORE_MIN", 60.0)
    return_days = _env_int("DIVERGENCE_RETURN_DAYS", 20)
    return_min = _env_float("DIVERGENCE_RETURN_MIN", -0.05)

    beta_lb = _env_int("BETA_LOOKBACK_DAYS", 60)
    beta_max = _env_float("BETA_MAX", 1.5)
    beta_min_rows = _env_int("BETA_MIN_ROWS", 30)
    top_k = _env_int("TOP_K_CANDIDATES", 75)
    corr_max = _env_float("CORR_MAX", 0.65)
    corr_lb = _env_int("CORR_LOOKBACK_DAYS", 30)
    corr_min_rows = _env_int("CORR_MIN_ROWS", 20)

    universe = [s for s in _load_universe(engine) if str(s) != GSPC_SYMBOL]
    if not universe:
        return set()

    snr_ok = _snr_passing_symbols(engine, score_date, universe, lookback, min_ratio)
    div_ok = _divergence_passing_symbols(
        engine, score_date, universe, scores_df, score_min, return_days, return_min
    )
    after_snr_div = snr_ok & div_ok

    beta_ok = _beta_passing_symbols(
        engine,
        score_date,
        list(after_snr_div),
        beta_lb,
        beta_max,
        beta_min_rows,
    )
    after_beta = after_snr_div & beta_ok

    if scores_df is not None and not scores_df.empty:
        sc = scores_df[scores_df["symbol"].astype(str).isin(after_beta)].copy()
        sc["symbol"] = sc["symbol"].astype(str)
        sc["total_heinrich"] = pd.to_numeric(sc["total_heinrich"], errors="coerce")
        sc = sc.sort_values("total_heinrich", ascending=False)
        top_k_symbols = sc.head(top_k)["symbol"].astype(str).tolist()
    else:
        top_k_symbols = list(after_beta)

    returns_df = _fetch_price_returns(engine, score_date, top_k_symbols, corr_lb)
    candidates = _corr_filter_symbols(top_k_symbols, returns_df, corr_max, corr_min_rows)

    logger.info(
        "risk_filter date=%s snr_div=%s beta=%s top_k=%s corr=%s",
        score_date,
        len(after_snr_div),
        len(after_beta),
        len(top_k_symbols),
        len(candidates),
    )
    return candidates
