"""
Track A (Original): 거래량·재무·리비전·밸류에이션 가중 스코어.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

W_VOLUME = 0.20
W_FINANCIAL = 0.35
W_REVISION = 0.30
W_VALUATION = 0.15

_LOOKBACK_CAL_DAYS = 120


def _minmax_0_100(s: pd.Series) -> pd.Series:
    lo = float(s.min())
    hi = float(s.max())
    if np.isnan(lo) or lo == hi:
        return pd.Series(50.0, index=s.index, dtype=float)
    return (s.astype(float) - lo) / (hi - lo) * 100.0


def _norm_sector(s: pd.Series) -> pd.Series:
    return s.fillna('__NA_SECTOR__').astype(str)


def _sector_median_fill(
    df: pd.DataFrame, value_cols: list[str], sector_col: str = 'gics_sector'
) -> pd.DataFrame:
    """결측은 동일 섹터 중앙값 → 여전히 NaN이면 전역 중앙값."""
    out = df.copy()
    sec = _norm_sector(out[sector_col])
    out['_sec'] = sec
    for c in value_cols:
        med_sec = out.groupby('_sec', sort=False)[c].transform('median')
        out[c] = pd.to_numeric(out[c], errors='coerce')
        out[c] = out[c].where(out[c].notna(), med_sec)
        gmed = float(out[c].median()) if out[c].notna().any() else float('nan')
        out[c] = out[c].fillna(gmed)
    return out.drop(columns=['_sec'], errors='ignore')


def _pct_high_better(series: pd.Series) -> pd.Series:
    r = series.rank(pct=True, method='average', ascending=False)
    return (r * 100.0).astype(float)


def _pct_low_better(series: pd.Series) -> pd.Series:
    r = series.rank(pct=True, method='average', ascending=True)
    return ((1.0 - r) * 100.0).astype(float)


def volume_score(engine: Engine, score_date: date) -> pd.DataFrame:
    start = score_date - timedelta(days=_LOOKBACK_CAL_DAYS)
    q = text("""
        SELECT symbol, date, volume
        FROM daily_prices
        WHERE date <= :score_date AND date >= :start_date
        ORDER BY symbol, date
    """)
    universe = pd.read_sql(
        text('SELECT symbol FROM tickers ORDER BY symbol'),
        engine,
    )['symbol'].astype(str).tolist()
    raw = pd.read_sql(q, engine, params={'score_date': score_date, 'start_date': start})
    ratio_by_sym: dict[str, float] = {}
    if not raw.empty:
        for sym, grp in raw.groupby('symbol', sort=False):
            sym = str(sym)
            g = grp.sort_values('date')
            row_today = g[g['date'] == score_date]
            if row_today.empty:
                ratio_by_sym[sym] = 1.0
                continue
            vol_today = float(row_today.iloc[-1]['volume'])
            if pd.isna(vol_today) or vol_today <= 0:
                ratio_by_sym[sym] = 1.0
                continue
            win = g[g['date'] <= score_date].tail(20)
            mv = win['volume'].mean()
            if pd.isna(mv) or float(mv) <= 0:
                ratio_by_sym[sym] = 1.0
            else:
                r = vol_today / float(mv)
                ratio_by_sym[sym] = 1.0 if pd.isna(r) else float(r)
    ratios = [ratio_by_sym.get(s, 1.0) for s in universe]
    ser = pd.Series(ratios, index=universe, dtype=float)
    out = _minmax_0_100(ser)
    out.name = 'volume_score'
    return out.reset_index().rename(columns={'index': 'symbol'})


def financial_score(engine: Engine, score_date: date) -> pd.DataFrame:
    f_sql = text("""
        SELECT DISTINCT ON (f.symbol)
            f.symbol,
            f.eps_actual,
            f.eps_consensus,
            f.operating_margin,
            f.debt_to_equity
        FROM fundamentals f
        WHERE f.report_date <= :score_date
        ORDER BY f.symbol, f.report_date DESC
    """)
    t_sql = text("SELECT symbol, gics_sector FROM tickers")
    fund = pd.read_sql(f_sql, engine, params={'score_date': score_date})
    tick = pd.read_sql(t_sql, engine)
    if fund.empty:
        return pd.DataFrame(columns=['symbol', 'financial_score'])
    df = fund.merge(tick, on='symbol', how='left')
    imp_cols = [
        'eps_actual',
        'eps_consensus',
        'operating_margin',
        'debt_to_equity',
    ]
    df = _sector_median_fill(df, imp_cols, 'gics_sector')
    sec = _norm_sector(df['gics_sector'])
    om_pct = df.groupby(sec, sort=False)['operating_margin'].transform(_pct_high_better)
    de_pct = df.groupby(sec, sort=False)['debt_to_equity'].transform(_pct_low_better)
    df['_spread'] = df['eps_actual'].astype(float) - df['eps_consensus'].astype(float)
    beat_pct = df.groupby(sec, sort=False)['_spread'].transform(_pct_high_better)
    df = df.drop(columns=['_spread'])
    fin = (om_pct.astype(float) + de_pct.astype(float) + beat_pct.astype(float)) / 3.0
    return pd.DataFrame({'symbol': df['symbol'], 'financial_score': fin})


def revision_score(engine: Engine, score_date: date) -> pd.DataFrame:
    q = text("""
        SELECT symbol, revision_score_7d, revision_score_30d
        FROM earnings_revisions
        WHERE date = :score_date
    """)
    rev = pd.read_sql(q, engine, params={'score_date': score_date})
    if rev.empty:
        return pd.DataFrame(columns=['symbol', 'revision_score'])

    def _one_row(r7: float | None, r30: float | None) -> float:
        a = r7 if r7 is not None and not (isinstance(r7, float) and np.isnan(r7)) else None
        b = r30 if r30 is not None and not (isinstance(r30, float) and np.isnan(r30)) else None
        if a is None and b is None:
            return 50.0
        if a is None:
            combined = float(b)
        elif b is None:
            combined = float(a)
        else:
            combined = 0.6 * float(a) + 0.4 * float(b)
        c = float(np.clip(combined, -1.0, 1.0))
        return (c + 1.0) / 2.0 * 100.0

    scores = [
        _one_row(
            float(r['revision_score_7d']) if pd.notna(r['revision_score_7d']) else None,
            float(r['revision_score_30d']) if pd.notna(r['revision_score_30d']) else None,
        )
        for _, r in rev.iterrows()
    ]
    return pd.DataFrame({'symbol': rev['symbol'], 'revision_score': scores})


def valuation_score(engine: Engine, score_date: date) -> pd.DataFrame:
    f_sql = text("""
        SELECT DISTINCT ON (f.symbol)
            f.symbol,
            f.eps_consensus
        FROM fundamentals f
        WHERE f.report_date <= :score_date
        ORDER BY f.symbol, f.report_date DESC
    """)
    p_sql = text("SELECT symbol, close FROM daily_prices WHERE date = :score_date")
    t_sql = text("SELECT symbol, gics_sector FROM tickers")
    fund = pd.read_sql(f_sql, engine, params={'score_date': score_date})
    px = pd.read_sql(p_sql, engine, params={'score_date': score_date})
    tick = pd.read_sql(t_sql, engine)
    if fund.empty or px.empty:
        return pd.DataFrame(columns=['symbol', 'valuation_score'])
    df = fund.merge(px, on='symbol', how='inner').merge(tick, on='symbol', how='left')
    df = _sector_median_fill(df, ['eps_consensus', 'close'], 'gics_sector')
    df['ey'] = df['eps_consensus'].astype(float) / df['close'].astype(float).replace(0, np.nan)
    df['ey'] = df['ey'].replace([np.inf, -np.inf], np.nan)
    df = _sector_median_fill(df, ['ey'], 'gics_sector')
    sec = _norm_sector(df['gics_sector'])
    df['_sec'] = sec
    val_pct = df.groupby('_sec', sort=False)['ey'].transform(_pct_high_better)
    df = df.drop(columns=['_sec'])
    return pd.DataFrame({'symbol': df['symbol'], 'valuation_score': val_pct})


def compute_track_a(engine: Engine, score_date: date) -> pd.DataFrame:
    tick = pd.read_sql(text("SELECT symbol FROM tickers"), engine)
    if tick.empty:
        return pd.DataFrame(columns=['symbol', 'score_original'])
    v = volume_score(engine, score_date)
    f = financial_score(engine, score_date)
    r = revision_score(engine, score_date)
    val = valuation_score(engine, score_date)
    out = tick[['symbol']].copy()
    out = out.merge(v, on='symbol', how='left')
    out = out.merge(f, on='symbol', how='left')
    out = out.merge(r, on='symbol', how='left')
    out = out.merge(val, on='symbol', how='left')
    out['revision_score'] = out['revision_score'].fillna(50.0)
    for col in ('volume_score', 'financial_score', 'valuation_score'):
        out[col] = out[col].fillna(50.0)
    out['score_original'] = (
        W_VOLUME * out['volume_score']
        + W_FINANCIAL * out['financial_score']
        + W_REVISION * out['revision_score']
        + W_VALUATION * out['valuation_score']
    )
    return out[['symbol', 'score_original']]
