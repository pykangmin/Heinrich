"""
Track D (Value Chain): VC 단계·포지션 그룹 내 operating_margin 상대 스코어 (score_bottleneck).

- operating_margin 결측 보간: gics_sector (`_sector_median_fill`)
- 백분위 폴백 (`MIN_GROUP_SIZE`): (stage, position) ≥ 임계 → 복합 그룹 → 아니면 stage 단독 ≥ 임계
  → 아니면 전체 유니버스(`_group_key == 'U'`)
- score_bottleneck = 50: vc_stage_num·vc_position **둘 다** NULL일 때만
- vc_stage_num NULL·vc_position만 있음: 그룹 불가 → 50
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from scoring.track_a import _minmax_0_100, _pct_high_better, _sector_median_fill

W_BOTTLENECK = 1.0
MIN_GROUP_SIZE = 3
_UNIVERSE_KEY = 'U'


def compute_track_d(engine: Engine, score_date: date) -> pd.DataFrame:
    f_sql = text("""
        SELECT DISTINCT ON (f.symbol)
            f.symbol,
            f.operating_margin
        FROM fundamentals f
        WHERE f.report_date <= :score_date
        ORDER BY f.symbol, f.report_date DESC
    """)
    t_sql = text(
        """
        SELECT symbol, vc_stage_num, vc_position, gics_sector
        FROM tickers
        ORDER BY symbol
        """
    )
    fund = pd.read_sql(f_sql, engine, params={'score_date': score_date})
    tick = pd.read_sql(t_sql, engine)
    if tick.empty:
        return pd.DataFrame(columns=['symbol', 'score_bottleneck'])

    df = tick.merge(fund, on='symbol', how='left')
    df = _sector_median_fill(df, ['operating_margin'], 'gics_sector')

    def _clean_vc_pos(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        t = str(x).strip()
        return t[:100] if t and t.lower() != 'nan' else None

    df['vc_position'] = df['vc_position'].apply(_clean_vc_pos)

    both_vc_null = df['vc_stage_num'].isna() & df['vc_position'].isna()
    bad_vc = df['vc_stage_num'].isna() & ~both_vc_null

    df['_n_stage'] = np.nan
    st_mask = df['vc_stage_num'].notna()
    if st_mask.any():
        df.loc[st_mask, '_n_stage'] = (
            df.loc[st_mask]
            .groupby('vc_stage_num', sort=False)['symbol']
            .transform('count')
        )

    df['_n_comp'] = np.nan
    both_dims = df['vc_stage_num'].notna() & df['vc_position'].notna()
    if both_dims.any():
        df.loc[both_dims, '_n_comp'] = (
            df.loc[both_dims]
            .groupby(['vc_stage_num', 'vc_position'], sort=False)['symbol']
            .transform('count')
        )

    df['_group_key'] = pd.NA

    only_stage = st_mask & df['vc_position'].isna()
    ge_st = df['_n_stage'] >= MIN_GROUP_SIZE
    if only_stage.any():
        ok = only_stage & ge_st
        if ok.any():
            df.loc[ok, '_group_key'] = 'S:' + df.loc[ok, 'vc_stage_num'].astype(str)
        need_u = only_stage & ~ge_st
        if need_u.any():
            df.loc[need_u, '_group_key'] = _UNIVERSE_KEY

    if both_dims.any():
        ok_c = both_dims & (df['_n_comp'] >= MIN_GROUP_SIZE)
        if ok_c.any():
            df.loc[ok_c, '_group_key'] = (
                'C:'
                + df.loc[ok_c, 'vc_stage_num'].astype(str)
                + ':'
                + df.loc[ok_c, 'vc_position'].astype(str)
            )
        fb_s = both_dims & (df['_n_comp'] < MIN_GROUP_SIZE) & ge_st
        if fb_s.any():
            df.loc[fb_s, '_group_key'] = 'S:' + df.loc[fb_s, 'vc_stage_num'].astype(str)
        fb_u = (
            both_dims
            & (df['_n_comp'] < MIN_GROUP_SIZE)
            & (df['_n_stage'] < MIN_GROUP_SIZE)
        )
        if fb_u.any():
            df.loc[fb_u, '_group_key'] = _UNIVERSE_KEY

    pct_mask = df['_group_key'].notna()
    scores = pd.Series(np.nan, index=df.index, dtype=float)
    if pct_mask.any():
        scores.loc[pct_mask] = (
            df.loc[pct_mask]
            .groupby('_group_key', sort=False)['operating_margin']
            .transform(_pct_high_better)
        )
    scores.loc[both_vc_null] = 50.0
    scores = scores.astype(float)
    scores.loc[bad_vc] = np.nan
    scores = scores.fillna(50.0)

    df = df.drop(columns=['_n_stage', '_n_comp', '_group_key'], errors='ignore')

    mm = _minmax_0_100(scores)
    return pd.DataFrame({'symbol': df['symbol'].astype(str), 'score_bottleneck': mm.values})
