"""
Track A + D 통합 → `daily_scores` upsert.

가중치: W_TRACK_A / W_TRACK_D. `earnings_revisions`·`daily_prices` 당일 행이 없어도
Track A 내부에서 revision 50·volume 비율 1.0 등으로 보간됨.
"""
from __future__ import annotations

import sys
from datetime import date, datetime

import pandas as pd
from sqlalchemy import text

from data_pipeline.db_manager import DBManager

W_TRACK_A = 0.60
W_TRACK_D = 0.40
W_A = W_TRACK_A
W_D = W_TRACK_D
REGIME_STUB = "관세/무역전쟁"


def _parse_score_date(argv: list[str]) -> date:
    if len(argv) < 2:
        return date.today()
    raw = argv[1].strip()
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return datetime.strptime(raw, "%Y-%m-%d").date()


def run_scoring(score_date: date) -> pd.DataFrame:
    from scoring.track_a import compute_track_a
    from scoring.track_d import compute_track_d

    db = DBManager()
    with db.engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    a_df = compute_track_a(db.engine, score_date)
    d_df = compute_track_d(db.engine, score_date)

    uni = pd.read_sql(text("SELECT symbol FROM tickers ORDER BY symbol"), db.engine)
    if uni.empty:
        return pd.DataFrame(
            columns=[
                'symbol',
                'date',
                'regime_stub',
                'score_original',
                'score_bottleneck',
                'total_heinrich',
            ]
        )

    out = uni.merge(a_df, on='symbol', how='left').merge(d_df, on='symbol', how='left')
    out['score_original'] = out['score_original'].fillna(50.0)
    out['score_bottleneck'] = out['score_bottleneck'].fillna(50.0)
    out['total_heinrich'] = (
        out['score_original'] * W_TRACK_A + out['score_bottleneck'] * W_TRACK_D
    )
    out['date'] = score_date
    out['regime_stub'] = REGIME_STUB

    upsert_cols = [
        'symbol',
        'date',
        'regime_stub',
        'score_original',
        'score_bottleneck',
        'total_heinrich',
    ]
    db.upsert_daily_scores(out[upsert_cols])
    return out[upsert_cols]


if __name__ == "__main__":
    run_scoring(_parse_score_date(sys.argv))
