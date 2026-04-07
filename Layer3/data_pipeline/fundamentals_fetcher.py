"""
yfinance `Ticker.info` 기반 재무 지표 스냅샷 → `fundamentals` upsert.
TTM 성격 지표는 수집일(`report_date = date.today()`)을 기준일로 둔다.
"""
from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _safe_float(x) -> float | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if pd.isna(v):
        return None
    return v


def fetch_single(symbol: str) -> dict | None:
    """
    `yf.Ticker(sym).info`에서 operatingMargins, grossMargins, trailingEps,
    forwardEps, debtToEquity 추출. 다섯 값이 **모두** None이면 None 반환.
    """
    sym = str(symbol).strip().replace('.', '-')
    if not sym:
        return None
    try:
        info = yf.Ticker(sym).info
        if not isinstance(info, dict):
            logger.warning("fetch_single(%s): info 없음 또는 dict 아님", sym)
            return None

        operating_margin = _safe_float(info.get('operatingMargins'))
        gross_margin = _safe_float(info.get('grossMargins'))
        eps_actual = _safe_float(info.get('trailingEps'))
        eps_consensus = _safe_float(info.get('forwardEps'))
        debt_to_equity = _safe_float(info.get('debtToEquity'))

        if all(
            v is None
            for v in (
                operating_margin,
                gross_margin,
                eps_actual,
                eps_consensus,
                debt_to_equity,
            )
        ):
            return None

        return {
            'symbol': sym,
            'report_date': date.today(),
            'eps_actual': eps_actual,
            'eps_consensus': eps_consensus,
            'operating_margin': operating_margin,
            'debt_to_equity': debt_to_equity,
            'gross_margin': gross_margin,
        }
    except Exception as e:
        logger.warning("fetch_single(%s) 실패: %s", sym, e)
        return None


def fetch_all(symbols: list[str], delay: float = 0.5) -> pd.DataFrame:
    """
    symbols 순회 후 성공 행만 DataFrame.
    delay: 심볼 간 sleep(레이트 리밋). 10% 단위 진행 로그.
    """
    rows: list[dict] = []
    n = len(symbols)
    next_milestone = 10

    for i, sym in enumerate(symbols):
        one = fetch_single(sym)
        if one is not None:
            rows.append(one)
        if delay and i + 1 < n:
            time.sleep(delay)
        if n <= 0:
            continue
        pct = int(100 * (i + 1) / n)
        while next_milestone <= 100 and pct >= next_milestone:
            logger.info("진행률 %s%% (%s/%s)", next_milestone, i + 1, n)
            next_milestone += 10

    if not rows:
        return pd.DataFrame(
            columns=[
                'symbol',
                'report_date',
                'eps_actual',
                'eps_consensus',
                'operating_margin',
                'debt_to_equity',
                'gross_margin',
            ]
        )
    return pd.DataFrame(rows)


def _load_symbols_from_db() -> list[str]:
    from data_pipeline.db_manager import DBManager, Ticker

    db = DBManager()
    session = db.Session()
    try:
        q = session.query(Ticker.symbol).order_by(Ticker.symbol)
        return [r[0] for r in q.all()]
    finally:
        session.close()


def _load_symbols_from_csv() -> list[str]:
    root = Path(__file__).resolve().parent.parent / 'data' / 'universe'
    if not root.is_dir():
        return []
    files = sorted(root.glob('sp500_universe_*.csv'))
    if not files:
        return []
    path = files[-1]
    df = pd.read_csv(path)
    col = 'ticker' if 'ticker' in df.columns else 'symbol'
    if col not in df.columns:
        return []
    sym = df[col].astype(str).str.strip().str.replace('.', '-', regex=False)
    return [s for s in sym.tolist() if s]


def _load_symbols_from_classification() -> list[str]:
    from data_pipeline.classification_loader import load_classification_df

    try:
        df = load_classification_df()
        if df.empty:
            return []
        return df['symbol'].unique().tolist()
    except Exception as e:
        logger.warning("분류.json에서 심볼 로드 실패: %s", e)
        return []


def load_symbols() -> list[str]:
    """DB tickers 우선, 없으면 S&P500 CSV, 없으면 분류.json."""
    try:
        syms = _load_symbols_from_db()
        if syms:
            return syms
    except Exception as e:
        logger.warning("DB에서 심볼 로드 실패, CSV 시도: %s", e)

    syms = _load_symbols_from_csv()
    if syms:
        return syms

    logger.warning("CSV에서 심볼 로드 실패, 분류.json 시도.")
    syms = _load_symbols_from_classification()
    if not syms:
        logger.warning("심볼 소스 없음(DB·CSV·분류.json). 빈 목록 반환.")
    return syms


if __name__ == "__main__":
    from data_pipeline.db_manager import DBManager

    symbols = load_symbols()
    if not symbols:
        raise SystemExit(1)
    df = fetch_all(symbols)
    if df.empty:
        logger.warning("적재할 fundamentals 행이 없습니다.")
    else:
        DBManager().upsert_fundamentals(df)
