import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _revision_score(up: float, down: float) -> float:
    """(up - down) / (up + down), 분모 0이면 0 반환."""
    u = 0.0 if up is None or (isinstance(up, float) and pd.isna(up)) else float(up)
    d = 0.0 if down is None or (isinstance(down, float) and pd.isna(down)) else float(down)
    denom = u + d
    if denom == 0:
        return 0.0
    return (u - d) / denom


def _ci_column(df: pd.DataFrame, *names: str) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for n in names:
        k = n.lower()
        if k in lower_map:
            return lower_map[k]
    return None


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


def _eps_est_current_from_ticker(t: yf.Ticker) -> float | None:
    ef = getattr(t, 'earnings_forecast', None)
    if isinstance(ef, pd.DataFrame) and not ef.empty:
        col = _ci_column(ef, 'earningsAverage', 'earningsaverage', 'avg')
        if col is not None:
            series = pd.to_numeric(ef[col], errors='coerce').dropna()
            if not series.empty:
                return float(series.mean())
    est = getattr(t, 'earnings_estimate', None)
    if isinstance(est, pd.DataFrame) and not est.empty:
        col = _ci_column(est, 'avg')
        if col is not None and len(est):
            v = _safe_float(est.iloc[0][col])
            return v
    return None


def fetch_single(symbol: str) -> dict | None:
    """
    yf.Ticker(symbol) 호출.
    eps_revisions 최신 행(iloc[0])에서 7d/30d up·down으로 revision_score_* 계산.
    earnings_forecast(또는 earnings_estimate 폴백)에서 eps_est_current.
    date = 오늘(로컬 기준일).
    예외 시 None + WARNING 로그.
    """
    sym = str(symbol).strip().replace('.', '-')
    if not sym:
        return None
    try:
        t = yf.Ticker(sym)
        rev = t.eps_revisions
        if rev is None or not isinstance(rev, pd.DataFrame) or rev.empty:
            logger.warning("fetch_single(%s): eps_revisions 없음", sym)
            return None

        row = rev.iloc[0]
        c_up7 = _ci_column(rev, 'upLast7days', 'uplast7days')
        c_dn7 = _ci_column(rev, 'downLast7days', 'downlast7days', 'downLast7Days')
        c_up30 = _ci_column(rev, 'upLast30days', 'uplast30days')
        c_dn30 = _ci_column(rev, 'downLast30days', 'downlast30days')

        up7 = _safe_float(row[c_up7]) if c_up7 else None
        dn7 = _safe_float(row[c_dn7]) if c_dn7 else None
        up30 = _safe_float(row[c_up30]) if c_up30 else None
        dn30 = _safe_float(row[c_dn30]) if c_dn30 else None

        eps_est = _eps_est_current_from_ticker(t)

        return {
            'symbol': sym,
            'date': date.today(),
            'eps_est_current': eps_est,
            'revision_score_7d': _revision_score(up7, dn7),
            'revision_score_30d': _revision_score(up30, dn30),
        }
    except Exception as e:
        logger.warning("fetch_single(%s) 실패: %s", sym, e)
        return None


def fetch_all(symbols: list[str], delay: float = 0.3) -> pd.DataFrame:
    """
    symbols 순회 후 성공 분만 DataFrame으로 반환.
    delay: 요 체 간 sleep(레이트 리밋).
    10% 단위 진행 로그.
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
        return pd.DataFrame(columns=['symbol', 'date', 'eps_est_current', 'revision_score_7d', 'revision_score_30d'])
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
    """분류.json에서 심볼 로드."""
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
        logger.warning("적재할 실적 개정 행이 없습니다.")
    else:
        DBManager().upsert_earnings_revisions(df)
