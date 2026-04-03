import os
import logging
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SP500Fetcher:
    """
    S&P 500 종목 리스트 및 메타데이터를 수집하는 클래스
    """
    WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    def __init__(self, output_dir='data/universe'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"디렉토리 생성됨: {self.output_dir}")

    def fetch_tickers(self):
        """
        Wikipedia에서 S&P 500 티커 리스트와 정보를 긁어옴
        """
        logger.info("Wikipedia에서 S&P 500 리스트 수집 중...")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(self.WIKI_URL, headers=headers, timeout=10)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            df = tables[0]

            if len(df.columns) != 8:
                raise ValueError(
                    f"위키 표 컬럼 수 변경 감지: {len(df.columns)}열 (예상 8열)"
                )

            # 위키 표 컬럼(2024~): Symbol, Security, GICS Sector, GICS Sub-Industry, …
            df.columns = [
                'ticker',
                'company_name',
                'sector',
                'industry',
                'headquarters',
                'date_added',
                'cik',
                'founded',
            ]

            # 티커 정규화 (BRK.B -> BRK-B)
            df['ticker'] = df['ticker'].str.replace('.', '-', regex=False)

            logger.info(f"총 {len(df)}개 종목 수집 완료.")
            return df

        except Exception as e:
            logger.error(f"S&P 500 리스트 수집 실패: {e}")
            return None

    def save_to_csv(self, df):
        """
        수집된 데이터를 CSV 파일로 저장
        """
        if df is not None:
            today = datetime.now().strftime('%Y%m%d')
            file_path = os.path.join(self.output_dir, f'sp500_universe_{today}.csv')
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.info(f"데이터 저장 완료: {file_path}")
            return file_path
        return None


class OHLCVFetcher:
    """
    yfinance 배치 다운로드 → long-format (db_manager.upsert_daily_prices 직접 호환).
    """

    _OUT_COLS = ('symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume')
    # reset_index 이후 열 이름: 공백·대소문자 차이 흡수 (yfinance/pandas 버전별)
    _LONG_COL_ALIASES = {
        'date': 'date',
        'ticker': 'symbol',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'adj close': 'adj_close',
        'adj_close': 'adj_close',
        'adjclose': 'adj_close',
        'volume': 'volume',
    }

    def fetch(self, symbols, start, end) -> pd.DataFrame:
        """
        Parameters
        ----------
        symbols : list[str] | str
        start, end : str | datetime | pd.Timestamp
            yfinance start/end (end는 배타적일 수 있음).

        Returns
        -------
        pd.DataFrame
            columns: symbol, date, open, high, low, close, adj_close, volume
        """
        tickers = self._normalize_symbol_list(symbols)
        if not tickers:
            logger.warning("OHLCVFetcher.fetch: 유효한 심볼이 없습니다.")
            return self._empty_output()

        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                group_by='ticker',
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception as e:
            logger.error("yfinance.download 실패: %s", e)
            return self._empty_output()

        if raw is None or raw.empty:
            logger.warning("yfinance가 빈 DataFrame을 반환했습니다. 기간·심볼·네트워크를 확인하세요.")
            return self._empty_output()

        return self._wide_multiindex_to_long(raw)

    def _empty_output(self) -> pd.DataFrame:
        return pd.DataFrame(columns=list(self._OUT_COLS))

    @staticmethod
    def _normalize_symbol_list(symbols) -> list[str]:
        if symbols is None:
            return []
        if isinstance(symbols, (str, bytes)):
            symbols = [symbols]
        out: list[str] = []
        seen: set[str] = set()
        for s in symbols:
            if s is None:
                continue
            t = str(s).strip().replace('.', '-')
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    @staticmethod
    def _stack_level0_preserve_na(raw: pd.DataFrame) -> pd.DataFrame:
        try:
            return raw.stack(level=0, future_stack=True)
        except TypeError:
            pass
        try:
            return raw.stack(level=0, dropna=False)
        except (TypeError, ValueError):
            pass
        logger.warning(
            "pandas가 stack(future_stack=True) / stack(dropna=False)를 지원하지 않습니다. "
            "결측·거래정지 일자 행이 누락될 수 있어 pandas 2.2+ 권장."
        )
        return raw.stack(level=0)

    @classmethod
    def _canonical_long_column(cls, col) -> str | None:
        norm = ' '.join(str(col).strip().split()).lower()
        if norm in cls._LONG_COL_ALIASES:
            return cls._LONG_COL_ALIASES[norm]
        collapsed = norm.replace(' ', '_')
        if collapsed in cls._LONG_COL_ALIASES:
            return cls._LONG_COL_ALIASES[collapsed]
        nospace = norm.replace(' ', '')
        if nospace in cls._LONG_COL_ALIASES:
            return cls._LONG_COL_ALIASES[nospace]
        return None

    @classmethod
    def _rename_stacked_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(
            columns={c: t for c in df.columns if (t := cls._canonical_long_column(c)) is not None}
        )

    def _wide_multiindex_to_long(self, raw: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(raw.columns, pd.MultiIndex):
            logger.warning(
                "yfinance 열이 MultiIndex가 아닙니다. download(tickers=<list>, ...)만 지원합니다."
            )
            return self._empty_output()

        if tuple(raw.columns.names[:2]) == ('Price', 'Ticker'):
            logger.warning(
                "열 순서가 Price-first입니다. download(tickers=<list>, group_by='ticker')만 사용하세요."
            )
            return self._empty_output()

        logger.debug(
            "yfinance wide MultiIndex: names=%s ncols=%s sample=%s",
            raw.columns.names,
            raw.shape[1],
            raw.columns[: min(12, raw.shape[1])].tolist(),
        )

        stacked = self._stack_level0_preserve_na(raw)
        stacked = stacked.reset_index()
        out = self._rename_stacked_columns(stacked)
        missing = [c for c in ('date', 'symbol', 'open', 'high', 'low', 'close', 'adj_close', 'volume')
                   if c not in out.columns]
        if missing:
            logger.error("long 변환 후 필수 컬럼 누락: %s", missing)
            return self._empty_output()

        out['date'] = pd.to_datetime(out['date']).dt.date
        out['symbol'] = out['symbol'].astype(str).str.strip().str.replace('.', '-', regex=False)
        out = out[list(self._OUT_COLS)].copy()
        out.columns.name = None
        return out


if __name__ == "__main__":
    import sys

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from data_pipeline.db_manager import DBManager

    db = DBManager()
    db.create_tables()

    fetcher = SP500Fetcher()
    sp500_df = fetcher.fetch_tickers()
    if sp500_df is None or sp500_df.empty:
        logger.error("유니버스 수집 실패 — 종료")
        raise SystemExit(1)

    db.upsert_tickers(sp500_df)

    symbols = sp500_df['ticker'].tolist()
    start_e = os.environ.get("OHLCV_START", "2025-01-01")
    end_e = os.environ.get("OHLCV_END", "2026-04-03")
    logger.info("OHLCV 구간: %s ~ %s (전체 백필은 OHLCV_START/OHLCV_END 조정)", start_e, end_e)

    ohlcv_df = OHLCVFetcher().fetch(symbols, start=start_e, end=end_e)
    if ohlcv_df.empty:
        logger.warning("OHLCV가 비어 있어 daily_prices 적재를 건너뜁니다.")
    else:
        db.upsert_daily_prices(ohlcv_df)

    fetcher.save_to_csv(sp500_df)
    logger.info("파이프라인 완료: tickers + daily_prices(비어 있지 않을 때)")
