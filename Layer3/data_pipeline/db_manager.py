import os
from dotenv import load_dotenv
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    SmallInteger,
    String,
    create_engine,
    text,
    update,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Base = declarative_base()

# 배치 upsert: 긴 작업에서도 단일 트랜잭션 내 시각 일관성·메모리 사용을 위해 500~1000행 권장
CHUNK_SIZE = 500


class Ticker(Base):
    __tablename__ = 'tickers'

    symbol = Column(String(10), primary_key=True)
    company_name = Column(String(100))
    gics_sector = Column(String(50))
    gics_industry = Column(String(50))
    is_active = Column(Boolean, nullable=False, server_default=text('true'))
    vc_sector = Column(String(50), nullable=True)
    vc_position = Column(String(100), nullable=True)
    vc_stage_num = Column(SmallInteger, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class DailyPrice(Base):
    """일봉: OHLC + 거래량 + 조정종가(표준 OHLCV 밖 컬럼)."""

    __tablename__ = 'daily_prices'

    symbol = Column(
        String(10),
        ForeignKey('tickers.symbol', ondelete='CASCADE'),
        primary_key=True,
    )
    date = Column(Date, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float, nullable=True)
    volume = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class EarningsRevision(Base):
    __tablename__ = 'earnings_revisions'

    symbol = Column(
        String(10),
        ForeignKey('tickers.symbol', ondelete='CASCADE'),
        primary_key=True,
    )
    date = Column(Date, primary_key=True)
    eps_est_current = Column(Float, nullable=True)
    revision_score = Column(Float, nullable=True)
    revision_score_7d = Column(Float, nullable=True)
    revision_score_30d = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Fundamentals(Base):
    __tablename__ = 'fundamentals'

    symbol = Column(
        String(10),
        ForeignKey('tickers.symbol', ondelete='CASCADE'),
        primary_key=True,
    )
    report_date = Column(Date, primary_key=True)
    eps_actual = Column(Float, nullable=True)
    eps_consensus = Column(Float, nullable=True)
    operating_margin = Column(Float, nullable=True)
    debt_to_equity = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class InsiderTrade(Base):
    __tablename__ = 'insider_trades'

    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=text('gen_random_uuid()'))
    symbol = Column(String(10), ForeignKey('tickers.symbol', ondelete='CASCADE'), nullable=False)
    filing_date = Column(Date, nullable=False)
    transaction_type = Column(String(10), nullable=True)
    value = Column(Float, nullable=True)
    accession_number = Column(String(25), unique=True, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class DailyScores(Base):
    __tablename__ = 'daily_scores'

    symbol = Column(
        String(10),
        ForeignKey('tickers.symbol', ondelete='CASCADE'),
        primary_key=True,
    )
    date = Column(Date, primary_key=True)
    regime_stub = Column(String(50), nullable=True)
    score_original = Column(Float, nullable=True)
    score_bottleneck = Column(Float, nullable=True)
    total_heinrich = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def _resolve_database_url() -> str:
    """
    Supabase 권장: DATABASE_URL 또는 DB_URL(Connection string).
    미설정 시 DB_* 분리 변수로 로컬/레거시 URL 조합.
    """
    url = (os.getenv("DATABASE_URL") or os.getenv("DB_URL") or "").strip()
    if url:
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://") :]
        return url
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "layer3_db")
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def _prepare_tickers_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'symbol' not in out.columns and 'ticker' in out.columns:
        out = out.rename(columns={'ticker': 'symbol'})
    if 'gics_sector' not in out.columns and 'sector' in out.columns:
        out = out.rename(columns={'sector': 'gics_sector'})
    if 'gics_industry' not in out.columns and 'industry' in out.columns:
        out = out.rename(columns={'industry': 'gics_industry'})
    required = ['symbol', 'company_name', 'gics_sector', 'gics_industry']
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"tickers 적재에 필요한 컬럼이 없습니다: {missing}")
    out = out[required].copy()
    out['symbol'] = (
        out['symbol'].astype(str).str.strip().str.replace('.', '-', regex=False)
    )
    def _cell_str(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        s = str(x).strip()
        return s if s else None

    for c in ['company_name', 'gics_sector', 'gics_industry']:
        out[c] = out[c].apply(_cell_str)
    out = out.dropna(subset=['symbol'])
    out = out[out['symbol'] != '']
    return out


def _prepare_daily_prices_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(' ', '_') for c in out.columns]
    if 'ticker' in out.columns and 'symbol' not in out.columns:
        out = out.rename(columns={'ticker': 'symbol'})
    required = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"daily_prices 적재에 필요한 컬럼이 없습니다: {missing}")
    if 'adj_close' not in out.columns:
        logger.info("adj_close 컬럼 없음 → close로 대체합니다.")
        out['adj_close'] = out['close']
    out['symbol'] = (
        out['symbol'].astype(str).str.strip().str.replace('.', '-', regex=False)
    )
    out['date'] = pd.to_datetime(out['date'], errors='coerce').dt.date
    for col in ('open', 'high', 'low', 'close', 'adj_close'):
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out['volume'] = pd.to_numeric(out['volume'], errors='coerce')
    out = out.dropna(subset=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
    out = out[out['symbol'] != '']
    out['volume'] = out['volume'].astype('int64')
    cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    return out[cols]


def _prepare_earnings_revisions_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(' ', '_') for c in out.columns]
    if 'ticker' in out.columns and 'symbol' not in out.columns:
        out = out.rename(columns={'ticker': 'symbol'})
    required = ['symbol', 'date', 'eps_est_current', 'revision_score_7d', 'revision_score_30d']
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"earnings_revisions 적재에 필요한 컬럼이 없습니다: {missing}")
    out['symbol'] = (
        out['symbol'].astype(str).str.strip().str.replace('.', '-', regex=False)
    )
    out['date'] = pd.to_datetime(out['date'], errors='coerce').dt.date
    for col in ('eps_est_current', 'revision_score_7d', 'revision_score_30d'):
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out = out.dropna(subset=['symbol', 'date'])
    out = out[out['symbol'] != '']
    return out[required]


def _prepare_fundamentals_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(' ', '_') for c in out.columns]
    if 'ticker' in out.columns and 'symbol' not in out.columns:
        out = out.rename(columns={'ticker': 'symbol'})
    if 'date' in out.columns and 'report_date' not in out.columns:
        out = out.rename(columns={'date': 'report_date'})
    required = ['symbol', 'report_date']
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"fundamentals 적재에 필요한 컬럼이 없습니다: {missing}")
    metric_cols = [
        'eps_actual',
        'eps_consensus',
        'operating_margin',
        'debt_to_equity',
    ]
    for c in metric_cols:
        if c not in out.columns:
            out[c] = None
    out['symbol'] = (
        out['symbol'].astype(str).str.strip().str.replace('.', '-', regex=False)
    )
    out['report_date'] = pd.to_datetime(out['report_date'], errors='coerce').dt.date
    for col in metric_cols:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out = out.dropna(subset=['symbol', 'report_date'])
    out = out[out['symbol'] != '']
    for col in metric_cols:
        out[col] = out[col].apply(
            lambda v: None
            if v is None or (isinstance(v, float) and pd.isna(v))
            else float(v)
        )
    cols = required + metric_cols
    return out[cols]


def _prepare_insider_trades_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(' ', '_') for c in out.columns]
    if 'ticker' in out.columns and 'symbol' not in out.columns:
        out = out.rename(columns={'ticker': 'symbol'})
    required = ['symbol', 'filing_date', 'transaction_type', 'value', 'accession_number']
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"insider_trades 적재에 필요한 컬럼이 없습니다: {missing}")
    out['symbol'] = (
        out['symbol'].astype(str).str.strip().str.replace('.', '-', regex=False)
    )
    out['accession_number'] = out['accession_number'].astype(str).str.strip()
    out['transaction_type'] = out['transaction_type'].apply(
        lambda x: None
        if x is None or (isinstance(x, float) and pd.isna(x))
        else str(x).strip()[:10]
        if str(x).strip()
        else None
    )
    out['filing_date'] = pd.to_datetime(out['filing_date'], errors='coerce').dt.date
    out['value'] = pd.to_numeric(out['value'], errors='coerce')
    out = out.dropna(subset=['symbol', 'filing_date'])
    out = out[out['symbol'] != '']
    out = out[out['accession_number'] != '']
    cols = ['symbol', 'filing_date', 'transaction_type', 'value', 'accession_number']
    return out[cols]


def _prepare_daily_scores_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(' ', '_') for c in out.columns]
    if 'ticker' in out.columns and 'symbol' not in out.columns:
        out = out.rename(columns={'ticker': 'symbol'})
    required = ['symbol', 'date']
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"daily_scores 적재에 필요한 컬럼이 없습니다: {missing}")
    opt = ['regime_stub', 'score_original', 'score_bottleneck', 'total_heinrich']
    for c in opt:
        if c not in out.columns:
            out[c] = None
    out['symbol'] = (
        out['symbol'].astype(str).str.strip().str.replace('.', '-', regex=False)
    )
    out['date'] = pd.to_datetime(out['date'], errors='coerce').dt.date
    out['regime_stub'] = out['regime_stub'].apply(
        lambda x: None
        if x is None or (isinstance(x, float) and pd.isna(x))
        else str(x).strip()[:50]
        if str(x).strip()
        else None
    )
    for col in ('score_original', 'score_bottleneck', 'total_heinrich'):
        out[col] = pd.to_numeric(out[col], errors='coerce')
        out[col] = out[col].apply(
            lambda v: None
            if v is None or (isinstance(v, float) and pd.isna(v))
            else float(v)
        )
    out = out.dropna(subset=['symbol', 'date'])
    out = out[out['symbol'] != '']
    return out[required + opt]


def _prepare_vc_classification_df(df: pd.DataFrame) -> pd.DataFrame:
    """symbol, vc_sector, vc_position, vc_stage_num 정규화."""
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(' ', '_') for c in out.columns]
    required = ['symbol', 'vc_sector', 'vc_position', 'vc_stage_num']
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"vc_classification 적재에 필요한 컬럼이 없습니다: {missing}")
    out['symbol'] = (
        out['symbol'].astype(str).str.strip().str.replace('.', '-', regex=False)
    )
    out['vc_sector'] = out['vc_sector'].apply(
        lambda x: None
        if x is None or (isinstance(x, float) and pd.isna(x))
        else str(x).strip()[:50] or None
    )
    out['vc_position'] = out['vc_position'].apply(
        lambda x: None
        if x is None or (isinstance(x, float) and pd.isna(x))
        else str(x).strip()[:100] or None
    )
    out['vc_stage_num'] = pd.to_numeric(out['vc_stage_num'], errors='coerce')
    out = out.dropna(subset=['symbol'])
    out = out[out['symbol'] != '']
    return out[required]


class DBManager:
    """Supabase(PostgreSQL) 연결 및 tickers / daily_prices 관리."""

    def __init__(self):
        self.db_url = _resolve_database_url()
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        logger.info("테이블 생성 중 (tickers, daily_prices)...")
        Base.metadata.create_all(self.engine)

    def upsert_tickers(self, df: pd.DataFrame) -> None:
        """S&P 500 CSV 등: symbol(또는 ticker), company_name, gics_sector/sector, gics_industry/industry."""
        if df is None or df.empty:
            logger.warning("upsert_tickers: 빈 DataFrame — 적재 생략")
            return
        norm = _prepare_tickers_df(df)
        if norm.empty:
            logger.warning("upsert_tickers: 유효한 행 없음")
            return
        records = norm.to_dict('records')
        session = self.Session()
        try:
            for i in range(0, len(records), CHUNK_SIZE):
                chunk = records[i : i + CHUNK_SIZE]
                stmt = insert(Ticker).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol'],
                    set_={
                        'company_name': stmt.excluded.company_name,
                        'gics_sector': stmt.excluded.gics_sector,
                        'gics_industry': stmt.excluded.gics_industry,
                        'updated_at': func.now(),
                    },
                )
                session.execute(stmt)
            session.commit()
            logger.info("tickers %s행 upsert 완료.", len(norm))
        except Exception as e:
            logger.error("tickers 적재 실패: %s", e)
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_daily_prices(self, df: pd.DataFrame) -> None:
        """OHLCV + adj_close(조정종가) → daily_prices. 기대: symbol|ticker, date, open, high, low, close, volume, [adj_close]."""
        if df is None or df.empty:
            logger.warning("upsert_daily_prices: 빈 DataFrame — 적재 생략")
            return
        norm = _prepare_daily_prices_df(df)
        if norm.empty:
            logger.warning("upsert_daily_prices: 유효한 행 없음")
            return
        records = norm.to_dict('records')
        session = self.Session()
        try:
            for i in range(0, len(records), CHUNK_SIZE):
                chunk = records[i : i + CHUNK_SIZE]
                stmt = insert(DailyPrice).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'date'],
                    set_={
                        'open': stmt.excluded.open,
                        'high': stmt.excluded.high,
                        'low': stmt.excluded.low,
                        'close': stmt.excluded.close,
                        'adj_close': stmt.excluded.adj_close,
                        'volume': stmt.excluded.volume,
                    },
                )
                session.execute(stmt)
            session.commit()
            logger.info("daily_prices %s행 upsert 완료.", len(norm))
        except Exception as e:
            logger.error("daily_prices 적재 실패: %s", e)
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_earnings_revisions(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            logger.warning("upsert_earnings_revisions: 빈 DataFrame — 적재 생략")
            return
        norm = _prepare_earnings_revisions_df(df)
        if norm.empty:
            logger.warning("upsert_earnings_revisions: 유효한 행 없음")
            return
        records = norm.to_dict('records')
        session = self.Session()
        try:
            for i in range(0, len(records), CHUNK_SIZE):
                chunk = records[i : i + CHUNK_SIZE]
                stmt = insert(EarningsRevision).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'date'],
                    set_={
                        'eps_est_current': stmt.excluded.eps_est_current,
                        'revision_score_7d': stmt.excluded.revision_score_7d,
                        'revision_score_30d': stmt.excluded.revision_score_30d,
                    },
                )
                session.execute(stmt)
            session.commit()
            logger.info("earnings_revisions %s행 upsert 완료.", len(norm))
        except Exception as e:
            logger.error("earnings_revisions 적재 실패: %s", e)
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_fundamentals(self, df: pd.DataFrame) -> None:
        """
        fundamentals 테이블 upsert.
        Conflict key: (symbol, report_date) → 지표 컬럼만 갱신.
        """
        if df is None or df.empty:
            logger.warning("upsert_fundamentals: 빈 DataFrame — 적재 생략")
            return
        norm = _prepare_fundamentals_df(df)
        if norm.empty:
            logger.warning("upsert_fundamentals: 유효한 행 없음")
            return
        records = norm.to_dict('records')
        session = self.Session()
        try:
            for i in range(0, len(records), CHUNK_SIZE):
                chunk = records[i : i + CHUNK_SIZE]
                stmt = insert(Fundamentals).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'report_date'],
                    set_={
                        'eps_actual': stmt.excluded.eps_actual,
                        'eps_consensus': stmt.excluded.eps_consensus,
                        'operating_margin': stmt.excluded.operating_margin,
                        'debt_to_equity': stmt.excluded.debt_to_equity,
                    },
                )
                session.execute(stmt)
            session.commit()
            logger.info("fundamentals %s행 upsert 완료.", len(norm))
        except Exception as e:
            logger.error("fundamentals 적재 실패: %s", e)
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_insider_trades(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            logger.warning("upsert_insider_trades: 빈 DataFrame — 적재 생략")
            return
        norm = _prepare_insider_trades_df(df)
        if norm.empty:
            logger.warning("upsert_insider_trades: 유효한 행 없음")
            return
        records = norm.to_dict('records')
        session = self.Session()
        try:
            for i in range(0, len(records), CHUNK_SIZE):
                chunk = records[i : i + CHUNK_SIZE]
                stmt = insert(InsiderTrade).values(chunk)
                stmt = stmt.on_conflict_do_nothing(index_elements=['accession_number'])
                session.execute(stmt)
            session.commit()
            logger.info("insider_trades %s행 insert(중복 건너뜀) 완료.", len(norm))
        except Exception as e:
            logger.error("insider_trades 적재 실패: %s", e)
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_daily_scores(self, df: pd.DataFrame) -> None:
        """
        daily_scores upsert. Conflict (symbol, date) → regime_stub, score_*, total_heinrich 갱신.
        """
        if df is None or df.empty:
            logger.warning("upsert_daily_scores: 빈 DataFrame — 적재 생략")
            return
        norm = _prepare_daily_scores_df(df)
        if norm.empty:
            logger.warning("upsert_daily_scores: 유효한 행 없음")
            return
        records = norm.to_dict('records')
        session = self.Session()
        try:
            for i in range(0, len(records), CHUNK_SIZE):
                chunk = records[i : i + CHUNK_SIZE]
                stmt = insert(DailyScores).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'date'],
                    set_={
                        'regime_stub': stmt.excluded.regime_stub,
                        'score_original': stmt.excluded.score_original,
                        'score_bottleneck': stmt.excluded.score_bottleneck,
                        'total_heinrich': stmt.excluded.total_heinrich,
                    },
                )
                session.execute(stmt)
            session.commit()
            logger.info("daily_scores %s행 upsert 완료.", len(norm))
        except Exception as e:
            logger.error("daily_scores 적재 실패: %s", e)
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_vc_classification(self, df: pd.DataFrame) -> None:
        """분류.json 등: vc_sector / vc_position / vc_stage_num만 기존 tickers 행 갱신."""
        if df is None or df.empty:
            logger.warning("upsert_vc_classification: 빈 DataFrame — 적재 생략")
            return
        norm = _prepare_vc_classification_df(df)
        if norm.empty:
            logger.warning("upsert_vc_classification: 유효한 행 없음")
            return
        records = norm.to_dict('records')
        session = self.Session()
        try:
            updated = 0
            for row in records:
                sn = row['vc_stage_num']
                if sn is None or (isinstance(sn, float) and pd.isna(sn)):
                    sn_i = None
                else:
                    sn_i = int(sn)
                res = session.execute(
                    update(Ticker)
                    .where(Ticker.symbol == row['symbol'])
                    .values(
                        vc_sector=row['vc_sector'],
                        vc_position=row['vc_position'],
                        vc_stage_num=sn_i,
                        updated_at=func.now(),
                    )
                )
                updated += res.rowcount or 0
            session.commit()
            logger.info(
                "tickers vc_* 갱신 요청 %s행 (존재하는 symbol만 rowcount 반영, 합계 %s).",
                len(norm),
                updated,
            )
        except Exception as e:
            logger.error("vc_classification 적재 실패: %s", e)
            session.rollback()
            raise
        finally:
            session.close()


if __name__ == "__main__":
    db_mgr = DBManager()
    db_mgr.create_tables()
    # 예: CSV → DataFrame → tickers
    # df = pd.read_csv('data/universe/sp500_universe_YYYYMMDD.csv')
    # db_mgr.upsert_tickers(df)
