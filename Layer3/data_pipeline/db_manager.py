import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Date, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
from datetime import datetime
import logging

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Base = declarative_base()

class Universe(Base):
    """
    S&P 500 종목 유니버스 테이블 정의
    """
    __tablename__ = 'universe_sp500'

    ticker = Column(String(20), primary_key=True)
    company_name = Column(String(255))
    sector = Column(String(100))
    industry = Column(String(100))
    date_added = Column(String(50))  # 위키피디아 날짜 형식이 다양하여 String으로 우선 수집
    cik = Column(String(20))
    last_updated = Column(DateTime, default=datetime.now, onupdate=datetime.now)

def _resolve_database_url() -> str:
    """
    Supabase 권장: 대시보드의 DATABASE_URL(Connection string).
    미설정 시 DB_* 분리 변수로 로컬/레거시 URL 조합.
    """
    url = (os.getenv("DATABASE_URL") or "").strip()
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


class DBManager:
    """
    Supabase(호스트 PostgreSQL) 연결 및 유니버스 테이블 관리.
    """

    def __init__(self):
        self.db_url = _resolve_database_url()
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        """환경에 필요한 테이블 생성"""
        logger.info("필요한 테이블 생성 중 (universe_sp500)...")
        Base.metadata.create_all(self.engine)

    def upsert_universe(self, csv_path):
        """
        CSV 파일로부터 데이터를 읽어 DB에 Upsert (Insert or Update)
        """
        if not os.path.exists(csv_path):
            logger.error(f"파일을 찾을 수 없습니다: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        session = self.Session()

        try:
            # PostgreSQL / Supabase 호환 upsert (ON CONFLICT DO UPDATE)
            for _, row in df.iterrows():
                stmt = insert(Universe).values(
                    ticker=row['ticker'],
                    company_name=row['company_name'],
                    sector=row['sector'],
                    industry=row['industry'],
                    date_added=str(row['date_added']),
                    cik=str(row['cik']),
                    last_updated=datetime.now()
                )
                
                # 티커가 겹치면 회사명, 섹터 등을 업데이트
                do_update_stmt = stmt.on_conflict_do_update(
                    index_elements=['ticker'],
                    set_={
                        'company_name': stmt.excluded.company_name,
                        'sector': stmt.excluded.sector,
                        'industry': stmt.excluded.industry,
                        'last_updated': datetime.now()
                    }
                )
                session.execute(do_update_stmt)

            session.commit()
            logger.info(f"성공적으로 {len(df)}개 유니버스 데이터를 DB에 적재/업데이트 완료.")
        
        except Exception as e:
            logger.error(f"DB 적재 실패: {e}")
            session.rollback()
        finally:
            session.close()

if __name__ == "__main__":
    db_mgr = DBManager()
    # 0. 테이블 생성 (최초 1회 필수)
    db_mgr.create_tables()
    
    # 1. 수집된 최신 CSV 파일 경로 (예시)
    # 실제 실행 시 fetcher가 만든 파일명을 넣어야 함
    # db_mgr.upsert_universe('data/universe/sp500_universe_YYYYMMDD.csv')
