import pandas as pd
import os
import logging
from datetime import datetime

# 로깅 설정
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
            # pandas의 read_html을 사용하여 위키피디아 테이블 파싱
            tables = pd.read_html(self.WIKI_URL)
            df = tables[0]
            
            # 컬럼명 정리
            df.columns = [
                'ticker', 'company_name', 'sec_filings', 'sector', 
                'industry', 'headquarters', 'date_added', 'cik', 'founded'
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

if __name__ == "__main__":
    fetcher = SP500Fetcher()
    sp500_df = fetcher.fetch_tickers()
    fetcher.save_to_csv(sp500_df)
