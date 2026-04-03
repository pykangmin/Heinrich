# 데이터 소스 및 환경 설정

| 용도 | 소스 | 설치/설정 |
| :--- | :--- | :--- |
| **매크로 지표 전체** | FRED API | `pip install fredapi` + API 키 발급 |
| **주가 데이터** | yfinance | `pip install yfinance` |
| **뉴스 헤드라인** | Reuters + CNBC + Yahoo Finance RSS | `pip install feedparser` |
| **공시/내부자** | SEC EDGAR API | `requests`로 직접 호출 |
| **NLP 감성분석** | FinBERT | `pip install transformers` |
| **뉴스 이벤트 분석** | GDELT BigQuery | Google Cloud 무료 계정 |
| **금리 기대** | Fed Funds Futures (yfinance) | ZQ 티커 조회 |
| **EPS 추정치·Revision** | yfinance | `Ticker.eps_revisions` (7일/30일 상향·하향 건수), `Ticker.earnings_forecast` (컨센서스 EPS) |