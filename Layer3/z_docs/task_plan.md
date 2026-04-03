# 현재 작업 목표 (Task Plan)

가장 시급하게 진행해야 할 최우선 개발 태스크입니다.

---

## 당면 과제: Phase 0 세 번째 단계 — SEC Form 4 내부자 거래 파이프라인

**목표**: SEC EDGAR Form 4 제출 데이터를 파싱하여 `insider_trades` 테이블에 적재.

### 배경
- `insider_trades` 테이블 스키마:
  `(id UUID PK gen_random_uuid(), symbol, filing_date, transaction_type, value, created_at)`
- 내부자 거래는 Track A 점수 모델의 보조 선행 지표.
- 데이터 소스: **SEC EDGAR** — `https://www.sec.gov/cgi-bin/browse-edgar` 또는 EDGAR Full-Text Search API.

### 결정해야 할 것
1. **수집 방법** — EDGAR RSS 피드(`/cgi-bin/browse-edgar?action=getcompany&type=4&dateb=&owner=include&count=40&search_text=`) vs EDGAR XBRL API(`/submissions/`)
2. **매핑 방식** — CIK → S&P 500 symbol 매핑 (EDGAR `company_tickers.json` 활용 가능)
3. **transaction_type** — 'P'(매수)·'S'(매도)·'A'(부여) 등 필터링 범위
4. **수집 구간** — 전체 백필(1년치) vs 최근 90일

### 구현 스펙
- `data_pipeline/insider_fetcher.py` 신규 작성
- EDGAR 요청 시 `User-Agent` 헤더 필수 (`User-Agent: <name> <email>`)
- 출력: `symbol, filing_date, transaction_type, value` DataFrame → `db_manager.upsert_insider_trades(df)` (신규 메서드)
- `db_manager.py`에 `InsiderTrade` ORM 모델 및 `upsert_insider_trades` 추가
- `id`는 DB 서버 `gen_random_uuid()` 기본값 사용 (클라이언트 미생성)

### 참고 파일
- `z_docs/schema.md` — insider_trades 테이블 정의
- `z_docs/source.md` — SEC EDGAR 항목
- `data_pipeline/earnings_fetcher.py` — fetcher 구조 참고 패턴
- `data_pipeline/db_manager.py` — ORM 추가 위치
