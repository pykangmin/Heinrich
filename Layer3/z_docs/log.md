# 영구 작업 히스토리 로그

완료된 주요 마일스톤 및 작업 내역을 요약하여 영구 보존합니다.

---
## [2026-04-03] Phase 0 — DB 스키마 확정 및 Supabase 테이블 생성

- 환경 설정: `.env_example` 을 `DATABASE_URL` 단일 변수로 확정. SQLAlchemy 직접 연결 방식 채택.
- 스키마 설계 완료: Gemini 피드백 반영하여 `daily_prices`에 OHLC 전체(`open/high/low/close/adj_close`) 및 `created_at` 추가, 전 테이블 수치형 컬럼 `NUMERIC → DOUBLE PRECISION` 변경.
- Supabase에 6개 테이블(`tickers`, `daily_prices`, `fundamentals`, `earnings_revisions`, `insider_trades`, `daily_scores`) + 인덱스 생성 완료.
- 잔여: `schema.md` DDL 업데이트, `db_manager.py` 신규 스키마 기준 재작성. → **이후 세션에서 완료**

---

## [2026-04-03] Phase 0 — `db_manager.py` 재설계 완료

- `Universe` ORM 제거 → `Ticker` / `DailyPrice` 모델로 교체 (`tickers` / `daily_prices` 테이블 기준).
- 전처리 함수 분리: `_prepare_tickers_df` (컬럼 매핑·정규화), `_prepare_daily_prices_df` (소문자화·adj_close 폴백·volume NaN 행 제거).
- `upsert_tickers`: conflict on `symbol` → `company_name`, `gics_sector`, `gics_industry`, `updated_at`(`func.now()`) 갱신. `created_at` 미갱신.
- `upsert_daily_prices`: conflict on `(symbol, date)` → OHLCV·`adj_close` 갱신. `created_at` 미갱신.
- 공통: `CHUNK_SIZE = 500` 배치 upsert, 실패 시 `rollback()` 후 예외 재전파.
- 주요 설계 결정: `adj_close = Column(Float, nullable=True)`, `updated_at`은 클라이언트 시각 대신 `func.now()`.

---

## [2026-04-03] Phase 0 — `OHLCVFetcher` 구현 및 E2E 파이프라인 완성

- `sp500_fetcher.py`에 `OHLCVFetcher` 클래스 추가.
  - `yfinance.download(group_by='ticker', auto_adjust=False, threads=True)` 배치 다운로드.
  - MultiIndex wide → long 변환: `_stack_level0_preserve_na`(`future_stack=True` → `dropna=False` → 폴백), `_canonical_long_column`(대소문자·공백 표기 흡수).
  - Price-first MultiIndex 감지 시 경고 + 빈 DF 반환.
- `SP500Fetcher.fetch_tickers`: `requests.get`(`User-Agent`, `timeout=10`) + `read_html(StringIO(...))` 로 Wikipedia 403 우회. 위키 8열 기준 컬럼 매핑, 열 수 변경 시 `ValueError`.
- Supabase 연결 이슈 해결: `DB_URL` → `DATABASE_URL` 변수명 수정, `=` 공백 제거, Session Pooler URL 사용, `gics_industry VARCHAR(50→100)` ALTER.
- **E2E 검증 완료**: `tickers` 503행, `daily_prices` 20,459행 적재 확인 (구간: 2025-01-01 ~ 2026-04-03 기본값).
- 실패 종목 4개(`Q`, `SOLV`, `GEV`, `SNDK`): yfinance 데이터 공백, 파이프라인 정상 동작.
