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

---

## [2026-04-03] Phase 0 — Earnings Revision, Form 4, Value Chain (`plan.md` Phase 0 체크리스트 대응)

`z_docs/plan.md` Phase 0 중 아래 항목과 직접 대응하는 작업을 한 번에 정리한다. 상세 파일·컬럼 단위 기록은 `z_docs/task_log.md` §1·§2·§3·§4.

- **실시간·일간 Earnings Revision 연동** (`plan.md` L12): `EarningsRevision` ORM 및 `upsert_earnings_revisions`, `earnings_fetcher.py`에서 yfinance 기반 추정·리비전 점수 적재. PK `(symbol, date)`, 청크 500.
- **SEC Form 4 내부자 거래 모듈** (`plan.md` L13): `insider_fetcher.py`(EDGAR submissions + Form 4 XML), `InsiderTrade` / `upsert_insider_trades`, `accession_number` UNIQUE·다건 거래 시 행 접미사 규칙. `SEC_USER_AGENT` 필수.
- **분류.json → `tickers.vc_*`** (`plan.md` L14): `classification_loader.py`, `upsert_vc_classification`(UPDATE-only 보호), `vc_stage_num` 룰 매핑. Track D 설계 입력으로 `plan.md` Phase 1 L18에서 참조.

**문서**: `task_plan.md`는 Form 4 스펙·미결정 사항 정리용; 실행 이력은 본 절 + `task_log.md`를 보면 된다.

---

## [2026-04-04] Phase 0 보강 — Fundamentals 수집 + Phase 1 — 스코어링 스택

**한 줄 요약**: yfinance 기반 `fundamentals` 배치 적재 파이프라인을 추가하고, Track A·Track D·통합 엔진을 `scoring/`에 구현해 `daily_scores`까지 upsert 가능한 상태로 맞춤.  
**상세(파일·컬럼·폴백 단계)**: `z_docs/task_log.md` [2026-04-04] 절 참고.

### Phase 0 — Fundamentals
- `fundamentals_fetcher.py`: `ticker.info` 스냅샷 → `Fundamentals` 테이블, 20종목마다 upsert, 연결 사전 확인.
- `db_manager`: `Fundamentals` ORM, `_prepare_fundamentals_df`, `upsert_fundamentals`.
- `fetch_all` 제거·`epsForward`만 consensus로 사용 등 스펙 단순화 반영.
- 문서: `task_plan.md`(Step 4 완료), `plan.md`(Phase 0 항목 반영).

### Phase 1 — Scoring & 적재
- `scoring/track_a.py`: 4서브스코어 가중합 → `score_original`.
- `scoring/track_d.py`: VC 메타 + OM 백분위 → `score_bottleneck`. **MIN_GROUP_SIZE=3**으로 복합 그룹→stage 단독→전역(`U`) 폴백 후 `_group_key` 단일 `groupby` 백분위.
- `scoring/engine.py`: `W_TRACK_A/W_TRACK_D = 0.6/0.4`, `REGIME_STUB`, 결측 50 보간 후 `total_heinrich` → `upsert_daily_scores`.
- `db_manager`: `DailyScores` ORM, `_prepare_daily_scores_df`, `upsert_daily_scores`.
- `pyproject.toml`: 패키지 `scoring` 등록.

### 결정·이슈 요약
- Track D에서 **OM 보간 축(gics_sector)** 과 **백분위 그룹 축(vc_*)** 을 분리 유지.
- 소규모 VC 셀에서 백분위가 왜곡되던 문제 → **최소 그룹 크기 폴백**으로 완화.
- 실행·SQL 검증: `python -m scoring.engine YYYY-MM-DD`, `COUNT(*) FROM daily_scores WHERE date = ...` ≈ tickers 수.

---

## [2026-04-03] Phase 0 — 수집 파이프라인 운영 보강 (연결 검증·중간 적재)

`plan.md` Phase 0는 기능 구현 완료 후에도 **데이터 인프라가 장시간 잡에서 안전하게 돌아가도록** 운영 층을 보강했다. 배경·파일별 디테일은 `task_log.md` §5·§6.

- **PostgreSQL 연결 문자열**: `db_manager._resolve_database_url`이 **`DATABASE_URL`과 `DB_URL` 모두** 인식(Session Pooler 등 단일 변수만 쓰는 `.env`와 호환).
- **`insider_fetcher.py`**: 수집 **전** `DBManager` 연결 ping, **20종목마다**(및 마지막) `insider_trades` 배치 `upsert`로 네트워크 단절 시 유실 구간 축소.

**의도**: Phase 0 파이프라인이 “끝까지 적재된다”는 전제를, 시작 시 연결 확인과 주기적 flush로 뒷받침한다. Phase 1 이후에도 동일 패턴을 스코어 적재·백필에 재사용할 수 있다.
