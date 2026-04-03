# 작업 로그 (세션 정리)

본 문서는 동일 세션에서 수행한 변경 사항을 주제별로 묶어 기록한다. (이미 `log.md` 등에 옮긴 내용은 중복하지 않음.)

---

## 1. Earnings Revision 파이프라인

- **DB (`data_pipeline/db_manager.py`)**: `EarningsRevision` ORM, `_prepare_earnings_revisions_df`, `upsert_earnings_revisions` (PK `symbol,date`, 갱신 필드 `eps_est_current` / `revision_score_7d` / `revision_score_30d`, 청크 500). `revision_score` 컬럼은 DataFrame에서 제외·DB NULL 유지.
- **수집 (`data_pipeline/earnings_fetcher.py`)**: yfinance `eps_revisions` 첫 행·`earnings_estimate.avg`(0q) 폴백, `_revision_score` 버그 수정(`else float(down)`). `load_symbols`는 DB `tickers` 우선·CSV 폴백.
- **운영 전제**: Supabase에 `revision_score_7d` / `revision_score_30d` 컬럼 ALTER (사용자 측 완료). 적재 검증 501/503행 수준까지 확인됨.

---

## 2. Task Plan 갱신

- **`z_docs/task_plan.md`**: Earnings 완료 반영, 당면 과제를 Phase 0 **SEC Form 4 내부자 거래 파이프라인**으로 교체.

---

## 3. SEC Form 4 · `insider_trades`

- **스키마 (`z_docs/schema.md`)**: `accession_number VARCHAR(25) UNIQUE`, 마이그레이션 SQL 예시.
- **DB**: `InsiderTrade` ORM (`id`는 `gen_random_uuid()`), `_prepare_insider_trades_df`, `upsert_insider_trades` — `ON CONFLICT DO NOTHING` on `accession_number`.
- **수집 (`data_pipeline/insider_fetcher.py`)**: `data.sec.gov/submissions/CIK{cik:010d}.json` + `www.sec.gov/files/company_tickers.json`(회사 맵; `data.sec.gov/.../company_tickers.json`은 404). Form 4 XML 파싱(nonDerivative/derivative), `SEC_USER_AGENT` 필수. **동일 제출 내 다건 거래**는 DB UNIQUE 대비 `접수번호#행번호` 형태(25자 이하)로 `accession_number` 저장.
- **환경 (`.env_example`)**: `SEC_USER_AGENT` 추가.

---

## 4. Value Chain 분류 · `tickers.vc_*`

- **DB**: `Ticker`에 `vc_sector`, `vc_position`, `vc_stage_num` (SmallInteger). `upsert_vc_classification`는 **UPDATE만** 수행하여 기존 티커 메타가 NULL로 덮어쓰이지 않도록 함; `_prepare_vc_classification_df`로 길이·타입 정규화.
- **`data_pipeline/classification_loader.py`**: 루트 `분류.json` 파싱, 티커 괄호 추출, `ORDERED_VC_STAGE_RULES`로 `vc_stage_num` 매핑(EDA·Midstream·1차 가공·EMS 등 룰 보강 후 457 심볼 전부 매핑됨), 중복 심볼은 `keep='last'`.
- **문서**: `schema.md`의 `tickers`에 `vc_*` 및 ALTER 예시; `plan.md`에 Phase 0 `분류.json` 항목·Phase 1 Track D에서 `vc_*` 활용 방향 명시.

---

## 5. 후속 확인 권장 (미실행 항목)

- Supabase `tickers`에 `vc_*` 컬럼 미추가 시 `schema.md`의 ALTER 적용 후 `python -m data_pipeline.classification_loader` 실행.
- 내부자·실적 파이프라인은 `.env`의 `DATABASE_URL` / `SEC_USER_AGENT` 유지 필요.
