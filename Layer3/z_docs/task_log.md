# 작업 로그 (세션 정리)

주제별 **상세 변경·파일 단위 메모**를 보관한다. 마일스톤 수준 요약과 로드맵 대응은 **`log.md`**에 기록한다.

---

## [2026-04-04] Phase 0 보강 — `fundamentals` 수집 파이프라인

### 배경
- `fundamentals` 테이블은 스키마만 있고 파이프라인이 없었음. Track A·D에서 `operating_margin`, `debt_to_equity`, `eps_actual`, `eps_consensus` 등 입력 필요.

### 신규: `data_pipeline/fundamentals_fetcher.py`
- `fetch_single(symbol)`: `yfinance.Ticker.info`에서 `trailingEps`→`eps_actual`, `epsForward`→`eps_consensus`(없으면 NULL), `operatingMargins` 소수 그대로, `debtToEquity`/100→`debt_to_equity`. `report_date`는 당일 스냅샷.
- 예외 시 WARNING 후 `None`; 부분 필드 누락은 NULL로 한 행이라도 적재.
- `load_symbols()`: `earnings_fetcher`와 동일( DB `tickers` → universe CSV → 분류.json ).
- `__main__`: `SELECT 1` 연결 확인 → 심볼 로드 → `fetch_single` 순회, **20종목마다** `upsert_fundamentals` + 마지막 flush. `delay=0.3`.
- **`fetch_all` 제거**: 전체 DF 수집과 20종목 배치 적재가 모순 → `insider_fetcher`와 같이 `__main__`에서만 순회+배치. (`eps_consensus_estimation`은 yfinance `info`에 없어 사용 안 함.)

### 수정: `data_pipeline/db_manager.py`
- ORM **`Fundamentals`**: PK `(symbol, report_date)`, Float 지표 4개, `created_at`.
- **`_prepare_fundamentals_df`**: 컬럼 정규화·`date`→`report_date` 별칭·지표 NaN→None.
- **`upsert_fundamentals`**: `(symbol, report_date)` ON CONFLICT → 4지표만 UPDATE.

### 문서
- `z_docs/task_plan.md`: Phase 0 Step 4 fundamentals 완료 `[x]` 블록 추가.
- `z_docs/plan.md`: Phase 0에 fundamentals fetcher 완료 체크.

---

## [2026-04-04] Phase 1 — `scoring` 패키지·Track A/D·엔진·`daily_scores` 적재

### 패키지·의존성
- **`pyproject.toml`**: `packages`에 `{include = "scoring"}` 추가.
- **`scoring/__init__.py`**: 빈 파일.

### `scoring/track_a.py` (Track A Original)
- 유틸: `_minmax_0_100`, `_norm_sector`, `_sector_median_fill`(gics_sector 기준 결측 보간), `_pct_high_better`, `_pct_low_better`.
- 서브스코어: `volume_score`(당일/20거래일 평균 비율→전체 tickers MinMax, 미데이터 비율 1.0), `financial_score`(섹터 내 OM·D/E 백분위 + EPS spread 백분위 평균), `revision_score`(당일 `earnings_revisions` 0.6×7d+0.4×30d → 0~100, 없으면 50), `valuation_score`(E/P 섹터 백분위).
- 가중치: Volume 0.20 / Financial 0.35 / Revision 0.30 / Valuation 0.15.
- **`compute_track_a(engine, score_date)`** → `DataFrame[symbol, score_original]`. 병합 시 미포함 서브스코어는 50 보간.

### `data_pipeline/db_manager.py` — `daily_scores` 적재 계층
- ORM **`DailyScores`**: PK `(symbol, date)`, `regime_stub`, `score_original`, `score_bottleneck`, `total_heinrich`, `created_at`(서버 기본값).
- **`_prepare_daily_scores_df`**: symbol strip, date, regime_stub(50자), Float 3개 coerce·None.
- **`upsert_daily_scores`**: `CHUNK_SIZE=500`, `(symbol,date)` 충돌 시 네 스코어 컬럼만 UPDATE(`created_at` 미터치).

### `scoring/track_d.py` (Track D Value Chain)
- fundamentals `DISTINCT ON (symbol)` + `report_date <= score_date`로 `operating_margin` + `tickers`(`vc_stage_num`, `vc_position`, `gics_sector`) 조인.
- **OM 결측 보간**: `_sector_median_fill(..., gics_sector)` — 백분위 그룹 축(vc_*)과 역할 분리.
- **Neutral 50**: `vc_stage_num`·`vc_position` **둘 다** NULL일 때만 명시 50. stage NULL·position만 있는 비정상 행은 백분위 불가 → 최종 `fillna(50)`.
- **`MIN_GROUP_SIZE = 3` 폴백(후속 반영)**  
  - `_n_comp`: `(stage, position)` 그룹 행 수(`transform('count')`).  
  - `_n_stage`: `vc_stage_num` 단독 그룹 행 수.  
  - `_group_key`: `n_comp≥3` → `C:...`, 아니고 `n_stage≥3` → `S:...`, 아니면 전역 `U`. stage-only 행도 동일 규칙.  
  - **`groupby('_group_key')` 한 번**에 `_pct_high_better` 적용 후 임시 컬럼 드롭 → `_minmax_0_100` → `score_bottleneck`.
- 상수 `W_BOTTLENECK = 1.0`(문서용).

### `scoring/engine.py`
- **`W_TRACK_A = 0.60`**, **`W_TRACK_D = 0.40`**, 별칭 `W_A`/`W_D`, **`REGIME_STUB = "관세/무역전쟁"`**.
- **`run_scoring(score_date)`**: `DBManager()` + `SELECT 1` → `compute_track_a` / `compute_track_d` → 전체 `tickers` LEFT JOIN → `score_original`·`score_bottleneck` 결측 50 → `total_heinrich` → `upsert_daily_scores`.
- **`__main__`**: `sys.argv[1]`을 `YYYY-MM-DD`로 파싱, 없으면 `date.today()`.

### 검증 메모(운영)
- 실행: `poetry run python -m scoring.engine 2026-04-04` (로그는 `tee scoring_log.txt` 등).
- SQL: `daily_scores`에서 해당 `date` 행 수 ≈ `tickers` 수(예: 503 전후).
- Track D 백분위가 단일 멤버 그룹에 몰려 `score_bottleneck≈100`이 되던 현상 → `MIN_GROUP_SIZE` 폴백으로緩和. 상세 분석은 필요 시 `analysis.txt` 등에 별도 기록.

---

## 기타 (본일 세션에서 다룬 설계 문서)

- Track D + Engine 플랜 문서(리뷰용): OM 보간 축 vs VC 백분위 그룹 축 분리, `earnings_revisions`/`daily_prices` 당일 행 부재 시 Track A 내부 보간 전제 등이 확정·문서화됨. 코드 주석·본 `task_log`와 일치.
