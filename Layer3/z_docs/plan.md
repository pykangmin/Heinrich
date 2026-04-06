# 하인리히 프로젝트 장기 로드맵

Layer 3 엔진 구축을 위한 전체 파이프라인 로드맵입니다.

**데이터 저장소**: Phase 0부터 **Supabase**(PostgreSQL 호환). 자체 호스트 Postgres는 필수 아님.

## Phase 0: Data First - 데이터 인프라 파이프라인 구축
- [x] DB 스키마 확정 및 Supabase 테이블 생성 (6개 테이블, DOUBLE PRECISION, created_at)
- [x] `db_manager.py` 재작성 — `Ticker` / `DailyPrice` ORM, `upsert_tickers` / `upsert_daily_prices`
- [x] `OHLCVFetcher` 구현 — `sp500_fetcher.py` 내 yfinance 배치 다운로드 + long-format reshape (`group_by='ticker'`, `auto_adjust=False`)
- [x] E2E 검증 — `tickers` 503행, `daily_prices` 20,459행 적재 확인 (2025-01-01 ~ 2026-04-03)
- [x] 실시간 / 일간 단위 Earnings Revision(이익 추정치) 업데이트 파이프라인 연동
- [x] SEC Form 4 내부자 거래 데이터 파싱 모듈 구현
- [x] 분류.json Value Chain 분류 → `tickers.vc_sector` / `vc_position` / `vc_stage_num` 적재 (`classification_loader.py`, Supabase 컬럼 반영 완료)

## Phase 1: Engine Core - Track A & D Scoring 엔진 개발

### Track A (Original) 점수 모델
- [x] **Step 1: Volume 서브스코어** — `daily_prices.volume` 기반. 20일 평균 대비 거래량 비율 정규화 (0~1).
- [x] **Step 2: 재무건전성 서브스코어** — `fundamentals`: `operating_margin` 섹터 내 상대 순위 + `debt_to_equity` 역수 정규화. EPS beat/miss (`eps_actual` vs `eps_consensus`).
- [x] **Step 3: Earnings Revision 서브스코어** — `earnings_revisions`: `revision_score_7d` + `revision_score_30d` 가중 합산 (7d×0.6 + 30d×0.4).
- [x] **Step 4: Valuation 서브스코어** — `fundamentals.eps_consensus` 기반 Earnings Yield. 섹터 내 상대 백분위.
- [x] **Step 5: Track A 결합** — 4개 서브스코어 가중합 → `score_original` (0~100 정규화). 초기 가중치: Volume 0.20 / 재무 0.35 / Revision 0.30 / Valuation 0.15.

### Track D (Value Chain) 점수 모델
- [x] **Step 6: 병목 프리미엄 산출** — `tickers.vc_stage_num` 기반 동 단계 내 `operating_margin` 분포 → 상위 분위 종목에 프리미엄. MIN_GROUP_SIZE=3 폴백 체계(복합→stage단독→전역).
- [x] **Step 7: Track D 결합** — 프리미엄 점수 → `score_bottleneck` (0~100).

### 통합
- [x] **Step 8: `daily_scores` 적재** — `total_heinrich = score_original × 0.6 + score_bottleneck × 0.4`. `regime_stub = "관세/무역전쟁"` 하드코딩. `scoring/engine.py` 구현 완료.

### 구현 파일
- `scoring/track_a.py` (신규) — 4개 서브스코어 함수 + `compute_track_a(date)`
- `scoring/track_d.py` (신규) — `compute_track_d(date)`
- `scoring/engine.py` (신규) — `run_scoring(date)` → `daily_scores` upsert
- `data_pipeline/db_manager.py` — `upsert_daily_scores()` 추가

## Phase 2: System Integration - Risk Filter 및 매매/방어 로직 통합
- [ ] **Step 7 연동**: 레짐 Stub 인터페이스 구축 ("관세/무역전쟁" 하드코딩) 및 스코어 테스트
- [ ] **Step 8 구현**: SNR 필터, Beta 및 상관관계 제어, Mixed Regime 시 30% 현금 방어 로직 하드코딩
- [ ] **IR Gap 트레이딩**: Post-Earnings Drift 및 테크니컬+실시간 Revision 기반 리밸런싱 연동
