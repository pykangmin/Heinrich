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
- [ ] Track A (Original) 점수 모델링 (거리량, 재무, 이익추정치, Valuation)
- [ ] Track D (Value Chain) 점수 모델링 (고마진율 기반 공급 병목 프리미엄). **입력:** `tickers.vc_stage_num`(상·하류 단계), `vc_position`(세부 병목 축), `vc_sector`; 동 단계·인접 단계 내 마진 분포 대비 상대 프리미엄·집중도 등으로 스코어 설계.
- [ ] 각 Track의 결합 점수 산출 로직 구현

## Phase 2: System Integration - Risk Filter 및 매매/방어 로직 통합
- [ ] **Step 7 연동**: 레짐 Stub 인터페이스 구축 ("관세/무역전쟁" 하드코딩) 및 스코어 테스트
- [ ] **Step 8 구현**: SNR 필터, Beta 및 상관관계 제어, Mixed Regime 시 30% 현금 방어 로직 하드코딩
- [ ] **IR Gap 트레이딩**: Post-Earnings Drift 및 테크니컬+실시간 Revision 기반 리밸런싱 연동
