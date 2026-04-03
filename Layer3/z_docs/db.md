# DB 및 데이터 인프라 파이프라인

하인리히 프로젝트 Layer 3의 모든 단계는 데이터 무결성과 최신성에 높은 의존도를 가집니다. Phase 0에서는 Data First 기반의 핵심 선행 지표 수집 데이터베이스와 파이프라인 구축에 집중합니다.

## 파이프라인 구축 핵심 목표
- 애널리스트 추정치 변화(**Earnings Revision**)를 선행 반영하기 위해 일간/실시간 수집 파이프라인 최우선 구축

## 데이터 소스 계획
1. **Price / Volume (OHLCV)**
   - 사용할 API: Yahoo Finance API 또는 Polygon.io 등
2. **Fundamental & Earnings Revision**
   - 사용할 API: Financial Modeling Prep (FMP) API 등
3. **내부자 거래 (Insider Trading)**
   - 수집 경로: SEC EDGAR Form 4 공시 파싱 및 sec4.net 활용
   - 목적: 분기 IR 공백기 동안의 핵심 모멘텀 시그널 및 매수 신호 활용

## 데이터 저장소 설계 (Supabase)
- **Primary**: [Supabase Database](https://supabase.com/docs/guides/database/overview) — 관리형 PostgreSQL, 백업·대시보드·SQL Editor 제공.
- **접속**: 애플리케이션은 `DATABASE_URL`(Connection string) 우선. RLS·API는 향후 필요 시 Supabase Client로 확장 가능하나, Phase 0 파이프라인은 서버 측에서 SQLAlchemy + `psycopg2`로 적재하는 전제.
- **스키마 관리**: 테이블 생성·변경은 Supabase **SQL Editor** 또는 추후 마이그레이션 도구로 버전 관리. **테이블·DDL 원문은 `schema.md`를 기준(Single Source of Truth)**으로 두고, 본 파일(`db.md`)은 파이프라인·저장소 전략과 동기화합니다.
- **시계열(OHLCV)**: 초기에는 일간 바 단위로 일반 테이블 + `(ticker, date)` 인덱스로 충분한 경우가 많다. 데이터량이 커지면 Supabase/Postgres에서 [TimescaleDB](https://supabase.com/docs/guides/database/extensions/timescaledb) 확장 사용 여부를 프로젝트 설정에서 검토한다.
