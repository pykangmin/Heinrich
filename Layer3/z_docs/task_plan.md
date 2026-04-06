# 현재 작업 목표 (Task Plan)

가장 시급하게 진행해야 할 최우선 개발 태스크입니다.

---

## Phase 1, Step 1~8: Track A & D 점수 엔진 — 완료 [x]

2026-04-04 기준 `daily_scores` 503개 종목 적재 확인. 상세 이력 → `log.md` [2026-04-04].

---

## 다음 과제: Phase 2 — Risk Filter 및 매매/방어 로직 통합

**목표**: 레짐 기반 필터·포트폴리오 방어 로직을 스코어 엔진에 연동.

### 항목 (plan.md Phase 2)
1. **레짐 Stub 인터페이스** — `"관세/무역전쟁"` 하드코딩 상태에서 레짐 변경 가능한 인터페이스로 추상화 및 스코어 테스트
2. **SNR 필터 & Beta/상관관계 제어** — 신호 대 잡음비 필터, 베타 상한, 종목 간 상관관계 제어
3. **Mixed Regime 방어 로직** — Mixed Regime 시 30% 현금 방어 하드코딩
4. **IR Gap 트레이딩** — Post-Earnings Drift + 테크니컬·실시간 Revision 기반 리밸런싱 연동

### 구현 예정 파일
- `scoring/regime.py` (신규) — 레짐 정의 및 인터페이스
- `scoring/risk_filter.py` (신규) — SNR·Beta·상관관계 필터
- `scoring/portfolio.py` (신규) — 방어 로직·리밸런싱

---

## (참고) Phase 0 — SEC Form 4 내부자 거래 파이프라인 — 완료 [x]

`data_pipeline/insider_fetcher.py`, `InsiderTrade` ORM, `upsert_insider_trades` 구현 완료.
상세 → `log.md` [2026-04-03].
