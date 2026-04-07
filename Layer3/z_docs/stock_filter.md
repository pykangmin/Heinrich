# 📊 기업 필터링 및 선정 방식 (Stock Filter & Selection)

## 1. 📌 필터링 계층 (Filtering Layers)

### 1.1 **Universe 정의 (Initial Universe)**

**기준**: `분류.json` 기반 SP500 구성

- **포함**: 503개 선정 종목 (GICS 섹터 + Value Chain 분류)
- **저장소**: `tickers` 테이블 (PK: symbol)
- **컬럼**:
  - `symbol`: 티커 (NVDA, AAPL, GOOGL 등)
  - `company_name`: 기업명
  - `gics_sector`: GICS 섹터 (IT, Financials, Health Care 등)
  - `vc_sector`: Value Chain 섹터 (웨이퍼 제조, 완성품·서비스, 유통·소매 등)
  - `vc_position`: Value Chain 포지션 (예: "시스템 통합 & 최종 조립")
  - `vc_stage_num`: VC 단계 번호 (1~3단계 또는 NULL)

**로딩 방식**:
- `data_pipeline/classification_loader.py` 
- `분류.json` → `tickers` upsert (vc_* 컬럼 포함)

---

## 2. 🎯 선정 점수 체계 (Scoring System)

### 2.1 **Track A: 원본 점수 (`score_original`)**

**목적**: 기업의 내재 가치 및 모멘텀 평가

#### **Step 1: Volume Score (거래량)**
- **계산**: 당일 거래량 / 최근 20거래일 평균 → MinMax 정규화 (0~100)
- **해석**: 
  - 고점수: 최근 거래량 급증 (관심 증가)
  - 저점수: 거래량 부진 (유동성 우려)
- **미포함 기업**: Volume 비율 1.0 (중립)
- **가중치**: **20%**

#### **Step 2: Financial Score (재무건전성)**
- **입력 데이터**:
  - `operating_margin`: 영업마진
  - `debt_to_equity`: 부채비율 (역수)
  - `eps_actual` vs `eps_consensus`: 실적 달성도
- **계산**:
  1. 각 지표별 **섹터 내 백분위** 계산
  2. 3개 지표 평균
  3. MinMax 정규화 (0~100)
- **미포함 기업**: 같은 섹터 중앙값 → 전체 중앙값 보간
- **가중치**: **35%** (가장 높음)

#### **Step 3: Revision Score (수익 추정 개정)**
- **입력 데이터**:
  - `revision_score_7d`: 7일 개정 점수 (-1~1)
  - `revision_score_30d`: 30일 개정 점수 (-1~1)
- **계산**:
  - 가중합: 0.6 × 7d + 0.4 × 30d
  - 정규화: (값 + 1) / 2 × 100 → 0~100
- **해석**:
  - 고점수: 애널리스트 상향 조정 중 (긍정 모멘텀)
  - 저점수: 애널리스트 하향 조정 중 (부정 모멘텀)
- **미포함 기업**: 50 (중립, 데이터 부재)
- **가중치**: **30%** (모멘텀 중심)

#### **Step 4: Valuation Score (밸류에이션)**
- **입력 데이터**:
  - Earnings Yield (E/P Ratio) = EPS Consensus / 종가
- **계산**:
  1. 섹터 내 E/P 백분위 (높을수록 고점수)
  2. MinMax 정규화 (0~100)
- **해석**:
  - 고점수: 저평가 상태 (수익력 대비 저평가)
  - 저점수: 고평가 상태
- **미포함 기업**: 섹터 중앙값 보간 → 50
- **가중치**: **15%**

#### **최종 Track A 점수**:
```
score_original = 0.20 × volume + 0.35 × financial + 0.30 × revision + 0.15 × valuation
범위: 0~100
```

---

### 2.2 **Track D: 병목 프리미엄 점수 (`score_bottleneck`)**

**목적**: Value Chain 내 병목(제약) 단계 기업에 프리미엄 부여

#### **분석 단위**: VC Stage × VC Position

- **VC Stage**: 1단계(원재료)→ 2단계(부품/소재)→ 3단계(완성품)
- **VC Position**: 웨이퍼 제조, 완성품·서비스, 유통·소매 등 세부 포지션

#### **점수 산출 알고리즘**:

1. **기업 분류**:
   - `(vc_stage_num, vc_position)` 조합별로 그룹 형성
   
2. **Operating Margin 백분위** (그룹 내):
   - 동일 그룹 내 기업의 OM 순위 (높을수록 고점수)
   - 섹터 결측값 보간: 동일 섹터 중앙값 → 전체 중앙값

3. **폴백 메커니즘** (MIN_GROUP_SIZE = 3):
   ```
   if 그룹 크기 ≥ 3:
       → 복합 그룹 (vc_stage + vc_position) 내 백분위
   elif stage 단독 크기 ≥ 3:
       → Stage 그룹 (vc_stage만) 내 백분위
   else:
       → 전체 유니버스 (U) 내 백분위
   ```
   
4. **중립값** (score_bottleneck = 50):
   - **조건**: vc_stage_num AND vc_position 모두 NULL
   - **목적**: Value Chain 미분류 기업 중립 처리

#### **최종 Track D 점수**:
```
score_bottleneck = MinMax(그룹별 operating_margin 백분위)
범위: 0~100
```

---

### 2.3 **통합 점수: Total Heinrich Score**

```
total_heinrich = 0.60 × score_original + 0.40 × score_bottleneck
범위: 0~100
```

**가중치 해석**:
- **60% (Track A)**: 기업 내재 가치·모멘텀
- **40% (Track D)**: Value Chain 상대적 강도

---

## 3. 🔄 선정 프로세스 (Selection Workflow)

### **일일 갱신 파이프라인** (`scoring/engine.py`)

```
daily_scores 적재 (매일 밤)
    ↓
1. compute_track_a(score_date)
   - volume_score, financial_score, revision_score, valuation_score 계산
   ↓
2. compute_track_d(score_date)
   - VC stage×position 그룹별 OM 백분위 계산
   ↓
3. 통합 계산
   - 결측값 50으로 보간
   - total_heinrich = 0.60 × score_original + 0.40 × score_bottleneck
   ↓
4. daily_scores 테이블 upsert
   - PK (symbol, date) → 기존 행 UPDATE 또는 신규 INSERT
   - 컬럼: symbol, date, regime_stub, score_original, score_bottleneck, total_heinrich
```

---

## 4. 🛠️ 구현 파일 (Implementation Files)

| 파일 | 역할 |
|------|------|
| `data_pipeline/classification_loader.py` | 분류.json → tickers 적재 (VC 분류) |
| `data_pipeline/fundamentals_fetcher.py` | yfinance → fundamentals (OM, D/E, EPS) |
| `data_pipeline/earnings_fetcher.py` | yfinance → earnings_revisions (7d, 30d 점수) |
| `data_pipeline/db_manager.py` | ORM & upsert 메서드 |
| `scoring/track_a.py` | 4개 서브스코어 & compute_track_a() |
| `scoring/track_d.py` | VC 병목 프리미엄 & compute_track_d() |
| `scoring/engine.py` | 통합 & daily_scores 적재 |

---

## 5. 📋 현재 상태 (Current State)

- **Universe**: 503개 종목 (분류.json 기준)
- **Daily Scores**: 매일 갱신 (Phase 1 완료)
- **Regime Stub**: 하드코딩 "관세/무역전쟁" (Phase 2 계획: 동적화)

---

## 6. 🚀 향후 개선 (Future Enhancements - Phase 2)

1. **Regime 인터페이스 추상화**
   - 현재: "관세/무역전쟁" 고정
   - 목표: 여러 레짐 시나리오 지원

2. **Risk Filter 통합**
   - SNR (Signal-to-Noise Ratio) 필터
   - Beta 상한선
   - 종목 간 상관관계 제어

3. **Portfolio Defense Logic**
   - Mixed Regime 시 30% 현금 방어
   - Dynamic Rebalancing

4. **IR Gap Trading**
   - Post-Earnings Drift 트레이딩
   - 실시간 Revision 기반 리밸런싱
