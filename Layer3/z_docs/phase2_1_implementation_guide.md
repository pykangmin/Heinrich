# Phase 2-1 상세 구현 가이드

## 1. Volume Score 정규화 변경

### 현재 코드 (scoring/track_a.py, ~96줄)
```python
def volume_score(engine: Engine, score_date: date) -> pd.DataFrame:
    # ... ratio_by_sym 계산 ...
    ratios = [ratio_by_sym.get(s, 1.0) for s in universe]
    ser = pd.Series(ratios, index=universe, dtype=float)
    out = _minmax_0_100(ser)  # ← 아웃라이어 취약!
    out.name = 'volume_score'
    return out.reset_index().rename(columns={'index': 'symbol'})
```

### 수정 로직
```python
def volume_score(engine: Engine, score_date: date) -> pd.DataFrame:
    # ... 기존 ratio_by_sym 계산 (그대로) ...
    ratios = [ratio_by_sym.get(s, 1.0) for s in universe]
    ser = pd.Series(ratios, index=universe, dtype=float)
    
    # 변경: MinMax → Percentile (Rank 기반)
    # 이 방식: ratio 값과 무관하게 순위(백분위)만 고려 → 아웃라이어 무시
    out = ser.reset_index()
    out.columns = ['symbol', 'ratio']
    out['volume_score'] = ser.rank(pct=True, method='average') * 100.0
    
    return out[['symbol', 'volume_score']]
```

**핵심 변경**:
- `_minmax_0_100(ser)` 제거
- `ser.rank(pct=True) * 100` 사용 (0~100, 순위 기반)

**효과**:
| 시나리오 | MinMax | Percentile |
|---------|--------|-----------|
| 502개 종목: 거래량 0~100 | 50점대 | 49.5점 |
| 1개 종목 거래량 5000 | 99점 / 나머지 0~1점 | 99점 / 나머지 50점대 |

---

## 2. Track D 정규화 변경

### 현재 코드 (scoring/track_d.py, ~125줄)
```python
mm = _minmax_0_100(scores)  # ← 아웃라이어 취약!
return pd.DataFrame({'symbol': df['symbol'].astype(str), 'score_bottleneck': mm.values})
```

### 수정 로직
```python
# scores는 이미 백분위로 계산된 값들의 Series
# 대신 마지막 MinMax를 제거하고 직접 사용

# 변경 전:
# mm = _minmax_0_100(scores)
# return pd.DataFrame({'symbol': df['symbol'].astype(str), 'score_bottleneck': mm.values})

# 변경 후:
# scores는 이미 _pct_high_better()의 출력(0~100)이므로
# 그대로 사용 (최종 MinMax 제거)
return pd.DataFrame({'symbol': df['symbol'].astype(str), 'score_bottleneck': scores.values})
```

**핵심 변경**:
- `_minmax_0_100(scores)` 제거
- `scores` 직접 사용 (이미 0~100 범위)

**사유**:
```python
# track_d.py 115줄 근처:
scores.loc[pct_mask] = (
    df.loc[pct_mask]
    .groupby('_group_key', sort=False)['operating_margin']
    .transform(_pct_high_better)  # ← 이미 0~100 범위
)
```

---

## 3. Data Quality Filter 추가

### scoring/engine.py에 신규 함수 추가 (run_scoring 앞에)

```python
import os
from collections import Counter

DATA_QUALITY_THRESHOLD = int(os.getenv('DATA_QUALITY_THRESHOLD', '3'))

def apply_data_quality_filter(
    tick: pd.DataFrame,
    v: pd.DataFrame,
    f: pd.DataFrame,
    r: pd.DataFrame,
    val: pd.DataFrame,
    threshold: int = DATA_QUALITY_THRESHOLD,
) -> tuple[set[str], list[dict]]:
    """
    Track A 4개 팩터에서 결측치 카운트.
    threshold개 이상 결측 → 필터링.
    
    Returns:
        filtered_symbols (set): 제외할 종목 set
        filter_log (list[dict]): 필터링 이유 기록
    """
    # 각 팩터별 결측 여부 (symbol 기준)
    v_missing = set(tick['symbol']) - set(v['symbol'])
    f_missing = set(tick['symbol']) - set(f['symbol'])
    r_missing = set(tick['symbol']) - set(r['symbol'])
    val_missing = set(tick['symbol']) - set(val['symbol'])
    
    filter_log = []
    filtered_symbols = set()
    
    for sym in tick['symbol']:
        missing_count = sum([
            sym in v_missing,
            sym in f_missing,
            sym in r_missing,
            sym in val_missing,
        ])
        
        if missing_count >= threshold:
            filtered_symbols.add(sym)
            filter_log.append({
                'symbol': sym,
                'missing_count': missing_count,
                'missing_factors': [
                    'volume' if sym in v_missing else '',
                    'financial' if sym in f_missing else '',
                    'revision' if sym in r_missing else '',
                    'valuation' if sym in val_missing else '',
                ],
            })
    
    return filtered_symbols, filter_log
```

### run_scoring() 함수 수정 (로직)

```python
def run_scoring(score_date: date) -> pd.DataFrame:
    from scoring.track_a import compute_track_a
    from scoring.track_d import compute_track_d

    db = DBManager()
    with db.engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    a_df = compute_track_a(db.engine, score_date)
    d_df = compute_track_d(db.engine, score_date)

    uni = pd.read_sql(text("SELECT symbol FROM tickers ORDER BY symbol"), db.engine)
    
    # ===== 신규: Data Quality Filter 적용 =====
    v = volume_score(db.engine, score_date)  # compute_track_a 내부에서 분리 필요하거나
    f = financial_score(db.engine, score_date)  # 개별 호출
    r = revision_score(db.engine, score_date)
    val = valuation_score(db.engine, score_date)
    
    filtered_symbols, filter_log = apply_data_quality_filter(uni, v, f, r, val)
    
    # 로그 기록
    import logging
    logger = logging.getLogger('scoring.engine')
    for entry in filter_log:
        missing_str = ','.join([x for x in entry['missing_factors'] if x])
        logger.warning(
            f"[{score_date}] Filtered: {entry['symbol']} - "
            f"{entry['missing_count']}/4 factors missing ({missing_str})"
        )
    
    logger.info(
        f"[{score_date}] Total filtered: {len(filtered_symbols)}/{len(uni)} "
        f"(threshold={DATA_QUALITY_THRESHOLD})"
    )
    # ===== Data Quality Filter 종료 =====

    if uni.empty:
        return pd.DataFrame(columns=[...])

    # 기존 merge 로직 (변경 없음)
    out = uni.merge(a_df, on='symbol', how='left').merge(d_df, on='symbol', how='left')
    
    # ===== 신규: 필터링된 종목 제거 =====
    out = out[~out['symbol'].isin(filtered_symbols)]
    # ===== 필터링 종료 =====
    
    out['score_original'] = out['score_original'].fillna(50.0)
    out['score_bottleneck'] = out['score_bottleneck'].fillna(50.0)
    out['total_heinrich'] = (
        out['score_original'] * W_TRACK_A + out['score_bottleneck'] * W_TRACK_D
    )
    out['date'] = score_date
    out['regime_stub'] = REGIME_STUB

    upsert_cols = [...(기존 그대로)]
    db.upsert_daily_scores(out[upsert_cols])
    return out[upsert_cols]
```

---

## 구현 순서

### Step 1: track_a.py 수정 (5분)
- 위 **섹션 1** 구현
- volume_score() 내 정규화 방식만 변경
- **테스트**: `poetry run python -m scoring.track_a` 타입 확인

### Step 2: track_d.py 수정 (3분)
- 위 **섹션 2** 구현
- MinMax 제거
- **테스트**: `poetry run python -m scoring.track_d` 범위 확인

### Step 3: engine.py 수정 (10분)
- 위 **섹션 3** 함수 추가 + run_scoring 수정
- logging 설정 (이미 있는지 확인)
- **테스트**: `poetry run python -m scoring.engine 2026-04-05` 실행

### Step 4: E2E 검증 (10분)
```bash
# daily_scores 레코드 수 확인 (503 → 500 정도로 감소 예상)
# SELECT COUNT(*) FROM daily_scores WHERE date = '2026-04-05';

# 점수 분포 확인 (더 균등해야 함)
# SELECT
#   PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_heinrich) AS p25,
#   PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_heinrich) AS median,
#   PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_heinrich) AS p75
# FROM daily_scores WHERE date = '2026-04-05';
```

---

## 검증 체크리스트

- [ ] volume_score 점수 범위 0~100 확인
- [ ] track_d score_bottleneck 범위 0~100 확인
- [ ] total_heinrich 범위 0~100 확인
- [ ] daily_scores 행 수 < 503 (필터링됨)
- [ ] 필터링 로그에 제외 종목 명시 확인
- [ ] `z_docs/task_log.md`에 기록: MinMax→Percentile 전환 완료, 필터링 종목 수, 실행 커맨드·검증 수치 (인사이트 수준이 아니면 `analysis.txt`에 쓰지 않음)

---

## 향후 조정 (유연성)

### threshold 조정
```python
# .env 파일 추가
DATA_QUALITY_THRESHOLD=3  # 나중에 2로 변경 가능
```

### threshold=2로 변경 시 영향도
- 필터링 종목 수 증가 (예상: 10~20개 추가)
- 데이터 품질 향상 (약한 신호 제거)
- 포트폴리오 축소 (선택성 강화)
