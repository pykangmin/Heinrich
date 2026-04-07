# Layer 1/2 → Layer 3 인터페이스 계약서

Layer 1(레짐 판별기)·Layer 2(레짐-지표 매핑)는 별도 팀이 구현.
본 문서는 **Layer 3가 Layer 1/2로부터 무엇을 받아야 하는지**를 정의한다.

---

## 1. Layer 1 → Layer 3: 레짐 문자열 전달

### Layer 3가 요구하는 값

| 항목 | 타입 | 설명 |
|------|------|------|
| `regime` | `str` | 현재 확정 레짐 또는 `"mixed"` |

### 허용 레짐 값 (`scoring/regime.py` 기준)

| 값 | 설명 |
|----|------|
| `"관세/무역전쟁"` | 무역 분쟁·관세 충격 국면 |
| `"금리 상승"` | 인플레이션·금리 상승 국면 |
| `"리세션"` | 경기침체 우려 국면 |
| `"정상"` | 특이 레짐 없는 정상 시장 |
| `"mixed"` | 두 채널 불일치 전환기 — 현금 30% 방어 트리거 |

> `REGIME_CONFIG`에 등록되지 않은 값이 들어오면 `"정상"` config로 자동 폴백됨 (WARNING 로그 출력).

### 현재 Stub (Layer 1 완성 전)
```python
# scoring/regime.py
def get_regime() -> str:
    return os.getenv("REGIME", "관세/무역전쟁").strip()
```
`.env`에서 `REGIME=관세/무역전쟁`으로 수동 설정 중.

### Swap 포인트 (Layer 1 완성 시 교체 위치)
- **파일**: `scoring/regime.py`
- **함수**: `get_regime()`
- 이 함수 내부만 교체하면 `engine.py` 등 상위 코드는 수정 불필요.

### 전달 방식 (미정 — 협의 필요)
- 후보 A: 환경변수 주입 (`REGIME=관세/무역전쟁`)
- 후보 B: DB 테이블 (`current_regime` 등) → `get_regime()`이 DB 조회
- 후보 C: REST API 호출

---

## 2. Layer 2 → Layer 3: 지표 Surprise 값

현재 Layer 3는 Layer 2의 지표 Surprise 값을 **직접 사용하지 않음**.

향후 Track A 서브스코어 가중치를 레짐별 Surprise 강도에 따라 동적 조정하는 기능 구현 시 연동 검토 예정. 현재는 미정.

---

## 3. Layer 3 내부 레짐 처리 현황 (참고)

| 항목 | 위치 | 내용 |
|------|------|------|
| 레짐별 가중치 | `scoring/regime.py` `REGIME_CONFIG` | Track A/D 비율 |
| Mixed 현금 방어 | `scoring/portfolio.py` `build_portfolio()` | `regime=="mixed"` 시 `cash_pct=0.30` |
| 가중치 폴백 | `get_regime_config()` | 미등록 레짐 → `"정상"` config |
| Mixed 가중치 | 미등록 | 폴백으로 `"관세/무역전쟁"` 가중치 사용 중 |
