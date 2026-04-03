# PostgreSQL 스키마 (Layer 3)

`pgcrypto` 확장과 6개 테이블로 구성됩니다. `insider_trades.id`는 `gen_random_uuid()`를 사용합니다.

## 확장

| 확장 | 용도 |
|------|------|
| `pgcrypto` | `insider_trades` UUID 기본값 (`gen_random_uuid`) |

```sql
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
```

---

## 1. `tickers`

종목 메타데이터 마스터.

| 컬럼 | 타입 | 제약 |
|------|------|------|
| `symbol` | `VARCHAR(10)` | PK |
| `company_name` | `VARCHAR(100)` | |
| `gics_sector` | `VARCHAR(50)` | |
| `gics_industry` | `VARCHAR(50)` | |
| `is_active` | `BOOLEAN` | 기본 `TRUE` |
| `vc_sector` | `VARCHAR(50)` | nullable, `분류.json`·Track D 보조 분류 |
| `vc_position` | `VARCHAR(100)` | nullable, Value Chain 세부 라벨 |
| `vc_stage_num` | `SMALLINT` | nullable, 밸류체인 단계 코드 (`classification_loader`) |
| `updated_at` | `TIMESTAMPTZ` | 기본 `NOW()` |
| `created_at` | `TIMESTAMPTZ` | 기본 `NOW()` |

**마이그레이션 예시:** `ALTER TABLE tickers ADD COLUMN vc_sector VARCHAR(50), ADD COLUMN vc_position VARCHAR(100), ADD COLUMN vc_stage_num SMALLINT;`

---

## 2. `daily_prices`

일별 OHLCV (조정종가 포함).

| 컬럼 | 타입 | 제약 |
|------|------|------|
| `symbol` | `VARCHAR(10)` | PK, FK → `tickers(symbol)` ON DELETE CASCADE |
| `date` | `DATE` | PK |
| `open` | `DOUBLE PRECISION` | |
| `high` | `DOUBLE PRECISION` | |
| `low` | `DOUBLE PRECISION` | |
| `close` | `DOUBLE PRECISION` | |
| `adj_close` | `DOUBLE PRECISION` | |
| `volume` | `BIGINT` | |
| `created_at` | `TIMESTAMPTZ` | 기본 `NOW()` |

**인덱스:** `idx_daily_prices_date` on `(date)`.

---

## 3. `fundamentals`

실적·재무 지표 (보고일 기준).

| 컬럼 | 타입 | 제약 |
|------|------|------|
| `symbol` | `VARCHAR(10)` | PK, FK → `tickers(symbol)` ON DELETE CASCADE |
| `report_date` | `DATE` | PK |
| `eps_actual` | `DOUBLE PRECISION` | |
| `eps_consensus` | `DOUBLE PRECISION` | |
| `operating_margin` | `DOUBLE PRECISION` | |
| `debt_to_equity` | `DOUBLE PRECISION` | |
| `created_at` | `TIMESTAMPTZ` | 기본 `NOW()` |

**인덱스:** `idx_fundamentals_report_date` on `(report_date)`.

---

## 4. `earnings_revisions`

EPS 추정 및 리비전 스코어(당일·단기 창 scoring).

| 컬럼 | 타입 | 제약 |
|------|------|------|
| `symbol` | `VARCHAR(10)` | PK, FK → `tickers(symbol)` ON DELETE CASCADE |
| `date` | `DATE` | PK |
| `eps_est_current` | `DOUBLE PRECISION` | |
| `revision_score` | `DOUBLE PRECISION` | |
| `revision_score_7d` | `DOUBLE PRECISION` | |
| `revision_score_30d` | `DOUBLE PRECISION` | |
| `created_at` | `TIMESTAMPTZ` | 기본 `NOW()` |

**인덱스:** `idx_earnings_revisions_date` on `(date)`.

**기존 테이블 확장(마이그레이션 예시):**

```sql
ALTER TABLE earnings_revisions
  ADD COLUMN revision_score_7d DOUBLE PRECISION,
  ADD COLUMN revision_score_30d DOUBLE PRECISION;
```

---

## 5. `insider_trades`

내부자 거래 제출.

| 컬럼 | 타입 | 제약 |
|------|------|------|
| `id` | `UUID` | PK, 기본 `gen_random_uuid()` |
| `symbol` | `VARCHAR(10)` | FK → `tickers(symbol)` ON DELETE CASCADE |
| `filing_date` | `DATE` | |
| `transaction_type` | `VARCHAR(10)` | |
| `value` | `DOUBLE PRECISION` | |
| `accession_number` | `VARCHAR(25)` | UNIQUE(행 단일화·재적재 시 `ON CONFLICT DO NOTHING`) |
| `created_at` | `TIMESTAMPTZ` | 기본 `NOW()` |

**인덱스:** `idx_insider_trades_symbol_date` on `(symbol, filing_date)`.

**마이그레이션 예시:**

```sql
ALTER TABLE insider_trades
  ADD COLUMN accession_number VARCHAR(25) UNIQUE;
```

---

## 6. `daily_scores`

일별 레짐·스코어 (하인리치 등).

| 컬럼 | 타입 | 제약 |
|------|------|------|
| `symbol` | `VARCHAR(10)` | PK, FK → `tickers(symbol)` ON DELETE CASCADE |
| `date` | `DATE` | PK |
| `regime_stub` | `VARCHAR(50)` | |
| `score_original` | `DOUBLE PRECISION` | |
| `score_bottleneck` | `DOUBLE PRECISION` | |
| `total_heinrich` | `DOUBLE PRECISION` | |
| `created_at` | `TIMESTAMPTZ` | 기본 `NOW()` |

**인덱스:** `idx_daily_scores_date` on `(date)`.

---

## 관계 요약

- 모든 시계열·팩트 테이블은 `tickers(symbol)`을 참조하며, 티커 삭제 시 **CASCADE**로 관련 행이 삭제됩니다.
- 복합 기본 키: `daily_prices`, `fundamentals`, `earnings_revisions`, `daily_scores`는 `(symbol, date 또는 report_date)`.
- `insider_trades`만 단일 UUID 기본 키입니다.
