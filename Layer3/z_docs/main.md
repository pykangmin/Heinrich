## 🔍 Directory Tree

문서 기준 프로젝트 루트(`layer3/`) 구조입니다. (가상환경 `.venv/` 제외)

```
layer3/
├── README.md
├── pyproject.toml
├── .env                          # 로컬 시크릿 (커밋 금지)
├── .env_example                  # 환경변수 템플릿
├── 분류.json                      # Value Chain 분류 정의
├── analysis.txt                  # 실행 결과 추론·분석 누적 기록
├── data_pipeline/
│   ├── __init__.py
│   ├── sp500_fetcher.py          # SP500Fetcher + OHLCVFetcher
│   ├── db_manager.py             # DBManager (Ticker, DailyPrice, Fundamentals, EarningsRevision, InsiderTrade ORM)
│   ├── fundamentals_fetcher.py   # yfinance 기반 재무지표 스냅샷 적재
│   ├── earnings_fetcher.py       # yfinance 기반 EPS 추정·리비전 적재
│   ├── insider_fetcher.py        # SEC EDGAR Form 4 내부자 거래 적재
│   └── classification_loader.py  # 분류.json → tickers.vc_* 적재
├── scoring/
│   ├── __init__.py
│   ├── track_a.py                # Track A 4개 서브스코어 + compute_track_a()
│   ├── track_d.py                # Track D 병목 프리미엄 (미구현)
│   └── engine.py                 # run_scoring() → daily_scores upsert (미구현)
└── z_docs/
    ├── main.md                   # 본 파일: 인덱스 + 워크플로우
    ├── rules.md                  # AI 협업 규칙
    ├── architecture.md
    ├── db.md
    ├── schema.md                 # 테이블·DDL 단일 기준
    ├── source.md                 # 데이터 소스 및 라이브러리 목록
    ├── setup.md
    ├── plan.md
    ├── task_plan.md
    ├── task_log.md
    └── log.md
```

---

### [시스템 명령: 프로젝트 초기화 및 무결성 관리 워크플로우]

너는 지금부터 수석 개발 파트너로서 아래의 '문서 기반 워크플로우'와 '파일 시스템 무결성 검사' 규칙을 엄격히 준수해야 한다.

#### 1. 📂 문서 구조 정의 (Source of Truth)
모든 협업 문서는 하단 배치를 위해 `z_docs/` 폴더 내에 위치하며, `z_docs/main.md`가 모든 파일 구조의 최상위 기준점이 된다.
- `z_docs/main.md`: 프로젝트 인덱스 및 **전체 파일 트리(Directory Tree)** 관리.
- `z_docs/rules.md`: AI 행동 지침 및 `task_log` -> `log` 요약 워크플로우.
- `z_docs/architecture.md`: 시스템 설계 및 로직 흐름.
- `z_docs/db.md`: DB 스키마 정보.
- `z_docs/schema.md`: 테이블 정의·인덱스·DDL (논리 스키마 단일 기준).
- `z_docs/source.md`: 데이터 소스 목록 (API·라이브러리·설치 방법).
- `z_docs/setup.md`: 환경 설정 및 의존성 가이드.
- `z_docs/plan.md`: 장기 로드맵.
- `z_docs/task_plan.md`: 현재 작업 목표.
- `z_docs/task_log.md`: [임시] 실시간 작업 로그 및 에러 기록.
- `z_docs/log.md`: [영구] 완료된 작업의 정제된 히스토리.
- `analysis.txt` (프로젝트 루트): [영구] 실행 결과에 대한 추론·분석 누적 기록. 스코어 이상, 데이터 품질 문제, 설계 결정 근거 등을 날짜 헤더와 함께 append.

#### 2. 🚀 분기점: 프로젝트 초기화 (New Project) vs 세션 재개 (Resume)
내가 너에게 첫 대화를 건넬 때, 현재 작업 환경에 `z_docs/main.md`가 존재하는지 여부에 따라 아래와 같이 다르게 행동하라.

**A. [신규 프로젝트인 경우 (파일이 없을 때)]**
1. 즉시 `z_docs/` 폴더를 생성하는 터미널 명령어를 제시하거나 파일 생성 모드를 가동하라.
2. 위 1번에 정의된 9개의 `.md` 파일을 생성하고, 각 파일 내부에는 가장 기초적인 뼈대(Markdown H1 제목과 1~2줄의 역할 설명)를 작성하라.
3. `z_docs/main.md` 최상단에는 **## 🔍 Directory Tree** 섹션을 만들고 전체 구조를 명시하라.
4. "프로젝트 초기화 완료: 모든 문서의 뼈대를 생성했습니다. `plan.md`에 로드맵을 작성하는 것부터 시작할까요?"라고 응답하라.

**B. [기존 프로젝트인 경우 (main.md가 제공될 때 - Auto Context Sync)]**
1. **규칙 숙지**: `z_docs/rules.md`를 읽고 코딩 컨벤션을 활성화하라.
2. **무결성 검사(Integrity Check)**: `z_docs/main.md`에 명시된 Directory Tree와 현재 제공된 파일 시스템을 대조하라. (누락 시 즉시 보고)
3. **목표 파악**: `z_docs/plan.md`를 읽어 전체 위치를 파악하고, `z_docs/task_plan.md`를 읽어 당장 해결할 스텝을 파악하라.
4. **진행 상황 동기화**: `z_docs/task_log.md`를 읽고 이전 세션에서 멈춘 지점(이슈, 에러 등)을 파악하라.
5. "무결성 검사 통과: 현재 `[task_plan.md의 세부 과제]`를 진행 중이며, `[task_log.md의 마지막 상태]`에서 멈추었습니다. 이어서 `[다음 추천 행동]`을 진행할까요?"라고 브리핑하라.

#### 3. 🔄 작업 및 기록 워크플로우 (공통)
- 작업을 시작하면 `z_docs/task_log.md`에 모든 시행착오를 기록한다.
- 하나의 태스크가 완료되면 `z_docs/task_log.md`의 내용을 요약하여 `z_docs/log.md`에 아카이빙하고, `z_docs/task_log.md`는 비운다.
- 실행 결과에서 이상·의문 사항이 발생하거나 설계 결정의 근거가 필요한 경우, 추론과 분석을 `analysis.txt`에 날짜 헤더(`[YYYY-MM-DD] 제목`)와 함께 append한다. 코드 변경 없이 관찰·판단 내용만 기록하는 분석 전용 파일이다.
