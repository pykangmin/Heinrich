# 🤖 AI Developer Collaboration Rules (`z_docs/rules.md`)

이 문서는 AI 어시스턴트가 이 프로젝트에서 코드를 작성하고 협업할 때 반드시 지켜야 할 핵심 원칙과 워크플로우를 정의합니다.

## 1. 🧭 핵심 행동 원칙 (Core Principles)
* **기존 동작 유지 (Preserve Existing Behavior)**: 명시적으로 요청받은 부분만 수정하며, 기존 코드의 정상적인 동작을 절대 건드리지 않는다.
* **추측 금지 (No Assumptions)**: 모호하거나 불확실한 부분이 있다면 임의로 코드를 작성하지 않고 반드시 사용자에게 먼저 질문하여 명확히 한다.
* **원자적 변경 (Atomic Changes)**: 한 번에 하나의 논리적 변경만 수행한다. 여러 수정이 필요한 경우 단계별로 나누어 진행한다.
* **영향도 분석 (Impact Analysis)**: 코드를 수정하기 전, 해당 변경이 다른 모듈에 미칠 수 있는 잠재적 영향을 파악하고 사용자에게 고지한다.

## 2. 💻 코드 작성 및 수정 표준 (Code Writing Standards)
* **단일 책임 원칙 (Single Responsibility)**: 각 함수와 클래스는 단 하나의 명확한 목적만 가져야 하며, 비대해지지 않도록 작성한다.
* **명확한 네이밍 (Clear Naming)**: 변수 및 함수명은 그 의도와 역할을 직관적으로 알 수 있도록 명시적으로 작성한다.
* **매직 넘버/스트링 금지 (No Magic Values)**: 코드 내에 의미를 알 수 없는 숫자나 문자열을 직접 사용하지 않고, 컨텍스트를 설명할 수 있는 이름의 상수(Constant)로 정의하여 사용한다.
* **견고한 예외 처리 (Robust Error Handling)**: 발생 가능한 모든 실패 경로(Failure paths)를 고려하여 예외 처리를 반드시 포함한다.

## 3. 🚫 절대 금지 사항 (Prohibitions)
* **무단 파일 조작 금지 (No Unauthorized File Ops)**: 사용자의 명시적인 승인 없이 파일을 삭제하거나 이름을 변경하지 않는다.
* **불필요한 리팩토링 금지 (No Stylistic Refactoring)**: 순전히 '스타일 개선'만을 목적으로 정상 작동하는 기능적 코드를 임의로 리팩토링하지 않는다.
* **데드 코드 방치 금지 (Dead Code Elimination)**: 사용하지 않는 Import, 변수, 함수 등을 남겨두지 않고 깔끔하게 정리한다.
* **미승인 의존성 추가 금지 (No Unrequested Dependencies)**: 요청받지 않은 새로운 라이브러리나 패키지를 임의로 패키지 매니저(package.json, pyproject.toml 등)에 추가하지 않는다.
* **사용 중단 API 사용 금지 (No Deprecated APIs)**: 공식 문서상 Deprecated 된 메서드나 라이브러리는 사용하지 않는다.

## 4. 📋 AI 워크플로우 및 플랜 관리 (Plan & Workflow)
AI는 코드 작성 전후로 다음의 문서를 활용하여 체계적으로 협업한다.

0. **메인 컨텍스트 확인 (`z_docs/main.md`)**: 코드를 작성하거나 수정하기 전, 가장 먼저 읽어 프로젝트의 전체 구조를 파악한다.
1. **메인 플랜 참조 (`z_docs/plan.md`)**: 작업 시작 전 전체 로드맵을 확인하여, 현재 진행해야 할 메인 과제(Main Task)의 위치를 파악한다.
2. **태스크 플랜 수립 (`z_docs/task_plan.md`)**: 해당 과제 달성을 위한 구체적인 세부 단계(Step-by-step), 수정할 파일 목록을 담은 '태스크 플랜'을 세운다.
3. **사용자 승인 (User Approval)**: 수립된 플랜을 제시하고 명시적인 실행 허락(Go-ahead)을 구한다.
4. **실행 및 task_log 기록 (`z_docs/task_log.md`)**: 플랜에 따라 실행하면서 진행 상황, 결정 사항, 발생한 이슈와 해결 과정을 실시간으로 상세히 기록한다.
5. **검증 (Verify)**: 코드 작성/수정 후 로직이 정상 작동하는지 검증한다.
6. **log.md 정리 및 업데이트**: 태스크가 종료되면 `z_docs/task_log.md`의 내용을 요약하여 `z_docs/log.md`에 이전하고, `z_docs/plan.md`의 상태를 업데이트한다. 이후 `task_log.md`는 초기화한다.
7. **규칙 최신화 (`z_docs/rules.md`)**: 작업 중 새로운 제약사항이 생기거나 반복되는 패턴이 발견되면 사용자 동의 하에 이 규칙 문서를 업데이트한다.

## 5. 🗣️ AI 응답 포맷 (Response Format)
* **Summary First**: 코드를 제시하기 전, 변경 사항을 [What / Why] 포맷으로 간결하게 요약한다.
* **File Paths**: 코드 블록을 출력할 때는 항상 최상단에 수정/생성되는 파일의 전체 경로(Full file path)를 명시한다.
* **Execution Order**: 여러 파일을 수정해야 할 경우, 어떤 순서로 적용해야 안전한지 권장 순서를 안내한다.

## 6. 📝 task_log.md 운영 규칙
`z_docs/task_log.md`는 현재 진행 중인 태스크의 실시간 작업 일지다.

### 기록 시점
- 각 Step 실행 시작 시: `### Step N 시작` + 무엇을 할지 한 줄 요약
- 이슈 발생 시: 즉시 원인과 시도한 해결 방법 기록
- 결정 사항 발생 시: 왜 그 방향을 선택했는지 근거 기록
- 각 Step 완료 시: 결과 및 다음 Step으로의 연결 기록

### log.md로 이전 조건
- 태스크 완료 선언 시
- 사용자가 "task_log 정리해줘" 요청 시
- task_log가 지나치게 길어져 가독성이 떨어질 때 (AI가 판단하여 제안)

### 이전 방식
1. `task_log.md` 전체 내용을 **[What / Why / Troubleshooting / Decision]** 형식으로 요약
2. 요약본을 `z_docs/log.md` 최상단에 날짜 헤더와 함께 추가
3. `z_docs/plan.md` 해당 항목 체크박스 업데이트
4. `z_docs/task_log.md` 초기화 (빈 템플릿 상태로 리셋)

---

## 7. 🤝 역할 분담 (Role Division)

* **코드 작성은 사용자 담당**: 구현·코드 작성은 사용자가 직접 한다. AI는 `z_docs/plan.md`와 `z_docs/task_plan.md` 업데이트만 담당하며, 소스 코드 파일에는 명시적 요청 없이 절대 손대지 않는다.
* **문서 간섭 최소화**: `main.md`·`rules.md` 등 문서 파일도 규칙 추가·무결성 업데이트 목적 외에는 수정하지 않는다.

---

## 8. 🛠️ 프로젝트 특화 컨벤션 (Project Specific Rules)

* **디렉토리/경로**: 협업 문서는 `z_docs/`, 파이프라인 코드는 `data_pipeline/`를 기준으로 한다.
* **환경 변수**: 시크릿은 `.env`에만 두고 저장소에 커밋하지 않는다. DB 접속은 우선 **`DATABASE_URL`** (Supabase 대시보드의 Connection string)을 사용한다.
* **데이터베이스**: 프로덕션·개발 저장소는 **Supabase**(호환 프로토콜: PostgreSQL). 스키마는 SQL Editor·마이그레이션으로 관리하고, 앱 계층에서는 기존 `DBManager`/SQLAlchemy 경로를 유지한다.
* **패키지 관리**: **Poetry** (`pyproject.toml`).
