# 환경 설정 및 가이드

하인리히 프로젝트 Layer 3 종목 선정 파이프라인 구축을 위한 로컬 및 서버 환경 구성 뼈대입니다.

## 기본 요구 사항
- Python 3.10 이상의 환경
- **[Supabase](https://supabase.com)** 프로젝트 1개 (엔진은 PostgreSQL 호환; 별도 로컬 Postgres 불필요)

## 가상환경 및 의존성 관리 (Poetry)

본 프로젝트는 **Poetry**를 사용하여 Mac과 Windows 환경에서 일관된 의존성을 관리합니다.

### 1. Poetry 설치 (미설치 시)
- **Mac/Linux/Windows**: [공식 설치 가이드](https://python-poetry.org/docs/#installing-with-the-official-installer) 참조

### 2. 초기 셋업 및 가상환경 활성화
```bash
# 의존성 설치 (pyproject.toml 기반)
poetry install

# 가상환경 내에서 쉘 실행 (활성화)
poetry shell
```

### 3. 주요 명령어
- 패키지 추가: `poetry add <package_name>`
- 스크립트 실행: `poetry run python <script.py>`

## pip으로 의존성 설치 (Poetry 없이)

Poetry를 쓰지 않을 때는 가상환경을 만든 뒤 `pip`으로 `pyproject.toml`과 동일한 패키지를 맞추면 됩니다.

### 1. 가상환경 생성 및 활성화
```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Mac/Linux
source .venv/bin/activate
```

### 2. 패키지 일괄 설치
`pyproject.toml`의 `[tool.poetry.dependencies]`와 동일한 목록입니다.

```bash
pip install --upgrade pip
pip install yfinance pandas beautifulsoup4 lxml sqlalchemy psycopg2-binary python-dotenv requests
```

Python 버전은 **3.10 이상**을 사용합니다.

### 3. 스크립트 실행 예시
```bash
python -m data_pipeline.sp500_fetcher
python data_pipeline/db_manager.py
```

이후 절차(Supabase `.env` 등)는 Poetry 경로와 같습니다.

## 초기 설정
- [x] Poetry 기반 `pyproject.toml` 및 `README.md` 생성
- [x] 가속 환경 및 패키지 구조(`data_pipeline/`) 세팅
- [x] API 키 등 `.env` 템플릿 환경변수 구성 가이드 작성 (Supabase 기반)

## Supabase / DB 초기 셋업
1. Supabase 대시보드에서 프로젝트 생성 → **Settings → Database**에서 비밀번호 확인.
2. **Connect**에서 앱용 연결 문자열 복사.
   - **Transaction pooler** (IPv4/서버리스 친화, 포트 `6543`) 또는 **Direct** (`5432`) 중 하나 선택. SQLAlchemy 배치 적재는 Direct가 단순한 경우가 많고, 동시성 높은 서비스는 Pooler 문서를 참고한다.
3. `.env`에 아래 중 **하나**를 설정한다 (권장: `DATABASE_URL` 단일 변수).
   ```bash
   # 권장: 대시보드에서 복사한 URI (sslmode 포함된 경우 그대로 유지)
   DATABASE_URL=postgresql://postgres.xxx:비밀번호@aws-0-xxx.pooler.supabase.com:6543/postgres
   ```
   레거시/로컬 호환을 위해 분리 변수도 지원한다: `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`.
4. 테이블은 Supabase SQL Editor에서 `z_docs/schema.md` DDL로 생성한다. 앱에서 ORM으로 보조 생성 시 `poetry run python data_pipeline/db_manager.py` 또는 venv 활성화 후 `python data_pipeline/db_manager.py`로 `tickers` / `daily_prices` 등 메타데이터 생성을 확인할 수 있다. E2E는 `python -m data_pipeline.sp500_fetcher`(`.env`에 `DATABASE_URL` 필요)로 검증한다.
