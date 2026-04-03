import json
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COL_RANK = "순위"
COL_NAME_TICKER = "종목명 (Ticker)"
COL_SECTOR = "섹터"
COL_VALUE_CHAIN = "세부 산업군 (Value Chain)"

# 긴 룰을 먼저 매칭 (부분 문자열 in vc_position)
ORDERED_VC_STAGE_RULES: tuple[tuple[str, int], ...] = (
    ("전임상 & 임상시험", 3),
    ("타깃 발굴 & Discovery", 3),
    ("FDA 승인 & 규제", 3),
    ("자본 조달 (IB & 보험)", 8),
    ("수송 & 저장 (Midstream)", 2),
    ("EDA & 코어 IP", 1),
    ("조립·패키징·테스트", 4),
    ("1차 가공 & 정련", 2),
    ("횡단 전력망 장비 & 건설", 9),
    ("토지 확보 & 자금 조달", 9),
    ("소비자 공급 & 소매", 7),
    ("아이디어 & 디자인", 6),
    ("채굴 & 탐사", 2),
    ("수송 & 저장", 2),
    ("장비 & 소재", 1),
    ("웨이퍼 제조", 2),
    ("생산 & 1차 처리", 2),
    ("특수소재 & 화학", 1),
    ("탐사 & 시추 (E&P)", 2),
    ("탐사 & 시추", 2),
    ("정제 & 소매 / 발전 & 판매", 2),
    ("원자재 & 부품 조달", 2),
    ("원자재 가공 & 정밀 부품", 2),
    ("건설·산업 적용", 1),
    ("상업 제조", 4),
    ("시스템 통합 & 최종 조립", 4),
    ("제조 & 조립", 4),
    ("제조 & 가공", 4),
    ("완성품 & 서비스", 5),
    ("소비자 서비스 & 개인화", 6),
    ("플랫폼 & 콘텐츠 제작", 6),
    ("유통 & 소비자 경험", 7),
    ("유통 & 소매", 7),
    ("유통 & 처방", 7),
    ("브랜딩 & 패키징", 7),
    ("브랜드 & 마케팅", 7),
    ("상업은행 & 결제 네트워크", 8),
    ("자산운용 & 자본시장", 8),
    ("자본 조달", 8),
    ("데이터 & 신용평가 인프라", 8),
    ("소비자 금융 & 핀테크", 8),
    ("물리 인프라", 9),
    ("네트워크 서비스 운영", 9),
    ("발전 (Generation)", 9),
    ("발전 (Energy Equipment)", 9),
    ("발전", 9),
    ("송전 & 배전", 9),
    ("물류 & 유통 인프라", 9),
    ("자산 소유 & 운용", 9),
    ("임차인 서비스 & 관리", 7),
    ("개발 & 건설", 4),
    ("유지보수 & 애프터서비스", 4),
)


def vc_stage_num_for_position(vc_position: str) -> int | None:
    """Value Chain 텍스트 → 단계 번호 (Track D 밸류체인 순서 대응). 미매칭 시 None."""
    if not vc_position or not str(vc_position).strip():
        return None
    s = str(vc_position).strip()
    for needle, num in ORDERED_VC_STAGE_RULES:
        if needle in s:
            return num
    return None


def _parse_symbol_from_label(label: str) -> str | None:
    m = re.search(r"\(([A-Za-z0-9.-]+)\)\s*$", (label or "").strip())
    if not m:
        return None
    return m.group(1).upper().replace(".", "-")


def default_classification_path() -> Path:
    return Path(__file__).resolve().parent.parent / "분류.json"


def load_classification_df(path: Path | None = None) -> pd.DataFrame:
    """
    분류.json → symbol, vc_sector, vc_position, vc_stage_num.
    동일 symbol 중복 시 마지막 행 유지.
    """
    p = path or default_classification_path()
    if not p.is_file():
        raise FileNotFoundError(f"분류 JSON 없음: {p}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    unmapped_vc: set[str] = set()

    for item in raw:
        if not isinstance(item, dict):
            continue
        name_t = item.get(COL_NAME_TICKER, "")
        sym = _parse_symbol_from_label(str(name_t))
        if not sym:
            logger.warning("티커 추출 실패: %s", name_t)
            continue
        vc = item.get(COL_VALUE_CHAIN, "")
        vc_str = str(vc).strip() if vc is not None else ""
        stage = vc_stage_num_for_position(vc_str)
        if stage is None and vc_str:
            unmapped_vc.add(vc_str)
        rows.append(
            {
                "symbol": sym,
                "vc_sector": str(item.get(COL_SECTOR, "")).strip() or None,
                "vc_position": vc_str or None,
                "vc_stage_num": stage,
            }
        )

    for u in sorted(unmapped_vc):
        logger.warning("vc_stage_num 미매칭 Value Chain (수동 매핑 검토): %s", u[:120])

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["symbol"], keep="last")
    return df


if __name__ == "__main__":
    from data_pipeline.db_manager import DBManager

    df = load_classification_df()
    if df.empty:
        raise SystemExit("분류 DataFrame 비어 있음")
    logger.info("%s 심볼 분류 로드 (vc_stage 지정 %s행)", len(df), df["vc_stage_num"].notna().sum())
    DBManager().upsert_vc_classification(df)
