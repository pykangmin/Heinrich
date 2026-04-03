import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import date

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SEC가 정적 제공하는 매핑 (data.sec.gov의 company_tickers 경로는 404)
CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FILING_XML_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"
START_DATE = date(2025, 1, 1)


def _get_headers() -> dict:
    """SEC_USER_AGENT 환경변수 필수. 없으면 RuntimeError."""
    ua = (os.getenv("SEC_USER_AGENT") or "").strip()
    if not ua:
        raise RuntimeError(
            "SEC_USER_AGENT 가 설정되지 않았습니다. "
            "SEC 정책에 따라 User-Agent(이름·이메일)가 필수입니다."
        )
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def _local_tag(tag: str) -> str:
    if not tag:
        return ""
    return tag.split("}", 1)[-1]


def _find_first_direct(parent: ET.Element, local_name: str):
    for c in parent:
        if _local_tag(c.tag) == local_name:
            return c
    return None


def _scalar_from_amounts(parent: ET.Element, local_name: str) -> str | None:
    node = _find_first_direct(parent, local_name)
    if node is None:
        return None
    val_el = _find_first_direct(node, "value")
    if val_el is not None and val_el.text and val_el.text.strip():
        return val_el.text.strip()
    return None


def _transaction_code(tx_el: ET.Element) -> str | None:
    coding = _find_first_direct(tx_el, "transactionCoding")
    if coding is None:
        return None
    tc = _find_first_direct(coding, "transactionCode")
    if tc is not None and tc.text and tc.text.strip():
        return tc.text.strip()
    return None


def _compute_value(tx_el: ET.Element) -> float | None:
    amounts = _find_first_direct(tx_el, "transactionAmounts")
    if amounts is None:
        return None
    shares_s = _scalar_from_amounts(amounts, "transactionShares")
    price_s = _scalar_from_amounts(amounts, "transactionPricePerShare")
    if not shares_s or not price_s:
        return None
    try:
        sh = float(shares_s)
        pr = float(price_s)
    except ValueError:
        return None
    return sh * pr


def _row_accession(accession: str, line_index: int) -> str:
    """
    SEC 접수번호는 제출당 동일하지만, 테이블 UNIQUE·거래 단위 적재를 위해
    접미사로 행을 구분한다. (최대 25자)
    """
    suf = f"#{line_index}"
    if len(accession) + len(suf) <= 25:
        return accession + suf
    compact = accession.replace("-", "")
    if len(compact) + len(suf) <= 25:
        return compact + suf
    return (compact + suf)[:25]


def load_cik_map(symbols: list[str]) -> dict[str, int]:
    """company_tickers.json → {symbol: cik}. 요청 심볼만 필터."""
    headers = _get_headers()
    r = requests.get(CIK_MAP_URL, headers=headers, timeout=60)
    r.raise_for_status()
    raw = r.json()
    want = {str(s).upper().strip().replace('.', '-') for s in symbols if s}
    out: dict[str, int] = {}
    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        t = str(entry.get("ticker", "")).upper().strip().replace(".", "-")
        if t in want:
            out[t] = int(entry["cik_str"])
    return out


def _parse_form4_xml(
    xml_text: str, symbol: str, filing_date: str, accession: str
) -> list[dict]:
    """
    nonDerivativeTable / derivativeTable 내 transaction 노드 파싱.
    transactionCode → transaction_type, shares * price → value (불가 시 None).
    accession_number: 제출 단위 키 + 행 접미사 (UNIQUE·25자 이하).
    """
    root = ET.fromstring(xml_text)
    if _local_tag(root.tag) != "ownershipDocument":
        return []

    rows: list[dict] = []
    line_idx = 0
    for el in root.iter():
        ln = _local_tag(el.tag)
        if ln not in ("nonDerivativeTransaction", "derivativeTransaction"):
            continue
        code = _transaction_code(el)
        val = _compute_value(el)
        acc_key = _row_accession(accession, line_idx)
        line_idx += 1
        rows.append(
            {
                "symbol": symbol,
                "filing_date": filing_date,
                "transaction_type": code,
                "value": val,
                "accession_number": acc_key,
            }
        )
    return rows


def fetch_filings_for_cik(cik: int, symbol: str, headers: dict) -> list[dict]:
    """
    /submissions/CIKXXXX.json에서 최근 Form 4만 필터( filingDate >= START_DATE ).
    각 접수별 XML 다운로드 후 파싱. XML 요청 간 0.1초 대기.
    """
    url = SUBMISSIONS_URL.format(cik=cik)
    logger.info("%s: submissions JSON 요청 (CIK %s)", symbol, cik)
    r = requests.get(url, headers=headers, timeout=90)
    r.raise_for_status()
    data = r.json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    cik_int = int(str(data.get("cik", cik)))

    work: list[tuple[str, str, str]] = []
    for form, fd, acc, doc in zip(forms, filing_dates, accession_numbers, primary_docs):
        if form != "4":
            continue
        try:
            fd_parts = fd.split("-")
            fdate = date(int(fd_parts[0]), int(fd_parts[1]), int(fd_parts[2]))
        except (ValueError, IndexError):
            continue
        if fdate < START_DATE:
            continue
        work.append((fd, acc, doc))

    n_files = len(work)
    if n_files == 0:
        logger.info("%s: 해당 기간(%s~) Form 4 없음", symbol, START_DATE.isoformat())
        return []

    logger.info("%s: Form 4 %s건 XML 수집 시작", symbol, n_files)
    step = max(1, n_files // 5)
    out: list[dict] = []
    for j, (fd, acc, doc) in enumerate(work):
        if j == 0 or (j + 1) % step == 0 or j == n_files - 1:
            logger.info("%s: Form 4 XML %s/%s", symbol, j + 1, n_files)

        basename = str(doc).rsplit("/", 1)[-1]
        acc_nodash = str(acc).replace("-", "")
        xml_url = FILING_XML_URL.format(
            cik=cik_int, accession=acc_nodash, filename=basename
        )
        time.sleep(0.1)
        try:
            xr = requests.get(xml_url, headers=headers, timeout=90)
            xr.raise_for_status()
            text = xr.text
            if "ownershipDocument" not in text:
                continue
            rows = _parse_form4_xml(text, symbol, fd, acc)
            out.extend(rows)
        except Exception as e:
            logger.warning("Form 4 XML 실패 symbol=%s accession=%s: %s", symbol, acc, e)

    logger.info("%s: 거래 행 %s건 추출 완료", symbol, len(out))
    return out


def fetch_all(
    symbols: list[str], cik_map: dict[str, int], delay: float = 0.5
) -> pd.DataFrame:
    """심볼 순회해 모든 제출·거래 행을 수집. 10% 단위 진행 로그."""
    headers = _get_headers()
    all_rows: list[dict] = []
    n = len(symbols)
    next_milestone = 10

    for i, sym in enumerate(symbols):
        nsym = str(sym).upper().strip().replace(".", "-")
        cik = cik_map.get(nsym)
        logger.info("[%s/%s] 종목=%s", i + 1, n, nsym)
        if not cik:
            logger.warning("CIK 매핑 없음: %s", nsym)
        else:
            try:
                all_rows.extend(fetch_filings_for_cik(cik, nsym, headers))
            except Exception as e:
                logger.warning("submissions 수집 실패 symbol=%s: %s", nsym, e)

        if delay and i + 1 < n:
            time.sleep(delay)

        if n <= 0:
            continue
        pct = int(100 * (i + 1) / n)
        while next_milestone <= 100 and pct >= next_milestone:
            logger.info("진행률 %s%% (%s/%s)", next_milestone, i + 1, n)
            next_milestone += 10

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "filing_date",
                "transaction_type",
                "value",
                "accession_number",
            ]
        )
    return pd.DataFrame(all_rows)


def load_symbols() -> list[str]:
    """DB 우선, 실패 시 CSV."""
    from data_pipeline.earnings_fetcher import load_symbols as _ls

    return _ls()


if __name__ == "__main__":
    from data_pipeline.db_manager import DBManager

    symbols = load_symbols()
    if not symbols:
        raise SystemExit(1)
    logger.info("로드된 심볼 %s개 — CIK 맵 조회 중...", len(symbols))
    cik_map = load_cik_map(symbols)
    logger.info("CIK 매핑 %s개 — 수집 루프 시작 (종목당 지연·Form 4 건수에 따라 수십 분 소요 가능)", len(cik_map))
    df = fetch_all(symbols, cik_map)
    if df.empty:
        logger.warning("적재할 내부자 거래 행이 없습니다.")
    else:
        DBManager().upsert_insider_trades(df)
