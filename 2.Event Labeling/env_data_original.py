# 1. 환경 설정

import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf
import os
from dotenv import load_dotenv

load_dotenv()
fred = Fred(os.getenv("FRED_API_KEY"))

# 상승,하락 국면 기간 정의
REGIMES = {
    "UP": [
        ("2020-03-23", "2022-01-03"),
        ("2022-10-10", "2023-07-19"),
        ("2023-10-26", "2024-12-31"),
    ],
    "DOWN": [
        ("2020-02-20", "2020-03-23"),
        ("2022-01-03", "2022-10-13"),
        ("2023-07-19", "2023-10-26"),
    ],
}

# 수집할 지표 정의 
SERIES_INFO = {
    "기준금리": ("fred", "FEDFUNDS"),
    "통화량": ("fred", "M2SL"),
    "연준대차대조표": ("fred", "WALCL"),
    "금융조건지수": ("fred", "NFCI"),
    "금융스트레스지수": ("fred", "STLFSI4"),
    "국채2년금리": ("fred", "DGS2"),
    "국채10년금리": ("fred", "DGS10"),
    "장단기금리차_10Y2Y": ("fred", "T10Y2Y"),
    "장단기금리차_10Y3M": ("fred", "T10Y3M"),
    "실질금리_10Y": ("fred", "DFII10"),
    "하이일드스프레드": ("fred", "BAMLH0A0HYM2"),
    "경기선행지수": ("fred", "USSLIND"),
    "TED스프레드": ("fred", "TEDRATE"),
    "소비자물가지수": ("fred", "CPIAUCSL"),
    "개인소비지출물가": ("fred", "PCEPI"),
    "생산자물가지수": ("fred", "PPIACO"),
    "제조업지수": ("fred", "IPMAN"),
    "미시간대소비자심리": ("fred", "UMCSENT"),
    "소매판매": ("fred", "RSAFS"),
    "주택착공건수": ("fred", "HOUST"),
    "비농업고용지수": ("fred", "PAYEMS"),
    "실업률": ("fred", "UNRATE"),
    "평균시간당임금": ("fred", "CES0500000003"),
    "채권변동성지수": ("yf", "^MOVE"),
    "변동성지수": ("yf", "^VIX"),
    "달러인덱스": ("fred", "DTWEXBGS"),
    "연준RRP": ("fred", "RRPONTSYD"),
    "WTI유가": ("fred", "DCOILWTICO"),
    "회사채스프레드_IG": ("fred", "BAMLC0A0CM"),
    "글로벌금융스트레스": ("fred", "KCFSI"),
}



# 2. 시계열 데이터 로딩 함수

def load_series(source, code, start="2020-01-01", end="2024-12-31"):
    """
    FRED: 일별/월별 시계열
    Yahoo Finance: 종가(Adj Close) 사용
    """
    try:
        if source == "fred":
            series = fred.get_series(code, observation_start=start, observation_end=end)
            return series.dropna()
        elif source == "yf":
            df = yf.download(code, start=start, end=end, progress=False)
            if df.empty:
                return None
            # 멀티인덱스 처리
            if isinstance(df.columns, pd.MultiIndex):
                if "Adj Close" in df.columns.levels[0]:
                    return df["Adj Close"].iloc[:, 0].dropna()
                else:
                    return df["Close"].iloc[:, 0].dropna()
            else:
                return df["Adj Close"].dropna() if "Adj Close" in df.columns else df["Close"].dropna()
        else:
            return None
    except Exception as e:
        print(f"데이터 로딩 실패 → {code} | {e}")
        return None



# 3. IQR 계산 함수

# 시리즈의 IQR(Q1, Q3) 반환
def compute_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return q1, q3

# 시리즈의 IQR(Q1, Q3) 반환
def intersect_iqr(iqr_list):
    # Q1 == Q3인 IQR 제거
    iqr_list = [iqr for iqr in iqr_list if iqr[0] != iqr[1]]

    if not iqr_list:  # 모두 Q1==Q3라면
        return np.nan, np.nan
    
    lower = max([iqr[0] for iqr in iqr_list])
    upper = min([iqr[1] for iqr in iqr_list])
    if lower <= upper:
        return lower, upper
    else:
        return np.nan, np.nan



# 4. 레짐별 공통 IQR 계산

results_up = []
results_down = []

for name, (source, code) in SERIES_INFO.items():
    series = load_series(source, code)
    if series is None or series.empty:
        continue

    for regime, periods in REGIMES.items():
        iqr_by_period = []

        # 각 구간 IQR 계산
        for start, end in periods:
            sliced = series.loc[start:end]
            if len(sliced) < 5:
                break
            iqr_by_period.append(compute_iqr(sliced))

        # 3개 구간 모두 IQR이 존재해야 공통 IQR 계산
        if len(iqr_by_period) != 3:
            continue

        lower, upper = intersect_iqr(iqr_by_period)
        if np.isnan(lower) or np.isnan(upper):
            continue

        row = {
            "지표": name,
            "공통_IQR": f"{lower:.3f} ~ {upper:.3f}",
        }

        if regime == "UP":
            results_up.append(row)
        else:
            results_down.append(row)



# 5. 결과 정리 및 엑셀 저장

up_df = pd.DataFrame(results_up).sort_values("지표").reset_index(drop=True)
down_df = pd.DataFrame(results_down).sort_values("지표").reset_index(drop=True)

output_path = "regime_common_iqr_2020_2024.xlsx"

with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    up_df.to_excel(writer, sheet_name="UP", index=False)
    down_df.to_excel(writer, sheet_name="DOWN", index=False)

print(f"엑셀 파일 저장 완료: {output_path}")
