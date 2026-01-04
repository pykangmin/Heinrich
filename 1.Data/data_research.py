import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import time

# 1. 환경 설정 및 API 로드

load_dotenv()
api_key = os.getenv("FRED_API_KEY")
fred = Fred(api_key)

RAW_START = "2019-12-27"
EXPORT_START = "2020-01-01"
END = "2024-12-31"


# 2. 매크로 지표 정의 (컬럼명은 한글, 티커는 내부용)
series_info = {
    "기준금리": ("fred", "FEDFUNDS"),           # FFR
    "통화량": ("fred", "M2SL"),                 # M2
    "연준대차대조표": ("fred", "WALCL"),          # Fed_BalanceSheet
    "금융조건지수": ("fred", "NFCI"),             # FCI
    "금융스트레스지수": ("fred", "STLFSI4"),       # STLFSI
    "국채2년금리": ("fred", "DGS2"),              # UST_2Y
    "국채10년금리": ("fred", "DGS10"),            # UST_10Y
    "장단기금리차_10Y2Y": ("fred", "T10Y2Y"),      # 10Y_2Y_Spread
    "장단기금리차_10Y3M": ("fred", "T10Y3M"),      # 10Y_3M_Spread
    "실질금리_10Y": ("fred", "DFII10"),           # TIPS_10Y
    "하이일드스프레드": ("fred", "BAMLH0A0HYM2"),   # HY_Spread
    "채권변동성지수": ("yf", "^MOVE"),             # MOVE
    "변동성지수": ("yf", "^VIX"),                 # VIX
    "경기선행지수": ("fred", "USSLIND"),           # LEI
    "TED스프레드": ("fred", "TEDRATE"),           # TEDRATE (FRAOIS 대체)
    "소비자물가지수": ("fred", "CPIAUCSL"),         # CPI
    "개인소비지출물가": ("fred", "PCEPI"),          # PCE
    "주거비": ("fred", "CUSR0000SEHA"),          # Shelter_CPI
    "생산자물가지수": ("fred", "PPIACO"),          # PPI
    "제조업지수_대체": ("fred", "IPMAN"),          # IPMAN (ISM 대체: 산업생산)
    "제조업가격_대체": ("fred", "WPSFD4111"),      # WPSFD4111 (ISM Price 대체: PPI 세부)
    "미시간대소비자심리": ("fred", "UMCSENT"),      # Michigan_Sentiment
    "소매판매": ("fred", "RSAFS"),               # Retail_Sales
    "주택착공건수": ("fred", "HOUST"),            # Housing_Starts
    "건축허가건수": ("fred", "PERMIT"),           # Housing_Permit
    "고용비용지수": ("fred", "ECIWAG"),           # ECI
    "민간고용_대체": ("fred", "NPPTTL"),          # NPPTTL (ADP 대체)
    "비농업고용지수": ("fred", "PAYEMS"),         # NFP
    "실업률": ("fred", "UNRATE"),               # Unemployment
    "평균시간당임금": ("fred", "CES0500000003"),  # AHE
}

# 3. 발표 시차 보정 규칙 (개월 단위)

LAG_RULES = {
    "통화량": 1,
    "소비자물가지수": 1,
    "개인소비지출물가": 1,
    "주거비": 1,
    "생산자물가지수": 1,
    "제조업지수_대체": 1,
    "제조업가격_대체": 1,
    "소매판매": 1,
    "주택착공건수": 1,
    "건축허가건수": 1,
    "민간고용_대체": 1,
    "비농업고용지수": 1,
    "실업률": 1,
    "평균시간당임금": 1,
    "고용비용지수": 3,
}

def shift_series_by_lag(series, name):
    lag = LAG_RULES.get(name, 0)
    if lag > 0:
        series = series.copy()
        series.index = series.index - pd.DateOffset(months=lag)
    return series


# 4. 데이터 수집

def fetch_macro_data():
    all_data = []
    print("--- 데이터 수집 시작 ---")

    for name, (source, code) in series_info.items():
        try:
            if source == "fred":
                s = fred.get_series(code, RAW_START, END)
            else:
                df_yf = yf.download(code, start=RAW_START, end=END,
                                    progress=False, auto_adjust=True)
                if not df_yf.empty:
                    s = df_yf["Close"].copy()
                else:
                    s = pd.Series(dtype="float64")

            s.name = name
            s = shift_series_by_lag(s, name)

            all_data.append(s)
            print(f"성공: {name}")

        except Exception as e:
            print(f"실패 ({name}): {e}")
            all_data.append(pd.Series(name=name, dtype="float64"))

        time.sleep(0.1)


    # 5. 일 단위 인덱스 + 보간 + ffill

    daily_index = pd.date_range(start=RAW_START, end=END, freq="D")

    df = pd.concat(all_data, axis=1)
    df = df.reindex(daily_index)

    # yfinance 지표 컬럼명 강제 정리
    df = df.rename(columns={
        "^MOVE": "채권변동성지수",
        "^VIX": "변동성지수"
    })  

    df = df.ffill()
    df = df.round(2)

    return df


# 6. 엑셀 저장 (지정된 날짜 이후 데이터만)

file_name = "indicator_data_2020_2024.xlsx"

def save_to_formatted_excel(df, file_name=file_name):
    print("--- 엑셀 저장 시작 ---")

    df = df.loc[EXPORT_START:].copy()
    df.index.name = "날짜"
    df.index = df.index.strftime("%Y-%m-%d")

    writer = pd.ExcelWriter(
        file_name,
        engine="xlsxwriter",
        engine_kwargs={"options": {"nan_inf_to_errors": True}},
    )

    df.to_excel(writer, sheet_name="MacroData")
    worksheet = writer.sheets["MacroData"]
    worksheet.set_column(0, len(df.columns), 18)

    writer.close()
    print(f"--- 저장 완료: {file_name} ---")


# 7. 실행

if __name__ == "__main__":
    macro_df = fetch_macro_data()
    save_to_formatted_excel(macro_df)
