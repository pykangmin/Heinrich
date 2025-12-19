import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import time

# 1. 환경 설정 및 API 로드
load_dotenv()
api_key = os.getenv("FRED_API_KEY") # FRED API 키를 가리기 위해서 .env 파일에 저장해 놓았음
# api_key = "Your Api Key Here" # 직접 입력하는 방법
fred = Fred(api_key)

# 데이터 수집 기간 설정
start = "2020-01-01"
end = "2024-12-31"

# 2. 수집 대상 매크로 지표 정의 (지표명: (데이터소스, 티커))
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

# 3. 데이터 수집 및 엑셀 저장 함수 정의
def fetch_macro_data():
    all_data = [] # 수집된 시리즈 저장 리스트
    print("--- 데이터 수집 시작 ---")

    for name, (source, code) in series_info.items():
        try:
            if source == "fred":
                s = fred.get_series(code, start, end)
            else:
                df_yf = yf.download(code, start=start, end=end, progress=False, auto_adjust=True)
                if not df_yf.empty:
                    s = df_yf["Close"][code] if isinstance(df_yf.columns, pd.MultiIndex) else df_yf["Close"]
                else:
                    s = pd.Series(dtype="float64")
            
            s.name = name
            all_data.append(s)
            print(f"성공: {name}")
        except Exception as e:
            print(f"실패 ({name}): {e}")
            all_data.append(pd.Series(name=name, dtype="float64"))
        time.sleep(0.1)

    # 일별 인덱스 병합 및 결측치 처리
    daily_index = pd.date_range(start=start, end=end, freq="D")
    df = pd.concat(all_data, axis=1).reindex(daily_index)
    df = df.ffill().round(2)
    
    return df

file_name = "indicator_data_2020_2024.xlsx" # 파일명

def save_to_formatted_excel(df, file_name=file_name):
    print("--- 엑셀 서식 적용 및 저장 시작 ---")
    
    df.index.name = "날짜"
    df.index = df.index.strftime("%Y-%m-%d")
    
    # nan_inf_to_errors 옵션을 추가하여 NaN 에러 방지
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter', engine_kwargs={'options': {'nan_inf_to_errors': True}})
    df.to_excel(writer, sheet_name='MacroData')

    workbook  = writer.book
    worksheet = writer.sheets['MacroData']
    
    # 공통 서식 설정
    base_format = workbook.add_format({
        'align': 'center', 
        'valign': 'vcenter', 
        'border': 1, 
        'num_format': '#,##0.00'
    })

    # 열 너비 설정
    worksheet.set_column(0, len(df.columns), 18) 

    # 데이터 변화에 따른 셀 병합 처리
    for col_idx, col_name in enumerate(df.columns, start=1):
        data = df[col_name].tolist()
        if not data: continue
        
        start_row = 0
        for i in range(1, len(data)):
            curr_val, prev_val = data[i], data[start_row]
            is_different = (curr_val != prev_val) if pd.notna(curr_val) and pd.notna(prev_val) else (pd.isna(curr_val) != pd.isna(prev_val))

            if is_different or i == len(data) - 1:
                end_row = i - 1 if is_different else i
                if pd.notna(prev_val):
                    if end_row > start_row:
                        worksheet.merge_range(start_row + 1, col_idx, end_row + 1, col_idx, prev_val, base_format)
                    else:
                        worksheet.write(start_row + 1, col_idx, prev_val, base_format)
                else:
                    for r in range(start_row, end_row + 1):
                        worksheet.write_blank(r + 1, col_idx, None, base_format)
                start_row = i

    writer.close()
    print(f"--- 저장 완료: {file_name} ---")

if __name__ == "__main__":
    macro_df = fetch_macro_data()
    save_to_formatted_excel(macro_df)
