import pandas as pd
import numpy as np
from math import sqrt
import yfinance as yf

#파일 불러오기
EXCEL_PATH = "C:\\Users\\User\\Documents\\GitHub\\Heinrich\\new_indicator_data_2020_2024.xlsx"
SHEET_NAME = "MacroData"

START = "2020-01-01"
END   = "2024-12-31"

#유틸 함수 생성

def pct_change_n(s: pd.Series, n: int) -> pd.Series:
    # n기간 전 대비 변화율: (현재/과거 - 1)
    den = s.shift(n)
    out = s / den - 1
    out = out.replace([np.inf, -np.inf], np.nan)

    # inf / -inf가 생기면 NaN 처리 (안정성)
    out = out.mask(den == 0, np.nan)
    return out

def zscore_rolling(s: pd.Series, window: int = 24, min_periods: int = 12) -> pd.Series:
    """
    롤링 z-score:
    - '그 시점 이전 데이터'만 사용해서 평균/표준편차 계산 → 누수 방지
    - window=24는 24개월 데이터 2년치 정도
    """
    mu = s.rolling(window=window, min_periods=min_periods).mean()
    sd = s.rolling(window=window, min_periods=min_periods).std()

    # sd가 0이면 분모가 0이 되므로 NaN 처리
    sd = sd.replace(0, np.nan)

    return (s - mu) / sd

#IQR일단 사용안해봄! (상의필요)
def proportion_ci_approx(p: float, n: int, z: float = 1.96):
    """상승확률 p의 근사 95% 신뢰구간(표본수 n이 너무 작으면 불안정)"""
    if n <= 0 or np.isnan(p):
        return (np.nan, np.nan)
    se = sqrt(max(p * (1 - p), 1e-12) / n)
    lo = max(0.0, p - z * se)
    hi = min(1.0, p + z * se)
    return (lo, hi)

def beta_binomial_smooth(k: int, n: int, alpha: float = 1.0, beta: float = 1.0):
    """
    베타-이항 스무딩:
    - 표본이 작을 때 확률이 0%/100%로 튀는 걸 완화
    - alpha=1, beta=1이면 라플라스 스무딩 느낌
    """
    return (k + alpha) / (n + alpha + beta)

# 엑셀 로드 후 일간 월말로 리샘플링

raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

# 엑셀에 저장된 날짜 컬럼명은 "날짜"
raw["날짜"] = pd.to_datetime(raw["날짜"])
raw = raw.sort_values("날짜").set_index("날짜")

# 분석 구간만 자르기
raw = raw.loc[START:END].copy()

# 월/분기 레짐을 만들 거라서, 일간을 월말 데이터로 맞춰야 함
# 월말(last): 해당 월의 마지막 관측치를 사용
m = raw.resample("ME").last()

# 이미 ffill을 했으니 여기선 추가 ffill을 최소화
# 그래도 월말 값이 비는 경우가 있을 수 있어, 직전값으로만 채우는 정도
m = m.ffill()

# =========================================================
# 4국면(점수 + 상태) 정의
# 상태(state): UP/DOWN 등 해석용
# 점수(score): 연속값(강도) 예측용
# =========================================================

# -------------------------
# (1) 인플레이션 국면
# CPI/PCE YoY(12개월 전 대비) 기반
# -------------------------
m["cpi_yoy"] = pct_change_n(m["소비자물가지수"], 12)
m["pce_yoy"] = pct_change_n(m["개인소비지출물가"], 12)

# 인플레 압력 점수(연속, level)
m["infl_score"] = (m["cpi_yoy"] + m["pce_yoy"]) / 2

# 인플레 모멘텀 (3개월 변화)
m["infl_mom_3m"] = m["infl_score"] - m["infl_score"].shift(3)

# 인플레이션 국면: 모멘텀이 가속되면 UP, 둔화되면 DOWN + 중립 추가(3차 수정)
#4차수정: NaN이 많은 구간에서 eps가 0이 되버리는 문제 해결

mom_abs_med = m["infl_mom_3m"].abs().median()
eps = (0 if pd.isna(mom_abs_med) else mom_abs_med) * 0.2
eps = max(eps, 1e-6)  # eps가 0으로 붕괴하는 걸 방지

m["inflation_regime"] = np.select(
    [m["infl_mom_3m"] > eps, m["infl_mom_3m"] < -eps],
    ["UP", "DOWN"],
    default="NEUTRAL"
)



# -------------------------
# (2) 통화정책 국면
# 기준금리 변화 + 연준대차대조표(QT/QE) 변화
# -------------------------
m["policy_rate_chg_3m"] = m["기준금리"] - m["기준금리"].shift(3)
m["fed_bs_pct_3m"]      = pct_change_n(m["연준대차대조표"], 3)

# 정책 긴축 점수(연속):
# 금리 상승이면 +, 하락이면 -
# 대차대조표 감소(QT)는 긴축이므로 +가 되도록 부호를 반대로
m["policy_score"] = (
    np.sign(m["policy_rate_chg_3m"]).fillna(0)
    + (-np.sign(m["fed_bs_pct_3m"])).fillna(0)
)

# 상태: 점수가 0보다 크면 긴축(TIGHT)
m["policy_regime"] = np.select(
    [m["policy_score"] > 0, m["policy_score"] < 0],
    ["TIGHT", "EASE"],
    default="NEUTRAL"
)



# -------------------------
# (3) 금융스트레스 국면
# 스트레스 관련 지표를 rolling z-score로 표준화 후 평균
# -------------------------
stress_cols = ["금융스트레스지수", "하이일드스프레드", "변동성지수", "TED스프레드", "채권변동성지수", "금융조건지수"]
stress_cols = [c for c in stress_cols if c in m.columns]  # 혹시 빠진 컬럼 대비

if len(stress_cols) == 0:
    raise ValueError("stress_cols가 비어있습니다. 엑셀 컬럼명/지표 수집을 확인하세요.")

WINDOW = 24  # 월데이터 기준 2년 롤링
MINP_STRESS   = 12 # 최소 1년치 데이터 요구

for c in stress_cols:
    m[f"z_{c}"] = zscore_rolling(m[c], window=WINDOW)

#1차 수정: zscore 초반 36개월이 NaN이 많아져서 stress_score빈 공간이 생겼는데 skipna=False로 변경해서 하나라도 NaN이면 stress_score도 NaN되도록 수정.
#2차 수정: 최소 4개 이상 z값이 있어야 stress_score 계산하도록 수정.
#3차 수정: 만약 stress_cols가 3개 이하가 될 수도 있고 그러면 notna().sum<4 조건 때문에 stress_score가 전부 NaN이 될 수 있기 때문에 min_k 변수를 만들어서 stress_cols 개수와 4 중 작은 값을 기준으로 수정.

z_cols = [f"z_{c}" for c in stress_cols]
m["stress_score"] = m[z_cols].mean(axis=1, skipna=True)

min_k = min(3, len(z_cols))  # 기존 4 -> 3 추천 (표본 살리기) 수정 5차
m.loc[m[z_cols].notna().sum(axis=1) < min_k, "stress_score"] = np.nan



# [선택] 2단계: 중립 버리기(0 근처 포함) → HIGH/LOW만 남김 = 스트레스를 총 2가지 단계로만 구분함 0 그리고 0 근삿값 전부 날려버리고 상승 하락 두개로만 분류.
STRESS_EPS = 1e-12  # 0 근처 잡음 제거용(원하면 1e-6도 가능)

# [2단계] stress_regime: HIGH / LOW (NaN은 NaN 유지)
m["stress_regime"] = pd.Series(np.nan, index=m.index, dtype="object")
m.loc[m["stress_score"].notna() & (m["stress_score"] >= 0), "stress_regime"] = "HIGH"
m.loc[m["stress_score"].notna() & (m["stress_score"] <  0), "stress_regime"] = "LOW"

# [선택] 3단계 레짐(LOW/MID/HIGH)도 저장: 표본수 늘리고 해석력 강화용 = 스트레스를 총 3가지 단계로 구분함.
stress_valid = m["stress_score"].dropna()

if len(stress_valid) >= 10:
    q1, q2 = stress_valid.quantile([0.33, 0.66])
    m["stress_regime_3"] = pd.cut(
        m["stress_score"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["LOW", "MID", "HIGH"]
    )
else:
    m["stress_regime_3"] = np.nan


#스트레스 지표 부호 점검 및 진단
ref = m["변동성지수"] if "변동성지수" in m.columns else None
if ref is not None:
    print("\n[DIAG] Stress direction check vs VIX:")
    for c in stress_cols:
        corr = m[c].corr(ref)
        print(f"  {c}: corr={corr:.3f}")

# -------------------------
# (4) 금리 국면
# 10Y 명목금리 + 10Y 실질금리 변화(3개월) 기반
# -------------------------
m["y10_chg_3m"] = m["국채10년금리"] - m["국채10년금리"].shift(3)
m["rr_chg_3m"]  = m["실질금리_10Y"] - m["실질금리_10Y"].shift(3)

m["rate_score"] = np.sign(m["y10_chg_3m"]).fillna(0) + np.sign(m["rr_chg_3m"]).fillna(0)
m["rate_regime"] = np.select(
    [m["rate_score"] > 0, m["rate_score"] < 0],
    ["UP", "DOWN"],
    default="FLAT"
)


# -------------------------
# 4국면 combo 만들기
# -------------------------

# 핵심 score가 모두 계산된 구간만 분석에 사용 (초기 NaN 구간 제거)
needed = ["infl_score", "policy_score", "stress_score", "rate_score"]

# 2단계 콤보는 stress_regime이 있어야만 표본으로 인정(=중립 제거 반영)
m_base = m.dropna(subset=needed + ["stress_regime"]).copy()

# 3단계 stress_regime_3 기반 콤보용
m_s3 = m.dropna(subset=needed + ["stress_regime_3"]).copy()

#4국면 combo 생성
REGIME_COLS = ["inflation_regime", "policy_regime", "stress_regime", "rate_regime"]
m_base["combo"] = m_base[REGIME_COLS].astype(str).agg(" | ".join, axis=1)

# [선택] stress 3단계를 사용한 combo(비교용)
REGIME_COLS_3 = ["inflation_regime", "policy_regime", "stress_regime_3", "rate_regime"]
m_s3["combo_stress3"] = m_s3[REGIME_COLS_3].astype(str).agg(" | ".join, axis=1)



# =========================================================
# S&P500 월말 종가 붙이기 + 3M/6M forward return 라벨 만들기
# =========================================================
# (1) S&P500 일간 다운로드 (로컬 실행 가능)
spx = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)


# MultiIndex 컬럼 평탄화 작업
#    예: ('Close','^GSPC') -> 'Close'
if isinstance(spx.columns, pd.MultiIndex):
    spx.columns = spx.columns.get_level_values(0)

# yfinance 결과 정리: 종가(close)만 사용 + close를 "series"로 변경
spx = spx.rename(columns={"Close": "close"}).copy()
spx = spx.reset_index().rename(columns={"Date": "date"})
spx["date"] = pd.to_datetime(spx["date"])
spx["close"] = pd.to_numeric(spx["close"], errors="coerce")
spx = spx.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

# (2) 300일 MA 추세필터(보조)
spx = spx.sort_values("date")
spx["ma300"] = spx["close"].rolling(window=300, min_periods=300).mean()
spx["trend_ma300"] = np.where(spx["ma300"].isna(), np.nan, (spx["close"] > spx["ma300"]).astype(int))


# (3) 월말로 리샘플
spx_m = spx.set_index("date")[["close", "trend_ma300"]].resample("ME").last().reset_index()

# (4) macro 월말 데이터와 조인 (m_base / m_s3 각각 생성)

# --- base(2단계 stress)용 ---
macro_m_base = m_base.reset_index().copy()
if "날짜" in macro_m_base.columns:
    macro_m_base = macro_m_base.rename(columns={"날짜": "date"})
else:
    macro_m_base = macro_m_base.rename(columns={macro_m_base.columns[0]: "date"})
macro_m_base["date"] = pd.to_datetime(macro_m_base["date"])

df = (
    pd.merge(macro_m_base, spx_m, on="date", how="inner")
    .sort_values("date")
    .reset_index(drop=True)
)

# --- stress3(3단계)용 ---
macro_m_s3 = m_s3.reset_index().copy()
if "날짜" in macro_m_s3.columns:
    macro_m_s3 = macro_m_s3.rename(columns={"날짜": "date"})
else:
    macro_m_s3 = macro_m_s3.rename(columns={macro_m_s3.columns[0]: "date"})
macro_m_s3["date"] = pd.to_datetime(macro_m_s3["date"])

df_s3 = (
    pd.merge(macro_m_s3, spx_m, on="date", how="inner")
    .sort_values("date")
    .reset_index(drop=True)
)

# (5) Forward Return 계산
df["fwd_ret_3m"] = df["close"].shift(-3) / df["close"] - 1
df["fwd_ret_6m"] = df["close"].shift(-6) / df["close"] - 1

# (5-1) Forward Return 계산 (df_s3도 동일하게)
df_s3["fwd_ret_3m"] = df_s3["close"].shift(-3) / df_s3["close"] - 1
df_s3["fwd_ret_6m"] = df_s3["close"].shift(-6) / df_s3["close"] - 1

# (6) 상승 라벨 정의 (핵심 / NaN은 NaN으로 유지)
df["label_up_3m"] = np.where(df["fwd_ret_3m"].isna(), np.nan, (df["fwd_ret_3m"] > 0).astype(int))
df["label_up_6m"] = np.where(df["fwd_ret_6m"].isna(), np.nan, (df["fwd_ret_6m"] > 0).astype(int))

# (6-1) 상승 라벨 정의 (df_s3도 동일하게)
df_s3["label_up_3m"] = np.where(df_s3["fwd_ret_3m"].isna(), np.nan, (df_s3["fwd_ret_3m"] > 0).astype(int))
df_s3["label_up_6m"] = np.where(df_s3["fwd_ret_6m"].isna(), np.nan, (df_s3["fwd_ret_6m"] > 0).astype(int))

# =========================================================
# 콤보별 상승확률 테이블 만들기 (스무딩 + Lift + 표본수)
# =========================================================
def combo_probability_table(
    data: pd.DataFrame,
    label_col: str,
    group_col: str = "combo",   # 어떤 컬럼으로 그룹핑할지
    min_n: int = 8,              # 이건 그냥 처음 코딩할 때 내가 희망하는 값이였는데 이렇게하면 경우의 수 존재 안해서 그냥 min_n_eff로 변경함 ==> 희망값 8로 일단 지정
    alpha: float = 1.0,
    beta: float = 1.0,
    add_reliability: bool = True #신뢰도 등급 컬럼 추가 여부 (TRUE / FALSE 로 조정)
):
    """
    콤보별로 '상승확률' 계산:
    - p_up_raw: 단순 평균
    - p_up_smoothed: 베타-이항 스무딩 확률(표본 작을 때 안정)
    - lift: 전체 평균 대비 얼마나 유리한지
    - ci_lo/ci_hi: 근사 신뢰구간(raw 기준)
    """
    d = data.dropna(subset=[label_col, group_col]).copy()
    if d.empty:
        return pd.DataFrame(columns=["combo", "n", "k_up", "p_up_raw"])

    base_rate = d[label_col].mean()
    
    # 1번. grouby로 g 일단 선언
    g = d.groupby(group_col)[label_col].agg(["count", "sum", "mean"]).rename(
        columns={"count": "n", "sum": "k_up", "mean": "p_up_raw"}
    )

    # 2번. 데이터가 허용하는 최대 표본수 확인
    max_count = int(g["n"].max()) if len(g) else 0

    # 3번. min_n 자동 보정: min_n(희망값)보다 크지 않게 + max_count 넘지 않게 + 최소 3
    min_n_eff = min(min_n, max_count)
    min_n_eff = max(3, min_n_eff)

    # 4번. 여기서만 필터링 (※ min_n으로 다시 필터링하지 말 것)
    g = g[g["n"] >= min_n_eff].copy()

    #5번. 표본이 없으면(필터링으로 전부 날아가면) 빈 테이블 반환
     # 그래도 비면(아주 극단 케이스), 최소 1까지 내려서라도 뽑기
    if g.empty:
        min_n_eff = max(1, max_count)
        g = d.groupby(group_col)[label_col].agg(["count", "sum", "mean"]).rename(
            columns={"count": "n", "sum": "k_up", "mean": "p_up_raw"}
        )
        g = g[g["n"] >= min_n_eff].copy()


    # 벡터화로: 안정 + 빠름 + 에러 방지
    g["p_up_smoothed"] = (g["k_up"].astype(float) + alpha) / (g["n"].astype(float) + alpha + beta)

    #6번. CI
    cis = [proportion_ci_approx(float(p), int(n)) for p, n in zip(g["p_up_raw"], g["n"])]
    g["ci_lo"], g["ci_hi"] = zip(*cis)

    g["base_rate"] = base_rate
    g["lift_smoothed"] = g["p_up_smoothed"] / base_rate if base_rate > 0 else np.nan

    # [추가] 신뢰도 등급: 표본수 기반
    if add_reliability:
        def grade(n):
            if n >= 20:
                return "A (Strong)"
            elif n >= 12:
                return "B (Moderate)"
            else:
                return "C (Weak)"
    
        g["reliability"] = g["n"].astype(int).apply(grade)

    #정렬
    g = g.sort_values(["p_up_smoothed", "n"], ascending=[False, False]).reset_index()
    g = g.rename(columns={group_col: "combo"})
    return g

# (A) 전체 구간 기준
tbl_3m_all = combo_probability_table(df, "label_up_3m", min_n=8)
tbl_6m_all = combo_probability_table(df, "label_up_6m", min_n=8)

# (B) MA300 위(추세상승장)만 따로 비교
df_ma = df[df["trend_ma300"].fillna(0).eq(1)].copy()


tbl_3m_ma = combo_probability_table(df_ma, "label_up_3m", min_n=6)
tbl_6m_ma = combo_probability_table(df_ma, "label_up_6m", min_n=6)

# (C) 스트레스 국면 2단계가 아니라 3단계로 분류한 콤보 확률표

#df_s3 표본이 너무 작을 수 있으니 min_n 조정
min_n_s3 = max(2, int(len(df_s3) * 0.10))  # df_s3의 10%, 최소 2
if df_s3["combo_stress3"].nunique() > 0 and len(df_s3) < 60:
    min_n_s3 = 5



tbl_3m_all_s3 = combo_probability_table(df_s3, "label_up_3m", group_col="combo_stress3", min_n=min_n_s3)
tbl_6m_all_s3 = combo_probability_table(df_s3, "label_up_6m", group_col="combo_stress3", min_n=min_n_s3)


# TOP COMBES index 비어있는 오류 원인 파악을 위해 잠시 사용한 코드========================== 하.. 진짜 힘드네..ㅅㅂ

print("df rows:", len(df))
print("combo nunique:", df["combo"].nunique())
print("top combo counts:\n", df["combo"].value_counts().head(10))
print("min count:", df["combo"].value_counts().min())
print("max count:", df["combo"].value_counts().max())

print("months total:", len(m))
print("stress_score non-null:", m["stress_score"].notna().sum())
print("m_base rows:", len(m_base))
print("m_s3 rows:", len(m_s3))
#=======================================================================

# =========================================================
# 결과 저장
# =========================================================

# 1) 월말 레짐 데이터 저장
OUTPUT_XLSX = "heinrich_regime_outputs.xlsx"

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
    # (1) 월말 레짐 마스터
    m.reset_index().to_excel(w, sheet_name="Macro_Monthly_Regimes", index=False)
    m_base.reset_index().to_excel(w, sheet_name="Macro_Monthly_Base", index=False)
    m_s3.reset_index().to_excel(w, sheet_name="Macro_Monthly_Stress3", index=False)

    # (2) 레짐 + SPX + 라벨까지 붙은 최종 데이터
    df.to_excel(w, sheet_name="MacroPlusSPX_Labels", index=False)
    df_s3.to_excel(w, sheet_name="MacroPlusSPX_Labels_Stress3", index=False)

    # (3) 콤보 확률표 4개
    tbl_3m_all.to_excel(w, sheet_name="3M_All", index=False)
    tbl_3m_ma.to_excel(w, sheet_name="3M_MA300", index=False)
    tbl_6m_all.to_excel(w, sheet_name="6M_All", index=False)
    tbl_6m_ma.to_excel(w, sheet_name="6M_MA300", index=False)

    #(4) Stress 국면을 LOW/MID/HIGH 3단계로 세분화한 경우의 콤보별 상승확률
    tbl_3m_all_s3.to_excel(w, sheet_name="3M_All_Stress3", index=False)
    tbl_6m_all_s3.to_excel(w, sheet_name="6M_All_Stress3", index=False)

    #(5) 상위 10개 콤보 요약
    tbl_3m_all.head(10).to_excel(w, sheet_name="TOP10_3M_All", index=False)
    tbl_6m_all.head(10).to_excel(w, sheet_name="TOP10_6M_All", index=False)
    tbl_3m_all_s3.head(10).to_excel(w, sheet_name="TOP10_3M_Stress3", index=False)
    tbl_6m_all_s3.head(10).to_excel(w, sheet_name="TOP10_6M_Stress3", index=False)



print("Saved Excel:", OUTPUT_XLSX)

# 상위 결과를 콘솔로도 확인
print("\nTop combos (3M, All):")
print(tbl_3m_all.head(10))

print("\nTop combos (6M, All):")
print(tbl_6m_all.head(10))
