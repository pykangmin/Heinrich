import pandas as pd
import numpy as np
from math import sqrt
import yfinance as yf
from openpyxl import load_workbook

# =========================================================
# 0. 설정값
# =========================================================
EXCEL_PATH = "./3.Regime/new_indicator_data_2020_2024.xlsx"
SHEET_NAME = "MacroData"
START, END = "2020-01-01", "2024-12-31"
OUTPUT_XLSX = "heinrich_regime_outputs.xlsx"


# =========================================================
# 1. 유틸 함수
# =========================================================
def pct_change_n(s: pd.Series, n: int) -> pd.Series:
    den = s.shift(n)
    out = s / den - 1
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.mask(den == 0, np.nan)

def zscore_rolling(s: pd.Series, window: int = 24, min_periods: int = 12) -> pd.Series:
    mu = s.rolling(window=window, min_periods=min_periods).mean()
    sd = s.rolling(window=window, min_periods=min_periods).std().replace(0, np.nan)
    return (s - mu) / sd

def proportion_ci_approx(p: float, n: int, z: float = 1.96):
    if n <= 0 or np.isnan(p):
        return (np.nan, np.nan)
    se = sqrt(max(p * (1 - p), 1e-12) / n)
    return (max(0.0, p - z * se), min(1.0, p + z * se))

def beta_binomial_smooth(k: int, n: int, alpha: float = 1.0, beta: float = 1.0):
    return (k + alpha) / (n + alpha + beta)


# =========================================================
# 2. 데이터 로드 & 전처리
# =========================================================
def load_macro_data():
    raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    raw["날짜"] = pd.to_datetime(raw["날짜"])
    raw = raw.sort_values("날짜").set_index("날짜").loc[START:END].copy()
    m = raw.resample("M").last().ffill()
    return m


# =========================================================
# 3. 국면 정의
# =========================================================

# (1) 인플레이션 국면
def define_inflation_regime(m):
    m["cpi_yoy"] = pct_change_n(m["소비자물가지수"], 12)
    m["pce_yoy"] = pct_change_n(m["개인소비지출물가"], 12)
    m["infl_score"] = (m["cpi_yoy"] + m["pce_yoy"]) / 2
    m["infl_mom_3m"] = m["infl_score"] - m["infl_score"].shift(3)

    mom_abs_med = m["infl_mom_3m"].abs().median()
    eps = max((0 if pd.isna(mom_abs_med) else mom_abs_med) * 0.2, 1e-6)

    m["inflation_regime"] = np.select(
        [m["infl_mom_3m"] > eps, m["infl_mom_3m"] < -eps],
        ["UP", "DOWN"],
        default="NEUTRAL"
    )
    return m


# (2) 통화정책 국면
def define_policy_regime(m):
    m["policy_rate_chg_3m"] = m["기준금리"] - m["기준금리"].shift(3)
    m["fed_bs_pct_3m"] = pct_change_n(m["연준대차대조표"], 3)
    m["policy_score"] = (
        np.sign(m["policy_rate_chg_3m"]).fillna(0)
        + (-np.sign(m["fed_bs_pct_3m"])).fillna(0)
    )
    m["policy_regime"] = np.select(
        [m["policy_score"] > 0, m["policy_score"] < 0],
        ["TIGHT", "EASE"],
        default="NEUTRAL"
    )
    return m

# (3) 금융스트레스 국면
#스트레스 지표 부호 점검 및 진단
def diagnostics_stress_direction(m, stress_cols):
    """스트레스 지표 방향성 진단: VIX와의 상관관계 출력"""
    if "변동성지수" not in m.columns:
        return
    ref = m["변동성지수"]
    print("\n[DIAG] Stress direction check vs VIX:")
    for c in stress_cols:
        corr = m[c].corr(ref)
        print(f"  {c}: corr={corr:.3f}")

# 금융스트레스 지표 기반 국면 정의
def define_stress_regime(m):
    stress_cols = ["금융스트레스지수", "하이일드스프레드", "변동성지수", 
                   "TED스프레드", "채권변동성지수", "금융조건지수"]
    stress_cols = [c for c in stress_cols if c in m.columns]
    if not stress_cols:
        raise ValueError("stress_cols가 비어있습니다. 엑셀 컬럼명 확인 필요")

    for c in stress_cols:
        m[f"z_{c}"] = zscore_rolling(m[c], window=24)

    z_cols = [f"z_{c}" for c in stress_cols]
    m["stress_score"] = m[z_cols].mean(axis=1, skipna=True)

    min_k = min(3, len(z_cols))
    m.loc[m[z_cols].notna().sum(axis=1) < min_k, "stress_score"] = np.nan

    STRESS_EPS = 1e-12
    m["stress_regime"] = pd.Series(np.nan, index=m.index, dtype="object")
    m.loc[m["stress_score"].notna() & (m["stress_score"] >= STRESS_EPS), "stress_regime"] = "HIGH"
    m.loc[m["stress_score"].notna() & (m["stress_score"] < -STRESS_EPS), "stress_regime"] = "LOW"

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

    diagnostics_stress_direction(m, stress_cols)

    return m


# (4) 금리 국면
def define_rate_regime(m):
    m["y10_chg_3m"] = m["국채10년금리"] - m["국채10년금리"].shift(3)
    m["rr_chg_3m"] = m["실질금리_10Y"] - m["실질금리_10Y"].shift(3)
    m["rate_score"] = np.sign(m["y10_chg_3m"]).fillna(0) + np.sign(m["rr_chg_3m"]).fillna(0)
    m["rate_regime"] = np.select(
        [m["rate_score"] > 0, m["rate_score"] < 0],
        ["UP", "DOWN"],
        default="FLAT"
    )
    return m


# =========================================================
# 4. SPX 처리
# =========================================================
def load_spx():
    spx = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)

    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)

    spx = spx.rename(columns={"Close": "close"}).reset_index().rename(columns={"Date": "date"})
    spx["date"] = pd.to_datetime(spx["date"])
    spx["close"] = pd.to_numeric(spx["close"], errors="coerce")
    spx = spx.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    spx["ma300"] = spx["close"].rolling(window=300, min_periods=300).mean()
    spx["trend_ma300"] = np.where(spx["ma300"].isna(), np.nan, (spx["close"] > spx["ma300"]).astype(int))

    spx_m = spx.set_index("date")[["close", "trend_ma300"]].resample("M").last().reset_index()
    return spx_m


# =========================================================
# 5. 콤보 생성, 병합, 라벨 정의
# =========================================================
def create_combos(m):
    needed = ["infl_score", "policy_score", "stress_score", "rate_score"]

    # 2단계 stress regime 기반
    m_base = m.dropna(subset=needed + ["stress_regime"]).copy()
    REGIME_COLS = ["inflation_regime", "policy_regime", "stress_regime", "rate_regime"]
    m_base["combo"] = m_base[REGIME_COLS].astype(str).agg(" | ".join, axis=1)

    # 3단계 stress regime 기반
    m_s3 = m.dropna(subset=needed + ["stress_regime_3"]).copy()
    REGIME_COLS_3 = ["inflation_regime", "policy_regime", "stress_regime_3", "rate_regime"]
    m_s3["combo_stress3"] = m_s3[REGIME_COLS_3].astype(str).agg(" | ".join, axis=1)

    return m_base, m_s3


def merge_macro_spx(m_regime, spx_m):
    macro_m = m_regime.reset_index().copy()
    if "날짜" in macro_m.columns:
        macro_m = macro_m.rename(columns={"날짜": "date"})
    else:
        macro_m = macro_m.rename(columns={macro_m.columns[0]: "date"})
    macro_m["date"] = pd.to_datetime(macro_m["date"])

    df = (
        pd.merge(macro_m, spx_m, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df


def add_forward_returns(df):
    df["fwd_ret_3m"] = df["close"].shift(-3) / df["close"] - 1
    df["fwd_ret_6m"] = df["close"].shift(-6) / df["close"] - 1

    df["label_up_3m"] = np.where(df["fwd_ret_3m"].isna(), np.nan, (df["fwd_ret_3m"] > 0).astype(int))
    df["label_up_6m"] = np.where(df["fwd_ret_6m"].isna(), np.nan, (df["fwd_ret_6m"] > 0).astype(int))
    return df

# =========================================================
# 6. 콤보별 확률표 생성
# =========================================================
def combo_probability_table(
    data: pd.DataFrame,
    label_col: str,
    group_col: str = "combo",
    min_n: int = 8,
    alpha: float = 1.0,
    beta: float = 1.0,
    add_reliability: bool = True
):
    """
    콤보별 상승확률 계산:
    - p_up_raw: 단순 평균
    - p_up_smoothed: 베타-이항 스무딩 확률
    - lift: 전체 평균 대비 상대적 유리함
    - ci_lo/ci_hi: 근사 신뢰구간
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
    min_n_eff = max(3, min(min_n, max_count))

    # 4번. 여기서만 필터링 (※ min_n으로 다시 필터링하지 말 것)
    g = g[g["n"] >= min_n_eff].copy()

    #5번. 표본이 없으면(필터링으로 전부 날아가면) 빈 테이블 반환
    if g.empty:
        min_n_eff = max(1, max_count)
        g = d.groupby(group_col)[label_col].agg(["count", "sum", "mean"]).rename(
            columns={"count": "n", "sum": "k_up", "mean": "p_up_raw"}
        )
        g = g[g["n"] >= min_n_eff].copy()

    # 벡터화로
    g["p_up_smoothed"] = (g["k_up"].astype(float) + alpha) / (g["n"].astype(float) + alpha + beta)

    #6번. Confidence Interval(신뢰구간)
    cis = [proportion_ci_approx(float(p), int(n)) for p, n in zip(g["p_up_raw"], g["n"])]
    g["ci_lo"], g["ci_hi"] = zip(*cis)

    g["base_rate"] = base_rate
    g["lift_smoothed"] = g["p_up_smoothed"] / base_rate if base_rate > 0 else np.nan

    if add_reliability:
        def grade(n):
            if n >= 20:
                return "A (Strong)"
            elif n >= 12:
                return "B (Moderate)"
            else:
                return "C (Weak)"
        g["reliability"] = g["n"].astype(int).apply(grade)

    g = g.sort_values(["p_up_smoothed", "n"], ascending=[False, False]).reset_index()
    g = g.rename(columns={group_col: "combo"})
    return g


# =========================================================
# 7. 결과 저장
# =========================================================
def save_results(m, m_base, m_s3, df, df_s3,
                 tbl_3m_all, tbl_6m_all, tbl_3m_ma, tbl_6m_ma,
                 tbl_3m_all_s3, tbl_6m_all_s3):

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        for data, name in [
            (m.reset_index(), "Macro_Monthly_Full"),
            (df, "MacroPlusSPX_Labels"),
            (df_s3, "MacroPlusSPX_Labels_Stress3")
        ]:
            temp_df = data.copy()
            date_col = '날짜' if '날짜' in temp_df.columns else 'date'
            if date_col in temp_df.columns:
                temp_df[date_col] = pd.to_datetime(temp_df[date_col])
            temp_df.to_excel(w, sheet_name=name, index=False)

        summary_list = []
        
        raw_tables = [
            ("3M_All", tbl_3m_all),
            ("6M_All", tbl_6m_all),
            ("3M_MA300", tbl_3m_ma),
            ("6M_MA300", tbl_6m_ma),
            ("3M_Stress3", tbl_3m_all_s3),
            ("6M_Stress3", tbl_6m_all_s3)
        ]

        for label, table in raw_tables:
            t_copy = table.copy()
            t_copy.insert(0, "Source_Type", label) 
            summary_list.append(t_copy)

        # 모든 테이블을 하나로 합침
        final_summary = pd.concat(summary_list, ignore_index=True)
        final_summary.to_excel(w, sheet_name="Combos_Summary", index=False)

    wb = load_workbook(OUTPUT_XLSX)
    date_style = 'yyyy-mm-dd'
    
    for ws in wb.worksheets:
        for col in ws.columns:
            # 헤더에 따라 열 너비 최적화
            header_val = col[0].value
            ws.column_dimensions[col[0].column_letter].width = 20 if header_val == "combo" else 15
            
            # 날짜 서식 적용
            if header_val in ['날짜', 'date']:
                for cell in col[1:]:
                    cell.number_format = date_style

    wb.save(OUTPUT_XLSX)

    print("Saved Excel:", OUTPUT_XLSX)
    print("\nTop combos (3M, All):")
    print(tbl_3m_all.head(10))
    print("\nTop combos (6M, All):")
    print(tbl_6m_all.head(10))


# =========================================================
# 8. 메인 실행부
# =========================================================
def main():
    # 1) 매크로 데이터 로드 및 국면 정의
    m = load_macro_data()
    m = define_inflation_regime(m)
    m = define_policy_regime(m)
    m = define_stress_regime(m)
    m = define_rate_regime(m)

    # 2) 콤보 생성
    m_base, m_s3 = create_combos(m)

    # 3) S&P500 데이터 로드
    spx_m = load_spx()

    # 4) Macro + SPX 병합
    df = merge_macro_spx(m_base, spx_m)
    df_s3 = merge_macro_spx(m_s3, spx_m)

    # 5) Forward Return + 라벨 정의
    df = add_forward_returns(df)
    df_s3 = add_forward_returns(df_s3)

    # 6) 콤보별 확률표 생성
    tbl_3m_all = combo_probability_table(df, "label_up_3m", min_n=8)
    tbl_6m_all = combo_probability_table(df, "label_up_6m", min_n=8)

    df_ma = df[df["trend_ma300"].fillna(0).eq(1)].copy()
    tbl_3m_ma = combo_probability_table(df_ma, "label_up_3m", min_n=6)
    tbl_6m_ma = combo_probability_table(df_ma, "label_up_6m", min_n=6)

    min_n_s3 = max(2, int(len(df_s3) * 0.10))
    if df_s3["combo_stress3"].nunique() > 0 and len(df_s3) < 60:
        min_n_s3 = 5
    tbl_3m_all_s3 = combo_probability_table(df_s3, "label_up_3m", group_col="combo_stress3", min_n=min_n_s3)
    tbl_6m_all_s3 = combo_probability_table(df_s3, "label_up_6m", group_col="combo_stress3", min_n=min_n_s3)

    # 7) 결과 저장
    save_results(m, m_base, m_s3, df, df_s3,
                 tbl_3m_all, tbl_6m_all, tbl_3m_ma, tbl_6m_ma,
                 tbl_3m_all_s3, tbl_6m_all_s3)


if __name__ == "__main__":
    main()
