import pandas as pd
import numpy as np
import yfinance as yf
from openpyxl import load_workbook
from scipy.stats import beta

# =========================================================
# 0. 설정값
# =========================================================
EXCEL_PATH = "./3.Regime/indicator_data_2006_2024.xlsx"
SHEET_NAME = "MacroData"

# 분석 구간
START, END = "2006-02-01", "2024-12-31"
OUTPUT_XLSX = "heinrich_regime_outputs.xlsx"

# release-aligned 전제면, 분석 단계에서 추가 shift(1) 같은 보정은 하면 안 됨
# (이미 관측가능 시점 기준으로 정렬된 데이터에 또 밀어버리면 룩어헤드/시점이 꼬임)
ASSUME_RELEASE_ALIGNED_DATA = True

# 안전마진: t월 의사결정에서 t-1월 정보만 쓴다고 가정하기 위한 추가 lag
# 구현: 국면/스코어 컬럼만 1개월 shift 해서 row(t)에 (t-1) 정보가 들어가게 함
SAFETY_LAG_MONTHS = 1


# =========================================================
# 1. 유틸 함수
# =========================================================
def pct_change_n(s: pd.Series, n: int) -> pd.Series:
    """
    n기간 변화율: s[t] / s[t-n] - 1

    - 분모가 0이면 폭발하므로 NaN 처리
    - inf/-inf도 NaN 처리
    """
    den = s.shift(n)
    out = s / den - 1
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.mask(den == 0, np.nan)


def zscore_rolling(s: pd.Series, window: int = 24, min_periods: int = 12) -> pd.Series:
    """
    롤링 Z-score 표준화

    목적:
    - 서로 단위/스케일이 다른 스트레스 지표들을 하나의 점수로 평균내기 위해
      동일 스케일(z)로 변환한다.

    주의:
    - rolling std가 0이면 분모가 0이므로 NaN 처리
    """
    mu = s.rolling(window=window, min_periods=min_periods).mean()
    sd = s.rolling(window=window, min_periods=min_periods).std().replace(0, np.nan)
    return (s - mu) / sd


def proportion_ci_jeffreys(k: int, n: int, level: float = 0.95):
    """
    Jeffreys 신뢰구간(이항 비율용 베이지안 CI)

    목적:
    - 표본 수가 적은 조합에서 k/n이 0% 또는 100%로 튀는 문제를 완화
    - 확률 추정값과 함께 불확실성(구간)을 같이 제공하기 위함

    Prior: Beta(0.5, 0.5)
    Posterior: Beta(k+0.5, n-k+0.5)
    """
    if n <= 0:
        return (np.nan, np.nan)

    k = int(k)
    if k < 0 or k > n:
        return (np.nan, np.nan)

    a = k + 0.5
    b = (n - k) + 0.5
    alpha = 1 - level

    lo = beta.ppf(alpha / 2, a, b)
    hi = beta.ppf(1 - alpha / 2, a, b)

    return (float(lo), float(hi))


def beta_binomial_smooth(k: int, n: int, alpha: float = 1.0, beta: float = 1.0):
    """
    베타-이항 스무딩(사후평균)

    목적:
    - 조합별 표본수가 작을 때 '관측비율(k/n)'이 과도하게 출렁이는 것을 완화
    - 기본값 alpha=1, beta=1이면 Laplace smoothing과 유사한 안정화 효과
    """
    return (k + alpha) / (n + alpha + beta)


# =========================================================
# 2. 데이터 로드 & 전처리
# =========================================================
def load_macro_data():
    """
    MacroData 로드 후 월말 기준으로 정렬(resample M last)

    전제:
    - 수집 파이프라인에서 발표시차(LAG_RULES)가 이미 반영되어
      각 시점 row가 "그 시점에 관측 가능했던 값"으로 맞춰져 있다고 가정한다.
    - 따라서 분석 단계에서 추가 shift(1)로 시점을 또 보정하면 안 된다.
    """
    raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    if "날짜" not in raw.columns:
        raise ValueError("엑셀에 '날짜' 컬럼이 없습니다. MacroData 포맷 확인 필요")

    raw["날짜"] = pd.to_datetime(raw["날짜"])
    raw = raw.sort_values("날짜").set_index("날짜").loc[START:END].copy()

    # 월말 관측치로 대표값을 선택 + 결측은 ffill로 이어붙임
    m = raw.resample("M").last().ffill()
    return m


def sanity_check_release_alignment(m: pd.DataFrame):
    """
    룩어헤드/시점 꼬임 방지를 위한 최소 점검

    - 기간, 행 수 확인
    - 핵심 컬럼(국면 계산 필수) 존재 여부 확인
    """
    print("\n[SANITY] Macro monthly range:", m.index.min().date(), "->", m.index.max().date())
    print("[SANITY] Macro monthly rows:", len(m))

    must_cols = ["소비자물가지수", "개인소비지출물가", "기준금리", "연준대차대조표", "국채10년금리", "실질금리_10Y"]
    missing = [c for c in must_cols if c not in m.columns]
    if missing:
        print("[WARN] Missing columns:", missing)
    else:
        print("[SANITY] Required columns OK.")


# =========================================================
# 3. 국면 정의 (인플레/정책/스트레스/금리)
# =========================================================

# (1) 인플레이션 국면
def define_inflation_regime(m):
    """
    인플레이션 국면 로직

    1) CPI YoY, PCE YoY를 구해 평균을 infl_score로 둔다.
    2) infl_mom_3m = infl_score의 3개월 변화량(가속/둔화)으로 방향을 본다.
    3) 변동성 크기에 따라 NEUTRAL 밴드를 만들기 위해
       infl_mom_3m의 절대값 rolling median(36M)을 기반으로 eps를 만든다.

    레짐:
    - infl_mom_3m > +eps  -> UP
    - infl_mom_3m < -eps  -> DOWN
    - 그 사이           -> NEUTRAL
    """
    m["cpi_yoy"] = pct_change_n(m["소비자물가지수"], 12)
    m["pce_yoy"] = pct_change_n(m["개인소비지출물가"], 12)
    m["infl_score"] = (m["cpi_yoy"] + m["pce_yoy"]) / 2

    # '인플레 압력 변화(가속/둔화)'를 보기 위한 3개월 차분
    m["infl_mom_3m"] = m["infl_score"] - m["infl_score"].shift(3)

    # 최근 몇 년간 변화폭의 "전형적인 크기"를 rolling median으로 잡고,
    # 그 일부(0.2배)를 중립 밴드로 사용
    rolling_scale = (
        m["infl_mom_3m"]
        .abs()
        .rolling(window=36, min_periods=18)
        .median()
    )
    eps = rolling_scale * 0.2

    m["inflation_regime"] = np.select(
        [m["infl_mom_3m"] > eps, m["infl_mom_3m"] < -eps],
        ["UP", "DOWN"],
        default="NEUTRAL"
    )
    return m


# (2) 통화정책 국면
def define_policy_regime(m):
    """
    통화정책 국면 로직

    정책을 2개 축으로 점수화:
    - 기준금리 3개월 변화: 인상(+), 인하(-)
    - 연준 대차대조표 3개월 변화: 확대는 완화 방향이므로 score에 (-)로 반영

    policy_score = sign(금리변화) + (-sign(대차대조표 변화))

    레짐:
    - score > 0  -> TIGHT
    - score < 0  -> EASE
    - score = 0  -> NEUTRAL
    """
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
def diagnostics_stress_direction(m, stress_cols, anchor_col="변동성지수", sign_map=None):
    """
    스트레스 지표 방향성 진단(디버깅)

    목적:
    - 여러 스트레스 지표를 평균내려면 "높을수록 스트레스↑" 방향이 동일해야 한다.
    - 앵커(기본 VIX)와 상관이 음(-)인 지표는 방향이 반대일 가능성이 있어
      sign_map으로 뒤집은 뒤, 어떤 지표가 뒤집혔는지 출력한다.
    """
    if anchor_col not in m.columns:
        print(f"\n[DIAG] anchor '{anchor_col}' 컬럼이 없어 방향성 진단/보정을 생략합니다.")
        return

    ref = m[anchor_col]
    print(f"\n[DIAG] Stress direction check vs {anchor_col}:")
    for c in stress_cols:
        corr = m[c].corr(ref)
        flip = ""
        if sign_map is not None and c in sign_map and sign_map[c] == -1:
            flip = "  (FLIPPED)"
        print(f"  {c}: corr={corr:.3f}{flip}")


def define_stress_regime(m, anchor_col="변동성지수"):
    """
    금융 스트레스 합성 국면 로직

    1) 여러 스트레스 지표를 롤링 Z-score로 표준화(스케일 통일)
    2) 앵커(VIX)와 상관이 음(-)이면 지표 방향을 뒤집어 "스트레스↑"로 정렬
    3) 정렬된 z-score들을 평균내 stress_score로 만든다.
    4) 결측이 많으면 평균이 의미 없으므로, 최소 지표 개수(min_k)를 충족할 때만 score 인정
    5) stress_score >= 0: HIGH, < 0: LOW (2분할)
    6) 추가로 33%/66% 분위수로 LOW/MID/HIGH 3분할도 제공(stress_regime_3)
    """
    stress_cols = ["금융스트레스지수", "하이일드스프레드", "변동성지수",
                   "TED스프레드", "채권변동성지수", "금융조건지수"]
    stress_cols = [c for c in stress_cols if c in m.columns]
    if not stress_cols:
        raise ValueError("stress_cols가 비어있습니다. 엑셀 컬럼명 확인 필요")

    # 1) 각 지표를 rolling z-score로 변환
    for c in stress_cols:
        m[f"z_{c}"] = zscore_rolling(m[c], window=24)

    # 2) 방향 정렬: VIX와 음의 상관이면 뒤집기
    sign_map = {c: 1 for c in stress_cols}
    if anchor_col in m.columns and anchor_col in stress_cols:
        ref = m[anchor_col]
        for c in stress_cols:
            if c == anchor_col:
                sign_map[c] = 1
                continue
            corr = m[c].corr(ref)
            # corr가 NaN이면 판단 불가 -> 그대로 둠
            if pd.notna(corr) and corr < 0:
                sign_map[c] = -1

    # 3) sign_map 반영한 z-score 생성
    z_cols_signed = []
    for c in stress_cols:
        z_signed = f"z_{c}_signed"
        m[z_signed] = m[f"z_{c}"] * sign_map[c]
        z_cols_signed.append(z_signed)

    # 4) 합성 스트레스 점수: signed z들의 평균
    m["stress_score"] = m[z_cols_signed].mean(axis=1, skipna=True)

    # 최소 지표 개수(min_k) 조건: 결측이 너무 많으면 score를 NaN 처리
    min_k = min(3, len(z_cols_signed))
    m.loc[m[z_cols_signed].notna().sum(axis=1) < min_k, "stress_score"] = np.nan

    # 5) 2분할(High/Low)
    STRESS_EPS = 1e-12
    m["stress_regime"] = pd.Series(np.nan, index=m.index, dtype="object")
    m.loc[m["stress_score"].notna() & (m["stress_score"] >= STRESS_EPS), "stress_regime"] = "HIGH"
    m.loc[m["stress_score"].notna() & (m["stress_score"] < -STRESS_EPS), "stress_regime"] = "LOW"

    # 6) 3분할(LOW/MID/HIGH): 분위수 컷
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

    # 7) 진단 출력(뒤집힌 지표 확인)
    diagnostics_stress_direction(m, stress_cols, anchor_col=anchor_col, sign_map=sign_map)

    return m


# (4) 금리 국면
def define_rate_regime(m):
    """
    금리 국면 로직

    - 10Y 명목금리 3개월 변화 + 10Y 실질금리 3개월 변화를 합산해 방향성만 본다.
      (둘 다 오르면 UP 쪽, 둘 다 내리면 DOWN 쪽)

    rate_score = sign(명목 10Y 변화) + sign(실질 10Y 변화)

    레짐:
    - score > 0  -> UP
    - score < 0  -> DOWN
    - score == 0 -> FLAT
    """
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
# 4. SPX 처리 (일봉 -> 월말 + 추세필터)
# =========================================================
def load_spx():
    """
    S&P500(^GSPC) 다운로드 후 월말 데이터로 변환한다.

    추가:
    - 300일 이동평균(ma300) 기준 추세필터(trend_ma300)
      close > ma300 이면 1, 아니면 0
      (ma300 계산 불가 구간은 NaN)
    """
    spx = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)

    # yfinance가 MultiIndex 컬럼으로 반환하는 경우 flatten 처리
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)

    spx = spx.rename(columns={"Close": "close"}).reset_index().rename(columns={"Date": "date"})
    spx["date"] = pd.to_datetime(spx["date"])
    spx["close"] = pd.to_numeric(spx["close"], errors="coerce")
    spx = spx.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    # 300일 MA 기반 추세
    spx["ma300"] = spx["close"].rolling(window=300, min_periods=300).mean()
    spx["trend_ma300"] = np.where(spx["ma300"].isna(), np.nan, (spx["close"] > spx["ma300"]).astype(int))

    # 월말 스냅샷으로 변환
    spx_m = spx.set_index("date")[["close", "trend_ma300"]].resample("M").last().reset_index()
    return spx_m


# =========================================================
# 5. 콤보 생성, 병합, 라벨 정의
# =========================================================
def create_combos(m, safety_lag_months: int = 1):
    """
    레짐(문자열)들을 '조합 키(combo)'로 묶어서 그룹별 확률을 계산하기 위한 전처리.

    룩어헤드 안전마진:
    - release-aligned 데이터라도, 실전 의사결정은 보통 '전월 데이터로 이달 포지션'을 잡는 형태가 많음
    - 그래서 국면/스코어 컬럼만 safety_lag_months 만큼 shift해서
      row(t)의 combo가 (t-1)의 정보로 구성되도록 만든다.
    """
    needed = ["infl_score", "policy_score", "stress_score", "rate_score"]

    REGIME_COLS = ["inflation_regime", "policy_regime", "stress_regime", "rate_regime"]
    REGIME_COLS_3 = ["inflation_regime", "policy_regime", "stress_regime_3", "rate_regime"]
    SCORE_COLS = ["infl_score", "policy_score", "stress_score", "rate_score"]

    # Safety margin 적용: derived feature(국면/스코어)만 shift
    m_lag = m.copy()
    lag_cols = list(set(REGIME_COLS + REGIME_COLS_3 + SCORE_COLS))
    m_lag[lag_cols] = m_lag[lag_cols].shift(safety_lag_months)

    # 2단계 스트레스(HIGH/LOW) 기반 combo
    m_base = m_lag.dropna(subset=needed + ["stress_regime"]).copy()
    m_base["combo"] = m_base[REGIME_COLS].astype(str).agg(" | ".join, axis=1)

    # 3단계 스트레스(LOW/MID/HIGH) 기반 combo_stress3
    m_s3 = m_lag.dropna(subset=needed + ["stress_regime_3"]).copy()
    m_s3["combo_stress3"] = m_s3[REGIME_COLS_3].astype(str).agg(" | ".join, axis=1)

    return m_base, m_s3


def merge_macro_spx(m_regime, spx_m):
    """
    매크로(월말)와 SPX(월말)를 date 기준으로 inner-join한다.
    - inner를 쓰는 이유: 양쪽 모두 존재하는 월만 분석 대상으로 잡기 위해
    """
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
    """
    Forward return(미래수익률)과 이진 라벨 생성

    - fwd_ret_3m: 3개월 후 종가 / 현재 종가 - 1
    - fwd_ret_6m: 6개월 후 종가 / 현재 종가 - 1
    - label_up_*: 미래수익률이 0보다 크면 1(상승), 아니면 0(비상승)

    해석:
    - row(t)의 label_up_3m은 "t월말 기준으로 3개월 뒤 수익률이 플러스였는가"를 뜻함
    """
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
    국면 조합(group_col)별로 label_col의 상승확률을 계산한다.

    출력 컬럼:
    - n: 해당 조합의 표본수
    - k_up: 상승(=1) 횟수
    - p_up_raw: 관측비율 k/n
    - p_up_smoothed: 베타-이항 스무딩 확률(표본 적을 때 안정화)
    - ci_lo/ci_hi: Jeffreys 95% 신뢰구간(불확실성)
    - base_rate: 전체 평균 상승확률
    - lift_smoothed: (조합 스무딩 확률) / (전체 base_rate)
    - reliability: 표본수 기반 신뢰도 등급(단순 가이드)

    min_n 처리:
    - 조합 수가 많거나 데이터가 짧으면 min_n을 그대로 쓰면 전부 필터될 수 있음
    - 그래서 min_n_eff를 "현재 가능한 최대 표본수"에 맞춰 자동 조정한다.
    """
    d = data.dropna(subset=[label_col, group_col]).copy()
    if d.empty:
        return pd.DataFrame(columns=["combo", "n", "k_up", "p_up_raw"])

    base_rate = d[label_col].mean()

    g = d.groupby(group_col)[label_col].agg(["count", "sum", "mean"]).rename(
        columns={"count": "n", "sum": "k_up", "mean": "p_up_raw"}
    )

    # 동적 min_n_eff: 너무 과도한 필터링을 방지
    max_count = int(g["n"].max()) if len(g) else 0
    min_n_eff = max(3, min(min_n, max_count))
    g = g[g["n"] >= min_n_eff].copy()

    # 전부 날아가면 완화해서라도 남김(탐색/디버깅 목적)
    if g.empty:
        min_n_eff = max(1, max_count)
        g = d.groupby(group_col)[label_col].agg(["count", "sum", "mean"]).rename(
            columns={"count": "n", "sum": "k_up", "mean": "p_up_raw"}
        )
        g = g[g["n"] >= min_n_eff].copy()

    # 스무딩 확률
    g["p_up_smoothed"] = (g["k_up"].astype(float) + alpha) / (g["n"].astype(float) + alpha + beta)

    # Jeffreys CI
    cis = [proportion_ci_jeffreys(int(k), int(n), level=0.95) for k, n in zip(g["k_up"], g["n"])]
    g["ci_lo"], g["ci_hi"] = zip(*cis)

    # base_rate 대비 lift
    g["base_rate"] = base_rate
    g["lift_smoothed"] = g["p_up_smoothed"] / base_rate if base_rate > 0 else np.nan

    # 표본수 기반 신뢰도(가이드)
    if add_reliability:
        def grade(n):
            if n >= 20:
                return "A (Strong)"
            elif n >= 12:
                return "B (Moderate)"
            else:
                return "C (Weak)"
        g["reliability"] = g["n"].astype(int).apply(grade)

    # 확률 높은 순 + 표본 많은 순으로 정렬
    g = g.sort_values(["p_up_smoothed", "n"], ascending=[False, False]).reset_index()
    g = g.rename(columns={group_col: "combo"})
    return g


# =========================================================
# 7. 결과 저장
# =========================================================
def save_results(m, m_base, m_s3, df, df_s3,
                 tbl_3m_all, tbl_6m_all, tbl_3m_ma, tbl_6m_ma,
                 tbl_3m_all_s3, tbl_6m_all_s3):
    """
    Excel로 결과를 저장한다.

    시트 구성:
    - Macro_Monthly_Full: 매크로 월별 전체(원본 + 파생 + 국면)
    - MacroPlusSPX_Labels: 2단계 스트레스(HIGH/LOW) 기준 병합 + 라벨
    - MacroPlusSPX_Labels_Stress3: 3단계 스트레스(LOW/MID/HIGH) 기준 병합 + 라벨
    - Combos_Summary: 확률표들을 Source_Type으로 구분해서 한 시트에 쌓아둠

    포맷:
    - 날짜 컬럼: yyyy-mm-dd
    - combo 컬럼은 길어서 폭을 넓게(20), 나머지는 15
    """
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        # 원본/라벨 데이터 시트 저장
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

        # 확률표들 요약(쌓기)
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

        final_summary = pd.concat(summary_list, ignore_index=True)
        final_summary.to_excel(w, sheet_name="Combos_Summary", index=False)

    # openpyxl로 간단 포맷(날짜/열너비) 적용
    wb = load_workbook(OUTPUT_XLSX)
    date_style = 'yyyy-mm-dd'

    for ws in wb.worksheets:
        for col in ws.columns:
            header_val = col[0].value

            # combo는 문자열이 길어서 폭을 넓게
            ws.column_dimensions[col[0].column_letter].width = 20 if header_val == "combo" else 15

            # 날짜 컬럼 표시 형식 통일
            if header_val in ['날짜', 'date']:
                for cell in col[1:]:
                    cell.number_format = date_style

    wb.save(OUTPUT_XLSX)
    print("Saved Excel:", OUTPUT_XLSX)


# =========================================================
# 8. 메인 실행부
# =========================================================
def main():
    """
    전체 파이프라인

    1) 매크로 데이터 로드(월말)
    2) 4개 국면 정의
    3) 안전마진 적용 + combo 생성
    4) SPX 월말 로드
    5) 병합 후 forward return 라벨 생성
    6) 조합별 확률표 생성(전체 / MA300 조건 / stress 3분할)
    7) 결과 저장
    """
    # 1) 매크로 데이터 로드
    m = load_macro_data()

    if ASSUME_RELEASE_ALIGNED_DATA:
        sanity_check_release_alignment(m)
        print("[INFO] ASSUME_RELEASE_ALIGNED_DATA=True -> 분석 단계에서 추가 shift(1) 금지")

    # 2) 국면 계산
    m = define_inflation_regime(m)
    m = define_policy_regime(m)
    m = define_stress_regime(m)
    m = define_rate_regime(m)

    # 3) combo 생성(안전마진 포함)
    m_base, m_s3 = create_combos(m, safety_lag_months=SAFETY_LAG_MONTHS)

    # 4) SPX 로드(월말)
    spx_m = load_spx()

    # 5) 매크로+SPX 병합
    df = merge_macro_spx(m_base, spx_m)
    df_s3 = merge_macro_spx(m_s3, spx_m)

    # 6) forward return & 라벨 생성
    df = add_forward_returns(df)
    df_s3 = add_forward_returns(df_s3)

    # 7) combo별 확률표(전체 구간)
    tbl_3m_all = combo_probability_table(df, "label_up_3m", min_n=8)
    tbl_6m_all = combo_probability_table(df, "label_up_6m", min_n=8)

    # 7-1) 추세 필터: SPX가 ma300 위(상승추세)인 구간만
    df_ma = df[df["trend_ma300"].fillna(0).eq(1)].copy()
    tbl_3m_ma = combo_probability_table(df_ma, "label_up_3m", min_n=6)
    tbl_6m_ma = combo_probability_table(df_ma, "label_up_6m", min_n=6)

    # 7-2) stress 3분할 combo_stress3 버전
    # - 데이터 길이에 따라 과도 필터링 방지 위해 min_n을 동적으로 잡음
    min_n_s3 = max(2, int(len(df_s3) * 0.10))
    if df_s3["combo_stress3"].nunique() > 0 and len(df_s3) < 60:
        min_n_s3 = 5

    tbl_3m_all_s3 = combo_probability_table(df_s3, "label_up_3m", group_col="combo_stress3", min_n=min_n_s3)
    tbl_6m_all_s3 = combo_probability_table(df_s3, "label_up_6m", group_col="combo_stress3", min_n=min_n_s3)

    # 8) 결과 저장
    save_results(m, m_base, m_s3, df, df_s3,
                 tbl_3m_all, tbl_6m_all, tbl_3m_ma, tbl_6m_ma,
                 tbl_3m_all_s3, tbl_6m_all_s3)


if __name__ == "__main__":
    main()
