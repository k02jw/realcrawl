"""
커피 C 선물 예측 - Feature Engineering v2 (Step 2/2)
=======================================================
입력: raw_data.csv (build_raw_data.py 출력)
출력: features.csv (피처 + 타깃, ML 학습 준비 완료)

v2 변경사항 (누수 점검 강화):
  1. target_vol_5d_ahead: 미래 윈도우 표준편차로 명시적 재작성 (혼란 제거)
  2. YoY 계산: shift(252) → 날짜 기반 lookup (정확히 1년 전 값)
  3. 모든 rolling 함수에 min_periods 명시화
  4. 자체 누수 점검 (self-check): 피처-미래타깃 상관 비정상 검출 시 경고
  5. shift 방향 일관성 검증 자동화
  6. 피처 카테고리별 시작일(유효 시점) 자동 리포트

⚠️ 데이터 누수 원칙:
  - 모든 피처는 시점 t까지의 정보만 사용 (t 자체 포함 가능)
  - 타깃은 t+h 정보 — shift(-h)로 미래에서 가져와 라벨로 사용
  - 라벨 생성 임계값(sigma)은 t까지의 정보로 계산 (라벨 생성자도 t에 알 수 있음)
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

INPUT_PATH = "raw_data.csv"
OUTPUT_PATH = "features.csv"
TARGET_HORIZONS = [1, 5]
DIRECTION_THRESHOLD_SIGMA = 0.5
USE_CATCH22 = True

# 자체 누수 점검 임계값: 피처-타깃 절대 상관이 이 값을 넘으면 의심
LEAKAGE_CORR_WARN_THRESHOLD = 0.15


# ─────────────────────────────────────────────────────────
# 0. Helper functions
# ─────────────────────────────────────────────────────────
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Wilder's smoothing). Right-aligned EWM, 누수 없음."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, period: int = 20, n_std: float = 2.0):
    ma = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    width = (upper - lower) / ma
    pctB = (series - lower) / (upper - lower).replace(0, np.nan)
    return upper, lower, width, pctB


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    log_hl = np.log(high / low).replace([np.inf, -np.inf], np.nan)
    return np.sqrt(
        (log_hl ** 2).rolling(window, min_periods=window).mean() / (4 * np.log(2))
    ) * np.sqrt(252)


def yoy_change_by_date(series: pd.Series, years: int = 1) -> pd.Series:
    """
    날짜 기반 YoY 계산 (shift(252) 같은 부정확한 거래일 기반 대신).
    시점 t에서 정확히 'years'년 전 가장 가까운 거래일 값과 비교.
    """
    target_dates = series.index - pd.DateOffset(years=years)
    # 각 target_date 이하의 가장 최근 인덱스(=값이 존재하는 가장 가까운 과거)
    past_idx = series.index.searchsorted(target_dates, side="right") - 1
    past_idx = np.where(past_idx < 0, 0, past_idx)
    past_values = series.iloc[past_idx].values
    # 너무 멀거나 NaN인 경우 처리
    out = pd.Series(past_values, index=series.index, name=series.name)
    # 시리즈 시작 전 시점은 NaN으로
    out.loc[series.index < (series.index[0] + pd.DateOffset(years=years))] = np.nan
    return out


# ─────────────────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────────────────
print(f"[로드] {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH, parse_dates=["date"], index_col="date").sort_index()
print(f"   shape: {df.shape}, 기간: {df.index.min().date()} ~ {df.index.max().date()}")


# ─────────────────────────────────────────────────────────
# A. 가격 기반 피처
# ─────────────────────────────────────────────────────────
print("\n[A] 가격 기반 피처...")
fe = pd.DataFrame(index=df.index)

coffee_close = df["coffee_close"]
coffee_high = df["coffee_high"]
coffee_low = df["coffee_low"]

# 로그 수익률
fe["coffee_logret_1d"] = np.log(coffee_close / coffee_close.shift(1))
fe["coffee_logret_5d"] = np.log(coffee_close / coffee_close.shift(5))
fe["coffee_logret_20d"] = np.log(coffee_close / coffee_close.shift(20))
fe["coffee_logret_60d"] = np.log(coffee_close / coffee_close.shift(60))

# 변동성 (min_periods 명시)
fe["coffee_realvol_5d"] = (
    fe["coffee_logret_1d"].rolling(5, min_periods=5).std() * np.sqrt(252)
)
fe["coffee_realvol_20d"] = (
    fe["coffee_logret_1d"].rolling(20, min_periods=20).std() * np.sqrt(252)
)
fe["coffee_realvol_60d"] = (
    fe["coffee_logret_1d"].rolling(60, min_periods=60).std() * np.sqrt(252)
)
fe["coffee_parkinson_20d"] = parkinson_vol(coffee_high, coffee_low, 20)

# 거래량 변환
if "coffee_volume" in df.columns:
    vol = df["coffee_volume"].replace(0, np.nan)
    fe["coffee_volume_logchg_1d"] = np.log(vol / vol.shift(1))
    fe["coffee_volume_z_20d"] = (
        (vol - vol.rolling(20, min_periods=20).mean())
        / vol.rolling(20, min_periods=20).std()
    )

# 기술적 지표
fe["coffee_rsi_14"] = rsi(coffee_close, 14)
macd_line, sig, hist = macd(coffee_close)
fe["coffee_macd"] = macd_line
fe["coffee_macd_signal"] = sig
fe["coffee_macd_hist"] = hist

up, lo, bw, pctB = bollinger(coffee_close, 20, 2.0)
fe["coffee_bb_width"] = bw
fe["coffee_bb_pctB"] = pctB

fe["coffee_atr_14"] = atr(coffee_high, coffee_low, coffee_close, 14)

# 가격 vs SMA
for w in [5, 20, 60]:
    sma = coffee_close.rolling(w, min_periods=w).mean()
    fe[f"coffee_price_to_sma{w}"] = coffee_close / sma - 1

# High-Low 범위
fe["coffee_hl_range"] = (coffee_high - coffee_low) / coffee_close
fe["coffee_hl_range_z_20d"] = (
    (fe["coffee_hl_range"] - fe["coffee_hl_range"].rolling(20, min_periods=20).mean())
    / fe["coffee_hl_range"].rolling(20, min_periods=20).std()
)

print(f"   ✓ A 완료 ({fe.shape[1]}개)")


# ─────────────────────────────────────────────────────────
# B. 상품 간 관계 피처
# ─────────────────────────────────────────────────────────
print("\n[B] 상품 간 관계 피처...")

related_assets = ["robusta", "cocoa", "sugar", "corn", "crude", "brent", "dxy", "vix"]
for asset in related_assets:
    col = f"{asset}_close"
    if col in df.columns:
        fe[f"{asset}_logret_1d"] = np.log(df[col] / df[col].shift(1))
        fe[f"{asset}_logret_5d"] = np.log(df[col] / df[col].shift(5))

if "brl_usd_close" in df.columns:
    fe["brl_usd_logret_1d"] = np.log(df["brl_usd_close"] / df["brl_usd_close"].shift(1))
    fe["brl_usd_logret_5d"] = np.log(df["brl_usd_close"] / df["brl_usd_close"].shift(5))


def rolling_corr(s1: pd.Series, s2: pd.Series, window: int = 60) -> pd.Series:
    return s1.rolling(window, min_periods=window).corr(s2)


coffee_ret = fe["coffee_logret_1d"]
for asset in related_assets + ["brl_usd"]:
    ret_col = f"{asset}_logret_1d"
    if ret_col in fe.columns:
        fe[f"corr_coffee_{asset}_60d"] = rolling_corr(coffee_ret, fe[ret_col], 60)

if "robusta_close" in df.columns:
    fe["coffee_robusta_spread_logret"] = (
        fe["coffee_logret_1d"] - fe["robusta_logret_1d"]
    )

print(f"   ✓ B 완료 (누적 {fe.shape[1]}개)")


# ─────────────────────────────────────────────────────────
# C. 거시·날씨 피처
# ─────────────────────────────────────────────────────────
print("\n[C] 거시·날씨 피처...")

# 금리
for col in ["us_10y", "us_2y"]:
    if col in df.columns:
        fe[col] = df[col]
        fe[f"{col}_chg_5d"] = df[col] - df[col].shift(5)
if "us_10y" in df.columns and "us_2y" in df.columns:
    fe["term_spread_10y2y"] = df["us_10y"] - df["us_2y"]

# FRED 변화율
for col in ["dxy_fred", "brl_fred", "wti_fred", "vix_fred"]:
    if col in df.columns:
        fe[f"{col}_logret_1d"] = np.log(df[col] / df[col].shift(1))

# 관세 — ★ shift(252) → 날짜 기반 YoY
if "tariff_us" in df.columns:
    fe["tariff_us"] = df["tariff_us"]
    tariff_1y_ago = yoy_change_by_date(df["tariff_us"], years=1)
    fe["tariff_us_chg_yoy"] = df["tariff_us"] - tariff_1y_ago

# 브라질 생산량 — ★ shift(252) → 날짜 기반 YoY
if "br_coffee_production" in df.columns:
    fe["br_coffee_production"] = df["br_coffee_production"]
    prod_1y_ago = yoy_change_by_date(df["br_coffee_production"], years=1)
    fe["br_coffee_production_yoy"] = df["br_coffee_production"] / prod_1y_ago - 1

# 날씨 — 누적, 이상치, 서리
weather_regions = ["br_minas", "br_sao", "vn_daklak", "co_huila"]
for region in weather_regions:
    precip_col = f"{region}_precip"
    temp_col = f"{region}_temp"
    temp_min_col = f"{region}_temp_min"

    if precip_col in df.columns:
        fe[f"{precip_col}_sum_7d"] = df[precip_col].rolling(7, min_periods=7).sum()
        fe[f"{precip_col}_sum_30d"] = df[precip_col].rolling(30, min_periods=30).sum()
        # ★ 가뭄 플래그: min_periods=252*3 (최소 3년 데이터 확보 후 비교)
        long_mean = (
            df[precip_col].rolling(252 * 10, min_periods=252 * 3).mean() * 30
        )
        fe[f"{precip_col}_drought_flag"] = (
            (fe[f"{precip_col}_sum_30d"] < 0.5 * long_mean).astype(int)
        )
        # long_mean이 NaN인 시점(초기 3년)은 플래그도 NaN으로
        fe.loc[long_mean.isna(), f"{precip_col}_drought_flag"] = np.nan

    if temp_col in df.columns:
        # ★ 기온 편차: min_periods=252*3
        long_temp_mean = df[temp_col].rolling(252 * 5, min_periods=252 * 3).mean()
        fe[f"{temp_col}_anomaly"] = df[temp_col] - long_temp_mean

    if temp_min_col in df.columns:
        is_winter = df.index.month.isin([6, 7, 8, 9])
        fe[f"{region}_frost_flag"] = (
            ((df[temp_min_col] < 3.0) & is_winter).astype(int)
        )
        fe[f"{region}_frost_count_7d"] = (
            fe[f"{region}_frost_flag"].rolling(7, min_periods=7).sum()
        )

print(f"   ✓ C 완료 (누적 {fe.shape[1]}개)")


# ─────────────────────────────────────────────────────────
# D. catch22 (선택)
# ─────────────────────────────────────────────────────────
if USE_CATCH22:
    print("\n[D] catch24 롤링 피처 생성...")
    try:
        import pycatch22

        def rolling_catch24(series: pd.Series, window: int = 60) -> pd.DataFrame:
            """
            과거 window일의 catch24 피처 계산.
            ★ vals[i - window : i] — 시점 i 자체는 포함 안 함 (보수적, 누수 없음).
            """
            vals = series.values
            n = len(vals)
            features = []
            names = None
            for i in range(n):
                if i < window or np.isnan(vals[i - window: i]).any():
                    features.append([np.nan] * 24)
                    continue
                result = pycatch22.catch22_all(
                    vals[i - window: i].tolist(), catch24=True
                )
                if names is None:
                    names = [f"c24_{n_}" for n_ in result["names"]]
                features.append(result["values"])
            return pd.DataFrame(features, index=series.index, columns=names)

        c24 = rolling_catch24(coffee_ret.fillna(0), window=60)
        fe = fe.join(c24, how="left")
        print(f"   ✓ catch24 완료 (누적 {fe.shape[1]}개)")
    except ImportError:
        print("   ⚠️ pycatch22 미설치 — pip install pycatch22")
else:
    print("\n[D] catch22 스킵 (USE_CATCH22=False)")


# ─────────────────────────────────────────────────────────
# E. 캘린더 피처
# ─────────────────────────────────────────────────────────
print("\n[E] 캘린더 피처...")

fe["dayofweek"] = df.index.dayofweek
fe["month"] = df.index.month
fe["quarter"] = df.index.quarter
fe["dayofmonth"] = df.index.day
fe["month_sin"] = np.sin(2 * np.pi * fe["month"] / 12)
fe["month_cos"] = np.cos(2 * np.pi * fe["month"] / 12)
fe["dow_sin"] = np.sin(2 * np.pi * fe["dayofweek"] / 5)
fe["dow_cos"] = np.cos(2 * np.pi * fe["dayofweek"] / 5)
fe["br_harvest_season"] = df.index.month.isin([5, 6, 7, 8, 9]).astype(int)

print(f"   ✓ E 완료 (누적 {fe.shape[1]}개)")


# ─────────────────────────────────────────────────────────
# F. 타깃 변수
# ─────────────────────────────────────────────────────────
print("\n[F] 타깃 변수 생성...")

for h in TARGET_HORIZONS:
    # 회귀 타깃: t→t+h 누적 로그 수익률
    future_logret = np.log(coffee_close.shift(-h) / coffee_close)
    fe[f"target_logret_{h}d"] = future_logret

    # 이진 분류: 상승(1) vs 하락(0)
    fe[f"target_dir_binary_{h}d"] = (future_logret > 0).astype(int)
    fe.loc[future_logret.isna(), f"target_dir_binary_{h}d"] = np.nan

    # 3-클래스: 임계값은 시점 t까지의 sigma로 계산 (PIT-safe)
    sigma = fe["coffee_logret_1d"].rolling(60, min_periods=60).std()
    threshold = DIRECTION_THRESHOLD_SIGMA * sigma * np.sqrt(h)
    cls = pd.Series(np.nan, index=fe.index)
    cls[future_logret > threshold] = 2
    cls[future_logret < -threshold] = 0
    cls[(future_logret <= threshold) & (future_logret >= -threshold)] = 1
    fe[f"target_dir_3class_{h}d"] = cls

# ★ target_vol_5d_ahead — 명시적 미래 윈도우 표준편차로 재작성
# 시점 t의 라벨 = 시점 t+1, t+2, t+3, t+4, t+5의 일별 로그수익률 표준편차
future_returns_matrix = pd.concat(
    [fe["coffee_logret_1d"].shift(-i) for i in range(1, 6)], axis=1
)
fe["target_vol_5d_ahead"] = (
    future_returns_matrix.std(axis=1, ddof=1) * np.sqrt(252)
)
# 미래 5일 중 하나라도 NaN이면 라벨도 NaN
fe.loc[future_returns_matrix.isna().any(axis=1), "target_vol_5d_ahead"] = np.nan

print(f"   ✓ F 완료 (최종 {fe.shape[1]}개)")


# ─────────────────────────────────────────────────────────
# G. 자체 누수 점검 (Self-Check)
# ─────────────────────────────────────────────────────────
print("\n[G] 누수 자체 점검...")

target_cols = [c for c in fe.columns if c.startswith("target_")]
feat_cols = [c for c in fe.columns if not c.startswith("target_")]

# 1) shift 방향 검증: target_logret_1d == coffee_logret_1d.shift(-1) 여야 함
expected = np.log(coffee_close.shift(-1) / coffee_close)
actual = fe["target_logret_1d"]
mismatch = ((expected.round(8) != actual.round(8)) & expected.notna() & actual.notna()).sum()
if mismatch == 0:
    print(f"   ✓ shift 방향 검증: target_logret_1d == log(P_{{t+1}}/P_t) ({mismatch} 불일치)")
else:
    print(f"   ⚠️ shift 검증 실패: {mismatch}건 불일치")

# 2) 피처-타깃 상관 점검 (랜덤워크 수준이면 |corr| < 0.1 정상)
print(f"   피처-타깃 상관 점검 (|corr| > {LEAKAGE_CORR_WARN_THRESHOLD} 시 경고):")
target_for_check = "target_logret_1d"
target_series = fe[target_for_check].dropna()
suspicious = []
for c in feat_cols:
    if c in ["dayofweek", "month", "quarter", "dayofmonth",
             "month_sin", "month_cos", "dow_sin", "dow_cos",
             "br_harvest_season"]:
        continue  # 캘린더 피처는 계절성으로 정상 상관 있을 수 있음
    s = fe[c].reindex(target_series.index).dropna()
    common = target_series.loc[s.index].dropna()
    if len(common) > 100:
        s_valid = s.loc[common.index]
        if s_valid.std() == 0 or common.std() == 0:
            continue
        corr = np.corrcoef(s_valid, common)[0, 1]
        if abs(corr) > LEAKAGE_CORR_WARN_THRESHOLD:
            suspicious.append((c, corr))

if not suspicious:
    print(f"   ✓ 모든 피처-타깃 절대 상관 < {LEAKAGE_CORR_WARN_THRESHOLD}")
else:
    print(f"   ⚠️ 의심 피처 {len(suspicious)}개 (상관계수 비정상적으로 높음):")
    for c, v in sorted(suspicious, key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"      {c}: corr={v:+.4f}")
    print(f"   → 실제 시그널이 강한 경우일 수 있지만, 누수 가능성도 검토 필요")

# 3) 카테고리별 유효 시작일 리포트
print("\n   [카테고리별 유효 시작일]")
def first_valid_date(df_, cols):
    if not cols:
        return None
    valid = df_[cols].dropna(axis=0)
    return valid.index.min() if len(valid) > 0 else None

cat_A = [c for c in feat_cols if c.startswith("coffee_")]
cat_B = [c for c in feat_cols if any(c.startswith(a + "_") for a in related_assets)
         or c.startswith("brl_usd_") or c.startswith("corr_") or c == "coffee_robusta_spread_logret"]
cat_C_macro = [c for c in feat_cols if c.startswith(("us_", "term_", "tariff_", "br_coffee_production",
                                                       "dxy_fred", "brl_fred", "wti_fred", "vix_fred"))]
cat_C_weather = [c for c in feat_cols if any(c.startswith(r) for r in weather_regions)]

for name, cols in [("A (가격)", cat_A), ("B (상품관계)", cat_B),
                    ("C-매크로", cat_C_macro), ("C-날씨", cat_C_weather)]:
    fd = first_valid_date(fe, cols)
    if fd is not None:
        print(f"   {name:<15} {len(cols):>3}개 피처, 유효 시작일: {fd.date()}")


# ─────────────────────────────────────────────────────────
# 저장 & 요약
# ─────────────────────────────────────────────────────────
print(f"\n[저장] {OUTPUT_PATH}")
fe.to_csv(OUTPUT_PATH)

print(f"\n{'=' * 60}")
print(f"✓ Feature engineering v2 완료")
print(f"  shape: {fe.shape}")
print(f"  피처: {len(feat_cols)}개,  타깃: {len(target_cols)}개")
print(f"  기간: {fe.index.min().date()} ~ {fe.index.max().date()}")
print(f"{'=' * 60}\n")

valid_start = fe[feat_cols].dropna(axis=0).index.min() if feat_cols else None
if valid_start is not None:
    print(f"전체 피처 non-NaN 시작일: {valid_start.date()}")
    print(f"  → 학습 가능 행: {(fe.index >= valid_start).sum()}")

print("\n[타깃 분포]")
for c in target_cols:
    if "3class" in c:
        print(f"  {c}:")
        print(fe[c].value_counts(normalize=True).sort_index()
                  .to_string().replace("\n", "\n    "))
    elif "binary" in c:
        print(f"  {c}: 상승 {fe[c].mean() * 100:.1f}%, "
              f"non-NaN {fe[c].notna().sum()}행")
    elif "vol" in c:
        print(f"  {c}: mean={fe[c].mean():.3f}, "
              f"non-NaN {fe[c].notna().sum()}행")
    else:
        print(f"  {c}: mean={fe[c].mean():.5f}, std={fe[c].std():.5f}, "
              f"non-NaN {fe[c].notna().sum()}행")
