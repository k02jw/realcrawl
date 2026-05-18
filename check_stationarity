"""
피처 정상성 자동 점검 (Step 3)
==================================
입력: features.csv (build_features.py 출력)
출력: stationarity_report.csv (피처별 검정 결과)

ADF (Augmented Dickey-Fuller) 검정:
  H0: 단위근 존재 (비정상)
  H1: 정상
  → p-value < 0.05 이면 정상

KPSS (Kwiatkowski-Phillips-Schmidt-Shin) 검정:
  H0: 정상
  H1: 비정상
  → p-value < 0.05 이면 비정상 (ADF와 가설 방향 반대)

두 검정을 함께 쓰는 이유:
  - ADF는 약한 정상성도 통과시킬 수 있음
  - KPSS는 추세 정상성도 비정상으로 판정할 수 있음
  - 두 검정 모두 통과해야 진짜 정상이라 판단 (Lopez de Prado 권고)

판정 결과:
  - STATIONARY        : ADF reject (정상) + KPSS not-reject (정상) → 안전
  - TREND_STATIONARY  : ADF reject + KPSS reject → 트렌드 제거 후 정상
  - UNIT_ROOT         : ADF not-reject + KPSS reject → 차분 필요
  - INCONCLUSIVE      : ADF not-reject + KPSS not-reject → 데이터 부족
  - SKIPPED           : 결측 너무 많거나 상수
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

INPUT_PATH = "features.csv"
OUTPUT_PATH = "stationarity_report.csv"
SIGNIFICANCE = 0.05      # 유의수준
MIN_VALID_OBS = 500      # 검정에 필요한 최소 non-NaN 행 수
MAX_LAGS = "AIC"         # ADF 자동 lag 선택 ("AIC", "BIC", or 정수)

# 점검에서 제외할 컬럼 (타깃, 캘린더 변수)
EXCLUDE_PATTERNS = ("target_", "dayofweek", "month", "quarter", "dayofmonth",
                    "month_sin", "month_cos", "dow_sin", "dow_cos",
                    "br_harvest_season", "_traded")


# ─────────────────────────────────────────────────────────
# 검정 함수
# ─────────────────────────────────────────────────────────
def adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller 검정.
    H0: 단위근 존재 (비정상). p < 0.05 이면 H0 기각 = 정상.
    """
    try:
        result = adfuller(series.dropna(), autolag=MAX_LAGS, regression="c")
        return {
            "adf_stat": result[0],
            "adf_pvalue": result[1],
            "adf_lags": result[2],
            "adf_nobs": result[3],
            "adf_reject_h0": result[1] < SIGNIFICANCE,
        }
    except Exception as e:
        return {"adf_stat": np.nan, "adf_pvalue": np.nan,
                "adf_lags": np.nan, "adf_nobs": np.nan,
                "adf_reject_h0": np.nan, "adf_error": str(e)}


def kpss_test(series: pd.Series) -> dict:
    """
    KPSS 검정.
    H0: 정상. p < 0.05 이면 H0 기각 = 비정상.
    regression='c' (level stationarity). 트렌드 정상성 보려면 'ct'.
    """
    try:
        result = kpss(series.dropna(), regression="c", nlags="auto")
        return {
            "kpss_stat": result[0],
            "kpss_pvalue": result[1],
            "kpss_lags": result[2],
            "kpss_reject_h0": result[1] < SIGNIFICANCE,
        }
    except Exception as e:
        return {"kpss_stat": np.nan, "kpss_pvalue": np.nan,
                "kpss_lags": np.nan, "kpss_reject_h0": np.nan,
                "kpss_error": str(e)}


def classify(adf_reject: bool, kpss_reject: bool) -> str:
    """ADF + KPSS 결과 조합으로 시계열 유형 판정."""
    if pd.isna(adf_reject) or pd.isna(kpss_reject):
        return "SKIPPED"
    if adf_reject and not kpss_reject:
        return "STATIONARY"            # 둘 다 정상 결론 → 안전
    if adf_reject and kpss_reject:
        return "TREND_STATIONARY"      # 추세 제거 후 정상 가능
    if not adf_reject and kpss_reject:
        return "UNIT_ROOT"             # 비정상 (단위근) → 차분 필요
    return "INCONCLUSIVE"              # 결정 불가 (데이터 부족 시사)


def suggest_action(verdict: str, col_name: str) -> str:
    """판정 결과에 따른 권장 조치."""
    if verdict == "STATIONARY":
        return "✓ 그대로 사용 가능"
    if verdict == "TREND_STATIONARY":
        return "추세 제거 (디트렌딩) 또는 1차 차분 권장"
    if verdict == "UNIT_ROOT":
        # 이미 변환된 피처면 더 차분, 수준 변수면 차분/로그수익률화
        if any(k in col_name for k in ["logret", "chg_", "yoy", "z_", "logchg",
                                          "anomaly", "spread", "rsi", "bb_", "macd",
                                          "atr", "to_sma"]):
            return "⚠️ 이미 변환됐는데도 비정상 — 추가 차분 또는 fractional diff 검토"
        return "⚠️ 차분(.diff()) 또는 로그수익률 변환 필요"
    if verdict == "INCONCLUSIVE":
        return "데이터 부족 — 더 긴 기간 수집 후 재검정"
    return "검정 불가"


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────
print(f"[로드] {INPUT_PATH}")
fe = pd.read_csv(INPUT_PATH, parse_dates=["date"], index_col="date").sort_index()
print(f"   shape: {fe.shape}")

# 점검 대상 컬럼 선정
feat_cols = [c for c in fe.columns if not c.startswith(EXCLUDE_PATTERNS)]
print(f"   점검 대상: {len(feat_cols)}개 컬럼\n")

print(f"[검정 시작]")
print(f"   ADF H0: 단위근(비정상) | p<{SIGNIFICANCE} → 정상")
print(f"   KPSS H0: 정상 | p<{SIGNIFICANCE} → 비정상")
print(f"   최소 관측치: {MIN_VALID_OBS}")
print()

records = []
for i, col in enumerate(feat_cols, 1):
    s = fe[col].dropna()

    # 사전 필터링
    if len(s) < MIN_VALID_OBS:
        records.append({
            "feature": col, "n_obs": len(s),
            "verdict": "SKIPPED", "reason": "결측 너무 많음",
        })
        continue
    if s.nunique() <= 1:
        records.append({
            "feature": col, "n_obs": len(s),
            "verdict": "SKIPPED", "reason": "상수 또는 거의 상수",
        })
        continue
    if s.std() < 1e-10:
        records.append({
            "feature": col, "n_obs": len(s),
            "verdict": "SKIPPED", "reason": "분산 사실상 0",
        })
        continue

    # 무한값/극단 결측 정리
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < MIN_VALID_OBS:
        records.append({
            "feature": col, "n_obs": len(s),
            "verdict": "SKIPPED", "reason": "유효 데이터 부족",
        })
        continue

    rec = {"feature": col, "n_obs": len(s)}
    rec.update(adf_test(s))
    rec.update(kpss_test(s))
    rec["verdict"] = classify(rec.get("adf_reject_h0"),
                              rec.get("kpss_reject_h0"))
    rec["action"] = suggest_action(rec["verdict"], col)
    records.append(rec)

    if i % 20 == 0:
        print(f"   진행: {i}/{len(feat_cols)}")

print(f"   진행: {len(feat_cols)}/{len(feat_cols)} 완료\n")

# ─────────────────────────────────────────────────────────
# 결과 정리 및 저장
# ─────────────────────────────────────────────────────────
report = pd.DataFrame(records)

# 컬럼 순서 정리
preferred_cols = ["feature", "verdict", "action", "n_obs",
                  "adf_pvalue", "adf_reject_h0", "adf_stat", "adf_lags",
                  "kpss_pvalue", "kpss_reject_h0", "kpss_stat", "kpss_lags",
                  "reason"]
report = report[[c for c in preferred_cols if c in report.columns]]
report.to_csv(OUTPUT_PATH, index=False)

# ─────────────────────────────────────────────────────────
# 요약 출력
# ─────────────────────────────────────────────────────────
print("=" * 70)
print("정상성 점검 결과 요약")
print("=" * 70)

verdict_counts = report["verdict"].value_counts()
total = len(report)
for v in ["STATIONARY", "TREND_STATIONARY", "UNIT_ROOT", "INCONCLUSIVE", "SKIPPED"]:
    n = verdict_counts.get(v, 0)
    pct = n / total * 100 if total > 0 else 0
    bar = "█" * int(pct / 2)
    print(f"  {v:<18} {n:>3}개 ({pct:>5.1f}%) {bar}")

print()

# ─────────────────────────────────────────────────────────
# 카테고리별 상세 출력
# ─────────────────────────────────────────────────────────
print("=" * 70)
print("판정별 피처 목록")
print("=" * 70)

for verdict in ["UNIT_ROOT", "TREND_STATIONARY", "INCONCLUSIVE", "SKIPPED"]:
    subset = report[report["verdict"] == verdict]
    if len(subset) == 0:
        continue
    print(f"\n▼ {verdict} ({len(subset)}개)")
    for _, row in subset.iterrows():
        adf_p = row.get("adf_pvalue", np.nan)
        kpss_p = row.get("kpss_pvalue", np.nan)
        action = row.get("action", "")
        if not pd.isna(adf_p):
            print(f"   {row['feature']:<40} "
                  f"ADF p={adf_p:.4f}  KPSS p={kpss_p:.4f}")
            print(f"     → {action}")
        else:
            print(f"   {row['feature']:<40} {row.get('reason', '')}")

# STATIONARY는 너무 많아질 수 있어서 개수만 출력
stat_count = len(report[report["verdict"] == "STATIONARY"])
if stat_count > 0:
    print(f"\n▼ STATIONARY ({stat_count}개) — 그대로 사용 가능, 목록 생략")
    print(f"   상세는 {OUTPUT_PATH} 참고")

print()
print("=" * 70)
print(f"전체 리포트 저장: {OUTPUT_PATH}")
print("=" * 70)

# ─────────────────────────────────────────────────────────
# 선형/로지스틱 회귀용 안전 피처 셋 추출
# ─────────────────────────────────────────────────────────
safe_features = report[report["verdict"] == "STATIONARY"]["feature"].tolist()
print(f"\n[선형/로지스틱 회귀에 안전한 피처 셋]")
print(f"   총 {len(safe_features)}개")
print(f"   사용법:")
print(f"     safe_features = pd.read_csv('{OUTPUT_PATH}')")
print(f"     safe_features = safe_features[safe_features['verdict']=='STATIONARY']['feature'].tolist()")
print(f"     X = fe[safe_features]")
