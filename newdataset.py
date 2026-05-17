"""
커피 선물 가격 방향성 예측 - 데이터셋 빌더
실행: python build_dataset.py
출력: coffee_futures_dataset.csv
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import time
from datetime import datetime

START = "2000-01-01"
END   = datetime.today().strftime("%Y-%m-%d")

print("=" * 55)
print("커피 선물 예측 데이터셋 빌더")
print("=" * 55)

# ── 1. 가격 데이터 (yfinance) ──────────────────────────────
print("\n[1/5] 가격 데이터 수집 중 (yfinance)...")
import yfinance as yf

TICKERS = {
    "coffee"  : "KC=F", #커피 선물 지수 2000.1~
    "brl_usd" : "BRL=X", #브라질-달러 환율 2003.12~
    "dxy"     : "DX-Y.NYB", #미국 달러 인덱스 - 미국 달러의 가치
    "crude"   : "CL=F", #원유 가격 지수 2000.08.23~
    "brent"   : "BZ=F", #브렌트유 가격 지수 2007.07.30~
    "cocoa"   : "CC=F", #코코아 (대체품) 지수
    "sugar"   : "SB=F", #설탕 (대체품) 지수
}

raw = yf.download(
    list(TICKERS.values()),
    start=START, end=END,
    interval="1d",
    auto_adjust=True,
    progress=False,
)

METRICS = ["Open", "High", "Low", "Close"]
ticker_list = list(TICKERS.values())
name_list   = list(TICKERS.keys())

frames = []
for metric in METRICS:
    df = raw[metric][ticker_list].copy()
    df.columns = [f"{name}_{metric.lower()}" for name in name_list]
    frames.append(df)

price_df = pd.concat(frames, axis=1).ffill()

print(f"   완료: {len(price_df)}행, {price_df.index[0].date()} ~ {price_df.index[-1].date()}")
print(price_df.head())

print("\n[2/5] 매크로 지표 수집 중 (FRED)...")
from fredapi import Fred

FRED_API_KEY = "f7ec6f22ff835fe3ec704ad3f8e86be6"
FRED_SERIES = {
    "us_10y": "DGS10",
}

try:
    fred = Fred(api_key=FRED_API_KEY)
    daily_index = pd.date_range(start=START, end=END, freq="B")  # 영업일 기준 일간 인덱스
    fred_frames = []
    for name, sid in FRED_SERIES.items():
        try:
            s = fred.get_series(sid, observation_start=START)
            s.name = name
            s.index = pd.to_datetime(s.index)
            # 일간 인덱스로 reindex 후 ffill (월간 데이터는 발표일 값을 다음 발표일까지 유지)
            s = s.reindex(daily_index).ffill()
            fred_frames.append(s)
            print(f"   {name} ({sid}): {len(s)}행")
        except Exception as e:
            print(f"   {name} ({sid}) 실패: {e}")
    if fred_frames:
        fred_df = pd.concat(fred_frames, axis=1)
        print(f"   완료: {fred_df.shape[1]}개 시리즈")
    else:
        fred_df = pd.DataFrame()
        print("   FRED 수집 실패 - 건너뜀")
except Exception as e:
    fred_df = pd.DataFrame()
    print(f"   FRED 초기화 실패 (API 키 확인): {e}")

# ── 3. 브라질 커피 생산량 (USDA PSD, WASDE 발표일 기준) ──
# 커피 마케팅연도(Oct-Sep) 첫 추정치는 해당 연도 5월 WASDE에 최초 공개.
# WASDE 실제 발표일을 anchor로 삼아 데이터 유출 방지.
print("\n[3/5] 브라질 커피 생산량 수집 중 (USDA PSD, WASDE 발표일 기준)...")

def fetch_brazil_coffee_production():
    """
    USDA PSD Bulk CSV 기준 발표 주기:
      - Month=0  : 1960~2001, 연 1회 (Calendar_Year 10월 기준)
      - Month=6  : 2002~,     6월 발표 (첫 추정치)
      - Month=12 : 일부 연도, 12월 발표 (최종 확정치)
    Calendar_Year + Month을 실제 발표일로 사용해 데이터 유출 방지.
    """
    url = "https://apps.fas.usda.gov/psdonline/downloads/psd_alldata_csv.zip"
    try:
        print("   USDA PSD Bulk CSV 다운로드 중 (수십 MB)...")
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)

        brazil = raw[
            raw["Commodity_Description"].astype(str).str.upper().str.contains("COFFEE") &
            (raw["Country_Name"] == "Brazil") &
            (raw["Attribute_Description"] == "Production")
        ].copy()

        if brazil.empty:
            print("   브라질 커피 생산량 데이터 없음")
            return pd.DataFrame()

        brazil["_cal_year"] = pd.to_numeric(brazil["Calendar_Year"], errors="coerce")
        brazil["_month"]    = pd.to_numeric(brazil["Month"],         errors="coerce")
        brazil["_value"]    = pd.to_numeric(brazil["Value"],         errors="coerce")

        # 발표일 계산
        # Month=0(연간): 해당 Calendar_Year 10월 1일 (커피 마케팅연도 시작)
        # Month=6/12   : Calendar_Year의 해당 월 1일
        def release_date(row):
            cal_yr = int(row["_cal_year"])
            mo     = int(row["_month"])
            if mo == 0:
                return pd.Timestamp(cal_yr, 10, 1)
            return pd.Timestamp(cal_yr, mo, 1)

        brazil["_release"] = brazil.apply(release_date, axis=1)
        brazil = brazil.dropna(subset=["_value", "_release"])

        # 동일 발표일에 여러 행이 있으면 합산 (연도별 중복 방지)
        anchors = (brazil.groupby("_release")["_value"]
                         .sum()
                         .rename("br_coffee_production")
                         .sort_index())

        daily_index = pd.date_range(start=START, end=END, freq="D")
        result = anchors.reindex(daily_index).ffill()
        result.index.name = "date"

        mo_counts = brazil["_month"].value_counts().sort_index().to_dict()
        print(f"   완료: {len(anchors)}개 발표 시점 → {len(result)}일")
        print(f"   발표 주기: Month=0(연간) {mo_counts.get(0,0)}건, "
              f"Month=6(6월) {mo_counts.get(6,0)}건, "
              f"Month=12(12월) {mo_counts.get(12,0)}건")
        return result.to_frame()
    except Exception as e:
        print(f"   브라질 커피 생산량 수집 실패: {e}")
        return pd.DataFrame()

brazil_prod_df = fetch_brazil_coffee_production()

# ── 4. 브라질 날씨 (Open-Meteo ERA5, 일 단위) ─────────────
print("\n[4/5] 브라질 날씨 수집 중 (Open-Meteo ERA5, 일 단위)...")

# 브라질 주요 커피 산지 두 곳의 일별 강수량·기온
BRAZIL_REGIONS = {
    "br_minas": (-18.5, -44.0),   # 미나스제라이스 (세하두)
    "br_sao"  : (-20.5, -47.5),   # 상파울루 (모지아나)
}

def fetch_weather_daily(lat, lon, label, retries=3):
    base = "https://archive-api.open-meteo.com/v1/archive"
    s_year = int(START[:4])
    e_year = int(END[:4])
    all_frames = []
    for y in range(s_year, e_year + 1, 5):
        chunk_start = f"{y}-01-01"
        chunk_end   = f"{min(y + 4, e_year)}-12-31"
        if chunk_end > END:
            chunk_end = END
        params = {
            "latitude"  : lat,
            "longitude" : lon,
            "start_date": chunk_start,
            "end_date"  : chunk_end,
            "daily"     : "precipitation_sum,temperature_2m_mean",
            "timezone"  : "UTC",
        }
        for attempt in range(retries):
            try:
                r = requests.get(base, params=params, timeout=120)
                r.raise_for_status()
                data = r.json()
                daily = data.get("daily")
                if daily and "time" in daily:
                    chunk_df = pd.DataFrame({
                        "date"              : pd.to_datetime(daily["time"]),
                        f"{label}_precip_d" : daily.get("precipitation_sum"),
                        f"{label}_temp_d"   : daily.get("temperature_2m_mean"),
                    }).set_index("date")
                    all_frames.append(chunk_df)
                break
            except requests.exceptions.Timeout:
                if attempt < retries - 1:
                    print(f"   [{label}] 타임아웃, 재시도 {attempt+2}/{retries}...")
                    time.sleep(5)
            except Exception as e:
                print(f"   [{label}] {chunk_start}~{chunk_end} 실패: {e}")
                break
    if not all_frames:
        return pd.DataFrame()
    merged = pd.concat(all_frames)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged

weather_frames = []
for label, (lat, lon) in BRAZIL_REGIONS.items():
    wdf = fetch_weather_daily(lat, lon, label)
    if not wdf.empty:
        weather_frames.append(wdf)
        print(f"   {label}: {len(wdf)}일 ({wdf.index[0].date()} ~ {wdf.index[-1].date()})")
    else:
        print(f"   {label}: 수집 실패")

weather_daily_df = pd.concat(weather_frames, axis=1) if weather_frames else pd.DataFrame()

# ── 5. 미국 관세율 (World Bank, 발표 lag 반영) ───────────
# World Bank는 연도 Y의 관세 데이터를 약 Y+1년 10월에 공개 (WTO 집계 후 게재).
# 연도 Y 값을 Y+1년 10월 1일에 배치해 데이터 유출 방지.
print("\n[5/5] 미국 관세율 수집 중 (World Bank, 발표 lag +10개월 적용)...")

def fetch_tariff_daily():
    url = "https://api.worldbank.org/v2/country/USA/indicator/TM.TAX.MRCH.WM.AR.ZS"
    params = {"format": "json", "per_page": 100,
              "date": f"{int(START[:4])}:{int(END[:4])}"}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if len(payload) < 2 or not payload[1]:
            return pd.DataFrame()
        records = [(int(item["date"]), item["value"])
                   for item in payload[1] if item["value"] is not None]
        if not records:
            return pd.DataFrame()
        annual = (pd.DataFrame(records, columns=["year", "value"])
                    .set_index("year").sort_index())

        daily_index = pd.date_range(start=START, end=END, freq="D")
        # 연도 Y 데이터 → Y+1년 10월 1일에 배치 (World Bank 실제 공개 시점)
        anchor_rows = [
            {"date": pd.Timestamp(int(yr) + 1, 10, 1), "tariff_us": row["value"]}
            for yr, row in annual.iterrows()
        ]
        anchor_df = pd.DataFrame(anchor_rows).set_index("date").sort_index()
        result = anchor_df.reindex(daily_index).ffill()
        result.index.name = "date"
        print(f"   완료: {len(annual)}개 연도 → {len(result)}일 (Y+1년 10월 공개 기준)")
        return result
    except Exception as e:
        print(f"   관세율 수집 실패: {e}")
        return pd.DataFrame()

tariff_daily_df = fetch_tariff_daily()

# ── CSV 저장 ──────────────────────────────────────────────
base_df = price_df.join(fred_df, how="left") if not fred_df.empty else price_df

extra_frames = [
    brazil_prod_df  if not brazil_prod_df.empty  else None,
    weather_daily_df if not weather_daily_df.empty else None,
    tariff_daily_df if not tariff_daily_df.empty else None,
]
result_df = base_df.copy()
for f in extra_frames:
    if f is not None:
        result_df = result_df.join(f, how="left")

OUTPUT = "price_df.csv"
result_df.to_csv(OUTPUT)
print(f"\n저장 완료: {OUTPUT} ({result_df.shape[0]}행 x {result_df.shape[1]}열)")
print("\n[컬럼 목록]")
for col in result_df.columns:
    null_pct = result_df[col].isna().mean() * 100
    print(f"  {col:<35} (결측 {null_pct:.1f}%)")
