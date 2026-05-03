"""
미국 시총 상위 ~1000개 보통주 유니버스를 생성해 data/tickers_us1000.csv로 저장한다.

실행:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/fetch_universe.py
"""

from __future__ import annotations

import re
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from yfinance import EquityQuery

OUTPUT_PATH = Path("data/tickers_us1000.csv")

GICS_ETF_MAP = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

WIKIPEDIA_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}


def fetch_top1000() -> pd.DataFrame:
    """yfinance screen으로 미국 시총 상위 1000개 수집 (우선주 제외)."""
    print("yfinance screen에서 시총 상위 1000개 수집 중...")
    q = EquityQuery(
        "and",
        [
            EquityQuery("eq", ["region", "us"]),
            EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
        ],
    )
    all_quotes: list[dict] = []
    for offset in [0, 250, 500, 750]:
        result = yf.screen(
            q, sortField="intradaymarketcap", sortAsc=False, size=250, offset=offset
        )
        all_quotes.extend(result.get("quotes", []))

    df = pd.DataFrame(
        [
            {
                "ticker": row["symbol"],
                "name": row.get("shortName", ""),
                "market_cap": row.get("marketCap", 0) or 0,
            }
            for row in all_quotes
        ]
    )

    # 우선주 제외: 티커가 -P 또는 -PA/-PB 등으로 끝나는 종목
    preferred_mask = df["ticker"].str.contains(r"-P[A-Z]?$", regex=True)
    n_before = len(df)
    df = df[~preferred_mask].reset_index(drop=True)
    print(f"  수집: {n_before}개 → 우선주 {preferred_mask.sum()}개 제외 → {len(df)}개")
    return df


def fetch_sector_map() -> pd.DataFrame:
    """Wikipedia S&P 500 + S&P 400에서 ticker → (sector, sector_etf) 매핑 수집."""
    print("Wikipedia에서 S&P 500 / S&P 400 섹터 정보 수집 중...")
    urls = {
        "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "S&P 400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    }
    frames = []
    for label, url in urls.items():
        resp = requests.get(url, headers=WIKIPEDIA_HEADERS, timeout=15)
        resp.raise_for_status()
        table = pd.read_html(StringIO(resp.text))[0]
        sub = table[["Symbol", "GICS Sector"]].rename(
            columns={"Symbol": "ticker", "GICS Sector": "sector"}
        )
        frames.append(sub)
        print(f"  {label}: {len(sub)}개")

    sector_df = pd.concat(frames).drop_duplicates("ticker").reset_index(drop=True)
    # Wikipedia 표기(BRK.B) → yfinance 표기(BRK-B)
    sector_df["ticker"] = sector_df["ticker"].str.replace(".", "-", regex=False)
    sector_df["sector_etf"] = sector_df["sector"].map(GICS_ETF_MAP).fillna("")
    return sector_df


def build_universe(top1000: pd.DataFrame, sector_map: pd.DataFrame) -> pd.DataFrame:
    merged = top1000.merge(sector_map[["ticker", "sector", "sector_etf"]], on="ticker", how="left")
    merged["sector"] = merged["sector"].fillna("Unknown")
    merged["sector_etf"] = merged["sector_etf"].fillna("")

    covered = (merged["sector_etf"] != "").sum()
    print(f"\n섹터 ETF 매핑: {covered}/{len(merged)}개 ({covered/len(merged)*100:.1f}%)")

    sector_dist = merged["sector"].value_counts()
    print("\n섹터 분포:")
    for sector, count in sector_dist.items():
        etf = GICS_ETF_MAP.get(sector, "-")
        print(f"  {sector:30s} {count:4d}개  {etf}")

    return merged[["ticker", "name", "sector", "sector_etf", "market_cap"]]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    top1000 = fetch_top1000()
    sector_map = fetch_sector_map()
    universe = build_universe(top1000, sector_map)

    universe.to_csv(OUTPUT_PATH, index=False)
    print(f"\n저장 완료: {OUTPUT_PATH}  ({len(universe)}개 종목)")


if __name__ == "__main__":
    main()
