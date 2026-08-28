"""Top pick(진입 검토 중 3~5 집중 선별) 규칙 변형의 포인트인타임 검증.

선행 조건: validate_verdicts.py + validate_earnings.py 실행 산출물
(earnings_validation_rows.csv — 후보 전체의 판정·점수·signal_type·어닝스 D-n·fwd_excess_20d).

변형 (모두 진입 검토 그룹에서 선별, 주간 스냅샷별 동일가중 바스켓):
  A: 점수 top-N (베이스라인)
  B: A + 상관 클러스터당 1종목
  C: B + 매집(acc/surge+acc)×어닝스 임박 우선 배치
  D: B + surge만×어닝스 임박 최대 1종목
  E: C + D
  F: B + 임박 전체 우선 배치 (참고용)

실행: UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_top_picks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import BENCHMARKS, ScreenerConfig, correlation_clusters, download_prices, load_universe  # noqa: E402

OUTPUT_DIR = Path("outputs/backtest")
IMMINENT_DAYS = 20  # 어닝스 임박 기준 (calendar days — 검증 버킷과 동일)


def nw_tstat(x: pd.Series, lags: int = 4) -> float:
    x = x.dropna().to_numpy()
    n = len(x)
    if n < lags + 3:
        return float("nan")
    xc = x - x.mean()
    var = (xc @ xc) / n
    for l in range(1, lags + 1):
        var += 2 * (1 - l / (lags + 1)) * ((xc[l:] @ xc[:-l]) / n)
    return float(x.mean() / np.sqrt(var / n))


def select_picks(g: pd.DataFrame, cluster_of: dict[str, int], variant: str, top_n: int) -> pd.DataFrame:
    g = g.copy()
    g["acc_imminent"] = g["signal_type"].isin(["acc", "surge+acc"]) & g["imminent"]
    g["surge_imminent"] = (g["signal_type"] == "surge") & g["imminent"]

    if variant in ("C", "E"):
        g["_tier"] = np.where(g["acc_imminent"], 0, 1)
    elif variant == "F":
        g["_tier"] = np.where(g["imminent"], 0, 1)
    else:
        g["_tier"] = 0
    g = g.sort_values(["_tier", "score"], ascending=[True, False])

    picks = []
    used_clusters: set[int] = set()
    surge_imm_count = 0
    for _, row in g.iterrows():
        if variant != "A":
            cid = cluster_of.get(row["ticker"])
            if cid is not None and cid in used_clusters:
                continue
        if variant in ("D", "E") and row["surge_imminent"] and surge_imm_count >= 1:
            continue
        picks.append(row)
        if variant != "A":
            cid = cluster_of.get(row["ticker"])
            if cid is not None:
                used_clusters.add(cid)
        if row["surge_imminent"]:
            surge_imm_count += 1
        if len(picks) >= top_n:
            break
    return pd.DataFrame(picks)


def main() -> None:
    rows_path = OUTPUT_DIR / "earnings_validation_rows.csv"
    if not rows_path.exists():
        sys.exit("earnings_validation_rows.csv 없음 — validate_earnings.py 먼저 실행")
    df = pd.read_csv(rows_path)
    df["imminent"] = df["days_to_earnings"].notna() & (df["days_to_earnings"] <= IMMINENT_DAYS)
    print(f"rows: {len(df)}행, 스냅샷 {df['snapshot_date'].nunique()}개")

    config = ScreenerConfig(period="2y")
    universe = load_universe(Path("data/tickers_us1000.csv"))
    symbols = sorted(set(universe["ticker"]) | set(BENCHMARKS))
    prices, report = download_prices(symbols, config)
    print(f"가격 로드 (cache hit: {report.from_cache})")

    variants = ["A", "B", "C", "D", "E", "F"]
    results: dict[int, dict[str, dict[str, list]]] = {
        n: {v: {"dates": [], "returns": []} for v in variants} for n in (3, 5)
    }

    snaps = sorted(df["snapshot_date"].unique())
    for i, date in enumerate(snaps, 1):
        snap = df[df["snapshot_date"] == date]
        jv = snap[(snap["verdict"] == "진입 검토") & snap["fwd_excess_20d"].notna()]
        if len(jv) < 5:
            continue
        ts = pd.Timestamp(date)
        truncated = {t: prices[t].loc[:ts] for t in snap["ticker"] if t in prices}
        clusters = correlation_clusters(snap["ticker"].tolist(), prices=truncated, config=config)
        cluster_of: dict[str, int] = {}
        for members in {id(v): v for v in clusters.values()}.values():
            cid = len(cluster_of) + 1000
            for t in members:
                cluster_of[t] = cid
        for top_n in (3, 5):
            for v in variants:
                picks = select_picks(jv, cluster_of, v, top_n)
                if len(picks):
                    results[top_n][v]["dates"].append(date)
                    results[top_n][v]["returns"].append(picks["fwd_excess_20d"].mean())
        print(f"\r진행 {i}/{len(snaps)}", end="", flush=True)
    print()

    labels = {
        "A": "점수 top-N",
        "B": "A + 클러스터당 1종목",
        "C": "B + 매집×임박 우선",
        "D": "B + surge×임박 ≤1",
        "E": "C + D",
        "F": "B + 임박 전체 우선 (참고)",
    }
    for top_n in (3, 5):
        print(f"\n=== top-{top_n} 바스켓 (스냅샷별 동일가중, 20d SPY 초과수익) ===")
        base = pd.Series(results[top_n]["A"]["returns"], index=results[top_n]["A"]["dates"])
        rows_out = []
        for v in variants:
            r = pd.Series(results[top_n][v]["returns"], index=results[top_n][v]["dates"])
            diff = (r - base).dropna()
            half = len(r) // 2
            rows_out.append({
                "변형": f"{v}: {labels[v]}",
                "mean%": r.mean() * 100,
                "win%": (r > 0).mean() * 100,
                "worst%": r.min() * 100,
                "전반%": r.iloc[:half].mean() * 100,
                "후반%": r.iloc[half:].mean() * 100,
                "vsA%p": diff.mean() * 100,
                "NW t": nw_tstat(diff) if v != "A" else np.nan,
                "n_snap": len(r),
            })
        table = pd.DataFrame(rows_out).set_index("변형")
        print(table.round(2).to_string())
        table.to_csv(OUTPUT_DIR / f"summary_top_picks_{top_n}.csv")
    print(f"\n저장: {OUTPUT_DIR}/summary_top_picks_3.csv, summary_top_picks_5.csv")


if __name__ == "__main__":
    main()
