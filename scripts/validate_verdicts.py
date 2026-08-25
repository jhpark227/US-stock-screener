"""판정 규칙(assign_verdicts) 포인트인타임 검증.

백테스트 rows(스냅샷별 후보 + 20d 초과수익)에 판정 규칙을 재적용해
진입 검토/대기/스킵 그룹의 실제 포워드 성과를 비교한다.

- 상관 클러스터는 스냅샷 날짜까지의 가격만 사용 (룩어헤드 방지).
- 어닝스(W1)는 과거 캘린더가 없어 검증 불가 — 해당 규칙만 비활성 상태로 평가됨을 명시.

실행: UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_verdicts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import ScreenerConfig, assign_verdicts, download_prices, load_universe, BENCHMARKS  # noqa: E402

OUTPUT_DIR = Path("outputs/backtest")


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


def main() -> None:
    rows_files = sorted(OUTPUT_DIR.glob("rows_*.csv"))
    rows_path = rows_files[-1]
    df = pd.read_csv(rows_path, low_memory=False)
    if "score_3m" not in df.columns:
        sys.exit(f"{rows_path}에 score_3m이 없습니다 — 최신 코드로 backtest.py를 재실행하세요.")
    print(f"rows: {rows_path.name}, 스냅샷 {df['snapshot_date'].nunique()}개")

    config = ScreenerConfig(period="2y")
    universe = load_universe(Path("data/tickers_us1000.csv"))
    symbols = sorted(set(universe["ticker"]) | set(BENCHMARKS))
    prices, report = download_prices(symbols, config)
    print(f"가격 로드 (cache hit: {report.from_cache})")

    records = []
    snaps = sorted(df["snapshot_date"].unique())
    for n, date in enumerate(snaps, 1):
        snap = df[(df["snapshot_date"] == date) & df["passed_hard_filters"].fillna(False)].copy()
        if snap.empty:
            continue
        ts = pd.Timestamp(date)
        truncated = {
            t: prices[t].loc[:ts]
            for t in snap["ticker"]
            if t in prices
        }
        judged = assign_verdicts(snap, truncated, config)
        records.append(judged[["snapshot_date", "ticker", "verdict", "fwd_excess_20d", "score"]])
        print(f"\r진행 {n}/{len(snaps)}", end="", flush=True)
    print()

    all_rows = pd.concat(records, ignore_index=True)
    all_rows = all_rows[all_rows["fwd_excess_20d"].notna()]

    print("\n=== 판정별 20d SPY 초과수익 (어닝스 규칙 비활성 상태) ===")
    agg = all_rows.groupby("verdict")["fwd_excess_20d"].agg(
        mean="mean", median="median", win=lambda s: (s > 0).mean(), n="count"
    )
    print((agg[["mean", "median", "win"]] * 100).round(2).join(agg["n"]).to_string())

    daily = all_rows.pivot_table(index="snapshot_date", columns="verdict", values="fwd_excess_20d", aggfunc="mean")
    if "진입 검토" in daily and "스킵" in daily:
        diff = (daily["진입 검토"] - daily["스킵"]).dropna()
        print(f"\n진입 검토 − 스킵 (스냅샷별 paired): {diff.mean()*100:+.2f}%p, NW t={nw_tstat(diff):+.2f} (n={len(diff)})")
    if "진입 검토" in daily and "대기" in daily:
        diff = (daily["진입 검토"] - daily["대기"]).dropna()
        print(f"진입 검토 − 대기: {diff.mean()*100:+.2f}%p, NW t={nw_tstat(diff):+.2f} (n={len(diff)})")

    # 진입 검토 top10 (점수순) — 실전 사용 형태
    top10 = []
    for date, g in all_rows[all_rows["verdict"] == "진입 검토"].groupby("snapshot_date"):
        if len(g) >= 5:
            top10.append(g.nlargest(min(10, len(g)), "score")["fwd_excess_20d"].mean())
    print(f"\n진입 검토 상위 10 바스켓: {np.mean(top10)*100:+.2f}% (n_snap={len(top10)})")

    out = OUTPUT_DIR / "summary_verdict.csv"
    agg.to_csv(out)
    rows_out = OUTPUT_DIR / "verdict_rows.csv"
    all_rows.to_csv(rows_out, index=False)
    print(f"저장: {out}, {rows_out}")


if __name__ == "__main__":
    main()
