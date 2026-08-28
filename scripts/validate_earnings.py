"""W1(어닝스 20일 내 → 대기) 규칙의 사후 검증 — 모델 미반영, 분석 전용.

백테스트 rows(스냅샷별 후보 + 20d 초과수익)에 yfinance 과거 어닝스 날짜를 결합해
"어닝스 임박" 버킷별 포워드 성과(평균·승률·꼬리)를 비교한다.

- 어닝스 발표 예정일은 실제로도 수주 전 공지되므로 과거 실제 발표일 사용은
  근사적으로 포인트인타임 안전 (리스케줄 소수 예외).
- fwd_excess_20d 윈도우가 발표일을 포함하므로 이벤트 갭 리스크가 그대로 측정됨.
- 판정은 verdict_rows.csv(어닝스 규칙 비활성 상태로 산출)를 사용 —
  즉 "W1이 없었다면 진입 검토였을 종목"의 성과를 직접 볼 수 있다.

실행: UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_earnings.py [--fetch-only]
캐시: .cache/yfinance/earnings_history/<TICKER>.json (심볼당 과거 발표일 목록)
출력: outputs/backtest/summary_earnings_validation.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path("outputs/backtest")
CACHE_DIR = Path(".cache/yfinance/earnings_history")

# 커버리지 판정: 스냅샷 전후 이 일수 안에 발표일이 하나도 없으면 데이터 누락으로 간주
COVERAGE_WINDOW_DAYS = 100


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


def fetch_earnings_history(tickers: list[str]) -> dict[str, list[str]]:
    """티커별 과거 어닝스 발표일 목록. 캐시 우선, 없으면 yfinance 조회."""
    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[str]] = {}
    to_fetch = []
    for t in tickers:
        path = CACHE_DIR / f"{t}.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data.get("dates") is not None:
                result[t] = data["dates"]
                continue
        to_fetch.append(t)

    print(f"어닝스 이력: 캐시 {len(result)}개, 신규 조회 {len(to_fetch)}개")
    for i, t in enumerate(to_fetch, 1):
        dates: list[str] | None = None
        for attempt in range(2):
            try:
                df = yf.Ticker(t).get_earnings_dates(limit=20)
                if df is None or df.empty:
                    dates = []
                else:
                    idx = df.index
                    if getattr(idx, "tz", None) is not None:
                        idx = idx.tz_localize(None)
                    dates = sorted({d.date().isoformat() for d in idx})
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 0:
                    time.sleep(2.0)
                else:
                    print(f"\n  실패: {t} ({e})")
        (CACHE_DIR / f"{t}.json").write_text(json.dumps({"dates": dates}))
        if dates is not None:
            result[t] = dates
        if i % 25 == 0 or i == len(to_fetch):
            print(f"\r조회 진행 {i}/{len(to_fetch)}", end="", flush=True)
        time.sleep(0.15)
    if to_fetch:
        print()
    return result


def bucket_stats(g: pd.Series) -> dict:
    return {
        "n": len(g),
        "mean%": g.mean() * 100,
        "median%": g.median() * 100,
        "win%": (g > 0).mean() * 100,
        "p5%": g.quantile(0.05) * 100,
        "min%": g.min() * 100,
        "share<-15%": (g < -0.15).mean() * 100,
    }


def main() -> None:
    fetch_only = "--fetch-only" in sys.argv

    rows_files = sorted(OUTPUT_DIR.glob("rows_*.csv"))
    rows_path = rows_files[-1]
    df = pd.read_csv(rows_path, low_memory=False)
    cand = df[df["passed_hard_filters"].fillna(False)].copy()
    print(f"rows: {rows_path.name}, 스냅샷 {cand['snapshot_date'].nunique()}개, 후보 {len(cand)}행")

    verdict_path = OUTPUT_DIR / "verdict_rows.csv"
    if verdict_path.exists():
        vr = pd.read_csv(verdict_path)[["snapshot_date", "ticker", "verdict"]]
        cand = cand.merge(vr, on=["snapshot_date", "ticker"], how="left")
    else:
        cand["verdict"] = np.nan
        print("경고: verdict_rows.csv 없음 — 판정 상호작용 분석 생략 (validate_verdicts.py 먼저 실행)")

    tickers = sorted(cand["ticker"].unique())
    earnings = fetch_earnings_history(tickers)
    if fetch_only:
        print("수집 완료 (--fetch-only)")
        return

    # 스냅샷별 days_to_earnings 계산
    dates_by_ticker = {t: pd.to_datetime(ds) for t, ds in earnings.items() if ds}
    snap_ts = pd.to_datetime(cand["snapshot_date"])

    days_next = np.full(len(cand), np.nan)
    days_prev = np.full(len(cand), np.nan)
    covered = np.zeros(len(cand), dtype=bool)
    for i, (t, s) in enumerate(zip(cand["ticker"].to_numpy(), snap_ts.to_numpy())):
        ds = dates_by_ticker.get(t)
        if ds is None:
            continue
        s = pd.Timestamp(s)
        deltas = (ds - s).days
        future = deltas[deltas >= 0]
        past = deltas[deltas < 0]
        if len(future):
            days_next[i] = future.min()
        if len(past):
            days_prev[i] = -past.max()
        covered[i] = bool(np.any(np.abs(deltas) <= COVERAGE_WINDOW_DAYS))
    cand["days_to_earnings"] = days_next
    cand["days_since_earnings"] = days_prev
    cand["earnings_covered"] = covered

    n_no_data = (~cand["ticker"].isin(dates_by_ticker)).sum()
    print(f"\n커버리지: 발표일 데이터 있는 행 {covered.mean()*100:.1f}% "
          f"(이력 자체 없음 {n_no_data}행, 윈도우 ±{COVERAGE_WINDOW_DAYS}일 밖 {len(cand)-covered.sum()-n_no_data}행)")

    ok = cand[cand["earnings_covered"] & cand["fwd_excess_20d"].notna()].copy()

    def to_bucket(d: float) -> str:
        if np.isnan(d) or d > 20:
            return "21일+/없음"
        if d <= 5:
            return "0-5일"
        return "6-20일"

    ok["bucket"] = ok["days_to_earnings"].map(to_bucket)
    order = ["0-5일", "6-20일", "21일+/없음"]

    print("\n=== 어닝스 임박 버킷별 20d SPY 초과수익 (후보 전체) ===")
    summary = pd.DataFrame({b: bucket_stats(g["fwd_excess_20d"]) for b, g in ok.groupby("bucket")}).T.reindex(order)
    print(summary.round(2).to_string())

    print("\n=== signal_type × 버킷 (mean% / win% / n) ===")
    for st, g in ok.groupby("signal_type"):
        parts = []
        for b in order:
            gb = g[g["bucket"] == b]["fwd_excess_20d"]
            parts.append(f"{b}: {gb.mean()*100:+.2f}%/{(gb>0).mean()*100:.0f}%/n={len(gb)}" if len(gb) else f"{b}: -")
        print(f"{st:>10}  " + "  |  ".join(parts))

    # W1이 실제로 강등했을 그룹: 어닝스 규칙 비활성 판정에서 "진입 검토"인데 6-20일 버킷
    jv = ok[ok["verdict"] == "진입 검토"]
    if len(jv):
        print("\n=== 진입 검토(어닝스 규칙 비활성) 내 버킷 비교 — W1이 강등했을 그룹의 실제 성과 ===")
        jsummary = pd.DataFrame({b: bucket_stats(g["fwd_excess_20d"]) for b, g in jv.groupby("bucket")}).T.reindex(order)
        print(jsummary.round(2).to_string())

        daily = jv.pivot_table(index="snapshot_date", columns="bucket", values="fwd_excess_20d", aggfunc="mean")
        if "6-20일" in daily and "21일+/없음" in daily:
            diff = (daily["6-20일"] - daily["21일+/없음"]).dropna()
            print(f"\n진입 검토 내 [6-20일] − [21일+/없음] (스냅샷별 paired): "
                  f"{diff.mean()*100:+.2f}%p, NW t={nw_tstat(diff):+.2f} (n_snap={len(diff)})")
        if "0-5일" in daily and "21일+/없음" in daily:
            diff = (daily["0-5일"] - daily["21일+/없음"]).dropna()
            print(f"진입 검토 내 [0-5일] − [21일+/없음]: "
                  f"{diff.mean()*100:+.2f}%p, NW t={nw_tstat(diff):+.2f} (n_snap={len(diff)})")

        # 전/후반 분할 안정성
        snaps = sorted(jv["snapshot_date"].unique())
        half = snaps[len(snaps) // 2]
        for label, part in [("전반", jv[jv["snapshot_date"] < half]), ("후반", jv[jv["snapshot_date"] >= half])]:
            m620 = part[part["bucket"] == "6-20일"]["fwd_excess_20d"]
            m21 = part[part["bucket"] == "21일+/없음"]["fwd_excess_20d"]
            print(f"  {label}: 6-20일 {m620.mean()*100:+.2f}% (n={len(m620)}) vs 21일+ {m21.mean()*100:+.2f}% (n={len(m21)})")

    # 발표 직후(참고): "발표 후 재평가" 구간의 실제 성과
    post = ok[ok["days_since_earnings"] <= 5]
    print(f"\n(참고) 발표 직후 ≤5일 후보: {post['fwd_excess_20d'].mean()*100:+.2f}% / "
          f"승률 {(post['fwd_excess_20d']>0).mean()*100:.1f}% (n={len(post)})")

    out = OUTPUT_DIR / "summary_earnings_validation.csv"
    summary.to_csv(out)
    rows_out = OUTPUT_DIR / "earnings_validation_rows.csv"
    ok[["snapshot_date", "ticker", "signal_type", "grade", "verdict", "score",
        "days_to_earnings", "days_since_earnings", "bucket", "fwd_excess_20d"]].to_csv(rows_out, index=False)
    print(f"저장: {out}, {rows_out}")


if __name__ == "__main__":
    main()
