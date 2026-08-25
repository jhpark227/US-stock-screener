"""포인트인타임 백테스트 — 과거 각 시점에 스크리너를 재실행하고 이후 수익률을 측정한다.

동작 원리:
  compute_feature_frame()이 전 기간 피처를 계산하므로, 스냅샷 날짜 t마다
  해당 행만 꺼내 evaluate_row()로 판정하면 과거 시점의 스크리너 결과가 재현된다.
  모든 피처는 backward-looking rolling이라 look-ahead가 없다.

한계 (해석 시 유의):
  - 유니버스가 "현재의" 상위 1000 종목이라 생존 편향 존재.
    → 절대 수익률 추정이 아니라 필터/점수 변형 간 상대 비교 용도.
  - auto_adjust 가격은 다운로드 시점 기준 조정 — 순위 비교에는 영향 미미.

사용법:
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/backtest.py
  옵션: --period 2y --step 5 --horizons 5 10 20 60 --min-dollar-volume 50000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import (  # noqa: E402
    FEATURE_REQUIRED_COLUMNS,
    BENCHMARKS,
    VIX_SYMBOLS,
    ScreenerConfig,
    add_scores,
    calculate_market_state_series,
    compute_feature_frame,
    download_prices,
    evaluate_row,
    load_universe,
)

DEFAULT_BACKTEST_DIR = Path("outputs/backtest")

# IC 분석 대상 피처 (스코어 구성요소 + 참고 지표)
IC_FEATURES = [
    "accumulation_days_10d",
    "close_position",
    "volume_trend",
    "base_stability",
    "ma_aligned",
    "rs_spy_20d",
    "rs_sector_20d",
    "rs_spy_50d",
    "distribution_days_10d",
    "close_to_50d_high",
    "volume_ratio",
    "rsi_14",
    "atr_ratio",
    "return_20d",
    "score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point-in-time screener backtest")
    parser.add_argument("--tickers", type=Path, default=Path("data/tickers_us1000.csv"))
    parser.add_argument("--period", default="2y", help="yfinance download period")
    parser.add_argument("--step", type=int, default=5, help="스냅샷 간격 (거래일)")
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20, 60])
    parser.add_argument("--min-dollar-volume", type=float, default=50_000_000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BACKTEST_DIR)
    parser.add_argument("--cost-bps", type=float, default=25.0, help="왕복 거래비용 (bp) — 포트폴리오 시뮬레이션용")
    parser.add_argument("--top-n", type=int, default=10, help="포트폴리오 시뮬레이션 종목 수")
    return parser.parse_args()


def collect_snapshots(
    universe: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    config: ScreenerConfig,
    step: int,
    horizons: list[int],
) -> pd.DataFrame:
    spy = prices["SPY"]
    qqq = prices["QQQ"]
    spy_close = spy["Close"]
    max_horizon = max(horizons)

    print("피처 계산 중 (종목당 1회)...")
    features: dict[str, pd.DataFrame] = {}
    metas: dict[str, pd.Series] = {}
    has_sector: dict[str, bool] = {}
    for _, meta in universe.iterrows():
        ticker = meta["ticker"]
        stock = prices.get(ticker)
        if stock is None or stock.empty:
            continue
        sector = prices.get(str(meta["sector_etf"]))
        features[ticker] = compute_feature_frame(stock, spy, qqq, sector, config)
        metas[ticker] = meta
        has_sector[ticker] = sector is not None

    spy_index = spy.index
    start = config.min_history_days
    end = len(spy_index) - max_horizon
    snapshot_positions = list(range(start, end, step))
    print(
        f"스냅샷 {len(snapshot_positions)}개: "
        f"{spy_index[start].date()} ~ {spy_index[snapshot_positions[-1]].date()} (step={step}일)"
    )

    regime_series, _ = calculate_market_state_series(prices, config)

    all_snapshots: list[pd.DataFrame] = []
    for n, pos in enumerate(snapshot_positions, 1):
        t = spy_index[pos]
        market_state = str(regime_series.asof(t))
        rows = []
        for ticker, ff in features.items():
            try:
                ip = ff.index.get_loc(t)
            except KeyError:
                continue
            if ip + 1 < config.min_history_days:
                continue
            row = ff.iloc[ip]
            if row[FEATURE_REQUIRED_COLUMNS].isna().any():
                continue
            rec = evaluate_row(ticker, metas[ticker], row, has_sector[ticker], market_state, config)
            close_series = ff["Close"]
            for k in horizons:
                if ip + k < len(close_series):
                    stock_fwd = close_series.iloc[ip + k] / close_series.iloc[ip] - 1
                    spy_fwd = spy_close.iloc[pos + k] / spy_close.iloc[pos] - 1
                    rec[f"fwd_{k}d"] = stock_fwd
                    rec[f"fwd_excess_{k}d"] = stock_fwd - spy_fwd
            rows.append(rec)
        if not rows:
            continue
        snap = add_scores(pd.DataFrame(rows), config)
        snap["snapshot_date"] = t.date().isoformat()
        all_snapshots.append(snap)
        print(f"\r  진행: {n}/{len(snapshot_positions)} ({t.date()}, 후보 {int(snap['passed_hard_filters'].sum())}개)", end="", flush=True)
    print()

    return pd.concat(all_snapshots, ignore_index=True)


def summarize(df: pd.DataFrame, horizons: list[int]) -> dict[str, pd.DataFrame]:
    df = df.copy()
    df["group"] = np.where(df["passed_hard_filters"].fillna(False), df["grade"], "fail")

    # 1) 등급별 성과
    grade_rows = []
    for k in horizons:
        col = f"fwd_excess_{k}d"
        if col not in df.columns:
            continue
        agg = df.groupby("group")[col].agg(
            mean="mean",
            median="median",
            hit_rate=lambda s: (s > 0).mean(),
            count="count",
        )
        agg["horizon"] = f"{k}d"
        grade_rows.append(agg.reset_index())
    grade_summary = pd.concat(grade_rows, ignore_index=True)

    # 2) 점수 십분위별 성과 (스냅샷별 십분위 → 전체 집계)
    def decile_table(sub: pd.DataFrame, label: str) -> pd.DataFrame:
        sub = sub.dropna(subset=["score"]).copy()
        sub["decile"] = sub.groupby("snapshot_date")["score"].transform(
            lambda s: pd.qcut(s, 10, labels=False, duplicates="drop")
        )
        out = []
        for k in horizons:
            col = f"fwd_excess_{k}d"
            if col not in sub.columns:
                continue
            t = sub.groupby("decile")[col].agg(mean="mean", hit_rate=lambda s: (s > 0).mean(), count="count")
            t["horizon"] = f"{k}d"
            t["universe"] = label
            out.append(t.reset_index())
        return pd.concat(out, ignore_index=True)

    decile_all = decile_table(df, "all")
    decile_candidates = decile_table(df[df["group"] != "fail"], "candidates")
    decile_summary = pd.concat([decile_all, decile_candidates], ignore_index=True)

    # 3) 피처별 IC (스냅샷별 Spearman → 평균)
    ic_rows = []
    target = "fwd_excess_20d"
    for label, sub in [("all", df), ("candidates", df[df["group"] != "fail"])]:
        if target not in sub.columns:
            continue
        for feature in IC_FEATURES:
            if feature not in sub.columns:
                continue
            daily_ics = []
            for _, day in sub.groupby("snapshot_date"):
                x = day[feature].astype(float) if day[feature].dtype == bool else pd.to_numeric(day[feature], errors="coerce")
                y = day[target]
                valid = x.notna() & y.notna()
                if valid.sum() < 30:
                    continue
                # rank 후 Pearson = Spearman (scipy 의존성 회피)
                daily_ics.append(x[valid].rank().corr(y[valid].rank()))
            ics = pd.Series(daily_ics).dropna()
            if ics.empty:
                continue
            ic_rows.append(
                {
                    "universe": label,
                    "feature": feature,
                    "mean_ic": ics.mean(),
                    "ic_std": ics.std(),
                    "ic_positive_pct": (ics > 0).mean(),
                    "n_snapshots": len(ics),
                    "t_stat": ics.mean() / ics.std() * np.sqrt(len(ics)) if ics.std() > 0 else np.nan,
                }
            )
    ic_summary = pd.DataFrame(ic_rows).sort_values(["universe", "mean_ic"], ascending=[True, False])

    # 4) 시장 국면별 등급 성과 (20d)
    regime_summary = pd.DataFrame()
    if "fwd_excess_20d" in df.columns:
        regime_summary = (
            df[df["group"] != "fail"]
            .groupby(["market_state", "group"])["fwd_excess_20d"]
            .agg(mean="mean", median="median", hit_rate=lambda s: (s > 0).mean(), count="count")
            .reset_index()
        )

    return {
        "grade": grade_summary,
        "decile": decile_summary,
        "ic": ic_summary,
        "regime": regime_summary,
    }


def portfolio_summary(df: pd.DataFrame, top_n: int = 10, cost_bps: float = 25.0) -> pd.DataFrame:
    """주간 리밸런스 top-N 포트폴리오의 비용 후 성과.

    스냅샷마다 후보 중 점수 상위 N개를 담고 다음 스냅샷까지 보유(fwd_excess_5d = 보유기간과
    스냅샷 간격 일치). 회전율 = 직전 바스켓 대비 교체 비율, 비용 = 회전율 × 왕복 cost_bps.
    첫 스냅샷은 회전율 1(전량 신규 매수)로 처리해 보수적으로 계산한다.
    """
    prev_holdings: set[str] = set()
    rows = []
    for date in sorted(df["snapshot_date"].unique()):
        snap = df[
            (df["snapshot_date"] == date)
            & df["passed_hard_filters"].fillna(False)
            & df["fwd_excess_5d"].notna()
        ]
        if len(snap) < top_n:
            continue
        basket = snap.nlargest(top_n, "score")
        holdings = set(basket["ticker"])
        turnover = 1.0 - len(holdings & prev_holdings) / top_n if prev_holdings else 1.0
        gross = basket["fwd_excess_5d"].mean()
        net = gross - turnover * cost_bps / 10_000
        rows.append({"snapshot_date": date, "gross_excess_5d": gross, "net_excess_5d": net, "turnover": turnover})
        prev_holdings = holdings
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    summary = pd.DataFrame([{
        "n_rebalances": len(out),
        "avg_turnover": out["turnover"].mean(),
        "gross_weekly_mean": out["gross_excess_5d"].mean(),
        "net_weekly_mean": out["net_excess_5d"].mean(),
        "cost_drag_weekly": out["gross_excess_5d"].mean() - out["net_excess_5d"].mean(),
        "gross_cum": (1 + out["gross_excess_5d"]).prod() - 1,
        "net_cum": (1 + out["net_excess_5d"]).prod() - 1,
        "cost_bps_roundtrip": cost_bps,
        "top_n": top_n,
    }])
    return summary


def main() -> None:
    args = parse_args()
    config = ScreenerConfig(period=args.period, min_dollar_volume=args.min_dollar_volume)

    universe = load_universe(args.tickers)
    stock_symbols = universe["ticker"].tolist()
    sector_symbols = sorted(set(universe["sector_etf"].dropna()) - {""})
    symbols = sorted(set(stock_symbols + list(BENCHMARKS) + list(VIX_SYMBOLS) + sector_symbols))

    print(f"가격 데이터 로드 중 ({len(symbols)}개 심볼, period={args.period})...")
    prices, report = download_prices(symbols, config)
    print(f"  로드 완료 (cache={report.from_cache}, {len(prices)}개)")

    df = collect_snapshots(universe, prices, config, args.step, args.horizons)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dates = df["snapshot_date"]
    rows_path = args.output_dir / f"rows_{dates.min()}_{dates.max()}_step{args.step}.csv"
    df.to_csv(rows_path, index=False)
    print(f"\n스냅샷 원본 저장: {rows_path} ({len(df):,} rows)")

    summaries = summarize(df, args.horizons)
    summaries["portfolio"] = portfolio_summary(df, top_n=args.top_n, cost_bps=args.cost_bps)
    for name, table in summaries.items():
        path = args.output_dir / f"summary_{name}.csv"
        table.to_csv(path, index=False)

    pd.set_option("display.width", 200)
    print("\n=== 등급별 SPY 대비 초과수익 ===")
    print(summaries["grade"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== 점수 십분위별 초과수익 (9=최상위) ===")
    print(summaries["decile"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== 피처별 IC (20일 초과수익 기준 Spearman) ===")
    print(summaries["ic"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== 시장 국면별 후보 성과 (20d) ===")
    print(summaries["regime"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n=== 주간 top{args.top_n} 포트폴리오 (왕복 {args.cost_bps:.0f}bp) ===")
    print(summaries["portfolio"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
