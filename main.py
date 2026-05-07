from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_TICKER_FILE = Path("data/tickers_us1000.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_YFINANCE_CACHE_DIR = Path(".cache/yfinance")
BENCHMARKS = ("SPY", "QQQ")


@dataclass(frozen=True)
class ScreenerConfig:
    period: str = "1y"
    interval: str = "1d"
    rs_short_window: int = 20
    rs_mid_window: int = 50
    rs_high_window: int = 50
    ma_short: int = 20
    ma_mid: int = 50
    ma_slope_lookback: int = 10
    volume_window: int = 20
    atr_window: int = 14
    rsi_window: int = 14
    min_dollar_volume: float = 100_000_000
    rs_near_high_threshold: float = 0.98
    near_50d_high_threshold: float = 0.90
    volume_ratio_min: float = 1.3
    volume_ratio_cap: float = 5.0
    close_position_min: float = 0.6
    max_close_to_ma20: float = 1.25
    max_return_5d: float = 0.40
    max_return_20d: float = 0.60
    max_daily_return: float = 0.25
    min_history_days: int = 80
    download_batch_size: int = 50
    download_workers: int = 5


@dataclass(frozen=True)
class ScreenerRun:
    results: pd.DataFrame
    universe_path: Path | None
    candidates_path: Path | None
    market_state: str
    universe_count: int
    evaluated_count: int
    candidates_count: int
    missing_symbols: list[str]
    config: ScreenerConfig
    from_cache: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="US stock RS + volume leadership screener"
    )
    parser.add_argument(
        "--tickers",
        type=Path,
        default=DEFAULT_TICKER_FILE,
        help="CSV file with ticker,name,sector,sector_etf columns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where screener results will be written",
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="yfinance download period, e.g. 1y, 2y",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of candidate rows to print",
    )
    parser.add_argument(
        "--min-dollar-volume",
        type=float,
        default=50_000_000,
        help="Minimum 20-day average dollar volume",
    )
    return parser.parse_args()


def load_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")

    universe = pd.read_csv(path)
    required = {"ticker", "name", "sector", "sector_etf"}
    missing = required.difference(universe.columns)
    if missing:
        raise ValueError(f"Ticker file is missing columns: {sorted(missing)}")

    universe = universe.copy()
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    universe["sector_etf"] = universe["sector_etf"].astype(str).str.strip().str.upper()
    universe = universe.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return universe


def _symbol_cache_dir(period: str) -> Path:
    today = date.today().isoformat()
    return DEFAULT_YFINANCE_CACHE_DIR / "prices" / period / today


def download_prices(symbols: list[str], config: ScreenerConfig) -> tuple[dict[str, pd.DataFrame], bool]:
    DEFAULT_YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(DEFAULT_YFINANCE_CACHE_DIR.resolve()))

    cache_dir = _symbol_cache_dir(config.period)
    cache_dir.mkdir(parents=True, exist_ok=True)

    prices: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for symbol in symbols:
        cache_file = cache_dir / f"{symbol}.pkl"
        if cache_file.exists():
            with cache_file.open("rb") as f:
                prices[symbol] = pickle.load(f)
        else:
            missing.append(symbol)

    if not missing:
        return prices, True

    def _download_batch(batch: list[str]) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}
        try:
            raw = yf.download(
                batch,
                period=config.period,
                interval=config.interval,
                auto_adjust=True,
                group_by="ticker",
                threads=False,
                progress=False,
                timeout=30,
            )
        except Exception:
            return result
        if raw.empty:
            return result
        for symbol in batch:
            frame = extract_symbol_frame(raw, symbol)
            if not frame.empty:
                result[symbol] = frame
                with (cache_dir / f"{symbol}.pkl").open("wb") as f:
                    pickle.dump(frame, f)
        return result

    batches = chunked(missing, config.download_batch_size)
    n_batches = len(batches)
    done = 0
    with ThreadPoolExecutor(max_workers=config.download_workers) as executor:
        futures = {executor.submit(_download_batch, b): b for b in batches}
        for future in as_completed(futures):
            prices.update(future.result())
            done += 1
            print(f"\r  다운로드 진행: {done}/{n_batches} 배치 완료", end="", flush=True)
    print()

    return prices, False


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def extract_symbol_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        frame = None
        for level in range(raw.columns.nlevels):
            matched_value = None
            for value in raw.columns.get_level_values(level).unique():
                if str(value).upper() == symbol.upper():
                    matched_value = value
                    break
            if matched_value is not None:
                frame = raw.xs(matched_value, axis=1, level=level, drop_level=True)
                break
        if frame is None:
            return pd.DataFrame()
    else:
        frame = raw.copy()

    frame = frame.rename(columns={column: str(column).title() for column in frame.columns})
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    if any(column not in frame.columns for column in required_columns):
        return pd.DataFrame()

    frame = frame.loc[:, required_columns].copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.sort_index()
    frame = frame.dropna(subset=["Close"])
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def calculate_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_atr(frame: pd.DataFrame, window: int) -> pd.Series:
    high = frame["High"]
    low = frame["Low"]
    previous_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def safe_latest(series: pd.Series) -> float | bool | pd.Timestamp:
    value = series.iloc[-1]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def calculate_market_state(spy: pd.DataFrame, qqq: pd.DataFrame, config: ScreenerConfig) -> str:
    """
    4단계 시장 환경 분류:
    - Confirmed Uptrend: SPY+QQQ 모두 MA50 위 + MA50 상승
    - Uptrend Under Pressure: 한쪽만 MA50 위 또는 MA50 하강
    - Market in Correction: SPY 또는 QQQ가 MA50 아래
    - Unknown: 데이터 부족
    """
    def _state(frame: pd.DataFrame) -> tuple[bool, bool]:
        close = frame["Close"]
        ma50 = close.rolling(config.ma_mid).mean()
        if len(close.dropna()) < config.ma_mid + config.ma_slope_lookback:
            return False, False
        above = close.iloc[-1] > ma50.iloc[-1]
        rising = ma50.iloc[-1] > ma50.shift(config.ma_slope_lookback).iloc[-1]
        return bool(above), bool(rising)

    spy_above, spy_rising = _state(spy)
    qqq_above, qqq_rising = _state(qqq)

    if spy_above and qqq_above and spy_rising and qqq_rising:
        return "Confirmed Uptrend"
    if not spy_above or not qqq_above:
        return "Market in Correction"
    return "Uptrend Under Pressure"


def calculate_rs_features(
    close: pd.Series,
    benchmark_close: pd.Series,
    prefix: str,
    config: ScreenerConfig,
) -> pd.DataFrame:
    aligned = pd.concat([close, benchmark_close], axis=1, join="inner").dropna()
    aligned.columns = ["stock", "benchmark"]
    rs = aligned["stock"] / aligned["benchmark"]
    result = pd.DataFrame(index=aligned.index)
    result[f"{prefix}_20d"] = rs / rs.shift(config.rs_short_window) - 1
    result[f"{prefix}_50d"] = rs / rs.shift(config.rs_mid_window) - 1
    result[f"{prefix}_near_high"] = (
        rs >= rs.rolling(config.rs_high_window).max() * config.rs_near_high_threshold
    )
    return result


def evaluate_stock(
    ticker: str,
    meta: pd.Series,
    prices: dict[str, pd.DataFrame],
    config: ScreenerConfig,
    market_state: str,
) -> dict[str, object] | None:
    stock = prices.get(ticker)
    spy = prices.get("SPY")
    qqq = prices.get("QQQ")
    sector_etf = str(meta["sector_etf"])
    sector = prices.get(sector_etf)

    if stock is None or spy is None or qqq is None:
        return None
    if len(stock.dropna(subset=["Close"])) < config.min_history_days:
        return None

    frame = stock.copy()
    close = frame["Close"]
    high = frame["High"]
    low = frame["Low"]
    volume = frame["Volume"]

    frame["daily_return"] = close.pct_change()
    frame["return_5d"] = close / close.shift(5) - 1
    frame["return_20d"] = close / close.shift(20) - 1
    frame["ma20"] = close.rolling(config.ma_short).mean()
    frame["ma50"] = close.rolling(config.ma_mid).mean()
    frame["ma50_rising"] = frame["ma50"] > frame["ma50"].shift(config.ma_slope_lookback)
    frame["high_50d"] = close.rolling(config.rs_high_window).max()
    frame["close_to_50d_high"] = close / frame["high_50d"]
    frame["avg_volume_20d"] = volume.rolling(config.volume_window).mean()
    frame["avg_volume_5d"] = volume.rolling(5).mean()
    frame["volume_ratio"] = volume / frame["avg_volume_20d"]
    frame["volume_trend"] = frame["avg_volume_5d"] / frame["avg_volume_20d"]
    frame["capped_volume_ratio"] = frame["volume_ratio"].clip(upper=config.volume_ratio_cap)
    range_size = (high - low).replace(0, np.nan)
    frame["close_position"] = ((close - low) / range_size).fillna(0.5)
    frame["avg_dollar_volume_20d"] = (close * volume).rolling(config.volume_window).mean()
    frame["atr_14"] = calculate_atr(frame, config.atr_window)
    frame["atr_ratio"] = frame["atr_14"] / close
    frame["rsi_14"] = calculate_rsi(close, config.rsi_window)

    rs_spy = calculate_rs_features(close, spy["Close"], "rs_spy", config)
    rs_qqq = calculate_rs_features(close, qqq["Close"], "rs_qqq", config)
    feature_frame = frame.join(rs_spy, how="left").join(rs_qqq, how="left")

    if sector is not None:
        rs_sector = calculate_rs_features(close, sector["Close"], "rs_sector", config)
        feature_frame = feature_frame.join(rs_sector, how="left")
        sector_etf_close = sector["Close"]
        sector_ma50 = sector_etf_close.rolling(config.ma_mid).mean()
        sector_high_52w = sector_etf_close.rolling(252).max()
        feature_frame["sector_etf_to_52w_high"] = (sector_etf_close / sector_high_52w).reindex(feature_frame.index)
    else:
        feature_frame["rs_sector_20d"] = np.nan
        feature_frame["rs_sector_50d"] = np.nan
        feature_frame["rs_sector_near_high"] = False
        feature_frame["sector_etf_to_52w_high"] = np.nan

    # 베이스 안정성: 최근 20일 변동성 대비 10~50일 전 변동성 비교
    returns = close.pct_change()
    vol_recent = returns.rolling(20).std()
    vol_base = returns.rolling(40).std().shift(10)
    feature_frame["base_stability"] = (vol_base / vol_recent.replace(0, np.nan)).clip(upper=3.0) / 3.0

    # 피벗 포인트: 최근 50일 고점 (VCP/컵 패턴의 돌파 기준가)
    feature_frame["pivot_price"] = close.rolling(config.rs_high_window).max()

    # 돌파 후 경과일: 종가가 피벗을 처음 넘은 날로부터 몇 거래일 지났는지
    above_pivot = close >= feature_frame["pivot_price"].shift(1)
    breakout_day = above_pivot & ~above_pivot.shift(1, fill_value=False)
    days_since = pd.Series(np.nan, index=close.index)
    last_bo = None
    for i, (idx, is_bo) in enumerate(breakout_day.items()):
        if is_bo:
            last_bo = i
        if last_bo is not None:
            days_since.iloc[i] = i - last_bo
    feature_frame["days_since_breakout"] = days_since

    latest = feature_frame.dropna(subset=["Close", "ma50", "rs_spy_20d", "rs_spy_50d"]).tail(1)
    if latest.empty:
        return None

    row = latest.iloc[0]
    liquidity_ok = row["avg_dollar_volume_20d"] >= config.min_dollar_volume
    rs_near_high = bool(row["rs_spy_near_high"])
    rs_positive = row["rs_spy_20d"] > 0
    rs_sector_positive = sector is None or row["rs_sector_20d"] > 0
    above_ma50 = row["Close"] > row["ma50"]
    ma50_rising = bool(row["ma50_rising"])
    near_50d_high = row["close_to_50d_high"] >= config.near_50d_high_threshold
    not_overheated = (
        row["Close"] <= row["ma20"] * config.max_close_to_ma20
        and row["return_5d"] < config.max_return_5d
        and row["return_20d"] < config.max_return_20d
        and row["daily_return"] < config.max_daily_return
    )
    volume_quality = (
        row["volume_trend"] >= config.volume_ratio_min
        and row["close_position"] >= 0.5
    )
    # 하드 필터: "절대 배제" 조건만 포함
    # - 유동성 미달, 추세 붕괴(MA50 이탈+하강), RS 음수, 과열
    # rs_near_high / near_50d_high / rs_sector_positive 는 스코어로 반영
    passed = all(
        [
            liquidity_ok,
            rs_positive,
            above_ma50,
            ma50_rising,
            not_overheated,
        ]
    )

    grade = ""
    if passed:
        grade = "A" if volume_quality else "B"

    pivot = float(row["pivot_price"]) if not pd.isna(row.get("pivot_price", np.nan)) else None
    current_close = float(row["Close"])
    pivot_distance = (current_close / pivot - 1) if pivot and pivot > 0 else None
    # 피벗 대비 5% 초과 시 추격 위험
    chasing_risk = pivot_distance is not None and pivot_distance > 0.05

    # 매수 구간: 피벗 기준 0~5% 이내
    if pivot is not None:
        buy_zone_low = round(pivot, 2)
        buy_zone_high = round(pivot * 1.05, 2)
    else:
        buy_zone_low = None
        buy_zone_high = None

    days_bo = row.get("days_since_breakout", np.nan)
    days_since_breakout = int(days_bo) if not pd.isna(days_bo) else None

    # 단계 분류
    if days_since_breakout is None:
        trend_stage = "Watch"
    elif days_since_breakout <= 7:
        trend_stage = "Early Breakout"
    elif days_since_breakout <= 35:
        trend_stage = "Trending"
    else:
        trend_stage = "Extended"

    base_stability = float(row["base_stability"]) if not pd.isna(row.get("base_stability", np.nan)) else None
    sector_etf_to_52w_high = float(row["sector_etf_to_52w_high"]) if not pd.isna(row.get("sector_etf_to_52w_high", np.nan)) else None

    return {
        "ticker": ticker,
        "name": meta["name"],
        "sector": meta["sector"],
        "sector_etf": sector_etf,
        "market_cap": float(meta.get("market_cap") or 0),
        "date": latest.index[-1].date().isoformat(),
        "close": current_close,
        "daily_return": row["daily_return"],
        "return_5d": row["return_5d"],
        "return_20d": row["return_20d"],
        "rs_spy_20d": row["rs_spy_20d"],
        "rs_spy_50d": row["rs_spy_50d"],
        "rs_spy_near_high": rs_near_high,
        "rs_qqq_20d": row["rs_qqq_20d"],
        "rs_qqq_50d": row["rs_qqq_50d"],
        "rs_sector_20d": row["rs_sector_20d"],
        "rs_sector_50d": row["rs_sector_50d"],
        "close_to_50d_high": row["close_to_50d_high"],
        "volume_ratio": row["volume_ratio"],
        "volume_trend": row["volume_trend"],
        "capped_volume_ratio": row["capped_volume_ratio"],
        "avg_dollar_volume_20d": row["avg_dollar_volume_20d"],
        "close_position": row["close_position"],
        "ma20": row["ma20"],
        "ma50": row["ma50"],
        "above_ma20": row["Close"] > row["ma20"],
        "above_ma50": above_ma50,
        "ma50_rising": ma50_rising,
        "atr_ratio": row["atr_ratio"],
        "rsi_14": row["rsi_14"],
        "liquidity_ok": liquidity_ok,
        "rs_positive": rs_positive,
        "rs_sector_positive": rs_sector_positive,
        "near_50d_high": near_50d_high,
        "volume_quality": volume_quality,
        "not_overheated": not_overheated,
        "passed_hard_filters": passed,
        "grade": grade,
        "market_state": market_state,
        # 신규 투자 맥락 필드
        "pivot_price": pivot,
        "buy_zone_low": buy_zone_low,
        "buy_zone_high": buy_zone_high,
        "pivot_distance": pivot_distance,
        "chasing_risk": chasing_risk,
        "days_since_breakout": days_since_breakout,
        "trend_stage": trend_stage,
        "base_stability": base_stability,
        "sector_etf_to_52w_high": sector_etf_to_52w_high,
    }


def add_scores(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    if results.empty:
        results["score"] = pd.Series(dtype=float)
        return results

    # 전체 종목에 스코어 부여 — 하드 필터 통과 여부와 무관하게 순위 파악 가능
    scored = results.copy()
    # RS 가속도: 단기 RS가 중기 RS를 상회하면 모멘텀 개선 중
    rs_acceleration = scored["rs_spy_20d"] - scored["rs_spy_50d"]

    scored["score"] = (
        0.20 * scored["rs_spy_20d"].rank(pct=True)          # 단기 RS 방향
        + 0.25 * scored["rs_spy_50d"].rank(pct=True)         # 중기 RS 수준 (안정성)
        + 0.15 * rs_acceleration.rank(pct=True)              # RS 가속도 (근접도 대체)
        + 0.15 * scored["rs_sector_20d"].rank(pct=True, na_option="bottom")  # 섹터 RS
        + 0.15 * scored["close_to_50d_high"].rank(pct=True)  # 가격 고점 근접도
        + 0.10 * scored["capped_volume_ratio"].rank(pct=True) # 거래량
    )
    results["score"] = scored["score"]
    return results.sort_values(
        ["passed_hard_filters", "score", "rs_spy_20d"],
        ascending=[False, False, False],
    )


def write_outputs(results: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_date = results["date"].dropna().max()
    universe_path = output_dir / f"screener_universe_{latest_date}.csv"
    candidates_path = output_dir / f"screener_candidates_{latest_date}.csv"

    results.to_csv(universe_path, index=False)
    results.loc[results["passed_hard_filters"]].to_csv(candidates_path, index=False)
    return universe_path, candidates_path


def run_screener(
    ticker_file: Path = DEFAULT_TICKER_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    period: str = "2y",
    min_dollar_volume: float = 20_000_000,
    write_files: bool = True,
) -> ScreenerRun:
    config = ScreenerConfig(
        period=period,
        min_dollar_volume=min_dollar_volume,
    )
    universe = load_universe(ticker_file)
    stock_symbols = universe["ticker"].tolist()
    sector_symbols = sorted(set(universe["sector_etf"].dropna()) - {""})
    symbols = sorted(set(stock_symbols + list(BENCHMARKS) + sector_symbols))

    prices, from_cache = download_prices(symbols, config)
    missing = sorted(set(symbols) - set(prices))
    missing_required = [symbol for symbol in BENCHMARKS if symbol not in prices]
    if missing_required:
        raise RuntimeError(
            "Required benchmark data missing: "
            + ", ".join(missing_required)
            + ". Check network access or yfinance availability."
        )

    market_state = calculate_market_state(prices["SPY"], prices["QQQ"], config)
    rows = [
        result
        for _, meta in universe.iterrows()
        if (result := evaluate_stock(meta["ticker"], meta, prices, config, market_state))
        is not None
    ]
    results = add_scores(pd.DataFrame(rows))
    if results.empty:
        raise RuntimeError("No stocks could be evaluated. Check ticker data and download results.")

    universe_path = None
    candidates_path = None
    if write_files:
        universe_path, candidates_path = write_outputs(results, output_dir)

    candidates_count = int(results["passed_hard_filters"].fillna(False).sum())
    return ScreenerRun(
        results=results,
        universe_path=universe_path,
        candidates_path=candidates_path,
        market_state=market_state,
        universe_count=len(stock_symbols),
        evaluated_count=len(results),
        candidates_count=candidates_count,
        missing_symbols=missing,
        config=config,
        from_cache=from_cache,
    )


def format_percent_columns(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    percent_columns = [
        "daily_return",
        "return_5d",
        "rs_spy_20d",
        "rs_spy_50d",
        "rs_qqq_20d",
        "rs_sector_20d",
        "close_to_50d_high",
        "volume_ratio",
        "close_position",
        "atr_ratio",
        "score",
    ]
    for column in percent_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )
    return formatted


def main() -> None:
    args = parse_args()
    universe = load_universe(args.tickers)
    print(f"Loading {len(universe)} stocks and benchmark/sector ETFs")
    run = run_screener(
        ticker_file=args.tickers,
        output_dir=args.output_dir,
        period=args.period,
        min_dollar_volume=args.min_dollar_volume,
        write_files=True,
    )
    results = run.results
    candidates = results.loc[results["passed_hard_filters"]].copy()
    if run.missing_symbols:
        print(f"Missing data skipped: {', '.join(run.missing_symbols[:20])}")
    print(f"Market state: {run.market_state}")
    print(f"Evaluated stocks: {run.evaluated_count}")
    print(f"Candidates: {run.candidates_count}")
    print(f"Wrote: {run.universe_path}")
    print(f"Wrote: {run.candidates_path}")

    if not candidates.empty:
        display_columns = [
            "ticker",
            "name",
            "sector",
            "grade",
            "score",
            "rs_spy_20d",
            "rs_spy_50d",
            "rs_sector_20d",
            "close_to_50d_high",
            "volume_ratio",
            "close_position",
            "rsi_14",
        ]
        print()
        print(format_percent_columns(candidates.head(args.top)[display_columns]).to_string(index=False))


if __name__ == "__main__":
    main()
