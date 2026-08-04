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
    ma_mid: int = 60
    ma_long: int = 120
    ma_slope_lookback: int = 10
    golden_cross_lookback: int = 5
    volume_window: int = 20
    atr_window: int = 14
    rsi_window: int = 14
    min_dollar_volume: float = 50_000_000
    rs_near_high_threshold: float = 0.98
    near_50d_high_threshold: float = 0.90
    volume_ratio_min: float = 1.3
    volume_ratio_cap: float = 5.0
    volume_strength_interest: float = 1.5
    volume_strength_attention: float = 2.0
    volume_strength_conviction: float = 5.0
    # 주목 트리거 (2026-08 백테스트로 결정: outputs/backtest/ 참조)
    surge_ratio_min: float = 1.5          # 당일 서지 하한 — 20일 평균 거래량 대비
    surge_ratio_max: float = 5.0          # 상한 — 이 이상은 어닝스 갭 의심, 트리거 제외+경고 라벨
    accumulation_trigger_days: int = 6    # 10일 내 매집일 이 값 이상이면 지속 매집 트리거
    distribution_warning_days: int = 3    # 10일 내 분산일 경고 라벨 기준
    accumulation_days_min: int = 3
    distribution_days_max: int = 1
    vp_lookback: int = 10
    close_position_min: float = 0.6
    max_close_to_ma20: float = 1.25
    max_return_5d: float = 0.40
    max_return_20d: float = 0.60
    max_daily_return: float = 0.25
    min_history_days: int = 130
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
    - Confirmed Uptrend: SPY+QQQ 모두 MA60 위 + MA60 상승
    - Uptrend Under Pressure: 한쪽만 MA60 위 또는 MA60 하강
    - Market in Correction: SPY 또는 QQQ가 MA60 아래
    - Unknown: 데이터 부족
    """
    def _state(frame: pd.DataFrame) -> tuple[bool, bool]:
        close = frame["Close"]
        ma_mid = close.rolling(config.ma_mid).mean()
        if len(close.dropna()) < config.ma_mid + config.ma_slope_lookback:
            return False, False
        above = close.iloc[-1] > ma_mid.iloc[-1]
        rising = ma_mid.iloc[-1] > ma_mid.shift(config.ma_slope_lookback).iloc[-1]
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


def compute_feature_frame(
    stock: pd.DataFrame,
    spy: pd.DataFrame,
    qqq: pd.DataFrame,
    sector: pd.DataFrame | None,
    config: ScreenerConfig,
) -> pd.DataFrame:
    """OHLCV → 전체 기간 피처 DataFrame. 마지막 행뿐 아니라 모든 날짜의 피처를 담는다."""
    frame = stock.copy()
    close = frame["Close"]
    high = frame["High"]
    low = frame["Low"]
    volume = frame["Volume"]

    frame["daily_return"] = close.pct_change()
    frame["return_5d"] = close / close.shift(5) - 1
    frame["return_20d"] = close / close.shift(20) - 1
    frame["ma20"] = close.rolling(config.ma_short).mean()
    frame["ma60"] = close.rolling(config.ma_mid).mean()
    frame["ma120"] = close.rolling(config.ma_long).mean()
    frame["ma60_rising"] = frame["ma60"] > frame["ma60"].shift(config.ma_slope_lookback)
    frame["ma120_rising"] = frame["ma120"] > frame["ma120"].shift(config.ma_slope_lookback)
    frame["ma_aligned"] = (
        (frame["ma20"] > frame["ma60"]) & (frame["ma60"] > frame["ma120"])
    )
    # 골든크로스: MA20이 MA60을 상향돌파한 시점 (최근 N일 내, MA60 상승 중일 때만)
    ma_defined = frame["ma20"].notna() & frame["ma60"].notna()
    ma20_above_ma60 = (frame["ma20"] > frame["ma60"]) & ma_defined
    # shift(1)의 NaN은 True로 처리 → 첫 정의된 시점에 ma20>ma60이어도 cross_up=False
    prev_above = ma20_above_ma60.shift(1)
    prev_above_fill = prev_above.where(prev_above.notna(), True).astype(bool)
    cross_up = ma20_above_ma60 & ~prev_above_fill
    cross_recent = cross_up.rolling(config.golden_cross_lookback, min_periods=1).max().fillna(0).astype(bool)
    frame["golden_cross_recent"] = cross_recent & frame["ma60_rising"].fillna(False)
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

    # 가격-거래량 4분면 일별 신호 + 강도 등급
    price_up = frame["daily_return"] > 0
    volume_up = volume > frame["avg_volume_20d"]
    frame["vp_signal"] = np.select(
        [
            price_up & volume_up,
            price_up & ~volume_up,
            ~price_up & volume_up,
        ],
        ["accumulation", "weak_rally", "distribution"],
        default="dry_up",
    )
    accumulation_flag = (frame["vp_signal"] == "accumulation").astype(int)
    distribution_flag = (frame["vp_signal"] == "distribution").astype(int)
    frame["accumulation_days_10d"] = accumulation_flag.rolling(config.vp_lookback).sum()
    frame["distribution_days_10d"] = distribution_flag.rolling(config.vp_lookback).sum()

    vr = frame["volume_ratio"]
    frame["volume_strength"] = np.select(
        [
            vr >= config.volume_strength_conviction,
            vr >= config.volume_strength_attention,
            vr >= config.volume_strength_interest,
        ],
        ["conviction", "attention", "interest"],
        default="normal",
    )

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

    return feature_frame


def evaluate_row(
    ticker: str,
    meta: pd.Series,
    row: pd.Series,
    has_sector: bool,
    market_state: str,
    config: ScreenerConfig,
) -> dict[str, object]:
    """피처 행 하나를 받아 필터 판정·등급·매수기준가를 계산한다. row.name은 날짜 인덱스."""
    sector_etf = str(meta["sector_etf"])
    liquidity_ok = row["avg_dollar_volume_20d"] >= config.min_dollar_volume
    rs_near_high = bool(row["rs_spy_near_high"])
    rs_positive = row["rs_spy_20d"] > 0
    rs_sector_positive = (not has_sector) or row["rs_sector_20d"] > 0
    above_ma60 = row["Close"] > row["ma60"]
    ma60_rising = bool(row["ma60_rising"])
    ma_aligned = bool(row["ma_aligned"]) if not pd.isna(row.get("ma_aligned", np.nan)) else False
    golden_cross_recent = bool(row["golden_cross_recent"]) if not pd.isna(row.get("golden_cross_recent", np.nan)) else False
    trend_structure_ok = ma_aligned or (golden_cross_recent and ma60_rising)
    near_50d_high = row["close_to_50d_high"] >= config.near_50d_high_threshold
    not_overheated = (
        row["Close"] <= row["ma20"] * config.max_close_to_ma20
        and row["return_5d"] < config.max_return_5d
        and row["return_20d"] < config.max_return_20d
        and row["daily_return"] < config.max_daily_return
    )
    accumulation_days = int(row["accumulation_days_10d"]) if not pd.isna(row.get("accumulation_days_10d", np.nan)) else 0
    distribution_days = int(row["distribution_days_10d"]) if not pd.isna(row.get("distribution_days_10d", np.nan)) else 0
    volume_quality = (
        accumulation_days >= config.accumulation_days_min
        and distribution_days <= config.distribution_days_max
    )

    # 주목 트리거: ① 당일 서지 (거래량 1.5~5배 + 양봉) ② 지속 매집 (10일 내 매집일 6+)
    daily_return_val = row["daily_return"]
    volume_ratio_val = row["volume_ratio"]
    surge_today = (
        not pd.isna(daily_return_val)
        and not pd.isna(volume_ratio_val)
        and daily_return_val > 0
        and config.surge_ratio_min <= volume_ratio_val < config.surge_ratio_max
    )
    sustained_accumulation = accumulation_days >= config.accumulation_trigger_days
    if surge_today and sustained_accumulation:
        signal_type = "surge+acc"
    elif surge_today:
        signal_type = "surge"
    elif sustained_accumulation:
        signal_type = "acc"
    else:
        signal_type = ""

    # 하드 필터: 유동성 + RS+ + 거래량 트리거. MA/과열은 배제하지 않고 등급·경고로 표시
    passed = liquidity_ok and rs_positive and (surge_today or sustained_accumulation)

    # 등급 = MA 컨텍스트: A = MA60 위 + MA60 상승, B = 그 외
    grade = ""
    if passed:
        grade = "A" if (above_ma60 and ma60_rising) else "B"

    # 경고 라벨 (참고 정보 — 배제 아님)
    warning_labels = []
    if not not_overheated:
        warning_labels.append("과열")
    if (
        not pd.isna(volume_ratio_val)
        and not pd.isna(daily_return_val)
        and volume_ratio_val >= config.surge_ratio_max
        and daily_return_val > 0
    ):
        warning_labels.append("거래량5x+")
    if distribution_days >= config.distribution_warning_days:
        warning_labels.append(f"분산{distribution_days}일")
    warnings = ",".join(warning_labels)

    current_close = float(row["Close"])
    ma20_val = float(row["ma20"]) if not pd.isna(row.get("ma20", np.nan)) else None
    ma60_val = float(row["ma60"]) if not pd.isna(row.get("ma60", np.nan)) else None

    # 매수기준가: MA20 위에 있으면 MA20*1.01, 아니면 MA60*1.01
    if ma20_val is not None and current_close > ma20_val:
        buy_price = round(ma20_val * 1.01, 2)
        buy_price_basis = "MA20"
    elif ma60_val is not None:
        buy_price = round(ma60_val * 1.01, 2)
        buy_price_basis = "MA60"
    else:
        buy_price = None
        buy_price_basis = None

    base_stability = float(row["base_stability"]) if not pd.isna(row.get("base_stability", np.nan)) else None
    sector_etf_to_52w_high = float(row["sector_etf_to_52w_high"]) if not pd.isna(row.get("sector_etf_to_52w_high", np.nan)) else None

    return {
        "ticker": ticker,
        "name": meta["name"],
        "sector": meta["sector"],
        "sector_etf": sector_etf,
        "market_cap": float(meta.get("market_cap") or 0),
        "date": row.name.date().isoformat(),
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
        "ma60": row["ma60"],
        "ma120": row["ma120"],
        "above_ma20": row["Close"] > row["ma20"],
        "above_ma60": above_ma60,
        "ma60_rising": ma60_rising,
        "ma120_rising": bool(row["ma120_rising"]) if not pd.isna(row.get("ma120_rising", np.nan)) else False,
        "ma_aligned": ma_aligned,
        "golden_cross_recent": golden_cross_recent,
        "vp_signal": row["vp_signal"],
        "accumulation_days_10d": accumulation_days,
        "distribution_days_10d": distribution_days,
        "volume_strength": row["volume_strength"],
        "surge_today": surge_today,
        "sustained_accumulation": sustained_accumulation,
        "signal_type": signal_type,
        "warnings": warnings,
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
        "buy_price": buy_price,
        "buy_price_basis": buy_price_basis,
        "base_stability": base_stability,
        "sector_etf_to_52w_high": sector_etf_to_52w_high,
    }


FEATURE_REQUIRED_COLUMNS = ["Close", "ma60", "rs_spy_20d", "rs_spy_50d"]


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

    feature_frame = compute_feature_frame(stock, spy, qqq, sector, config)
    latest = feature_frame.dropna(subset=FEATURE_REQUIRED_COLUMNS).tail(1)
    if latest.empty:
        return None

    return evaluate_row(ticker, meta, latest.iloc[0], sector is not None, market_state, config)


def add_scores(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    if results.empty:
        results["score"] = pd.Series(dtype=float)
        return results

    # 전체 종목에 스코어 부여 — 하드 필터 통과 여부와 무관하게 순위 파악 가능
    # v2 (2026-08): 백테스트 IC 근거 재가중 — RS 계열만 유의(t>3.4), 당일 봉/베이스 피처는 예측력 없어 제외
    scored = results.copy()

    scored["score"] = (
        0.30 * scored["rs_spy_20d"].rank(pct=True)                                  # 단기 RS — 최대 IC
        + 0.25 * scored["rs_spy_50d"].rank(pct=True)                                # 중기 RS
        + 0.25 * scored["rs_sector_20d"].rank(pct=True, na_option="bottom")         # 섹터 RS
        + 0.10 * scored["accumulation_days_10d"].rank(pct=True, na_option="bottom")  # 매집 지속성
        + 0.10 * scored["volume_trend"].rank(pct=True, na_option="bottom")          # 거래량 증가 추세
    )

    results["score"] = scored["score"]
    return results.sort_values(
        ["passed_hard_filters", "score", "close_to_50d_high"],
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
    min_dollar_volume: float = 50_000_000,
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
            "signal_type",
            "score",
            "rs_spy_20d",
            "rs_spy_50d",
            "rs_sector_20d",
            "volume_ratio",
            "accumulation_days_10d",
            "buy_price",
            "warnings",
        ]
        print()
        print(format_percent_columns(candidates.head(args.top)[display_columns]).to_string(index=False))


if __name__ == "__main__":
    main()
