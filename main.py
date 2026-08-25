from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from price_store import SyncReport, extract_symbol_frame, sync_prices  # noqa: F401 (재수출)


DEFAULT_TICKER_FILE = Path("data/tickers_us1000.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_YFINANCE_CACHE_DIR = Path(".cache/yfinance")
BENCHMARKS = ("SPY", "QQQ")
VIX_SYMBOLS = ("^VIX", "^VIX3M")  # 국면 분류 3축 중 변동성 축 (없으면 해당 축 중립 처리)


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
    # 거래량 신호 (2026-08 백테스트로 결정: outputs/backtest/ 참조)
    surge_ratio_min: float = 1.5          # 당일 서지 하한 — 20일 평균 거래량 대비
    surge_ratio_max: float = 5.0          # 상한 — 이 이상은 어닝스 갭 의심, 신호 제외+경고 라벨
    accumulation_trigger_days: int = 6    # 10일 내 매집일 이 값 이상이면 지속 매집 신호
    # 관찰기간: 신호 후 이 거래일 수 동안 후보 유지 (당일 포함). 지연 진입 백테스트(2026-08-21,
    # 신규 신호 7,739건) 근거 — 알파 감쇠는 첫 2일에 집중되고 k=3~7일에도 당일의 ~70% 유지.
    trigger_window_days: int = 5
    distribution_warning_days: int = 3    # 10일 내 분산일 경고 라벨 기준
    regime_fit_bonus: float = 0.10        # 국면-등급 정합 후보 점수 보너스 (CU→A, Correction→B)
    # 3축 국면 분류 (2026-08-22 적용): 추세(히스테리시스) + 시장 폭 + VIX 기간구조, 합산 후 K일 확정
    regime_hysteresis_down: float = 0.99  # MA60×이 값 미만 종가가 아래 연속일수만큼 나와야 '이탈'
    regime_hysteresis_up: float = 1.01    # MA60×이 값 초과 종가 하루면 '복귀'
    regime_down_confirm_days: int = 2     # 이탈 확정에 필요한 연속일
    regime_breadth_low: float = 0.40      # 유니버스 MA60 상회 비율 — 이 미만이면 폭 축 -1.
                                          # 폭은 하한 감지기 전용(+1표 없음): 후보 성과가 40~60%와 >60%에서
                                          # 동등(+2.06% vs +1.52%)하고 <40%에서만 붕괴(-3.17%) — 2026-08-22 검증
    regime_vix_contango: float = 0.95     # VIX/VIX3M 이 미만(콘탱고)이면 변동성 축 +1
    regime_vix_backwardation: float = 1.00  # 이 초과(백워데이션)면 -1
    regime_confirm_days: int = 3          # 새 국면이 이 일수 연속 유지돼야 전환 확정 (전환 27→14회)
    # 판정 규칙 (2026-08-22): 진입 검토/대기/스킵을 코드로 결정론적으로 산출. 임계값 근거는 CLAUDE.md.
    verdict_corr_window: int = 60         # 클러스터 판정용 수익률 상관 윈도우
    verdict_corr_threshold: float = 0.70  # 이 이상 상관이면 같은 클러스터 (금은 6종목 실측 0.92+)
    verdict_cluster_min_size: int = 3     # 이 크기 이상 클러스터만 중복 규칙 적용
    verdict_reps_per_cluster: int = 2     # 클러스터당 진입 검토 대표 수
    verdict_s2_score3m: float = 0.30      # 국면 부정합 등급 스킵 기준 (중기 근거 없으면)
    verdict_s4_score3m: float = 0.10      # 단기 이벤트성 + 분산 매물 스킵 기준
    verdict_w3_score3m: float = 0.15      # 단기 이벤트성 대기 기준
    verdict_min_score_pct: float = 0.40   # 후보 내 점수 percentile 하한 — 미만이면 스킵.
                                          # 검증: 하위 40% 진입 검토는 -0.23%(승률 46%)로 스킵 그룹보다 나쁨
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
    stale_symbols: list[str] = field(default_factory=list)  # 당일 다운로드 실패 → 전일 저장분으로 평가된 종목


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
        default="2y",
        help="yfinance download period, e.g. 1y, 2y (mom_12_1/score_3m은 252+21일 이력이 필요해 2y 권장)",
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


def download_prices(symbols: list[str], config: ScreenerConfig) -> tuple[dict[str, pd.DataFrame], SyncReport]:
    """증분 parquet 저장소(price_store)를 동기화하고 가격 dict와 동기화 리포트를 반환한다."""
    prices, report = sync_prices(
        symbols,
        period=config.period,
        batch_size=config.download_batch_size,
        workers=config.download_workers,
    )
    print(f"  가격 동기화: {report.summary()}")
    return prices, report


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


def calculate_market_state_series(
    prices: dict[str, pd.DataFrame],
    config: ScreenerConfig,
) -> tuple[pd.Series, dict[str, object]]:
    """3축 국면 분류 (2026-08-22): 일별 국면 시계열과 최신 축 상세를 반환.

    축 1 추세 — SPY·QQQ 각각 MA60 히스테리시스 상태머신: 이탈은 MA60×0.99 미만 2일 연속,
        복귀는 MA60×1.01 초과 1일. 둘 다 위 + MA60 상승 → +1 / 어느 하나 아래 → -1 / 그 외 0.
        (단순 위/아래 대비 경계 왕복 제거)
    축 2 시장 폭 — 유니버스 중 종가>MA60 비율. 지수는 메가캡 몇 개로 왜곡될 수 있어
        "종목 대다수"의 상태를 따로 본다. 하한 감지기 전용: <40% → -1, 그 외 0.
        (+1표는 정보 없음이 검증됨. 따라서 CU는 추세+VIX 둘 다 양호해야 성립 —
        VIX 데이터가 수일 결손되면 K일 지연 후 Pressure로 보수적 강등됨.)
    축 3 VIX 기간구조 — VIX/VIX3M. 옵션 시장의 스트레스: <0.95(콘탱고, 평온) → +1 /
        >1.0(백워데이션, 즉각적 공포) → -1. 데이터 없으면 0(중립).

    합산 ≥+2 → Confirmed Uptrend / ≤-1 → Market in Correction / 그 외 Uptrend Under Pressure.
    새 국면이 regime_confirm_days(3일) 연속 유지될 때만 전환 확정 — 2y 검증에서 전환 27→14회,
    Correction의 A/B 분리 개선(A -1.55% vs B +0.65%), 판정 성과 동등(진입-스킵 +2.70%p, t=4.72).
    """
    spy, qqq = prices["SPY"], prices["QQQ"]
    index = spy.index

    def hysteresis_above(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        close = frame["Close"]
        ma = close.rolling(config.ma_mid).mean()
        below_trigger = close < ma * config.regime_hysteresis_down
        above_trigger = close > ma * config.regime_hysteresis_up
        state, out, below_run = True, [], 0
        for i in range(len(close)):
            if pd.isna(ma.iloc[i]):
                out.append(np.nan)
                continue
            if state:
                below_run = below_run + 1 if below_trigger.iloc[i] else 0
                if below_run >= config.regime_down_confirm_days:
                    state, below_run = False, 0
            elif above_trigger.iloc[i]:
                state = True
            out.append(state)
        return pd.Series(out, index=close.index), ma

    spy_above, spy_ma = hysteresis_above(spy)
    qqq_above, qqq_ma = hysteresis_above(qqq)
    spy_rising = spy_ma > spy_ma.shift(config.ma_slope_lookback)
    qqq_rising = qqq_ma > qqq_ma.shift(config.ma_slope_lookback)
    trend_axis = pd.Series(0.0, index=index)
    trend_axis[(spy_above == True) & (qqq_above == True) & spy_rising & qqq_rising] = 1  # noqa: E712
    trend_axis[(spy_above == False) | (qqq_above == False)] = -1  # noqa: E712

    stock_symbols = [
        t for t in prices
        if t not in BENCHMARKS and not t.startswith("^") and len(prices[t]) > config.ma_mid + 10
    ]
    above_frame = pd.DataFrame({
        t: (prices[t]["Close"] > prices[t]["Close"].rolling(config.ma_mid).mean()).reindex(index)
        for t in stock_symbols
    })
    breadth = above_frame.mean(axis=1)
    breadth_axis = pd.Series(0.0, index=index)
    breadth_axis[breadth < config.regime_breadth_low] = -1

    vix, vix3m = prices.get("^VIX"), prices.get("^VIX3M")
    if vix is not None and vix3m is not None and not vix.empty and not vix3m.empty:
        vix_ratio = (vix["Close"] / vix3m["Close"]).reindex(index).ffill()
    else:
        vix_ratio = pd.Series(np.nan, index=index)
    vix_axis = pd.Series(0.0, index=index)
    vix_axis[vix_ratio < config.regime_vix_contango] = 1
    vix_axis[vix_ratio > config.regime_vix_backwardation] = -1

    total = trend_axis + breadth_axis + vix_axis
    raw = pd.Series("Uptrend Under Pressure", index=index)
    raw[total >= 2] = "Confirmed Uptrend"
    raw[total <= -1] = "Market in Correction"
    warmup = spy_above.isna() | breadth.isna()
    raw[warmup] = "Unknown"

    # K일 확정 지연: 새 라벨이 연속 유지될 때만 전환
    confirmed, current, pending, run = [], None, None, 0
    for label in raw:
        if current is None or current == "Unknown":
            current = label
        if label == current:
            pending, run = None, 0
        elif label == pending:
            run += 1
            if run >= config.regime_confirm_days:
                current, pending, run = label, None, 0
        else:
            pending, run = label, 1
        confirmed.append(current)
    series = pd.Series(confirmed, index=index)

    detail = {
        "trend": int(trend_axis.iloc[-1]),
        "breadth": round(float(breadth.iloc[-1]), 3) if not pd.isna(breadth.iloc[-1]) else None,
        "vix_ratio": round(float(vix_ratio.iloc[-1]), 3) if not pd.isna(vix_ratio.iloc[-1]) else None,
        "score": int(total.iloc[-1]),
    }
    return series, detail


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
    # 상승일/하락일 거래량 비율 (20일) — 매수·매도 압력의 비대칭을 하나의 값으로.
    # 매집일수 카운트보다 정보량이 많아 점수 v3에서 거래량 축을 담당. 상한 5로 극단값 억제.
    up_volume_20d = volume.where(frame["daily_return"] > 0, 0.0).rolling(config.volume_window).sum()
    down_volume_20d = volume.where(frame["daily_return"] < 0, 0.0).rolling(config.volume_window).sum()
    frame["updown_vol_ratio_20d"] = (up_volume_20d / down_volume_20d.replace(0, np.nan)).clip(upper=5.0)
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

    # 3M(60d) 관점 느린 모멘텀 — SPY 상대 비율의 12-1 / 6-1 개월 변화율.
    # 최근 1개월(21일)은 단기 반전 소음이라 제외하는 학계 표준형. 상장 12개월 미만이면 NaN.
    benchmark_ratio = close / spy["Close"].reindex(frame.index)
    feature_frame["mom_12_1"] = benchmark_ratio.shift(21) / benchmark_ratio.shift(252) - 1
    feature_frame["mom_6_1"] = benchmark_ratio.shift(21) / benchmark_ratio.shift(126) - 1

    # 거래량 신호 이력(관찰기간): 발생 여부·경과일·종류를 전 기간 벡터로 계산.
    # evaluate_row가 행 하나만 받아도 "며칠 전에 신호가 나왔는지" 알 수 있고, 백테스트도 그대로 재현된다.
    # 조건은 evaluate_row의 당일 판정(유동성+RS+서지/매집)과 동일해야 한다.
    surge_series = (
        (feature_frame["daily_return"] > 0)
        & (feature_frame["volume_ratio"] >= config.surge_ratio_min)
        & (feature_frame["volume_ratio"] < config.surge_ratio_max)
    ).fillna(False)
    acc_series = feature_frame["accumulation_days_10d"].fillna(0) >= config.accumulation_trigger_days
    trigger_series = (
        (feature_frame["avg_dollar_volume_20d"] >= config.min_dollar_volume)
        & (feature_frame["rs_spy_20d"] > 0)
        & (surge_series | acc_series)
    ).fillna(False)
    positions = pd.Series(np.arange(len(feature_frame), dtype=float), index=feature_frame.index)
    last_trigger_position = positions.where(trigger_series).ffill()
    feature_frame["trigger_today"] = trigger_series
    feature_frame["days_since_trigger"] = positions - last_trigger_position
    trigger_signal = pd.Series(
        np.select(
            [surge_series & acc_series, surge_series, acc_series],
            ["surge+acc", "surge", "acc"],
            default="",
        ),
        index=feature_frame.index,
    )
    feature_frame["trigger_signal_type"] = trigger_signal.where(trigger_series).ffill().fillna("")
    # 신호일 종가 = 발생가. 신호 대비 등락률(현재가/발생가-1)은 표시용 — 예측력 없음이 백테스트로
    # 확인됐으므로(IC -0.03, t=-1.3) 판정·점수에 쓰지 말 것. 참조 앵커 전용.
    feature_frame["trigger_price"] = close.where(trigger_series).ffill()

    return feature_frame


def evaluate_row(
    ticker: str,
    meta: pd.Series,
    row: pd.Series,
    has_sector: bool,
    market_state: str,
    config: ScreenerConfig,
) -> dict[str, object]:
    """피처 행 하나를 받아 필터 판정·등급·발생가를 계산한다. row.name은 날짜 인덱스."""
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

    # 거래량 신호: ① 당일 서지 (거래량 1.5~5배 + 양봉) ② 지속 매집 (10일 내 매집일 6+)
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

    # 관찰기간: 최근 trigger_window_days(당일 포함) 내 신호 발생 여부.
    # 신호 당일이 아니어도 기간 내에 있고 오늘 유동성·RS가 유지되면 후보로 남긴다.
    days_since_val = row.get("days_since_trigger", np.nan)
    days_since_trigger = int(days_since_val) if not pd.isna(days_since_val) else None
    in_trigger_window = (
        days_since_trigger is not None and days_since_trigger <= config.trigger_window_days - 1
    )
    triggered_today = bool(row.get("trigger_today", False))
    if in_trigger_window and not signal_type:
        # 신호 당일이 아닌 유지 상태 — 관찰기간을 연 신호의 종류를 표시
        signal_type = str(row.get("trigger_signal_type", "") or "")

    # 하드 필터: 유동성 + RS+ + 관찰기간 내 거래량 신호. MA/과열은 배제하지 않고 등급·경고로 표시
    passed = liquidity_ok and rs_positive and in_trigger_window

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

    # 발생가(신호일 종가)와 신호 대비 등락률 — 참고용.
    # 구 buy_price(MA20/60×1.01 눌림 기준)는 폐기: 백테스트에서 MA 괴리가 클수록 오히려
    # 20d 수익이 좋았고(IC +0.06, t=+1.8) "눌림 대기" 서사를 지지하는 근거가 없었다.
    trigger_price_val = row.get("trigger_price", np.nan)
    if not pd.isna(trigger_price_val):
        trigger_price = round(float(trigger_price_val), 2)
        ext_from_trigger = round(current_close / float(trigger_price_val) - 1, 4)
    else:
        trigger_price = None
        ext_from_trigger = None

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
        "updown_vol_ratio_20d": row["updown_vol_ratio_20d"],
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
        "triggered_today": triggered_today,
        "days_since_trigger": days_since_trigger,
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
        "trigger_price": trigger_price,
        "ext_from_trigger": ext_from_trigger,
        "base_stability": base_stability,
        "sector_etf_to_52w_high": sector_etf_to_52w_high,
        "mom_12_1": row.get("mom_12_1", np.nan),
        "mom_6_1": row.get("mom_6_1", np.nan),
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


# 점수 v3 percentile 가중치 — add_scores()와 대시보드 문서(툴팁·퍼널 칩·필터링 방식)가 공유하는 단일 소스.
# 값 변경 시 scripts/backtest.py 전/후 비교 필수.
SCORE_WEIGHTS_V3: dict[str, float] = {
    "rs_spy_50d": 0.25,
    "rs_spy_20d": 0.20,
    "updown_vol_ratio_20d": 0.20,
    "rs_sector_20d": 0.15,
    "rs_sector_50d": 0.10,
    "rs_spy_near_high": 0.10,
}


def add_scores(results: pd.DataFrame, config: ScreenerConfig | None = None) -> pd.DataFrame:
    results = results.copy()
    config = config or ScreenerConfig()
    if results.empty:
        results["score"] = pd.Series(dtype=float)
        return results

    # 전체 종목에 스코어 부여 — 하드 필터 통과 여부와 무관하게 순위 파악 가능
    # v3 (2026-08-21): 워크포워드 + Newey-West 보정 + 리스크 조정 IC 재검증 후 재구성.
    #   - 매집일수·거래량추세 제거: NW 보정 t=1.27/0.46, 리스크 조정 후 예측력 0.
    #   - rs_spy_50d 상향·rs_sector_50d 신규: 베타조정·변동성잔차 타깃에서 가장 견고 (t=2.3~3.6).
    #   - rs_spy_near_high 신규: NW t=3.31, 양수 스냅샷 74%.
    #   - updown_vol_ratio_20d 신규: 거래량 축 대체 (NW t=2.97, 양수 71%).
    #   - rs_qqq는 미채택: 횡단면 percentile에서 rs_spy와 순위가 수학적으로 동일 (벤치마크 항이 공통 상수).
    #   - atr_ratio 계속 보류: IC 최상위지만 절반이 베타 노출 (베타조정 시 +0.092→+0.041, t=1.85).
    # 근거: 62 주간 스냅샷(2025-02-11~2026-04-30), 전반 학습/후반 검증 분할에서 v2 대비
    # 유니버스 IC 0.048→0.052 (후반 0.045→0.056), 후보 top10 차이는 비유의(paired t<1.3).
    scored = results.copy()

    w = SCORE_WEIGHTS_V3
    scored["score"] = (
        w["rs_spy_20d"] * scored["rs_spy_20d"].rank(pct=True)                                    # 단기 RS
        + w["rs_spy_50d"] * scored["rs_spy_50d"].rank(pct=True)                                  # 중기 RS — 리스크 조정 후 최강
        + w["rs_sector_20d"] * scored["rs_sector_20d"].rank(pct=True, na_option="bottom")        # 단기 섹터 RS
        + w["rs_sector_50d"] * scored["rs_sector_50d"].rank(pct=True, na_option="bottom")        # 중기 섹터 RS
        + w["rs_spy_near_high"] * scored["rs_spy_near_high"].astype(float).rank(pct=True)        # RS선 50일 고점 근접
        + w["updown_vol_ratio_20d"] * scored["updown_vol_ratio_20d"].rank(pct=True, na_option="bottom")  # 상승/하락일 거래량비
    )

    # 국면별 A/B 가중 (2026-08): 국면-등급 정합 후보에 소프트 보너스. 게이팅(후보 제거)은 하지 않는다.
    # 근거: Confirmed Uptrend에서 A(+2.51%) > B(+1.10%), Correction에서 B(+2.67%) > A(+0.50%).
    # 저장된 스냅샷 재정렬 비교에서 top10 바스켓 +2.82%→+3.12%, 전/후반 분할 모두 개선 (paired t=1.86).
    # UUP는 스냅샷 4개뿐이라 중립 처리.
    regime_fit = (
        ((results["market_state"] == "Confirmed Uptrend") & (results["grade"] == "A"))
        | ((results["market_state"] == "Market in Correction") & (results["grade"] == "B"))
    )
    scored["score"] = scored["score"] + config.regime_fit_bonus * regime_fit.astype(float)

    # score_3m (2026-08-21, 실험적): 60d(3M) 초과수익 관점의 느린 모멘텀 순위. 정렬에는 쓰지 않는 참고 컬럼.
    # 근거: fwd_excess_60d IC — mom_12_1 +0.093(NW t=2.44, 양수 79%), mom_6_1 +0.074(t=1.67).
    # 십분위 스프레드(60d): 최상위 +9.4% vs 최하위 -1.1% (단조). 현행 score 구성요소는 60d에서 무력.
    # 주의: 장기 모멘텀은 생존 편향(현재 시총 상위 유니버스)이 가장 크게 부풀리는 지표 —
    # 절대 수익 기대가 아니라 종목 간 상대 순위 전용. 상장 12개월 미만 종목은 NaN.
    if "mom_12_1" in scored.columns:
        results["score_3m"] = (
            0.60 * scored["mom_12_1"].rank(pct=True)
            + 0.40 * scored["mom_6_1"].rank(pct=True)
        )

    results["score"] = scored["score"]
    return results.sort_values(
        ["passed_hard_filters", "score", "close_to_50d_high"],
        ascending=[False, False, False],
    )


def fetch_earnings_flags(tickers: list[str], horizon_days: int = 20) -> dict[str, str]:
    """후보 종목의 다음 어닝스 날짜를 조회한다. 반환: {ticker: ISO 날짜} (없으면 키 없음).

    yfinance calendar는 종목당 API 호출이라 후보(수십 개)에만 사용하고 일별 캐시를 둔다.
    조회 실패는 조용히 건너뛴다 — 어닝스 플래그는 보조 정보라 실행을 막으면 안 된다.
    """
    import json

    cache_dir = DEFAULT_YFINANCE_CACHE_DIR / "earnings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{pd.Timestamp.today():%Y-%m-%d}.json"
    keep_after = (pd.Timestamp.today() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    for old in cache_dir.glob("*.json"):
        if old.stem < keep_after:
            old.unlink(missing_ok=True)
    cached: dict[str, str] = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except Exception:
            cached = {}

    to_fetch = [t for t in tickers if t not in cached]

    def _get(ticker: str) -> tuple[str, str | None]:
        try:
            calendar = yf.Ticker(ticker).calendar
            dates = calendar.get("Earnings Date") if isinstance(calendar, dict) else None
            if dates:
                return ticker, pd.Timestamp(dates[0]).date().isoformat()
        except Exception:
            pass
        return ticker, None

    if to_fetch:
        with ThreadPoolExecutor(max_workers=10) as executor:
            for ticker, date in executor.map(_get, to_fetch):
                cached[ticker] = date or ""
        try:
            cache_path.write_text(json.dumps(cached))
        except Exception:
            pass
    return {t: d for t, d in cached.items() if d}


def add_earnings_flags(results: pd.DataFrame, horizon_days: int = 20) -> pd.DataFrame:
    """후보 종목에 next_earnings_date / earnings_within_20d 컬럼과 경고 라벨을 붙인다."""
    results = results.copy()
    results["next_earnings_date"] = ""
    results["earnings_within_20d"] = False
    candidate_mask = results["passed_hard_filters"].fillna(False)
    tickers = results.loc[candidate_mask, "ticker"].tolist()
    if not tickers:
        return results
    earnings = fetch_earnings_flags(tickers, horizon_days)
    if not earnings:
        return results
    as_of = pd.Timestamp(results["date"].dropna().max())
    cutoff = as_of + pd.tseries.offsets.BDay(horizon_days)
    for idx in results.index[candidate_mask]:
        date_str = earnings.get(results.at[idx, "ticker"], "")
        if not date_str:
            continue
        earnings_ts = pd.Timestamp(date_str)
        if earnings_ts < as_of:
            continue  # yfinance가 직전(과거) 실적일을 반환하는 경우 — 다음 일정 미공표
        results.at[idx, "next_earnings_date"] = date_str
        if as_of <= earnings_ts <= cutoff:
            results.at[idx, "earnings_within_20d"] = True
            existing = results.at[idx, "warnings"] or ""
            label = "어닝스20일내"
            results.at[idx, "warnings"] = f"{existing},{label}" if existing else label
    return results


def correlation_clusters(
    tickers: list[str],
    prices: dict[str, pd.DataFrame],
    config: ScreenerConfig,
) -> dict[str, list[str]]:
    """후보 간 수익률 상관 클러스터. 반환: {ticker: 클러스터 멤버 리스트(자신 포함)}.

    섹터 라벨 대신 실제 상관을 쓴다 — 금은주는 섹터가 같아도 상관 0.92+로 잡히고,
    같은 중국 ADR이라도 상관이 낮으면(BABA-HTHT 0.15 실측) 묶이지 않는다.
    """
    returns = {}
    for ticker in tickers:
        p = prices.get(ticker)
        if p is not None and not p.empty:
            returns[ticker] = p["Close"].pct_change().tail(config.verdict_corr_window)
    if len(returns) < 2:
        return {}
    corr = pd.DataFrame(returns).corr(min_periods=int(config.verdict_corr_window * 0.6))

    parent = {t: t for t in corr.columns}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            value = corr.loc[a, b]
            if not pd.isna(value) and value >= config.verdict_corr_threshold:
                parent[find(a)] = find(b)

    groups: dict[str, list[str]] = {}
    for t in cols:
        groups.setdefault(find(t), []).append(t)
    result: dict[str, list[str]] = {}
    for members in groups.values():
        if len(members) >= config.verdict_cluster_min_size:
            for t in members:
                result[t] = members
    return result


REGIME_FIT_STATES = {"Confirmed Uptrend": "A", "Market in Correction": "B"}


def assign_verdicts(
    results: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    config: ScreenerConfig | None = None,
) -> pd.DataFrame:
    """후보 전체에 결정론적 판정(진입 검토/대기/스킵)과 사유를 부여한다.

    규칙 순서(위에서부터, 먼저 걸리는 것이 판정):
      S1 acc 신호 D+3 이상 — acc는 감쇠가 빨라 소멸 단계
      S2 국면 부정합 등급 + score_3m < 0.30 — 조정장 A +0.5% vs B +2.7%
      S3 상관 클러스터(≥3개) 중복 — 국면 정합 등급 우선·점수순 상위 2개만 대표
      S4 score_3m < 0.10 + 분산 경고 — 단기 이벤트 소진
      S5 후보 내 점수 percentile < 40% — 검증상 이 구간의 "진입 검토"는 스킵보다도 나빴음
      W1 어닝스 20일 내 — 이벤트 베팅 분리 (다음 어닝스가 없으면 해당 없음)
      W2 과열 경고 + 당일 저가권 마감(<0.5) — 눌림 대기
      W3 score_3m < 0.15 — 신호 재발생/중기 추세 대기
      통과 → 진입 검토
    판정은 백테스트로 검증 가능해야 하므로 여기서만 결정한다. AI는 설명문만 작성.
    """
    config = config or ScreenerConfig()
    results = results.copy()
    results["verdict"] = ""
    results["verdict_reason"] = ""
    candidate_mask = results["passed_hard_filters"].fillna(False)
    cand = results.loc[candidate_mask]
    if cand.empty:
        return results

    clusters = correlation_clusters(cand["ticker"].tolist(), prices, config)
    score_pct = cand["score"].rank(pct=True)

    # 클러스터 대표 선정: 국면 정합 등급 우선, 그 다음 점수순
    market_state = str(cand["market_state"].dropna().iloc[0]) if "market_state" in cand.columns else ""
    fit_grade = REGIME_FIT_STATES.get(market_state)
    reps: dict[str, list[str]] = {}
    for members in {id(v): v for v in clusters.values()}.values():
        sub = cand[cand["ticker"].isin(members)].copy()
        sub["_fit"] = (sub["grade"] == fit_grade).astype(int) if fit_grade else 0
        ranked = sub.sort_values(["_fit", "score"], ascending=[False, False])["ticker"].tolist()
        for t in members:
            reps[t] = ranked[: config.verdict_reps_per_cluster]

    for idx in cand.index:
        row = results.loc[idx]
        ticker = row["ticker"]
        warnings = str(row.get("warnings") or "")
        s3m_val = row.get("score_3m", np.nan)
        s3m = float(s3m_val) if not pd.isna(s3m_val) else 0.0
        days = row.get("days_since_trigger")
        mismatch = fit_grade is not None and row["grade"] != fit_grade
        in_cluster = ticker in clusters

        if row["signal_type"] == "acc" and days is not None and days >= 3:
            verdict, reason = "스킵", f"acc D+{int(days)} 관찰 종료 — 신호 재발생 전까지 관망"
        elif mismatch and s3m < config.verdict_s2_score3m:
            verdict, reason = "스킵", f"국면({market_state}) 부정합 {row['grade']}등급 + 중기 근거 부족(3M {s3m:.2f})"
        elif in_cluster and ticker not in reps.get(ticker, []):
            verdict, reason = "스킵", f"상관 클러스터 중복 — 대표: {'·'.join(reps[ticker])}"
        elif s3m < config.verdict_s4_score3m and "분산" in warnings:
            verdict, reason = "스킵", f"단기 이벤트성(3M {s3m:.2f}) + 분산 매물"
        elif score_pct[idx] < config.verdict_min_score_pct:
            verdict, reason = "스킵", f"점수 후순위 (후보 내 하위 {score_pct[idx]:.0%}) — 우선순위 미달"
        elif (lambda v: False if pd.isna(v) else bool(v))(row.get("earnings_within_20d", False)):
            verdict, reason = "대기", f"어닝스 {row.get('next_earnings_date', '')} — 발표 후 재평가"
        elif "과열" in warnings and float(row.get("close_position") or 1.0) < 0.5:
            verdict, reason = "대기", "과열 + 당일 저가권 마감 — 단기 소화 후 재평가"
        elif s3m < config.verdict_w3_score3m:
            verdict, reason = "대기", f"단기 이벤트성(3M {s3m:.2f}) — 신호 재발생 또는 중기 추세 형성 대기"
        else:
            parts = []
            if fit_grade and row["grade"] == fit_grade:
                parts.append(f"국면 정합 {row['grade']}등급")
            parts.append(f"{row['signal_type']} D+{int(days) if days is not None else '?'}")
            if in_cluster:
                parts.append(f"클러스터 대표({len(clusters[ticker])}종목 중)")
            verdict, reason = "진입 검토", " · ".join(parts)

        results.at[idx, "verdict"] = verdict
        results.at[idx, "verdict_reason"] = reason
    return results


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
    symbols = sorted(set(stock_symbols + list(BENCHMARKS) + list(VIX_SYMBOLS) + sector_symbols))

    prices, sync_report = download_prices(symbols, config)
    missing = sorted(set(symbols) - set(prices))
    missing_required = [symbol for symbol in BENCHMARKS if symbol not in prices]
    if missing_required:
        raise RuntimeError(
            "Required benchmark data missing: "
            + ", ".join(missing_required)
            + ". Check network access or yfinance availability."
        )

    regime_series, regime_detail = calculate_market_state_series(prices, config)
    market_state = str(regime_series.iloc[-1])
    rows = [
        result
        for _, meta in universe.iterrows()
        if (result := evaluate_stock(meta["ticker"], meta, prices, config, market_state))
        is not None
    ]
    results = add_scores(pd.DataFrame(rows), config)
    if results.empty:
        raise RuntimeError("No stocks could be evaluated. Check ticker data and download results.")
    results = add_earnings_flags(results)
    results = assign_verdicts(results, prices, config)
    results["regime_trend"] = regime_detail["trend"]
    results["regime_breadth"] = regime_detail["breadth"]
    results["regime_vix_ratio"] = regime_detail["vix_ratio"]

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
        from_cache=sync_report.from_cache,
        stale_symbols=sorted(sync_report.stale),
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
            "days_since_trigger",
            "score",
            "rs_spy_20d",
            "rs_spy_50d",
            "rs_sector_20d",
            "volume_ratio",
            "accumulation_days_10d",
            "trigger_price",
            "ext_from_trigger",
            "warnings",
        ]
        print()
        print(format_percent_columns(candidates.head(args.top)[display_columns]).to_string(index=False))


if __name__ == "__main__":
    main()
