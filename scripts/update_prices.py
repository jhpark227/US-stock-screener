"""가격 저장소 수동 동기화 CLI.

스크리너(main.py)가 매일 자동으로 동기화하므로 평소엔 필요 없다.
전체 리프레시(--full)나 스크리너 실행 전 미리 받아두고 싶을 때 사용.

    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/update_prices.py [--full] [--period 2y]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import BENCHMARKS, VIX_SYMBOLS, DEFAULT_TICKER_FILE, load_universe  # noqa: E402
from price_store import sync_prices  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="parquet 가격 저장소 동기화")
    parser.add_argument("--full", action="store_true", help="증분 대신 전 종목 전체 다운로드")
    parser.add_argument("--period", default="2y", help="전체 다운로드 시 기간 (기본 2y)")
    parser.add_argument("--tickers", type=Path, default=DEFAULT_TICKER_FILE)
    args = parser.parse_args()

    universe = load_universe(args.tickers)
    sector_symbols = sorted(set(universe["sector_etf"].dropna()) - {""})
    symbols = sorted(
        set(universe["ticker"]) | set(BENCHMARKS) | set(VIX_SYMBOLS) | set(sector_symbols)
    )

    print(f"동기화 시작: {len(symbols)}개 심볼 (full={args.full})")
    prices, report = sync_prices(symbols, period=args.period, force_full=args.full)
    print(f"완료 — {report.summary()}")
    if report.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
