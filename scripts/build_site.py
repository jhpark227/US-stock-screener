"""정적 대시보드 빌더 — Cloudflare Pages 배포용.

최신 스크리너 결과(outputs/)와 parquet 가격 저장소를 읽어 `site/`에 정적 미러를 생성한다.
서버(dashboard.py) 없이 동작하도록 index.html의 getJson을 정적 JSON 라우터로 패치한다.

    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/build_site.py

산출물:
    site/index.html          패치된 대시보드 (스크리너 실행 버튼 숨김)
    site/api/latest.json     /api/latest 응답 (코멘터리 포함)
    site/api/diff.json       /api/diff 응답
    site/api/universe.json   /api/universe 응답
    site/api/tickers.json    {ticker: /api/ticker 응답} — 유니버스 전 종목
    site/api/charts.json     {ticker: /api/chart 응답(252일)} — 후보·ETF·벤치마크만
    site/outputs/*.csv       최신 결과 CSV (다운로드 링크용)

launchd(run_screener.sh)가 코멘터리 생성 후 이 스크립트를 실행하고 site/를 커밋·푸시하면
Cloudflare Pages가 자동으로 재배포한다.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from dashboard import (  # noqa: E402
    INDEX_HTML,
    OUTPUT_DIR,
    TICKER_FILE,
    chart_data,
    config_payload,
    diff_response,
    latest_output_response,
    screener_row_payload,
    sector_counts,
)
from main import BENCHMARKS, ScreenerConfig, load_universe  # noqa: E402

SITE_DIR = PROJECT_ROOT / "site"
CHART_DAYS = 252  # 차트 모달 최대 기간(1Y) — 프론트가 클라이언트에서 잘라 쓴다

# index.html 패치 앵커 — dashboard.py의 INDEX_HTML이 바뀌어 앵커가 사라지면 빌드가 실패한다(의도).
GETJSON_ANCHOR = """    async function getJson(url, options = {}) {
      const response = await fetch(url, options);"""

GETJSON_STATIC = """    // --- 정적 미러 모드: /api/* 호출을 미리 구운 JSON으로 라우팅 ---
    const STATIC_BUNDLES = {};
    function loadBundle(path) {
      if (!STATIC_BUNDLES[path]) {
        STATIC_BUNDLES[path] = fetch(path).then(r => {
          if (!r.ok) throw new Error(`데이터 파일 로드 실패 (HTTP ${r.status})`);
          return r.json();
        });
      }
      return STATIC_BUNDLES[path];
    }
    async function getJson(url, options = {}) {
      const [path, query] = url.split("?");
      const params = new URLSearchParams(query || "");
      if (path === "/api/latest")   return loadBundle("api/latest.json");
      if (path === "/api/diff")     return loadBundle("api/diff.json");
      if (path === "/api/universe") return loadBundle("api/universe.json");
      if (path === "/api/ticker") {
        const bundle = await loadBundle("api/tickers.json");
        const sym = (params.get("symbol") || "").toUpperCase();
        return bundle[sym] || { symbol: sym, in_universe: false };
      }
      if (path === "/api/chart") {
        const bundle = await loadBundle("api/charts.json");
        const sym = (params.get("symbol") || "").toUpperCase();
        const d = bundle[sym];
        if (!d) return { error: "정적 미러에는 후보·ETF 차트만 포함됩니다." };
        const days = Math.min(parseInt(params.get("days") || "63", 10) || 63, d.dates.length);
        return {
          ...d,
          dates: d.dates.slice(-days),
          prices: d.prices.slice(-days),
          volumes: d.volumes ? d.volumes.slice(-days) : null,
        };
      }
      const response = await fetch(url, options);"""

RUNBUTTON_ANCHOR = 'const runButton        = document.getElementById("runButton");'
RUNBUTTON_STATIC = (
    RUNBUTTON_ANCHOR
    + '\n    runButton.style.display = "none";  // 정적 미러 — 서버 없음'
    + '\n    refreshButton.style.display = "none";  // 정적 미러 — 데이터가 빌드 시점에 고정, F5로 충분'
)


def _patch_index_html() -> str:
    html = INDEX_HTML
    for anchor, replacement in ((GETJSON_ANCHOR, GETJSON_STATIC), (RUNBUTTON_ANCHOR, RUNBUTTON_STATIC)):
        if html.count(anchor) != 1:
            raise RuntimeError(
                f"INDEX_HTML 패치 앵커를 찾지 못했습니다 (count={html.count(anchor)}). "
                "dashboard.py 변경 시 scripts/build_site.py 앵커를 함께 갱신하세요:\n" + anchor[:80]
            )
        html = html.replace(anchor, replacement)
    return html


def _json_default(value: object) -> object:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, default=_json_default)
    text = text.replace(": NaN", ": null")  # pandas 유래 NaN 방어
    path.write_text(text)


def build_tickers_bundle(universe: pd.DataFrame, results: pd.DataFrame | None) -> dict[str, object]:
    bundle: dict[str, object] = {}
    results_by_ticker = (
        results.set_index("ticker", drop=False) if results is not None else None
    )
    for _, meta in universe.iterrows():
        symbol = str(meta["ticker"])
        name = str(meta.get("name", "") or "")
        sector = str(meta.get("sector", "") or "")
        if results_by_ticker is None or symbol not in results_by_ticker.index:
            bundle[symbol] = {
                "symbol": symbol, "in_universe": True, "name": name,
                "sector": sector, "has_screener_result": False,
            }
        else:
            bundle[symbol] = screener_row_payload(
                symbol, name, sector, results_by_ticker.loc[symbol]
            )
    return bundle


def main() -> None:
    latest = latest_output_response()
    if not latest.get("has_result"):
        sys.exit("outputs/에 스크리너 결과가 없습니다 — main.py를 먼저 실행하세요.")

    universe = load_universe(TICKER_FILE)
    universe_files = sorted(OUTPUT_DIR.glob("screener_universe_*.csv"))
    results = pd.read_csv(universe_files[-1]) if universe_files else None

    # 차트 대상: 후보 전 종목 + 전일 대비 diff 등장 종목 + 벤치마크 + 섹터 ETF
    diff = diff_response()
    chart_symbols = {str(c["ticker"]) for c in latest.get("candidates", [])}
    for key in ("new_entries", "dropped", "upgraded", "downgraded"):
        chart_symbols.update(diff.get(key) or [])
    chart_symbols.update(BENCHMARKS)
    chart_symbols.update(set(universe["sector_etf"].dropna()) - {""})

    charts: dict[str, object] = {}
    for symbol in sorted(chart_symbols):
        data = chart_data(symbol, days=CHART_DAYS)
        if "error" not in data:
            charts[symbol] = data

    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    (SITE_DIR / "api").mkdir(parents=True)
    (SITE_DIR / "outputs").mkdir()

    (SITE_DIR / "index.html").write_text(_patch_index_html())
    _write_json(SITE_DIR / "api" / "latest.json", latest)
    _write_json(SITE_DIR / "api" / "diff.json", diff)
    _write_json(SITE_DIR / "api" / "universe.json", {
        "ticker_file": str(TICKER_FILE),
        "universe_count": len(universe),
        "sector_counts": sector_counts(TICKER_FILE),
        "config": config_payload(ScreenerConfig()),
    })
    _write_json(SITE_DIR / "api" / "tickers.json", build_tickers_bundle(universe, results))
    _write_json(SITE_DIR / "api" / "charts.json", charts)

    # 다운로드 링크(/outputs/*.csv) 대상 파일 복사 — 최신 universe/candidates 한 쌍만
    for key in ("universe_csv", "candidates_csv"):
        link = latest.get(key)
        if link:
            src = OUTPUT_DIR / Path(str(link)).name
            if src.exists():
                shutil.copy2(src, SITE_DIR / "outputs" / src.name)

    total_kb = sum(f.stat().st_size for f in SITE_DIR.rglob("*") if f.is_file()) / 1024
    print(
        f"site/ 빌드 완료 — 날짜 {latest.get('date')}, 차트 {len(charts)}종목, "
        f"총 {total_kb:,.0f}KB"
    )


if __name__ == "__main__":
    main()
