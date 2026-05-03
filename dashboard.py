from __future__ import annotations

import argparse
import json
import math
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd

from main import DEFAULT_OUTPUT_DIR, DEFAULT_TICKER_FILE, ScreenerConfig, load_universe, run_screener


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / DEFAULT_OUTPUT_DIR
TICKER_FILE = PROJECT_ROOT / DEFAULT_TICKER_FILE


INDEX_HTML = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RS Volume Screener</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f6f3;
      --surface: #ffffff;
      --line: #deded7;
      --text: #171a16;
      --muted: #666d63;
      --accent: #18745a;
      --accent-soft: #dcebe5;
      --warn: #a35f00;
      --danger: #a83a35;
      --shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }

    button, input, select { font: inherit; }

    .shell {
      width: min(1320px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }

    header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 18px;
    }

    h1 { margin: 0; font-size: 24px; line-height: 1.1; }

    .subtle { color: var(--muted); }

    .controls {
      display: grid;
      grid-template-columns: minmax(180px, 240px) auto;
      gap: 10px;
      align-items: end;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 14px;
      margin-bottom: 16px;
    }

    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
    }

    input, select {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 8px 10px;
    }

    button {
      min-height: 38px;
      border: 1px solid #12634c;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      padding: 8px 14px;
      cursor: pointer;
      white-space: nowrap;
    }

    button:disabled { cursor: wait; opacity: 0.65; }

    .status {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 32px;
      margin-bottom: 16px;
      color: var(--muted);
    }

    .status strong { color: var(--text); font-weight: 650; }

    .grid {
      display: grid;
      grid-template-columns: 1.1fr 1.6fr;
      gap: 16px;
      margin-bottom: 16px;
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
    }

    .panel h2 { margin: 0 0 12px; font-size: 15px; }

    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }

    .metric {
      min-height: 82px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 13px;
    }

    .metric .label { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
    .metric .value { font-size: 24px; font-weight: 700; line-height: 1.1; }
    .metric .note { color: var(--muted); margin-top: 7px; font-size: 12px; }

    .funnel { display: grid; gap: 9px; }

    .step {
      display: grid;
      grid-template-columns: 140px 1fr 52px 52px 52px;
      gap: 10px;
      align-items: center;
      min-height: 30px;
    }

    .bar { height: 10px; border-radius: 999px; background: #ededeb; overflow: hidden; }
    .bar span { display: block; height: 100%; width: 0%; background: var(--accent); }
    .step .count   { text-align: right; font-variant-numeric: tabular-nums; }
    .step .removed { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
    .step .pct     { text-align: right; color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
    .step.bottleneck .removed { color: var(--danger); font-weight: 700; }

    .funnel-params {
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .funnel-params .chip { font-size: 11px; padding: 4px 8px; }

    /* Sector table */
    .sector-table {
      width: 100%;
      min-width: 0;
      border-collapse: collapse;
      font-size: 13px;
    }

    .sector-table th {
      position: static;
      background: none;
      color: var(--muted);
      font-size: 11px;
      font-weight: 650;
      padding: 0 6px 6px;
      border-bottom: 1px solid var(--line);
      cursor: default;
    }

    .sector-table th:hover { background: none; }

    .sector-table td {
      padding: 6px 6px;
      border-bottom: 1px solid #f0f0ec;
      white-space: nowrap;
    }

    .sector-table tr:last-child td { border-bottom: 0; }

    .sector-table .hit-bar {
      width: 60px;
      height: 6px;
      border-radius: 999px;
      background: #ededeb;
      overflow: hidden;
      display: inline-block;
      vertical-align: middle;
      margin-right: 6px;
    }

    .sector-table .hit-bar span {
      display: block;
      height: 100%;
      background: var(--accent);
    }

    .sector-table .hit-rate {
      font-size: 12px;
      color: var(--accent);
      font-weight: 650;
    }

    .sector-table .zero { color: var(--muted); }

    .chips { display: flex; flex-wrap: wrap; gap: 8px; }

    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fbfbfa;
      padding: 6px 9px;
      color: var(--muted);
      font-size: 12px;
    }

    /* Candidates panel */
    .candidate-header {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }

    .candidate-header h2 { margin: 0; font-size: 15px; }

    .grade-group {
      display: flex;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }

    .grade-btn {
      min-height: 30px;
      padding: 4px 12px;
      border: none;
      border-radius: 0;
      background: #fff;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      border-right: 1px solid var(--line);
    }

    .grade-btn:last-child { border-right: none; }

    .grade-btn.active {
      background: var(--accent);
      color: #fff;
    }

    .sector-select {
      min-height: 30px;
      padding: 4px 8px;
      font-size: 12px;
      width: auto;
      min-width: 160px;
    }

    .candidate-count {
      margin-left: auto;
      font-size: 12px;
      color: var(--muted);
    }

    /* Table */
    .table-wrap {
      overflow-x: auto;
      overflow-y: auto;
      max-height: 820px;
      border: 1px solid var(--line);
      border-radius: 8px;
    }

    table {
      width: 100%;
      min-width: 1040px;
      border-collapse: collapse;
      background: var(--surface);
    }

    th, td {
      padding: 10px 11px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;
    }

    th {
      position: sticky;
      top: 0;
      z-index: 1;
      color: var(--muted);
      background: #fafaf8;
      font-size: 12px;
      font-weight: 650;
      cursor: pointer;
      user-select: none;
    }

    th:hover { background: #f0f0ec; }

    th .sort-arrow {
      display: inline-block;
      margin-left: 4px;
      opacity: 0.3;
      font-size: 10px;
    }

    th.sort-asc .sort-arrow,
    th.sort-desc .sort-arrow { opacity: 1; color: var(--accent); }

    tr:last-child td { border-bottom: 0; }

    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .rank { text-align: right; color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; width: 32px; }
    .ticker { font-weight: 700; }

    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 28px;
      height: 24px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #0e5b45;
      font-weight: 700;
    }

    .badge.b { background: #eee9d9; color: var(--warn); }

    .empty { padding: 22px; color: var(--muted); text-align: center; }
    .error { color: var(--danger); }

    .links { display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
    .links a { color: var(--accent); text-decoration: none; font-weight: 650; }

    .desc {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 14px 18px;
      margin-bottom: 16px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }

    .desc summary {
      cursor: pointer;
      font-weight: 650;
      color: var(--text);
      font-size: 13px;
      user-select: none;
      list-style: none;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .desc summary::before { content: "▶"; font-size: 10px; transition: transform 0.15s; }
    .desc[open] summary::before { transform: rotate(90deg); }
    .desc ul { margin: 10px 0 0; padding-left: 18px; }
    .desc li { margin-bottom: 4px; }
    .desc li strong { color: var(--text); }

    /* Insight bar */
    .insight-bar {
      background: var(--surface);
      border: 1px solid var(--line);
      border-left: 3px solid var(--accent);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 14px 18px;
      margin-bottom: 16px;
      display: none;
    }

    .insight-bar ul {
      margin: 0;
      padding: 0;
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 7px;
    }

    .insight-bar li {
      font-size: 13px;
      line-height: 1.55;
      color: var(--text);
      padding-left: 16px;
      position: relative;
    }

    .insight-bar li::before {
      content: "·";
      position: absolute;
      left: 4px;
      color: var(--accent);
      font-weight: 700;
    }

    .insight-bar li .tag {
      display: inline-block;
      font-size: 11px;
      font-weight: 700;
      padding: 1px 6px;
      border-radius: 4px;
      margin-right: 6px;
      background: var(--accent-soft);
      color: var(--accent);
      vertical-align: middle;
    }

    .insight-bar li .tag.warn {
      background: #eee9d9;
      color: var(--warn);
    }

    .insight-bar li .tag.neutral {
      background: #ededeb;
      color: var(--muted);
    }

    /* Guide accordion */
    .guide-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0 32px;
      margin-top: 10px;
    }

    .guide-term {
      padding: 9px 0;
      border-bottom: 1px solid #f0f0ec;
    }

    .guide-term:nth-last-child(-n+2) { border-bottom: 0; }

    .guide-term dt {
      font-size: 13px;
      font-weight: 650;
      color: var(--text);
      margin-bottom: 3px;
    }

    .guide-term dd {
      margin: 0;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.55;
    }

    /* Ticker search */
    .ticker-search {
      display: flex;
      gap: 8px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 12px 14px;
      margin-bottom: 16px;
      align-items: center;
    }

    .ticker-search label {
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
      display: block;
    }

    #tickerInput {
      width: 120px;
      min-height: 34px;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.03em;
    }

    #tickerResult {
      flex: 1;
      font-size: 13px;
    }

    .tr-name    { font-weight: 650; }
    .tr-sector  { color: var(--muted); font-size: 12px; margin-left: 8px; }

    .tr-filters {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }

    .tr-filter {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      font-size: 12px;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fbfbfa;
      color: var(--muted);
    }

    .tr-filter.pass { border-color: #b3d9cc; background: var(--accent-soft); color: #0e5b45; }
    .tr-filter.fail { border-color: #f0c4c2; background: #fdf0ef; color: var(--danger); font-weight: 650; }

    .tr-metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }

    .tr-metrics span strong { color: var(--text); }

    @media (max-width: 980px) {
      .controls, .grid, .metrics { grid-template-columns: 1fr; }
      header, .status { flex-direction: column; align-items: stretch; }
      .step { grid-template-columns: 110px 1fr 44px 44px 44px; }
      .guide-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>RS Volume Screener</h1>
        <div class="subtle" id="universeLine">Universe loading</div>
      </div>
      <div style="display:flex;align-items:center;gap:12px">
        <div class="links" id="fileLinks"></div>
        <button id="refreshButton" type="button" style="background:none;border:1px solid var(--line);color:var(--text);font-size:13px;padding:6px 14px;min-height:34px">↻ Refresh</button>
      </div>
    </header>

    <details class="desc">
      <summary>필터링 방식</summary>
      <ul>
        <li><strong>유동성</strong> — 20일 평균 거래대금 ≥ 기준값. 저유동성 종목 제외.</li>
        <li><strong>RS 고점 근접</strong> — SPY 대비 상대강도(RS)가 최근 50일 RS 최고값의 98% 이상.</li>
        <li><strong>RS 20D 양수</strong> — 20일 SPY 대비 RS 변화율 &gt; 0.</li>
        <li><strong>MA50 위 + 상승</strong> — 종가 &gt; 50일 이동평균, MA50이 10일 전보다 높음.</li>
        <li><strong>50일 고점 근접</strong> — 종가 ≥ 최근 50일 최고가의 90%.</li>
        <li><strong>과열 없음</strong> — 종가 ≤ MA20의 125%, 5일 수익률 &lt; 40%, 당일 수익률 &lt; 25%.</li>
        <li><strong>등급 A</strong> — 위 조건 통과 + 거래량 품질(거래량비율 ≥ 1.3배, 양봉, 종가위치 ≥ 60%).</li>
        <li><strong>등급 B</strong> — 위 조건만 통과, 거래량 품질 미충족.</li>
        <li><strong>점수</strong> — RS 20D(35%) + RS 50D(25%) + 50일 고점 근접도(20%) + 거래량비율(15%) + 종가위치(5%).</li>
      </ul>
    </details>

    <details class="desc">
      <summary>지표 이해하기</summary>
      <dl class="guide-grid">
        <div class="guide-term">
          <dt>상대강도 (RS, Relative Strength)</dt>
          <dd>SPY(미국 전체 시장)를 기준으로 이 주식이 얼마나 더 강하게 움직이고 있는지 나타내는 비율입니다. RS &gt; 0이면 시장보다 더 많이 올랐다는 의미, RS &lt; 0이면 시장보다 뒤처지고 있다는 의미입니다.</dd>
        </div>
        <div class="guide-term">
          <dt>RS 20D / RS 50D</dt>
          <dd>각각 최근 20일·50일 동안 SPY 대비 상대강도 변화율입니다. 단기(20D)는 최근 모멘텀, 중기(50D)는 추세의 지속성을 봅니다. 둘 다 양수면 단·중기 모두 시장 대비 강한 흐름.</dd>
        </div>
        <div class="guide-term">
          <dt>RS 고점 근접 (RS Near High)</dt>
          <dd>최근 50일 중 RS가 가장 높았던 시점의 98% 수준 이상에 있다는 의미입니다. 과거에 한 번 강세를 보였고, 지금도 그 강도를 유지하고 있는 종목을 걸러냅니다. 조금 전에 RS 최고점을 찍고 지금 밀리는 종목은 탈락합니다.</dd>
        </div>
        <div class="guide-term">
          <dt>MA50 위 + 상승</dt>
          <dd>50일 이동평균선(최근 50일 평균 주가)보다 현재 주가가 위에 있고, 그 평균선 자체도 10일 전보다 높아진 상태입니다. 단순히 평균 위에 있는 게 아니라 추세 자체가 위를 향하고 있다는 의미입니다.</dd>
        </div>
        <div class="guide-term">
          <dt>50일 고점 근접</dt>
          <dd>최근 50일 최고가의 90% 이상에 위치해 있다는 의미입니다. "저점에서 반등 중"인 종목이 아니라, 고점 근처에서 버티고 있는 강한 종목만 선별합니다.</dd>
        </div>
        <div class="guide-term">
          <dt>과열 없음</dt>
          <dd>종가가 MA20의 125% 이하, 최근 5일 수익률 40% 미만, 당일 수익률 25% 미만. 이미 단기간에 급등한 종목을 제외합니다. 추격 매수 위험이 높은 종목보다 아직 상승 여력이 남은 종목을 봅니다.</dd>
        </div>
        <div class="guide-term">
          <dt>등급 A / B</dt>
          <dd><strong>A등급</strong>: 위 모든 조건 충족 + 당일 거래량이 20일 평균의 1.3배 이상이고 양봉이며 종가가 당일 범위 상단 60% 이상. 추세와 거래량이 동시에 확인된 가장 신뢰도 높은 신호.<br><strong>B등급</strong>: 추세 조건은 충족하나 거래량 확인 미충족. 진입 전 거래량 추이 별도 확인 권장.</dd>
        </div>
        <div class="guide-term">
          <dt>점수 (Score) · 섹터 RS</dt>
          <dd><strong>점수</strong>: RS 20D·50D, 고점 근접도, 거래량, 종가위치를 percentile 가중합으로 계산한 0~1 사이 값. 높을수록 현재 장에서 상대적으로 강한 종목.<br><strong>섹터 RS</strong>: SPY 대신 해당 섹터 ETF(IT→XLK 등)와 비교한 상대강도. 섹터 전체가 강한 건지, 그 안에서 특히 강한 종목인지 파악하는 데 활용.</dd>
        </div>
      </dl>
    </details>

    <div class="ticker-search">
      <label>티커 검색</label>
      <input type="text" id="tickerInput" placeholder="예: AAPL" autocomplete="off" spellcheck="false" maxlength="10">
      <button type="button" id="tickerSearchBtn">조회</button>
      <div id="tickerResult"></div>
    </div>

    <div class="status">
      <div id="statusText">Ready</div>
      <div id="runMeta" class="subtle"></div>
    </div>

    <section class="metrics" id="metrics"></section>

    <div class="insight-bar" id="insightBar"><ul id="insightList"></ul></div>

    <section class="grid">
      <div class="panel">
        <h2>Universe</h2>
        <div class="chips" id="universeChips"></div>
      </div>
      <div class="panel">
        <h2>Filter Funnel</h2>
        <div class="funnel" id="filterFunnel"></div>
      </div>
    </section>

    <section class="panel">
      <div class="candidate-header">
        <h2>Candidates</h2>
        <div class="grade-group">
          <button class="grade-btn active" data-grade="all">전체</button>
          <button class="grade-btn" data-grade="A">A등급</button>
          <button class="grade-btn" data-grade="B">B등급</button>
        </div>
        <select class="sector-select" id="sectorFilter">
          <option value="">전체 섹터</option>
        </select>
        <span class="candidate-count" id="candidateCount"></span>
      </div>
      <div class="table-wrap" id="candidateTable"></div>
    </section>
  </main>

  <script>
    const insightBar       = document.getElementById("insightBar");
    const insightList      = document.getElementById("insightList");
    const tickerInput      = document.getElementById("tickerInput");
    const tickerSearchBtn  = document.getElementById("tickerSearchBtn");
    const tickerResult     = document.getElementById("tickerResult");
    const refreshButton    = document.getElementById("refreshButton");
    const statusText = document.getElementById("statusText");
    const runMeta = document.getElementById("runMeta");
    const metrics = document.getElementById("metrics");
    const universeLine = document.getElementById("universeLine");
    const universeChips = document.getElementById("universeChips");
    const filterFunnel = document.getElementById("filterFunnel");
    const candidateTable = document.getElementById("candidateTable");
    const candidateCount = document.getElementById("candidateCount");
    const fileLinks = document.getElementById("fileLinks");
    const sectorFilter = document.getElementById("sectorFilter");

    const columns = [
      ["_rank",              "#",          "rank"],
      ["ticker",             "Ticker",     "text"],
      ["name",               "Name",       "text"],
      ["sector",             "Sector",     "text"],
      ["grade",              "Grade",      "grade"],
      ["score",              "Score",      "decimal"],
      ["market_cap",         "Mkt Cap",    "marketcap"],
      ["rs_spy_20d",         "RS 20D",     "pct"],
      ["rs_spy_50d",         "RS 50D",     "pct"],
      ["rs_sector_20d",      "Sector RS",  "pct"],
      ["close_to_50d_high",  "50D High",   "ratio"],
      ["volume_ratio",       "Volume",     "ratio"],
      ["close_position",     "Close Pos",  "ratio"],
      ["rsi_14",             "RSI",        "number"],
    ];

    // --- State ---
    let allCandidates = [];
    let sortKey = "score";
    let sortAsc = false;
    let activeGrade = "all";

    function escapeHtml(v) {
      return String(v ?? "")
        .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
    }

    function formatValue(value, type) {
      if (value === null || value === undefined || (typeof value === "number" && Number.isNaN(value))) return "";
      if (type === "pct")       return `${(value * 100).toFixed(1)}%`;
      if (type === "ratio")     return Number(value).toFixed(2);
      if (type === "decimal")   return Number(value).toFixed(3);
      if (type === "number")    return Number(value).toFixed(1);
      if (type === "money")     return `$${Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
      if (type === "rank")      return value;
      if (type === "marketcap") {
        const b = value / 1e9;
        return b >= 1000 ? `$${(b / 1000).toFixed(1)}T` : `$${Math.round(b)}B`;
      }
      return value;
    }

    function setBusy(isBusy) {
      refreshButton.disabled = isBusy;
      refreshButton.textContent = isBusy ? "↻ Loading…" : "↻ Refresh";
    }

    async function getJson(url, options = {}) {
      const response = await fetch(url, options);
      const payload = await response.json();
      if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`);
      return payload;
    }

    // --- Market insight ---
    function generateInsight(data) {
      const steps      = data.filter_steps || [];
      const total      = Math.max(data.universe_count || 1, 1);
      const candidates = data.candidates_count || 0;
      const gradeA     = data.grade_counts?.A || 0;

      // RS Near High 통과율
      const rsStep    = steps.find(s => s.label === "RS Near High");
      const rsPassPct = rsStep ? rsStep.count / total * 100 : 0;

      // 섹터 집중도
      const candBySector = {};
      (data.candidates || []).forEach(r => {
        const s = r.sector || "Unknown";
        candBySector[s] = (candBySector[s] || 0) + 1;
      });
      const sortedSec = Object.entries(candBySector).sort((a, b) => b[1] - a[1]);
      const top1 = sortedSec[0];
      const top2 = sortedSec[1];
      const top2Count = (top1?.[1] || 0) + (top2?.[1] || 0);
      const top2Pct   = candidates > 0 ? top2Count / candidates * 100 : 0;
      const numSectors = sortedSec.filter(([,c]) => c > 0).length;

      const items = [];

      // 1. 시장 환경 (RS 통과율 기반)
      if (rsPassPct < 10) {
        items.push(`<span class="tag warn">좁은 시장</span>전체 ${total}개 중 ${rsStep?.count || 0}개(${rsPassPct.toFixed(0)}%)만 SPY 대비 상대강도 고점 유지 — 지수가 개별주를 앞서는 장세. 후보군이 압축될수록 통과 종목의 신뢰도는 높아집니다.`);
      } else if (rsPassPct < 25) {
        items.push(`<span class="tag neutral">선별적 강세</span>전체의 ${rsPassPct.toFixed(0)}%가 RS 고점 근처 — 일부 섹터에서 주도주가 형성되는 국면. 섹터 선택이 중요합니다.`);
      } else {
        items.push(`<span class="tag">광범위 강세</span>전체의 ${rsPassPct.toFixed(0)}%가 RS 고점 유지 — 개별주 전반에 걸쳐 강세 흐름. 종목 선택의 폭이 넓습니다.`);
      }

      // 2. 섹터 집중도
      if (candidates > 0) {
        if (top2Pct >= 55 && top1 && top2) {
          items.push(`<span class="tag">섹터 집중</span>후보 ${candidates}개 중 ${top2Count}개(${top2Pct.toFixed(0)}%)가 ${escapeHtml(top1[0])}·${escapeHtml(top2[0])}에 몰림 — 현재 장의 주도 섹터가 뚜렷합니다.`);
        } else if (top2Pct >= 55 && top1) {
          items.push(`<span class="tag">섹터 집중</span>후보 ${candidates}개 중 ${top1[1]}개(${(top1[1]/candidates*100).toFixed(0)}%)가 ${escapeHtml(top1[0])}에 집중 — 이 섹터가 현재 시장 주도.`);
        } else {
          items.push(`<span class="tag neutral">분산</span>후보가 ${numSectors}개 섹터에 분산 — 특정 테마 없이 개별 종목 장세. 섹터보다 종목 자체의 RS·거래량 확인이 중요합니다.`);
        }
      }

      // 3. 거래량 확인 (A등급 비율)
      if (candidates > 0) {
        const aPct = gradeA / candidates * 100;
        if (aPct < 15) {
          items.push(`<span class="tag warn">거래량 미확인</span>최종 후보 중 A등급(거래량 확인) ${gradeA}개(${aPct.toFixed(0)}%) — 추세는 있으나 강한 매수세가 실린 종목은 소수. B등급 진입 시 거래량 재확인 권장.`);
        } else if (aPct >= 35) {
          items.push(`<span class="tag">거래량 확인</span>A등급 ${gradeA}개(${aPct.toFixed(0)}%) — 거래량까지 뒷받침된 종목 비율이 높아 매수 환경 우호적.`);
        } else {
          items.push(`<span class="tag neutral">거래량 보통</span>A등급 ${gradeA}개(${aPct.toFixed(0)}%) — 거래량 확인 종목과 미확인 종목이 혼재. A등급 우선 검토 권장.`);
        }
      }

      return items;
    }

    function renderInsight(data) {
      const items = generateInsight(data);
      if (!items.length) { insightBar.style.display = "none"; return; }
      insightList.innerHTML = items.map(t => `<li>${t}</li>`).join("");
      insightBar.style.display = "block";
    }

    // --- Filtering & sorting ---
    function applyFilters() {
      let rows = allCandidates;
      if (activeGrade !== "all") rows = rows.filter(r => r.grade === activeGrade);
      const sector = sectorFilter.value;
      if (sector) rows = rows.filter(r => r.sector === sector);

      rows = [...rows].sort((a, b) => {
        let va = a[sortKey], vb = b[sortKey];
        const nullA = va === null || va === undefined;
        const nullB = vb === null || vb === undefined;
        if (nullA && nullB) return 0;
        if (nullA) return 1;
        if (nullB) return -1;
        if (va < vb) return sortAsc ? -1 : 1;
        if (va > vb) return sortAsc ? 1 : -1;
        return 0;
      });

      renderTable(rows);
      const total = allCandidates.length;
      candidateCount.textContent = rows.length === total
        ? `${total}개`
        : `${rows.length} / ${total}개`;
    }

    function populateSectorOptions() {
      const sectors = [...new Set(allCandidates.map(r => r.sector).filter(Boolean))].sort();
      const current = sectorFilter.value;
      sectorFilter.innerHTML = `<option value="">전체 섹터</option>` +
        sectors.map(s => `<option value="${escapeHtml(s)}"${s === current ? " selected" : ""}>${escapeHtml(s)}</option>`).join("");
    }

    // --- Render ---
    function renderMetrics(data) {
      const items = [
        ["Market",     data.market_state || "-",  data.date || ""],
        ["Universe",   data.universe_count,        data.ticker_file || ""],
        ["Evaluated",  data.evaluated_count,       `${data.missing_symbols?.length || 0} missing`],
        ["Candidates", data.candidates_count,      `${data.grade_counts?.A || 0} A / ${data.grade_counts?.B || 0} B`],
        ["Liquidity",  formatValue(data.config?.min_dollar_volume, "marketcap"), "20D average"],
      ];
      metrics.innerHTML = items.map(([label, value, note]) => `
        <div class="metric">
          <div class="label">${escapeHtml(label)}</div>
          <div class="value">${escapeHtml(value)}</div>
          <div class="note">${escapeHtml(note)}</div>
        </div>`).join("");
    }

    function renderUniverse(data) {
      universeLine.textContent = `US 1000 · ${data.universe_count ?? "-"} stocks`;
      const sectorTotals = data.sector_counts || {};

      // 섹터별 후보 수 집계 (allCandidates 전역 변수 사용)
      const candBySector = {};
      allCandidates.forEach(r => {
        const s = r.sector || "Unknown";
        candBySector[s] = (candBySector[s] || 0) + 1;
      });

      const hasCandidates = allCandidates.length > 0;

      const rows = Object.entries(sectorTotals)
        .sort((a, b) => {
          // 후보 있을 때는 후보 수 내림차순, 없으면 전체 수 내림차순
          if (hasCandidates) {
            const diff = (candBySector[b[0]] || 0) - (candBySector[a[0]] || 0);
            if (diff !== 0) return diff;
          }
          return b[1] - a[1];
        })
        .map(([sector, total]) => {
          const cand = candBySector[sector] || 0;
          const rate = total > 0 ? (cand / total * 100) : 0;
          const barWidth = Math.round(rate * 2);  // max 100% → 50% = full bar at 50% hit rate
          if (hasCandidates) {
            const rateCell = cand > 0
              ? `<span class="hit-bar"><span style="width:${Math.min(barWidth,100)}%"></span></span><span class="hit-rate">${rate.toFixed(1)}%</span>`
              : `<span class="zero">-</span>`;
            return `<tr>
              <td>${escapeHtml(sector)}</td>
              <td class="num">${total}</td>
              <td class="num">${cand || "-"}</td>
              <td>${rateCell}</td>
            </tr>`;
          } else {
            return `<tr>
              <td>${escapeHtml(sector)}</td>
              <td class="num">${total}</td>
            </tr>`;
          }
        }).join("");

      const thead = hasCandidates
        ? `<tr><th>섹터</th><th class="num">전체</th><th class="num">후보</th><th>적중률</th></tr>`
        : `<tr><th>섹터</th><th class="num">종목 수</th></tr>`;

      universeChips.innerHTML = `<table class="sector-table"><thead>${thead}</thead><tbody>${rows}</tbody></table>`;
    }

    const funnelTooltips = {
      "Universe":       "전체 입력 종목",
      "Evaluated":      "데이터 충분한 종목",
      "Liquidity":      "거래대금 기준 충족",
      "RS Near High":   "RS가 50일 고점 근처",
      "RS 20D > 0":     "20일 RS 양수",
      "Close > MA50":   "종가가 MA50 위",
      "MA50 Rising":    "MA50 상승 중",
      "Near 50D High":  "50일 고점 90% 이상",
      "Not Overheated": "과열·급등 없음",
      "A Grade":        "거래량 품질 충족",
    };

    function renderFunnel(data) {
      const total = Math.max(data.universe_count || 0, 1);
      const steps = data.filter_steps || [];

      // 최대 탈락 스텝 찾기 (bottleneck)
      const maxRemoved = Math.max(...steps.map(s => s.removed || 0));

      const stepsHtml = steps.map(step => {
        const width = Math.max(2, Math.round((step.count / total) * 100));
        const pct = (step.count / total * 100).toFixed(1) + "%";
        const removedText = step.removed > 0 ? `-${step.removed}` : "";
        const isBottleneck = step.removed > 0 && step.removed === maxRemoved;
        return `
          <div class="step${isBottleneck ? " bottleneck" : ""}" title="${escapeHtml(funnelTooltips[step.label] || "")}">
            <div>${escapeHtml(step.label)}</div>
            <div class="bar"><span style="width:${width}%"></span></div>
            <div class="count">${step.count}</div>
            <div class="removed">${removedText}</div>
            <div class="pct">${pct}</div>
          </div>`;
      }).join("");

      const cfg = data.config || {};
      const params = [
        `RS near high ${Math.round((cfg.rs_near_high_threshold ?? 0.98) * 100)}%`,
        `50D High ≥ ${Math.round((cfg.near_50d_high_threshold ?? 0.90) * 100)}%`,
        `Volume ≥ ${(cfg.volume_ratio_min ?? 1.3).toFixed(1)}x`,
        `Close Pos ≥ ${Math.round((cfg.close_position_min ?? 0.6) * 100)}%`,
      ].map(t => `<span class="chip">${escapeHtml(t)}</span>`).join("");

      filterFunnel.innerHTML = stepsHtml +
        `<div class="funnel-params">${params}</div>`;
    }

    function renderFiles(data) {
      const links = [];
      if (data.universe_csv)   links.push(`<a href="${escapeHtml(data.universe_csv)}">Universe CSV</a>`);
      if (data.candidates_csv) links.push(`<a href="${escapeHtml(data.candidates_csv)}">Candidates CSV</a>`);
      fileLinks.innerHTML = links.join("");
    }

    function renderTable(rows) {
      if (!rows || rows.length === 0) {
        candidateTable.innerHTML = '<div class="empty">No candidates</div>';
        return;
      }
      const head = columns.map(([key, label, type]) => {
        const isNum = ["pct", "ratio", "decimal", "number", "marketcap"].includes(type);
        let cls = isNum ? "num" : type === "rank" ? "rank" : "";
        let sortCls = key === sortKey ? (sortAsc ? " sort-asc" : " sort-desc") : "";
        const arrow = key === sortKey ? (sortAsc ? "↑" : "↓") : "↕";
        return `<th class="${cls}${sortCls}" data-key="${escapeHtml(key)}">${escapeHtml(label)}<span class="sort-arrow">${arrow}</span></th>`;
      }).join("");
      const body = rows.map(row => {
        const cells = columns.map(([key, , type]) => {
          if (type === "rank") {
            return `<td class="rank">${row[key] ?? ""}</td>`;
          }
          if (type === "grade") {
            const grade = row[key] || "";
            return `<td><span class="badge${grade === "B" ? " b" : ""}">${escapeHtml(grade)}</span></td>`;
          }
          const isNum = ["pct", "ratio", "decimal", "number", "marketcap"].includes(type);
          const cls = isNum ? "num" : key === "ticker" ? "ticker" : "";
          return `<td class="${cls}">${escapeHtml(formatValue(row[key], type))}</td>`;
        }).join("");
        return `<tr>${cells}</tr>`;
      }).join("");
      candidateTable.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;

      // bind sort click
      candidateTable.querySelectorAll("th[data-key]").forEach(th => {
        th.addEventListener("click", () => {
          const key = th.dataset.key;
          if (sortKey === key) {
            sortAsc = !sortAsc;
          } else {
            sortKey = key;
            sortAsc = key === "ticker" || key === "name" || key === "sector";
          }
          applyFilters();
        });
      });
    }

    function renderResult(data) {
      renderMetrics(data);
      renderFunnel(data);
      renderFiles(data);
      renderInsight(data);
      allCandidates = data.candidates || [];
      // score 내림차순으로 순위 부여 (필터/정렬과 무관하게 고정)
      [...allCandidates]
        .sort((a, b) => (b.score ?? -1) - (a.score ?? -1))
        .forEach((r, i) => { r._rank = i + 1; });
      renderUniverse(data);  // allCandidates 세팅 후 호출해야 섹터 적중률 계산 가능
      populateSectorOptions();
      applyFilters();
      runMeta.textContent = data.source === "latest" ? "Loaded latest output" : data.from_cache ? "Cached (today)" : "Fresh download";
    }

    // --- Ticker search ---
    const filterLabels = {
      "liquidity_ok":        "유동성",
      "rs_spy_near_high":    "RS 고점 근접",
      "rs_positive":         "RS 20D > 0",
      "above_ma50":          "MA50 위",
      "ma50_rising":         "MA50 상승",
      "near_50d_high":       "50일 고점 근접",
      "not_overheated":      "과열 없음",
      "passed_hard_filters": "하드 필터 통과",
      "volume_quality":      "거래량 품질 (A등급)",
    };

    const filterFailReasons = {
      "liquidity_ok":        (m) => `20일 평균 거래대금이 기준 미달 — 유동성이 낮아 제외됩니다.`,
      "rs_spy_near_high":    (m) => {
        const v = m.close_to_50d_high;
        return `최근 50일 RS 최고점에서 멀어진 상태 — SPY 대비 상대강도가 최근 고점의 98% 아래로 떨어졌습니다. 과거엔 강했어도 지금은 모멘텀이 식고 있다는 신호입니다.`;
      },
      "rs_positive":         (m) => `최근 20일간 SPY 대비 상대강도(RS)가 마이너스 — 시장 전체보다 더 많이 하락했습니다.`,
      "above_ma50":          (m) => `현재 주가가 50일 이동평균선 아래 — 중기 추세가 하락 방향입니다.`,
      "ma50_rising":         (m) => `50일 이동평균선이 10일 전보다 낮아짐 — 평균선 자체가 하향 중이어서 추세 약화 신호입니다.`,
      "near_50d_high":       (m) => {
        const v = m.close_to_50d_high;
        const pct = v != null ? ` (현재 ${(v*100).toFixed(1)}%, 기준 90%)` : "";
        return `최근 50일 최고가의 90% 미만${pct} — 고점에서 많이 밀린 상태로, 저점 반등 종목에 가깝습니다.`;
      },
      "not_overheated":      (m) => `단기 급등 감지 — MA20 대비 125% 초과이거나, 5일 수익률 40% 이상이거나, 당일 수익률 25% 이상입니다. 추격 매수 위험이 높은 구간입니다.`,
      "passed_hard_filters": (m) => `위 필터 중 하나 이상 탈락 — 최종 후보군에 포함되지 않았습니다.`,
      "volume_quality":      (m) => `거래량 품질 미충족 — 거래량비율 1.3배 미만이거나, 음봉이거나, 종가위치가 당일 범위 하단에 있습니다. B등급으로 분류됩니다.`,
    };

    function renderTickerResult(d) {
      if (d.error) {
        tickerResult.innerHTML = `<span style="color:var(--danger)">${escapeHtml(d.error)}</span>`;
        return;
      }
      if (!d.in_universe) {
        tickerResult.innerHTML = `<span style="color:var(--muted)"><strong>${escapeHtml(d.symbol)}</strong> — 유니버스에 없는 종목입니다.</span>`;
        return;
      }

      const hasResult = d.has_screener_result;
      const filters = d.filters || {};
      const metrics = d.metrics || {};

      let header = `<span class="tr-name">${escapeHtml(d.symbol)}</span>`;
      if (d.name) header += ` <span class="tr-name" style="font-weight:400">${escapeHtml(d.name)}</span>`;
      if (d.sector) header += `<span class="tr-sector">${escapeHtml(d.sector)}</span>`;

      let body = "";

      if (!hasResult) {
        body = `<div style="color:var(--muted);font-size:12px;margin-top:6px">스크리너 결과가 없습니다. Run Screener를 먼저 실행해 주세요.</div>`;
      } else {
        const filterKeys = ["liquidity_ok","rs_spy_near_high","rs_positive","above_ma50","ma50_rising","near_50d_high","not_overheated","passed_hard_filters","volume_quality"];
        const chips = filterKeys.map(key => {
          const val = filters[key];
          if (val === null || val === undefined) return "";
          const cls = val ? "pass" : "fail";
          const icon = val ? "✓" : "✗";
          return `<span class="tr-filter ${cls}">${icon} ${escapeHtml(filterLabels[key] || key)}</span>`;
        }).join("");
        body += `<div class="tr-filters">${chips}</div>`;

        // 탈락 필터 사유 설명
        const failedKeys = filterKeys.filter(k => filters[k] === false);
        if (failedKeys.length > 0) {
          const reasons = failedKeys.map(key => {
            const fn = filterFailReasons[key];
            const text = fn ? fn(metrics) : "";
            return text ? `<li><strong>${escapeHtml(filterLabels[key])}</strong> — ${escapeHtml(text)}</li>` : "";
          }).filter(Boolean).join("");
          if (reasons) {
            body += `<ul style="margin:8px 0 0;padding-left:16px;font-size:12px;color:var(--muted);line-height:1.6">${reasons}</ul>`;
          }
        }

        const fmt = (v, type) => v === null || v === undefined ? "–" : formatValue(v, type);
        const metricItems = [
          ["종가", fmt(metrics.close, "number")],
          ["RS 20D", fmt(metrics.rs_spy_20d, "pct")],
          ["RS 50D", fmt(metrics.rs_spy_50d, "pct")],
          ["50D 고점", fmt(metrics.close_to_50d_high, "ratio")],
          ["거래량비율", fmt(metrics.volume_ratio, "ratio")],
          ["Score", fmt(metrics.score, "decimal")],
          ["시가총액", fmt(metrics.market_cap, "marketcap")],
        ].map(([k, v]) => `<span><strong>${escapeHtml(v)}</strong> ${escapeHtml(k)}</span>`).join("");
        body += `<div class="tr-metrics">${metricItems}</div>`;
      }

      tickerResult.innerHTML = `<div>${header}${body}</div>`;
    }

    async function lookupTicker() {
      const sym = tickerInput.value.trim().toUpperCase();
      if (!sym) return;
      tickerResult.innerHTML = `<span style="color:var(--muted)">조회 중…</span>`;
      try {
        const data = await getJson(`/api/ticker?symbol=${encodeURIComponent(sym)}`);
        renderTickerResult(data);
      } catch (err) {
        tickerResult.innerHTML = `<span style="color:var(--danger)">${escapeHtml(err.message)}</span>`;
      }
    }

    tickerSearchBtn.addEventListener("click", lookupTicker);
    tickerInput.addEventListener("keydown", e => { if (e.key === "Enter") lookupTicker(); });

    // --- Grade buttons ---
    document.querySelectorAll(".grade-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".grade-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        activeGrade = btn.dataset.grade;
        applyFilters();
      });
    });

    sectorFilter.addEventListener("change", applyFilters);

    // --- API calls ---
    async function loadUniverse() {
      try {
        const data = await getJson("/api/universe");
        renderUniverse(data);
      } catch (error) {
        statusText.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
      }
    }

    async function loadLatest() {
      try {
        const data = await getJson("/api/latest");
        if (data.has_result) {
          renderResult(data);
          statusText.innerHTML = `Latest result · <strong>${escapeHtml(data.date || "")}</strong>`;
        }
      } catch (error) {
        statusText.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
      }
    }

    async function refresh() {
      setBusy(true);
      statusText.innerHTML = "Loading latest result…";
      runMeta.textContent = "";
      try {
        const data = await getJson("/api/latest");
        if (data.has_result) {
          renderResult(data);
          statusText.innerHTML = `Latest result · <strong>${escapeHtml(data.date || "")}</strong>`;
        } else {
          statusText.innerHTML = `<span class="error">결과 없음 — main.py를 먼저 실행해 주세요.</span>`;
        }
      } catch (error) {
        statusText.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
      } finally {
        setBusy(false);
      }
    }

    refreshButton.addEventListener("click", refresh);
    loadUniverse();
    loadLatest();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local RS volume screener dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def json_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def frame_records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    records = []
    for _, row in frame.iterrows():
        records.append({col: json_value(row.get(col)) for col in columns})
    return records


def sector_counts(ticker_file: Path) -> dict[str, int]:
    try:
        universe = load_universe(ticker_file)
    except Exception:
        return {}
    return {
        str(sector): int(count)
        for sector, count in universe["sector"].value_counts().sort_values(ascending=False).items()
    }


def filter_steps(results: pd.DataFrame, universe_count: int) -> list[dict[str, object]]:
    if results.empty:
        return [{"label": "Universe", "count": universe_count, "removed": 0}]

    def bool_filter(column: str, fallback: pd.Series | None = None) -> pd.Series:
        if column in results.columns:
            return results[column].fillna(False).astype(bool)
        if fallback is not None:
            return fallback.fillna(False).astype(bool)
        return pd.Series(False, index=results.index)

    steps: list[dict[str, object]] = [
        {"label": "Universe",  "count": universe_count,                        "removed": 0},
        {"label": "Evaluated", "count": len(results),                          "removed": max(universe_count - len(results), 0)},
    ]
    mask = pd.Series(True, index=results.index)
    filters = [
        ("Liquidity",      bool_filter("liquidity_ok")),
        ("RS Near High",   bool_filter("rs_spy_near_high")),
        ("RS 20D > 0",     bool_filter("rs_positive", results["rs_spy_20d"] > 0)),
        ("Close > MA50",   bool_filter("above_ma50")),
        ("MA50 Rising",    bool_filter("ma50_rising")),
        ("Near 50D High",  bool_filter("near_50d_high", results["close_to_50d_high"] >= 0.90)),
        ("Not Overheated", bool_filter("not_overheated")),
    ]
    previous = len(results)
    for label, condition in filters:
        mask = mask & condition.astype(bool)
        count = int(mask.sum())
        steps.append({"label": label, "count": count, "removed": max(previous - count, 0)})
        previous = count
    steps.append({
        "label": "A Grade",
        "count": int((bool_filter("passed_hard_filters") & bool_filter("volume_quality")).sum()),
        "removed": 0,
    })
    return steps


def output_link(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        name = path.resolve().relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        return None
    return f"/outputs/{name.as_posix()}"


def config_payload(config: ScreenerConfig) -> dict[str, object]:
    return {
        "period": config.period,
        "min_dollar_volume": config.min_dollar_volume,
        "rs_near_high_threshold": config.rs_near_high_threshold,
        "near_50d_high_threshold": config.near_50d_high_threshold,
        "volume_ratio_min": config.volume_ratio_min,
        "close_position_min": config.close_position_min,
    }


def response_from_results(
    results: pd.DataFrame,
    *,
    ticker_file: Path,
    config: ScreenerConfig,
    market_state: str,
    universe_count: int,
    missing_symbols: list[str] | None = None,
    universe_path: Path | None = None,
    candidates_path: Path | None = None,
    source: str,
    from_cache: bool = False,
) -> dict[str, object]:
    candidates = results.loc[results["passed_hard_filters"].fillna(False)].copy()
    candidate_columns = [
        "ticker", "name", "sector", "grade", "score",
        "market_cap",
        "rs_spy_20d", "rs_spy_50d", "rs_sector_20d",
        "close_to_50d_high", "volume_ratio", "close_position",
        "rsi_14", "avg_dollar_volume_20d",
    ]
    date = None
    if "date" in results.columns and not results["date"].dropna().empty:
        date = str(results["date"].dropna().max())

    grade_counts = {
        str(grade): int(count)
        for grade, count in candidates["grade"].value_counts().items()
        if str(grade)
    }
    return {
        "has_result": True,
        "source": source,
        "date": date,
        "ticker_file": str(ticker_file.relative_to(PROJECT_ROOT)) if ticker_file.is_relative_to(PROJECT_ROOT) else str(ticker_file),
        "market_state": market_state,
        "universe_count": universe_count,
        "evaluated_count": len(results),
        "candidates_count": len(candidates),
        "grade_counts": grade_counts,
        "missing_symbols": missing_symbols or [],
        "sector_counts": sector_counts(ticker_file),
        "config": config_payload(config),
        "filter_steps": filter_steps(results, universe_count),
        "candidates": frame_records(candidates, candidate_columns),
        "universe_csv": output_link(universe_path),
        "candidates_csv": output_link(candidates_path),
        "from_cache": from_cache,
    }


def latest_output_response() -> dict[str, object]:
    files = sorted(OUTPUT_DIR.glob("screener_universe_*.csv"))
    if not files:
        return {
            "has_result": False,
            "ticker_file": str(DEFAULT_TICKER_FILE),
            "universe_count": len(load_universe(TICKER_FILE)),
            "sector_counts": sector_counts(TICKER_FILE),
            "config": config_payload(ScreenerConfig()),
        }

    universe_path = files[-1]
    date = universe_path.stem.removeprefix("screener_universe_")
    candidates_path = OUTPUT_DIR / f"screener_candidates_{date}.csv"
    results = pd.read_csv(universe_path)
    market_state = str(results["market_state"].dropna().iloc[0]) if "market_state" in results.columns else "Unknown"
    return response_from_results(
        results,
        ticker_file=TICKER_FILE,
        config=ScreenerConfig(),
        market_state=market_state,
        universe_count=len(load_universe(TICKER_FILE)),
        universe_path=universe_path,
        candidates_path=candidates_path if candidates_path.exists() else None,
        source="latest",
    )


def ticker_lookup(symbol: str) -> dict[str, object]:
    symbol = symbol.strip().upper()

    # 1. Check universe membership
    try:
        universe = load_universe(TICKER_FILE)
    except Exception as exc:
        return {"error": f"유니버스 로드 실패: {exc}"}

    row = universe[universe["ticker"] == symbol]
    if row.empty:
        return {"symbol": symbol, "in_universe": False}

    name   = str(row.iloc[0].get("name",   "") or "")
    sector = str(row.iloc[0].get("sector", "") or "")

    # 2. Find latest screener universe CSV
    files = sorted(OUTPUT_DIR.glob("screener_universe_*.csv"))
    if not files:
        return {"symbol": symbol, "in_universe": True, "name": name, "sector": sector, "has_screener_result": False}

    results = pd.read_csv(files[-1])
    sym_row = results[results["ticker"] == symbol]
    if sym_row.empty:
        return {"symbol": symbol, "in_universe": True, "name": name, "sector": sector, "has_screener_result": False}

    r = sym_row.iloc[0]

    def safe(col: str) -> object:
        v = r.get(col)
        if v is None:
            return None
        try:
            fv = float(v)
            return None if (math.isnan(fv) or math.isinf(fv)) else fv
        except (TypeError, ValueError):
            return str(v) if str(v) else None

    def safe_bool(col: str) -> bool | None:
        v = r.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return bool(v)

    filters = {
        "liquidity_ok":        safe_bool("liquidity_ok"),
        "rs_spy_near_high":    safe_bool("rs_spy_near_high"),
        "rs_positive":         safe_bool("rs_positive"),
        "above_ma50":          safe_bool("above_ma50"),
        "ma50_rising":         safe_bool("ma50_rising"),
        "near_50d_high":       safe_bool("near_50d_high"),
        "not_overheated":      safe_bool("not_overheated"),
        "passed_hard_filters": safe_bool("passed_hard_filters"),
        "volume_quality":      safe_bool("volume_quality"),
    }

    grade = str(r.get("grade") or "")
    metrics = {
        "close":                 safe("close"),
        "rs_spy_20d":            safe("rs_spy_20d"),
        "rs_spy_50d":            safe("rs_spy_50d"),
        "close_to_50d_high":     safe("close_to_50d_high"),
        "volume_ratio":          safe("volume_ratio"),
        "avg_dollar_volume_20d": safe("avg_dollar_volume_20d"),
        "market_cap":            safe("market_cap"),
        "score":                 safe("score"),
        "grade":                 grade if grade else None,
    }

    return {
        "symbol":              symbol,
        "in_universe":         True,
        "name":                name,
        "sector":              sector,
        "has_screener_result": True,
        "filters":             filters,
        "metrics":             metrics,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "RSVolumeDashboard/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(INDEX_HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/universe":
            self.send_json({
                "ticker_file": str(DEFAULT_TICKER_FILE),
                "universe_count": len(load_universe(TICKER_FILE)),
                "sector_counts": sector_counts(TICKER_FILE),
                "config": config_payload(ScreenerConfig()),
            })
            return
        if parsed.path == "/api/latest":
            self.send_json(latest_output_response())
            return
        if parsed.path == "/api/ticker":
            params = parse_qs(parsed.query)
            symbol = (params.get("symbol") or [""])[0]
            if not symbol:
                self.send_json({"error": "symbol 파라미터가 필요합니다."}, status=HTTPStatus.BAD_REQUEST)
            else:
                self.send_json(ticker_lookup(symbol))
            return
        if parsed.path.startswith("/outputs/"):
            self.send_output_file(parsed.path.removeprefix("/outputs/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self.read_json()
            run = run_screener(
                ticker_file=TICKER_FILE,
                output_dir=OUTPUT_DIR,
                min_dollar_volume=float(payload.get("min_dollar_volume", 20_000_000)),
                write_files=True,
            )
            self.send_json(response_from_results(
                run.results,
                ticker_file=TICKER_FILE,
                config=run.config,
                market_state=run.market_state,
                universe_count=run.universe_count,
                missing_symbols=run.missing_symbols,
                universe_path=run.universe_path,
                candidates_path=run.candidates_path,
                source="run",
                from_cache=run.from_cache,
            ))
        except Exception as error:
            traceback.print_exc()
            self.send_json({"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        return json.loads(body.decode("utf-8"))

    def send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=json_value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_output_file(self, relative_path: str) -> None:
        requested = (OUTPUT_DIR / unquote(relative_path)).resolve()
        if OUTPUT_DIR.resolve() not in requested.parents or not requested.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = requested.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
