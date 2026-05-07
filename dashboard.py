from __future__ import annotations

import argparse
import json
import math
import pickle
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd

from main import DEFAULT_OUTPUT_DIR, DEFAULT_TICKER_FILE, DEFAULT_YFINANCE_CACHE_DIR, ScreenerConfig, _symbol_cache_dir, load_universe, run_screener


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / DEFAULT_OUTPUT_DIR
TICKER_FILE = PROJECT_ROOT / DEFAULT_TICKER_FILE


INDEX_HTML = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>APEX — Alpha Pulse Equity eXplorer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
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
      font-family: 'DM Sans', ui-sans-serif, system-ui, sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }

    button, input, select { font: inherit; }

    .shell {
      width: min(1680px, calc(100vw - 32px));
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

    h1 { margin: 0; font-size: 28px; font-weight: 700; line-height: 1.1; letter-spacing: -0.02em; }
    h1 .apex-sub { font-size: 11px; font-weight: 500; letter-spacing: 0.08em; color: var(--muted); text-transform: uppercase; display: block; margin-bottom: 4px; }

    .btn-ghost {
      background: none;
      border: 1px solid var(--line);
      color: var(--text);
      font-size: 13px;
      padding: 6px 14px;
      min-height: 34px;
      border-radius: 6px;
      cursor: pointer;
      white-space: nowrap;
    }
    .btn-ghost:hover { background: #f0f0ec; }

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
      min-height: 96px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px 14px;
    }

    .metric .label { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
    .metric .value { font-size: 24px; font-weight: 700; line-height: 1.1; font-family: 'DM Mono', monospace; }
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

    /* Market cap buttons */
    .mcap-filter {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }

    .mcap-group {
      display: flex;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: visible;
    }

    .mcap-btn {
      position: relative;
      min-height: 30px;
      padding: 4px 11px;
      border: none;
      border-radius: 0;
      border-right: 1px solid var(--line);
      background: #fff;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }

    .mcap-btn:first-child { border-radius: 5px 0 0 5px; }
    .mcap-btn:last-child  { border-right: none; border-radius: 0 5px 5px 0; }

    .mcap-btn.active {
      background: var(--accent);
      color: #fff;
    }

    .mcap-btn:not(.active):hover { background: #f0f0ec; }

    .mcap-btn .mcap-tooltip {
      display: none;
      position: absolute;
      bottom: calc(100% + 7px);
      left: 50%;
      transform: translateX(-50%);
      background: #2a2e29;
      color: #fff;
      font-size: 11px;
      font-weight: 400;
      white-space: nowrap;
      padding: 4px 8px;
      border-radius: 5px;
      pointer-events: none;
      z-index: 10;
    }

    .mcap-btn .mcap-tooltip::after {
      content: "";
      position: absolute;
      top: 100%;
      left: 50%;
      transform: translateX(-50%);
      border: 5px solid transparent;
      border-top-color: #2a2e29;
    }

    .mcap-btn:hover .mcap-tooltip { display: block; }

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
      background:
        linear-gradient(to right, var(--surface) 20px, transparent) left center,
        linear-gradient(to left,  var(--surface) 20px, transparent) right center,
        radial-gradient(farthest-side at 0 50%, rgba(0,0,0,.08), transparent) left center,
        radial-gradient(farthest-side at 100% 50%, rgba(0,0,0,.08), transparent) right center;
      background-repeat: no-repeat;
      background-size: 60px 100%, 60px 100%, 16px 100%, 16px 100%;
      background-attachment: local, local, scroll, scroll;
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

    tbody tr:hover { background: #f6f9f7; }
    tfoot tr { border-top: 2px solid var(--border); background: #f6f9f7; }

    .num { text-align: right; font-variant-numeric: tabular-nums; font-family: 'DM Mono', monospace; font-size: 13px; }
    .rank { text-align: right; color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; font-family: 'DM Mono', monospace; width: 32px; }
    .ticker { font-weight: 700; font-family: 'DM Mono', monospace; letter-spacing: 0.02em; }

    /* diff badge column */
    .diff-badge-cell { width: 36px; padding: 0 4px 0 8px; }
    .db { display: inline-flex; align-items: center; justify-content: center;
          font-size: 10px; font-weight: 700; letter-spacing: 0.02em;
          padding: 1px 5px; border-radius: 4px; white-space: nowrap; }
    .db-new  { background: #dcebe5; color: var(--accent); }
    .db-up   { background: #dcebe5; color: var(--accent); }
    .db-down { background: #fef0dc; color: var(--warn); }

    /* diff summary section */
    .diff-section {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 14px 18px;
      margin-bottom: 16px;
    }
    .diff-section h3 { margin: 0 0 10px; font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
    .diff-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
    .diff-chip { display: inline-flex; align-items: center; gap: 5px; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 500; border: 1px solid var(--line); background: #f6f9f7; }
    .diff-chip .count { font-weight: 700; font-family: 'DM Mono', monospace; }
    .diff-chip.new-in  { border-color: #b2d5c8; background: #edf6f2; color: var(--accent); }
    .diff-chip.dropped { border-color: #f0c4c2; background: #fdf0ef; color: var(--danger); }
    .diff-chip.upgraded { border-color: #b2d5c8; background: #edf6f2; color: var(--accent); }
    .diff-chip.downgraded { border-color: #f5dbb5; background: #fef6e8; color: var(--warn); }
    .diff-tickers { font-size: 12px; color: var(--muted); line-height: 1.8; }
    .diff-tickers strong { color: var(--text); font-family: 'DM Mono', monospace; font-size: 11px; }

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

    .desc summary::-webkit-details-marker { display: none; }
    .desc summary::before { content: "▶"; font-size: 10px; transition: transform 0.15s; }
    .desc[open] summary::before { transform: rotate(90deg); }
    .desc ul { margin: 10px 0 0; padding-left: 18px; }
    .desc li { margin-bottom: 4px; }
    .desc li strong { color: var(--text); }

    /* Market environment banner */
    .market-banner {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 18px;
      border-radius: 8px;
      border: 1px solid var(--line);
      border-left-width: 4px;
      margin-bottom: 16px;
      font-size: 13px;
      line-height: 1.5;
    }

    .market-banner.uptrend {
      background: #edf6f2;
      border-color: #b2d5c8;
      border-left-color: var(--accent);
    }

    .market-banner.pressure {
      background: #fef6e8;
      border-color: #f0d8a0;
      border-left-color: var(--warn);
    }

    .market-banner.correction {
      background: #fdf0ef;
      border-color: #f0c4c2;
      border-left-color: var(--danger);
    }

    .market-banner .mb-icon { font-size: 18px; flex-shrink: 0; }

    .market-banner .mb-state {
      font-weight: 700;
      font-size: 14px;
    }

    .market-banner.uptrend  .mb-state { color: var(--accent); }
    .market-banner.pressure .mb-state { color: var(--warn); }
    .market-banner.correction .mb-state { color: var(--danger); }

    .market-banner .mb-desc { color: var(--muted); }

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

    /* Watchlist panel */
    .watchlist-panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px 18px;
      margin-bottom: 16px;
      display: none;
    }

    .watchlist-header {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 14px;
    }

    .watchlist-header h3 {
      margin: 0;
      font-size: 14px;
      font-weight: 700;
    }

    .watchlist-header .wl-date {
      font-size: 12px;
      color: var(--muted);
      margin-left: auto;
    }

    .watchlist-groups {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
    }

    @media (max-width: 980px) {
      .watchlist-groups { grid-template-columns: 1fr; }
    }

    .wl-group {
      border: 1px solid var(--line);
      border-radius: 7px;
      overflow: hidden;
    }

    .wl-group-title {
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      padding: 7px 12px;
    }

    .wl-group-title.g-entry { background: #edf6f2; color: #0e5b45; border-bottom: 1px solid #c8e8dc; }
    .wl-group-title.g-watch { background: #fef6e8; color: var(--warn); border-bottom: 1px solid #f0d8a0; }
    .wl-group-title.g-list  { background: #f0f0ec; color: var(--muted); border-bottom: 1px solid var(--line); }

    .wl-rows { padding: 4px 0; max-height: 380px; overflow-y: auto; }

    .wl-row {
      padding: 8px 12px;
      border-bottom: 1px solid #f4f4f0;
    }

    .wl-row:last-child { border-bottom: 0; }

    .wl-row-top {
      display: flex;
      align-items: center;
      gap: 7px;
      margin-bottom: 3px;
    }

    .wl-ticker {
      font-family: 'DM Mono', monospace;
      font-weight: 700;
      font-size: 13px;
      letter-spacing: 0.03em;
      cursor: pointer;
      color: var(--accent);
      border-bottom: 1px dotted var(--accent);
    }

    .wl-name {
      font-size: 12px;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      flex: 1;
    }

    .wl-grade {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 20px;
      height: 20px;
      border-radius: 999px;
      font-size: 10px;
      font-weight: 700;
      background: var(--accent-soft);
      color: #0e5b45;
      flex-shrink: 0;
    }

    .wl-grade.b { background: #eee9d9; color: var(--warn); }

    .wl-comment {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.5;
    }

    .wl-empty {
      padding: 10px 12px;
      font-size: 12px;
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
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 12px 14px;
      margin-bottom: 16px;
    }

    .ticker-search-row {
      display: flex;
      gap: 8px;
      align-items: center;
    }

    .ticker-search label {
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }

    #tickerInput {
      width: 120px;
      min-height: 34px;
      text-transform: uppercase;
      font-family: 'DM Mono', monospace;
      font-weight: 500;
      letter-spacing: 0.05em;
    }

    #tickerResult {
      font-size: 13px;
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
    }

    #tickerResult:empty {
      display: none;
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

    .tr-layout {
      display: flex;
      gap: 12px;
      align-items: stretch;
    }

    .tr-info { flex: 0 0 auto; min-width: 0; max-width: 55%; }

    .tr-inline-charts {
      display: flex;
      flex-direction: row;
      gap: 8px;
      flex: 1;
      min-width: 0;
      padding-top: 2px;
    }

    .tr-inline-chart-cell {
      display: flex;
      flex-direction: column;
      flex: 1;
      min-width: 0;
      min-height: 90px;
    }

    .tr-inline-chart-header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      margin-bottom: 3px;
    }

    .tr-inline-chart-period {
      font-size: 10px;
      font-weight: 700;
      font-family: 'DM Mono', monospace;
      letter-spacing: 0.04em;
      color: var(--muted);
    }

    .tr-inline-chart-return {
      font-size: 11px;
      font-family: 'DM Mono', monospace;
    }

    .tr-inline-chart-return.pos { color: var(--accent); }
    .tr-inline-chart-return.neg { color: var(--danger); }

    .tr-inline-chart-wrap {
      flex: 1;
      min-height: 60px;
    }

    .tr-inline-chart-wrap svg { width: 100%; height: 100%; display: block; overflow: visible; }

    .tr-inline-chart-wrap svg { width: 100%; height: 100%; overflow: visible; }

    /* Chart popup modal */
    .chart-overlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.35);
      z-index: 100;
      align-items: center;
      justify-content: center;
    }

    .chart-overlay.open { display: flex; }

    .chart-modal {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.18);
      width: min(860px, calc(100vw - 32px));
      padding: 20px 22px 18px;
      position: relative;
    }

    .chart-modal-header {
      display: flex;
      align-items: baseline;
      gap: 10px;
      margin-bottom: 4px;
    }

    .chart-modal-ticker {
      font-family: 'DM Mono', monospace;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0.03em;
    }

    .chart-modal-name {
      font-size: 13px;
      color: var(--muted);
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .chart-modal-close {
      position: absolute;
      top: 14px;
      right: 16px;
      background: none;
      border: none;
      font-size: 18px;
      color: var(--muted);
      cursor: pointer;
      padding: 2px 6px;
      line-height: 1;
      min-height: unset;
    }

    .chart-modal-close:hover { color: var(--text); background: none; }

    .chart-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-bottom: 14px;
    }

    .chart-cell {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px 8px;
    }

    .chart-cell-header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      margin-bottom: 6px;
    }

    .chart-cell-period {
      font-family: 'DM Mono', monospace;
      font-size: 11px;
      font-weight: 700;
      color: var(--muted);
      letter-spacing: 0.05em;
    }

    .chart-cell-return {
      font-family: 'DM Mono', monospace;
      font-size: 12px;
      font-weight: 650;
    }

    .chart-cell-return.pos { color: var(--accent); }
    .chart-cell-return.neg { color: var(--danger); }

    .chart-svg-wrap {
      width: 100%;
      height: 130px;
    }

    .chart-svg-wrap svg { width: 100%; height: 100%; overflow: visible; }

    .chart-footer {
      display: flex;
      justify-content: flex-end;
    }

    .chart-perplexity-btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 600;
      color: var(--accent);
      text-decoration: none;
      border: 1px solid var(--accent);
      border-radius: 6px;
      padding: 5px 12px;
      min-height: unset;
      background: none;
      cursor: pointer;
    }

    .chart-perplexity-btn:hover { background: var(--accent-soft); }

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
        <h1><span class="apex-sub">Alpha Pulse Equity eXplorer</span>APEX</h1>
        <div class="subtle" id="universeLine">Universe loading</div>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <button id="refreshButton" type="button" class="btn-ghost">↻ Refresh</button>
        <button id="runButton" type="button" class="btn-ghost" style="font-weight:600">▶ Run Screener</button>
      </div>
    </header>

    <details class="desc">
      <summary>필터링 방식</summary>
      <ul>
        <li><strong>시장 환경</strong> — SPY·QQQ 기반 4단계 분류: Confirmed Uptrend / Uptrend Under Pressure / Market in Correction. 하락장에서는 후보 출력 시 경고 배너 표시.</li>
        <li><strong>유동성</strong> — 20일 평균 거래대금 ≥ 기준값. 저유동성 종목 제외.</li>
        <li><strong>RS 고점 근접</strong> — SPY 대비 상대강도(RS)가 최근 50일 RS 최고값의 98% 이상.</li>
        <li><strong>RS 20D 양수</strong> — 20일 SPY 대비 RS 변화율 &gt; 0.</li>
        <li><strong>MA50 위 + 상승</strong> — 종가 &gt; 50일 이동평균, MA50이 10일 전보다 높음.</li>
        <li><strong>50일 고점 근접</strong> — 종가 ≥ 최근 50일 최고가의 90%.</li>
        <li><strong>과열 없음</strong> — 종가 ≤ MA20의 125%, 5일 수익률 &lt; 40%, 당일 수익률 &lt; 25%.</li>
        <li><strong>등급 A</strong> — 위 조건 통과 + 거래량 품질(거래량비율 ≥ 1.3배, 양봉, 종가위치 ≥ 60%).</li>
        <li><strong>등급 B</strong> — 위 조건만 통과, 거래량 품질 미충족.</li>
        <li><strong>점수</strong> — RS 20D(25%) + RS 50D(20%) + 섹터 RS(15%) + 50일 고점 근접도(20%) + 거래량비율(15%) + 종가위치(5%) percentile 가중합.</li>
        <li><strong>Stage</strong> — 돌파 후 경과일 기준: Early Breakout(≤7일) / Trending(≤35일) / Extended(35일+). 조기 돌파 종목이 가장 안전한 구간.</li>
        <li><strong>Pivot / vs Pivot</strong> — 50일 고점을 피벗 기준가로 사용. vs Pivot이 +5% 초과(⚠)면 추격 위험.</li>
        <li><strong>Base</strong> — 베이스 기간 변동성 대비 최근 변동성 비율(높을수록 안정적 베이스 후 돌파).</li>
        <li><strong>Sector 52W</strong> — 섹터 ETF의 52주 고점 대비 현재 위치. 섹터 전체 건강도 파악용.</li>
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
      <div class="ticker-search-row">
        <label>티커 검색</label>
        <input type="text" id="tickerInput" placeholder="예: AAPL" autocomplete="off" spellcheck="false" maxlength="10">
        <button type="button" id="tickerSearchBtn">조회</button>
      </div>
      <div id="tickerResult"></div>
    </div>

    <div class="status">
      <div id="statusText">Ready</div>
      <div id="runMeta" class="subtle"></div>
    </div>

    <div id="marketBanner" style="display:none" class="market-banner">
      <span class="mb-icon" id="marketBannerIcon"></span>
      <div>
        <div class="mb-state" id="marketBannerState"></div>
        <div class="mb-desc" id="marketBannerDesc"></div>
      </div>
    </div>

    <section class="metrics" id="metrics"></section>

    <div class="insight-bar" id="insightBar"><ul id="insightList"></ul></div>

    <div id="diffSection" style="display:none" class="diff-section">
      <h3>전일 대비 변경사항 <span id="diffDateLabel" style="font-weight:400;text-transform:none;letter-spacing:0"></span></h3>
      <div class="diff-chips" id="diffChips"></div>
      <div class="diff-tickers" id="diffTickers"></div>
    </div>

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

    <div class="watchlist-panel" id="watchlistPanel">
      <div class="watchlist-header">
        <h3>오늘의 종목 해석</h3>
        <span class="wl-date" id="watchlistDate"></span>
      </div>
      <div class="watchlist-groups" id="watchlistGroups"></div>
    </div>

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
        <div class="mcap-filter">
          <span>Mkt Cap</span>
          <div class="mcap-group">
            <button class="mcap-btn active" data-mcap="all">All<span class="mcap-tooltip">전체</span></button>
            <button class="mcap-btn" data-mcap="small">Small<span class="mcap-tooltip">< $2B</span></button>
            <button class="mcap-btn" data-mcap="mid">Mid<span class="mcap-tooltip">$2B – $10B</span></button>
            <button class="mcap-btn" data-mcap="large">Large<span class="mcap-tooltip">$10B – $200B</span></button>
            <button class="mcap-btn" data-mcap="mega">Mega<span class="mcap-tooltip">≥ $200B</span></button>
          </div>
        </div>
        <span class="candidate-count" id="candidateCount"></span>
      </div>
      <div class="table-wrap" id="candidateTable"></div>
    </section>
  </main>

  <div class="chart-overlay" id="chartOverlay">
    <div class="chart-modal" id="chartModal">
      <button class="chart-modal-close" id="chartModalClose">✕</button>
      <div class="chart-modal-header">
        <span class="chart-modal-ticker" id="chartModalTicker"></span>
        <span class="chart-modal-name" id="chartModalName"></span>
      </div>
      <div class="chart-grid" id="chartGrid">
        <div class="chart-cell" id="chartCell-21">
          <div class="chart-cell-header">
            <span class="chart-cell-period">1M</span>
            <span class="chart-cell-return" id="chartReturn-21"></span>
          </div>
          <div class="chart-svg-wrap" id="chartSvgWrap-21"></div>
        </div>
        <div class="chart-cell" id="chartCell-63">
          <div class="chart-cell-header">
            <span class="chart-cell-period">3M</span>
            <span class="chart-cell-return" id="chartReturn-63"></span>
          </div>
          <div class="chart-svg-wrap" id="chartSvgWrap-63"></div>
        </div>
        <div class="chart-cell" id="chartCell-126">
          <div class="chart-cell-header">
            <span class="chart-cell-period">6M</span>
            <span class="chart-cell-return" id="chartReturn-126"></span>
          </div>
          <div class="chart-svg-wrap" id="chartSvgWrap-126"></div>
        </div>
        <div class="chart-cell" id="chartCell-252">
          <div class="chart-cell-header">
            <span class="chart-cell-period">1Y</span>
            <span class="chart-cell-return" id="chartReturn-252"></span>
          </div>
          <div class="chart-svg-wrap" id="chartSvgWrap-252"></div>
        </div>
      </div>
      <div class="chart-footer">
        <a id="chartPerplexityBtn" class="chart-perplexity-btn" target="_blank" rel="noopener">
          Perplexity에서 보기 ↗
        </a>
      </div>
    </div>
  </div>

  <script>
    const insightBar       = document.getElementById("insightBar");
    const insightList      = document.getElementById("insightList");
    const tickerInput      = document.getElementById("tickerInput");
    const tickerSearchBtn  = document.getElementById("tickerSearchBtn");
    const tickerResult     = document.getElementById("tickerResult");
    const refreshButton    = document.getElementById("refreshButton");
    const runButton        = document.getElementById("runButton");
    const statusText = document.getElementById("statusText");
    const runMeta = document.getElementById("runMeta");
    const metrics = document.getElementById("metrics");
    const universeLine = document.getElementById("universeLine");
    const universeChips = document.getElementById("universeChips");
    const filterFunnel = document.getElementById("filterFunnel");
    const candidateTable = document.getElementById("candidateTable");
    const candidateCount = document.getElementById("candidateCount");
    const sectorFilter = document.getElementById("sectorFilter");

    const columns = [
      ["_diff",                  "",             "diff"],
      ["_rank",                  "#",            "rank"],
      ["ticker",                 "Ticker",       "text"],
      ["name",                   "Name",         "text"],
      ["sector",                 "Sector",       "text"],
      ["grade",                  "Grade",        "grade"],
      ["trend_stage",            "Stage",        "stage"],
      ["score",                  "Score",        "decimal"],
      ["market_cap",             "Mkt Cap",      "marketcap"],
      ["rs_spy_20d",             "RS 20D",       "pct"],
      ["rs_spy_50d",             "RS 50D",       "pct"],
      ["rs_sector_20d",          "Sector RS",    "pct"],
      ["close_to_50d_high",      "50D High",     "ratio"],
      ["volume_ratio",           "Volume",       "ratio"],
      ["close_position",         "Close Pos",    "ratio"],
      ["rsi_14",                 "RSI",          "number"],
      ["pivot_price",            "Pivot",        "money"],
      ["pivot_distance",         "vs Pivot",     "pctdist"],
      ["base_stability",         "Base",         "ratio"],
      ["sector_etf_to_52w_high", "Sector 52W",   "ratio"],
    ];

    // --- State ---
    let allCandidates = [];
    let sortKey = "score";
    let sortAsc = false;
    let activeGrade = "all";

    // Market cap button filter
    const mcapRanges = {
      all:   [0,        Infinity],
      small: [0,        2e9],
      mid:   [2e9,      10e9],
      large: [10e9,     200e9],
      mega:  [200e9,    Infinity],
    };
    let activeMcap = "all";

    function initMcapButtons() {
      document.querySelectorAll(".mcap-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          document.querySelectorAll(".mcap-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          activeMcap = btn.dataset.mcap;
          applyFilters();
        });
      });
    }

    function resetMcapFromData() {
      activeMcap = "all";
      document.querySelectorAll(".mcap-btn").forEach(b => {
        b.classList.toggle("active", b.dataset.mcap === "all");
      });
    }

    function escapeHtml(v) {
      return String(v ?? "")
        .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
    }

    function formatValue(value, type) {
      if (value === null || value === undefined || (typeof value === "number" && Number.isNaN(value))) return "";
      if (type === "pct")       return `${(value * 100).toFixed(1)}%`;
      if (type === "pctdist")   return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;
      if (type === "ratio")     return Number(value).toFixed(2);
      if (type === "decimal")   return Number(value).toFixed(3);
      if (type === "number")    return Number(value).toFixed(1);
      if (type === "money")     return `$${Number(value).toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 })}`;
      if (type === "rank")      return value;
      if (type === "stage")     return value || "";
      if (type === "marketcap") {
        const b = value / 1e9;
        return b >= 1000 ? `$${(b / 1000).toFixed(1)}T` : `$${Math.round(b)}B`;
      }
      return value;
    }

    function setBusy(isBusy) {
      refreshButton.disabled = isBusy;
      runButton.disabled = isBusy;
      refreshButton.textContent = isBusy ? "↻ Loading…" : "↻ Refresh";
      runButton.textContent = isBusy ? "▶ Running…" : "▶ Run Screener";
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

    // --- Watchlist ---
    function buildComment(r, group) {
      const rs20  = r.rs_spy_20d ?? 0;
      const secRS = r.rs_sector_20d ?? 0;
      const high  = r.close_to_50d_high ?? 0;
      const vol   = r.volume_ratio ?? 0;
      const cp    = r.close_position ?? 0;
      const ret20 = r.return_20d ?? 0;
      const isA   = r.grade === "A";

      const parts = [];

      // 워치리스트 A등급: 왜 진입 검토가 아닌지 먼저 명시
      if (group === "list" && isA) {
        if (ret20 >= 0.40) {
          parts.push(`A등급이지만 20일 수익률 ${(ret20*100).toFixed(0)}%로 단기 급등 피로 구간 — 눌림 확인 후 재진입 검토`);
        } else if (high < 0.97) {
          parts.push(`A등급이지만 50일 고점의 ${(high*100).toFixed(0)}%로 아직 고점에서 멀어진 상태 — 고점 재접근 시 재분류 예정`);
        }
      }

      // RS 강도
      if (rs20 >= 0.30) parts.push(`RS 20D ${(rs20*100).toFixed(0)}%로 강한 모멘텀`);
      else parts.push(`RS 20D ${(rs20*100).toFixed(0)}%`);

      // 섹터 내 위치
      if (secRS >= 0.30) parts.push(`섹터 내 최상위 RS(${(secRS*100).toFixed(0)}%)`);
      else if (secRS >= 0.15) parts.push(`섹터 RS ${(secRS*100).toFixed(0)}%로 양호`);

      // 고점 근접 (워치리스트 A등급 첫 줄에서 이미 언급했으면 생략)
      if (!(group === "list" && isA)) {
        if (high >= 0.99) parts.push("50일 고점 돌파 중");
        else if (high >= 0.97) parts.push(`50일 고점의 ${(high*100).toFixed(0)}%로 신고점 직전`);
        else parts.push(`50일 고점의 ${(high*100).toFixed(0)}%`);
      }

      // 등급 판단 근거
      if (isA) {
        parts.push(`거래량 ${vol.toFixed(2)}배, 종가위치 ${(cp*100).toFixed(0)}%로 당일 매수세 확인`);
      } else {
        if (vol < 1.3 && cp < 0.6) {
          parts.push(`거래량 ${vol.toFixed(2)}배·종가위치 ${(cp*100).toFixed(0)}% 모두 미흡 — 추세만 확인된 상태`);
        } else if (vol < 1.3) {
          parts.push(`거래량 ${vol.toFixed(2)}배로 평소 수준 — 거래량 확인 후 진입 검토`);
        } else {
          parts.push(`종가위치 ${(cp*100).toFixed(0)}%로 장중 밀림 — 내일 종가 위치 재확인 필요`);
        }
      }

      // 과열 경고 (워치리스트 A등급 첫 줄에서 이미 언급했으면 생략)
      if (!(group === "list" && isA && ret20 >= 0.40)) {
        if (ret20 >= 0.50) parts.push(`20일 수익률 ${(ret20*100).toFixed(0)}%로 단기 급등 구간, 눌림 주의`);
        else if (ret20 >= 0.35) parts.push(`20일 수익률 ${(ret20*100).toFixed(0)}%로 다소 과열`);
      }

      return parts.join(". ") + ".";
    }

    function buildWatchlist(candidates) {
      function priorityScore(r) {
        let s = 0;
        if (r.grade === "A") s += 40;
        if ((r.close_to_50d_high ?? 0) >= 0.97) s += 25;
        s += Math.min((r.rs_sector_20d ?? 0) * 100, 20);
        const ret20 = r.return_20d ?? 0;
        if (ret20 < 0.20) s += 15;
        else if (ret20 < 0.40) s += 8;
        return s;
      }

      const scored = candidates.map(r => ({ ...r, _ps: priorityScore(r) }))
        .sort((a, b) => b._ps - a._ps);

      const entry = [], watch = [], list = [];

      for (const r of scored) {
        const isA      = r.grade === "A";
        const nearHigh = (r.close_to_50d_high ?? 0) >= 0.97;
        const ret20ok  = (r.return_20d ?? 0) < 0.40;

        if (isA && nearHigh && ret20ok) {
          entry.push(r);
        } else if (!isA && nearHigh && ret20ok) {
          watch.push(r);
        } else {
          list.push(r);
        }
      }

      return { entry, watch, list };
    }

    function renderWatchlistGroup(title, cls, group, rows) {
      const titleHtml = `<div class="wl-group-title ${cls}">${escapeHtml(title)}</div>`;
      if (!rows.length) {
        return `<div class="wl-group">${titleHtml}<div class="wl-empty">해당 종목 없음</div></div>`;
      }
      const rowsHtml = rows.map(r => {
        const gradeCls = r.grade === "B" ? " b" : "";
        return `<div class="wl-row">
          <div class="wl-row-top">
            <span class="wl-grade${gradeCls}">${escapeHtml(r.grade)}</span>
            <span class="wl-ticker ticker-link" data-ticker="${escapeHtml(r.ticker)}" data-name="${escapeHtml(r.name||"")}">${escapeHtml(r.ticker)}</span>
            <span class="wl-name">${escapeHtml(r.name || "")}</span>
          </div>
          <div class="wl-comment">${escapeHtml(buildComment(r, group))}</div>
        </div>`;
      }).join("");
      return `<div class="wl-group">${titleHtml}<div class="wl-rows">${rowsHtml}</div></div>`;
    }

    function renderWatchlist(data) {
      const panel  = document.getElementById("watchlistPanel");
      const groups = document.getElementById("watchlistGroups");
      const dateEl = document.getElementById("watchlistDate");
      const candidates = data.candidates || [];
      if (!candidates.length) { panel.style.display = "none"; return; }

      const { entry, watch, list } = buildWatchlist(candidates);
      dateEl.textContent = data.date || "";
      groups.innerHTML = [
        renderWatchlistGroup("오늘 진입 검토",   "g-entry", "entry", entry),
        renderWatchlistGroup("거래량 확인 대기", "g-watch", "watch", watch),
        renderWatchlistGroup("워치리스트",       "g-list",  "list",  list),
      ].join("");

      groups.querySelectorAll(".ticker-link").forEach(el => {
        el.addEventListener("click", () => openChartModal(el.dataset.ticker, el.dataset.name));
      });

      panel.style.display = "block";
    }

    // --- Filtering & sorting ---
    function applyFilters() {
      let rows = allCandidates;
      if (activeGrade !== "all") rows = rows.filter(r => r.grade === activeGrade);
      const sector = sectorFilter.value;
      if (sector) rows = rows.filter(r => r.sector === sector);

      // Market cap filter
      if (activeMcap !== "all") {
        const [capMin, capMax] = mcapRanges[activeMcap];
        rows = rows.filter(r => {
          const cap = r.market_cap;
          if (cap == null) return false;
          return cap >= capMin && (!isFinite(capMax) || cap < capMax);
        });
      }

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
        ["Liquidity",  (() => { const v = data.config?.min_dollar_volume; if (v == null) return "-"; const m = v / 1e6; return m >= 1000 ? `$${(m/1000).toFixed(1)}B` : `$${Math.round(m)}M`; })(), "20D average"],
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

      const totalAll  = Object.values(sectorTotals).reduce((s, v) => s + v, 0);
      const totalCand = Object.values(candBySector).reduce((s, v) => s + v, 0);
      const totalRate = totalAll > 0 ? (totalCand / totalAll * 100) : 0;
      const tfoot = hasCandidates
        ? `<tfoot><tr>
            <td><strong>합계</strong></td>
            <td class="num"><strong>${totalAll}</strong></td>
            <td class="num"><strong>${totalCand}</strong></td>
            <td><span class="hit-bar"><span style="width:${Math.min(Math.round(totalRate*2),100)}%"></span></span><span class="hit-rate">${totalRate.toFixed(1)}%</span></td>
           </tr></tfoot>`
        : `<tfoot><tr><td><strong>합계</strong></td><td class="num"><strong>${totalAll}</strong></td></tr></tfoot>`;

      universeChips.innerHTML = `<table class="sector-table"><thead>${thead}</thead><tbody>${rows}</tbody>${tfoot}</table>`;
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

    const columnTooltips = {
      "_rank":                  "점수 기준 순위",
      "ticker":                 "종목 티커. 클릭하면 차트 팝업",
      "name":                   "종목명",
      "sector":                 "GICS 섹터",
      "grade":                  "A: 추세 + 거래량 모두 확인\nB: 추세만 확인, 거래량 미충족",
      "trend_stage":            "50일 고점 돌파 후 경과 거래일 기준 단계\n· Early Breakout (≤7일): 가장 안전한 매수 구간\n· Trending (8~35일): 추세 중반, 리스크 다소 증가\n· Extended (36일+): 많이 올라온 상태, 추격 주의\n· Watch: 돌파 이력 없음",
      "score":                  "RS·거래량·고점 근접도 등을 percentile 가중합한 0~1 점수\n높을수록 현재 장에서 상대적으로 강한 종목",
      "market_cap":             "시가총액",
      "rs_spy_20d":             "최근 20일간 SPY 대비 상대강도 변화율\n양수 = 시장보다 더 많이 오름",
      "rs_spy_50d":             "최근 50일간 SPY 대비 상대강도 변화율\n단기(20D)와 함께 보면 모멘텀 지속성 확인 가능",
      "rs_sector_20d":          "최근 20일간 섹터 ETF 대비 상대강도 변화율\n섹터 안에서도 특히 강한 종목인지 확인",
      "close_to_50d_high":      "현재 종가 ÷ 최근 50일 최고가\n1.0에 가까울수록 고점 근처에서 버티는 강한 종목",
      "volume_ratio":           "당일 거래량 ÷ 20일 평균 거래량\n1.3 이상이면 평소보다 강한 매수세",
      "close_position":         "당일 범위(고가-저가) 내 종가 위치\n1.0 = 당일 최고가 마감, 0.0 = 최저가 마감",
      "rsi_14":                 "14일 RSI (상대강도지수)\n70 이상 과매수, 30 이하 과매도 구간",
      "pivot_price":            "50일 고점을 기준으로 한 매수 기준가(피벗)\n이 가격 부근이 가장 이상적인 매수 타이밍",
      "pivot_distance":         "현재 종가가 피벗 대비 몇 % 위/아래인지\n0~+5%: 정상 매수 구간\n+5% 초과(⚠): 추격 위험 — 피벗에서 너무 멀어짐",
      "base_stability":         "돌파 전 베이스 안정성 점수 (0~1)\n높을수록 최근 조용히 쉰 후 돌파 → 신뢰도 높음\n낮을수록 최근 오히려 변동성이 커진 상태",
      "sector_etf_to_52w_high": "섹터 ETF의 52주 고점 대비 현재 위치\n0.95+: 섹터 자체가 신고점 근처 → 섹터 강세\n0.80 이하: 섹터 전체가 약세, 개별 종목 신뢰도 낮아짐",
    };

    function renderTable(rows) {
      if (!rows || rows.length === 0) {
        candidateTable.innerHTML = '<div class="empty">No candidates</div>';
        return;
      }
      const head = columns.map(([key, label, type]) => {
        if (type === "diff") return `<th class="diff-badge-cell"></th>`;
        const isNum = ["pct", "ratio", "decimal", "number", "marketcap"].includes(type);
        let cls = isNum ? "num" : type === "rank" ? "rank" : "";
        let sortCls = key === sortKey ? (sortAsc ? " sort-asc" : " sort-desc") : "";
        const arrow = key === sortKey ? (sortAsc ? "↑" : "↓") : "↕";
        const tooltip = columnTooltips[key] ? ` title="${escapeHtml(columnTooltips[key])}"` : "";
        return `<th class="${cls}${sortCls}" data-key="${escapeHtml(key)}"${tooltip}>${escapeHtml(label)}<span class="sort-arrow">${arrow}</span></th>`;
      }).join("");
      const body = rows.map(row => {
        const cells = columns.map(([key, , type]) => {
          if (type === "diff") {
            const d = row._diff;
            if (!d) return `<td class="diff-badge-cell"></td>`;
            if (d === "new")  return `<td class="diff-badge-cell"><span class="db db-new">NEW</span></td>`;
            if (d === "up")   return `<td class="diff-badge-cell"><span class="db db-up">↑A</span></td>`;
            if (d === "down") return `<td class="diff-badge-cell"><span class="db db-down">↓B</span></td>`;
            return `<td class="diff-badge-cell"></td>`;
          }
          if (type === "rank") {
            return `<td class="rank">${row[key] ?? ""}</td>`;
          }
          if (type === "grade") {
            const grade = row[key] || "";
            return `<td><span class="badge${grade === "B" ? " b" : ""}">${escapeHtml(grade)}</span></td>`;
          }
          if (type === "stage") {
            const stage = row[key] || "";
            const stageColors = {
              "Early Breakout": "color:#0e5b45;background:#edf6f2;border-color:#b2d5c8",
              "Trending":       "color:#666d63;background:#f0f0ec;border-color:var(--line)",
              "Extended":       "color:var(--warn);background:#fef6e8;border-color:#f0d8a0",
              "Watch":          "color:var(--muted);background:#fbfbfa;border-color:var(--line)",
            };
            const style = stageColors[stage] || "";
            return `<td><span style="font-size:11px;padding:2px 7px;border-radius:999px;border:1px solid;white-space:nowrap;${style}">${escapeHtml(stage)}</span></td>`;
          }
          if (type === "pctdist") {
            const v = row[key];
            if (v === null || v === undefined) return `<td class="num">–</td>`;
            const pct = (v * 100).toFixed(1);
            const isChasing = v > 0.05;
            const color = isChasing ? "color:var(--warn);font-weight:700" : v < 0 ? "color:var(--muted)" : "color:var(--accent)";
            const prefix = v >= 0 ? "+" : "";
            const warn = isChasing ? " ⚠" : "";
            return `<td class="num" style="${color}" title="${isChasing ? "피벗 대비 5% 초과 — 추격 위험" : ""}">${prefix}${pct}%${warn}</td>`;
          }
          const isNum = ["pct", "ratio", "decimal", "number", "marketcap", "money"].includes(type);
          const cls = isNum ? "num" : key === "ticker" ? "ticker" : "";
          if (key === "ticker" && row[key]) {
            return `<td class="${cls}"><span class="ticker-link" data-ticker="${escapeHtml(row[key])}" data-name="${escapeHtml(row.name || "")}" style="cursor:pointer;border-bottom:1px dotted var(--accent)">${escapeHtml(row[key])}</span></td>`;
          }
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

    function renderMarketBanner(data) {
      const banner = document.getElementById("marketBanner");
      const state  = data.market_state || "";
      if (!state || state === "Unknown") { banner.style.display = "none"; return; }

      const configs = {
        "Confirmed Uptrend": {
          cls: "uptrend", icon: "●",
          desc: "SPY·QQQ 모두 MA50 위 + 상승 중 — 매수 환경 우호적. 후보 종목 정상 출력.",
        },
        "Uptrend Under Pressure": {
          cls: "pressure", icon: "◐",
          desc: "지수 일부 약화 중 — 추세 훼손 초기. A등급 위주로 검토하고 포지션 크기 축소 권장.",
        },
        "Market in Correction": {
          cls: "correction", icon: "○",
          desc: "SPY 또는 QQQ가 MA50 아래 — 하락장. 후보 종목은 참고용으로만 활용, 신규 매수 자제.",
        },
      };
      const cfg = configs[state] || { cls: "", icon: "?", desc: state };

      banner.className = `market-banner ${cfg.cls}`;
      document.getElementById("marketBannerIcon").textContent  = cfg.icon;
      document.getElementById("marketBannerState").textContent = state;
      document.getElementById("marketBannerDesc").textContent  = cfg.desc;
      banner.style.display = "";
    }

    function renderResult(data) {
      renderMarketBanner(data);
      renderMetrics(data);
      renderFunnel(data);
      renderInsight(data);
      renderWatchlist(data);
      allCandidates = data.candidates || [];
      // score 내림차순으로 순위 부여 (필터/정렬과 무관하게 고정)
      [...allCandidates]
        .sort((a, b) => (b.score ?? -1) - (a.score ?? -1))
        .forEach((r, i) => { r._rank = i + 1; });
      renderUniverse(data);  // allCandidates 세팅 후 호출해야 섹터 적중률 계산 가능
      populateSectorOptions();
      resetMcapFromData();
      applyFilters();
      const srcLabel = data.source === "latest" ? "파일 로드" : data.from_cache ? "캐시 사용" : "신규 다운로드";
      const runLabel = data.last_run_at ? `마지막 실행 ${escapeHtml(data.last_run_at)}` : "";
      runMeta.textContent = [srcLabel, runLabel].filter(Boolean).join(" · ");
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

      const chartsHtml = `
        <div class="tr-inline-charts" id="trInlineCharts">
          <div class="tr-inline-chart-cell">
            <div class="tr-inline-chart-header">
              <span class="tr-inline-chart-period">1M</span>
              <span class="tr-inline-chart-return" id="trChartReturn-21"></span>
            </div>
            <div class="tr-inline-chart-wrap" id="trChartWrap-21"></div>
          </div>
          <div class="tr-inline-chart-cell">
            <div class="tr-inline-chart-header">
              <span class="tr-inline-chart-period">3M</span>
              <span class="tr-inline-chart-return" id="trChartReturn-63"></span>
            </div>
            <div class="tr-inline-chart-wrap" id="trChartWrap-63"></div>
          </div>
        </div>`;

      tickerResult.innerHTML = `<div class="tr-layout"><div class="tr-info">${header}${body}</div>${chartsHtml}</div>`;

      // Load inline charts
      [21, 63].forEach(days => {
        const retEl  = document.getElementById(`trChartReturn-${days}`);
        const wrapEl = document.getElementById(`trChartWrap-${days}`);
        retEl.textContent = "…";
        getJson(`/api/chart?symbol=${encodeURIComponent(d.symbol)}&days=${days}`).then(data => {
          if (data.error) { retEl.textContent = ""; return; }
          const prices = data.prices;
          const ret = prices.length >= 2
            ? ((prices[prices.length - 1] - prices[0]) / prices[0] * 100)
            : null;
          if (ret !== null) {
            retEl.textContent = (ret >= 0 ? "+" : "") + ret.toFixed(1) + "%";
            retEl.className = "tr-inline-chart-return " + (ret >= 0 ? "pos" : "neg");
          } else {
            retEl.textContent = "";
          }
          requestAnimationFrame(() => requestAnimationFrame(() => drawChartSmall(wrapEl, data.dates, prices)));
        }).catch(() => { retEl.textContent = ""; });
      });
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

    // --- Chart popup ---
    const chartOverlay   = document.getElementById("chartOverlay");
    const chartModalClose = document.getElementById("chartModalClose");

    function closeChartModal() {
      chartOverlay.classList.remove("open");
    }

    chartModalClose.addEventListener("click", closeChartModal);
    chartOverlay.addEventListener("click", e => { if (e.target === chartOverlay) closeChartModal(); });
    document.addEventListener("keydown", e => { if (e.key === "Escape") closeChartModal(); });

    function drawChartSmall(wrapEl, dates, prices) {
      const W = wrapEl.clientWidth || 150;
      const H = wrapEl.clientHeight || 80;
      const pad = { top: 4, right: 6, bottom: 14, left: 28 };
      const iW = W - pad.left - pad.right;
      const iH = H - pad.top - pad.bottom;

      const minP = Math.min(...prices);
      const maxP = Math.max(...prices);
      const range = maxP - minP || 1;

      const x = i => pad.left + (i / (prices.length - 1)) * iW;
      const y = v => pad.top + iH - ((v - minP) / range) * iH;

      const pts = prices.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
      const areaBottom = pad.top + iH;
      const areaPath = `M${x(0).toFixed(1)},${areaBottom} ` +
        prices.map((v, i) => `L${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ") +
        ` L${x(prices.length-1).toFixed(1)},${areaBottom} Z`;

      const isUp = prices[prices.length - 1] >= prices[0];
      const lineColor = isUp ? "var(--accent)" : "var(--danger)";
      const gradId = "areaGradS_" + Math.random().toString(36).slice(2);

      const yTicks = [minP, maxP].map(v => ({ v, yPx: y(v).toFixed(1) }));
      const xTicks = [0, prices.length - 1].map(i => ({
        label: dates[i] ? dates[i].slice(5) : "",
        xPx: x(i).toFixed(1),
      }));

      wrapEl.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
        <defs>
          <linearGradient id="${gradId}" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="${lineColor}" stop-opacity="0.18"/>
            <stop offset="100%" stop-color="${lineColor}" stop-opacity="0"/>
          </linearGradient>
        </defs>
        <path d="${areaPath}" fill="url(#${gradId})"/>
        <polyline points="${pts}" fill="none" stroke="${lineColor}" stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round"/>
        ${yTicks.map(t => `
          <line x1="${pad.left}" y1="${t.yPx}" x2="${pad.left + iW}" y2="${t.yPx}" stroke="var(--line)" stroke-width="0.8"/>
          <text x="${pad.left - 4}" y="${t.yPx}" text-anchor="end" dominant-baseline="middle" fill="var(--muted)" font-size="8" font-family="DM Mono,monospace">$${t.v >= 1000 ? (t.v/1000).toFixed(1)+"K" : t.v.toFixed(0)}</text>
        `).join("")}
        ${xTicks.map(t => `
          <text x="${t.xPx}" y="${pad.top + iH + 10}" text-anchor="middle" fill="var(--muted)" font-size="8" font-family="DM Mono,monospace">${t.label}</text>
        `).join("")}
      </svg>`;
    }

    function drawChart(wrapEl, dates, prices) {
      const W = wrapEl.clientWidth || 380;
      const H = 130;
      const pad = { top: 8, right: 10, bottom: 24, left: 40 };
      const iW = W - pad.left - pad.right;
      const iH = H - pad.top - pad.bottom;

      const minP = Math.min(...prices);
      const maxP = Math.max(...prices);
      const range = maxP - minP || 1;

      const x = i => pad.left + (i / (prices.length - 1)) * iW;
      const y = v => pad.top + iH - ((v - minP) / range) * iH;

      const pts = prices.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
      const areaBottom = pad.top + iH;
      const areaPath = `M${x(0).toFixed(1)},${areaBottom} ` +
        prices.map((v, i) => `L${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ") +
        ` L${x(prices.length-1).toFixed(1)},${areaBottom} Z`;

      const isUp = prices[prices.length - 1] >= prices[0];
      const lineColor = isUp ? "var(--accent)" : "var(--danger)";
      const gradId = "areaGrad_" + Math.random().toString(36).slice(2);

      const yTicks = [minP, minP + range * 0.5, maxP].map(v => ({
        v, yPx: y(v).toFixed(1)
      }));
      const xTickIdxs = [0, Math.floor((prices.length - 1) / 2), prices.length - 1];
      const xTicks = xTickIdxs.map(i => ({
        label: dates[i] ? dates[i].slice(5) : "",
        xPx: x(i).toFixed(1),
      }));

      wrapEl.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
        <defs>
          <linearGradient id="${gradId}" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="${lineColor}" stop-opacity="0.15"/>
            <stop offset="100%" stop-color="${lineColor}" stop-opacity="0"/>
          </linearGradient>
        </defs>
        <path d="${areaPath}" fill="url(#${gradId})"/>
        <polyline points="${pts}" fill="none" stroke="${lineColor}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>
        ${yTicks.map(t => `
          <line x1="${pad.left}" y1="${t.yPx}" x2="${pad.left + iW}" y2="${t.yPx}" stroke="var(--line)" stroke-width="1"/>
          <text x="${pad.left - 5}" y="${t.yPx}" text-anchor="end" dominant-baseline="middle" fill="var(--muted)" font-size="9" font-family="DM Mono,monospace">$${t.v >= 1000 ? (t.v/1000).toFixed(1)+"K" : t.v.toFixed(0)}</text>
        `).join("")}
        ${xTicks.map(t => `
          <text x="${t.xPx}" y="${pad.top + iH + 14}" text-anchor="middle" fill="var(--muted)" font-size="9" font-family="DM Mono,monospace">${t.label}</text>
        `).join("")}
      </svg>`;
    }

    let chartCurrentTicker = null;

    async function loadChartCell(ticker, days) {
      const retEl  = document.getElementById(`chartReturn-${days}`);
      const wrapEl = document.getElementById(`chartSvgWrap-${days}`);
      retEl.textContent = "…";
      retEl.className = "chart-cell-return";
      wrapEl.innerHTML = "";
      try {
        const data = await getJson(`/api/chart?symbol=${encodeURIComponent(ticker)}&days=${days}`);
        if (data.error) { retEl.textContent = data.error; return; }
        const prices = data.prices;
        const ret = prices.length >= 2
          ? ((prices[prices.length - 1] - prices[0]) / prices[0] * 100)
          : null;
        if (ret !== null) {
          retEl.textContent = (ret >= 0 ? "+" : "") + ret.toFixed(1) + "%";
          retEl.className = "chart-cell-return " + (ret >= 0 ? "pos" : "neg");
        } else {
          retEl.textContent = "";
        }
        drawChart(wrapEl, data.dates, prices);
      } catch (err) {
        retEl.textContent = err.message;
      }
    }

    function openChartModal(ticker, name) {
      chartCurrentTicker = ticker;
      document.getElementById("chartModalTicker").textContent = ticker;
      document.getElementById("chartModalName").textContent = name || "";
      document.getElementById("chartPerplexityBtn").href =
        `https://www.perplexity.ai/finance/${encodeURIComponent(ticker)}`;
      chartOverlay.classList.add("open");
      [21, 63, 126, 252].forEach(days => loadChartCell(ticker, days));
    }

    // Delegate ticker-link clicks from the table
    candidateTable.addEventListener("click", e => {
      const link = e.target.closest(".ticker-link");
      if (!link) return;
      openChartModal(link.dataset.ticker, link.dataset.name);
    });

    // --- API calls ---
    async function loadUniverse() {
      try {
        const data = await getJson("/api/universe");
        renderUniverse(data);
      } catch (error) {
        statusText.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
      }
    }

    // --- Diff ---
    let diffData = null;

    function renderDiff(diff) {
      diffData = diff;
      const section = document.getElementById("diffSection");
      if (!diff || (!diff.new_entries.length && !diff.dropped.length && !diff.upgraded.length && !diff.downgraded.length)) {
        section.style.display = "none";
        return;
      }
      section.style.display = "";
      document.getElementById("diffDateLabel").textContent =
        diff.prev_date ? `${diff.prev_date} → ${diff.curr_date}` : diff.curr_date;

      const chips = [];
      if (diff.new_entries.length)  chips.push(`<span class="diff-chip new-in"><span class="count">+${diff.new_entries.length}</span> 신규 진입</span>`);
      if (diff.dropped.length)      chips.push(`<span class="diff-chip dropped"><span class="count">-${diff.dropped.length}</span> 이탈</span>`);
      if (diff.upgraded.length)     chips.push(`<span class="diff-chip upgraded"><span class="count">${diff.upgraded.length}</span> B→A 상향</span>`);
      if (diff.downgraded.length)   chips.push(`<span class="diff-chip downgraded"><span class="count">${diff.downgraded.length}</span> A→B 하향</span>`);
      document.getElementById("diffChips").innerHTML = chips.join("");

      const parts = [];
      if (diff.new_entries.length)  parts.push(`신규: ${diff.new_entries.map(t => `<strong>${escapeHtml(t)}</strong>`).join(", ")}`);
      if (diff.dropped.length)      parts.push(`이탈: ${diff.dropped.map(t => `<strong>${escapeHtml(t)}</strong>`).join(", ")}`);
      if (diff.upgraded.length)     parts.push(`B→A: ${diff.upgraded.map(t => `<strong>${escapeHtml(t)}</strong>`).join(", ")}`);
      if (diff.downgraded.length)   parts.push(`A→B: ${diff.downgraded.map(t => `<strong>${escapeHtml(t)}</strong>`).join(", ")}`);
      document.getElementById("diffTickers").innerHTML = parts.join("<br>");

      // apply _diff flags to allCandidates
      const newSet      = new Set(diff.new_entries);
      const upgradedSet = new Set(diff.upgraded);
      const downgradedSet = new Set(diff.downgraded);
      allCandidates.forEach(r => {
        if (newSet.has(r.ticker))        r._diff = "new";
        else if (upgradedSet.has(r.ticker))   r._diff = "up";
        else if (downgradedSet.has(r.ticker)) r._diff = "down";
        else r._diff = null;
      });
      applyFilters();
    }

    async function loadDiff() {
      try {
        const diff = await getJson("/api/diff");
        renderDiff(diff);
      } catch (_) { /* diff 없으면 조용히 무시 */ }
    }

    async function loadLatest() {
      try {
        const data = await getJson("/api/latest");
        if (data.has_result) {
          renderResult(data);
          statusText.innerHTML = `Latest result · <strong>${escapeHtml(data.date || "")}</strong>`;
          await loadDiff();
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
          await loadDiff();
        } else {
          statusText.innerHTML = `<span class="error">결과 없음 — main.py를 먼저 실행해 주세요.</span>`;
        }
      } catch (error) {
        statusText.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
      } finally {
        setBusy(false);
      }
    }

    async function runScreener() {
      setBusy(true);
      statusText.innerHTML = "스크리닝 실행 중… (약 1~2분 소요)";
      runMeta.textContent = "";
      try {
        const data = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        }).then(r => r.json());
        if (data.error) throw new Error(data.error);
        renderResult(data);
        statusText.innerHTML = `스크리닝 완료 · <strong>${escapeHtml(data.date || "")}</strong>`;
        await loadDiff();
      } catch (error) {
        statusText.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
      } finally {
        setBusy(false);
      }
    }

    refreshButton.addEventListener("click", refresh);
    runButton.addEventListener("click", runScreener);
    initMcapButtons();
    loadUniverse();
    loadLatest();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local RS volume screener dashboard")
    parser.add_argument("--host", default="0.0.0.0")
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
        "rsi_14", "avg_dollar_volume_20d", "return_20d",
        # 신규 투자 맥락 필드
        "trend_stage", "pivot_price", "pivot_distance", "chasing_risk",
        "buy_zone_low", "buy_zone_high", "days_since_breakout",
        "base_stability", "sector_etf_to_52w_high",
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
        "last_run_at": last_run_at(),
    }


def last_run_at() -> str | None:
    log_file = PROJECT_ROOT / "launchd" / "screener.log"
    if not log_file.exists():
        return None
    try:
        with log_file.open("r") as f:
            for line in reversed(f.readlines()):
                if "screener 완료" in line or "screener 실패" in line:
                    return line.split("===")[1].strip().split(" screener")[0]
    except Exception:
        return None
    return None


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


def diff_response() -> dict[str, object]:
    """최근 2개 candidates CSV를 비교해 신규/이탈/등급변경을 반환."""
    files = sorted(OUTPUT_DIR.glob("screener_candidates_*.csv"))
    if len(files) < 2:
        return {"has_diff": False}

    prev_path, curr_path = files[-2], files[-1]
    prev_date = prev_path.stem.removeprefix("screener_candidates_")
    curr_date = curr_path.stem.removeprefix("screener_candidates_")

    prev = pd.read_csv(prev_path)[["ticker", "grade"]].set_index("ticker")["grade"]
    curr = pd.read_csv(curr_path)[["ticker", "grade"]].set_index("ticker")["grade"]

    prev_set, curr_set = set(prev.index), set(curr.index)
    new_entries = sorted(curr_set - prev_set)
    dropped     = sorted(prev_set - curr_set)
    both        = curr_set & prev_set
    upgraded    = sorted(t for t in both if prev[t] == "B" and curr[t] == "A")
    downgraded  = sorted(t for t in both if prev[t] == "A" and curr[t] == "B")

    return {
        "has_diff": True,
        "prev_date": prev_date,
        "curr_date": curr_date,
        "new_entries": new_entries,
        "dropped": dropped,
        "upgraded": upgraded,
        "downgraded": downgraded,
    }


def chart_data(symbol: str, days: int = 63) -> dict[str, object]:
    symbol = symbol.strip().upper()
    cache_file = PROJECT_ROOT / _symbol_cache_dir(ScreenerConfig().period) / f"{symbol}.pkl"
    if not cache_file.exists():
        return {"error": "캐시 없음 — 스크리너를 먼저 실행해 주세요."}
    try:
        with cache_file.open("rb") as f:
            df = pickle.load(f)
        if df is None or df.empty or "Close" not in df.columns:
            return {"error": "데이터 없음"}
        close = df["Close"].dropna().tail(days)
        dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in close.index]
        prices = [round(float(v), 2) for v in close.values]
        return {"symbol": symbol, "dates": dates, "prices": prices}
    except Exception as exc:
        return {"error": str(exc)}


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
        if parsed.path == "/api/diff":
            self.send_json(diff_response())
            return
        if parsed.path == "/api/ticker":
            params = parse_qs(parsed.query)
            symbol = (params.get("symbol") or [""])[0]
            if not symbol:
                self.send_json({"error": "symbol 파라미터가 필요합니다."}, status=HTTPStatus.BAD_REQUEST)
            else:
                self.send_json(ticker_lookup(symbol))
            return
        if parsed.path == "/api/chart":
            params = parse_qs(parsed.query)
            symbol = (params.get("symbol") or [""])[0]
            if not symbol:
                self.send_json({"error": "symbol 파라미터가 필요합니다."}, status=HTTPStatus.BAD_REQUEST)
            else:
                try:
                    days = int((params.get("days") or ["63"])[0])
                except ValueError:
                    days = 63
                self.send_json(chart_data(symbol, days=days))
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
