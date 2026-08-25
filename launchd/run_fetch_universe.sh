#!/bin/bash
# set -e를 쓰지 않는다 — 실패 시에도 아래 분기 로그가 남아야 한다.
set -uo pipefail

PROJECT_DIR="/Users/jhpark/Documents/Claude Code/US-stock-screener"
LOG_FILE="$PROJECT_DIR/launchd/fetch_universe.log"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') 유니버스 갱신 시작 ===" >> "$LOG_FILE"

cd "$PROJECT_DIR" || exit 1

if UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run python scripts/fetch_universe.py >> "$LOG_FILE" 2>&1; then
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') 유니버스 갱신 완료 (exit 0) ===" >> "$LOG_FILE"
else
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') 유니버스 갱신 실패 (exit $?) ===" >> "$LOG_FILE"
fi
