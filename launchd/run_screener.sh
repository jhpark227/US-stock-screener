#!/bin/bash
set -euo pipefail

PROJECT_DIR="/Users/jhpark/Documents/Claude Code/US-stock-screener"
LOG_FILE="$PROJECT_DIR/launchd/screener.log"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') screener 시작 ===" >> "$LOG_FILE"

cd "$PROJECT_DIR"
UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run python main.py >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') screener 완료 (exit 0) ===" >> "$LOG_FILE"
else
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') screener 실패 (exit $EXIT_CODE) ===" >> "$LOG_FILE"
fi
