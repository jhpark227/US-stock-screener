#!/bin/bash
# set -e를 쓰지 않는다 — 실패 시에도 아래 분기 로그가 남아야 한다.
set -uo pipefail

PROJECT_DIR="/Users/jhpark/Documents/Claude Code/US-stock-screener"
LOG_FILE="$PROJECT_DIR/launchd/screener.log"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') screener 시작 ===" >> "$LOG_FILE"

cd "$PROJECT_DIR" || exit 1

if UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run python main.py >> "$LOG_FILE" 2>&1; then
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') screener 완료 (exit 0) ===" >> "$LOG_FILE"
    # AI 데일리 코멘터리 (claude -p 구독 인증, 하루 1회 — 이미 있으면 스킵)
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') commentary 시작 ===" >> "$LOG_FILE"
    if UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run python scripts/daily_commentary.py >> "$LOG_FILE" 2>&1; then
        echo "=== $(date '+%Y-%m-%d %H:%M:%S') commentary 완료 ===" >> "$LOG_FILE"
    else
        echo "=== $(date '+%Y-%m-%d %H:%M:%S') commentary 실패 (대시보드는 코멘트 없이 정상 동작) ===" >> "$LOG_FILE"
    fi
else
    EXIT_CODE=$?
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') screener 실패 (exit $EXIT_CODE) — 30분 후 1회 재시도 ===" >> "$LOG_FILE"
    sleep 1800
    if UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run python main.py >> "$LOG_FILE" 2>&1; then
        echo "=== $(date '+%Y-%m-%d %H:%M:%S') screener 재시도 성공 ===" >> "$LOG_FILE"
        UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run python scripts/daily_commentary.py >> "$LOG_FILE" 2>&1 \
            && echo "=== $(date '+%Y-%m-%d %H:%M:%S') commentary 완료 ===" >> "$LOG_FILE" \
            || echo "=== $(date '+%Y-%m-%d %H:%M:%S') commentary 실패 ===" >> "$LOG_FILE"
    else
        echo "=== $(date '+%Y-%m-%d %H:%M:%S') screener 재시도 실패 (exit $?) ===" >> "$LOG_FILE"
    fi
fi
