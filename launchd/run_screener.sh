#!/bin/bash
# set -e를 쓰지 않는다 — 실패 시에도 아래 분기 로그가 남아야 한다.
set -uo pipefail

PROJECT_DIR="/Users/jhpark/Documents/Claude Code/US-stock-screener"
LOG_FILE="$PROJECT_DIR/launchd/screener.log"
UV="/opt/homebrew/bin/uv"

log() { echo "=== $(date '+%Y-%m-%d %H:%M:%S') $1 ===" >> "$LOG_FILE"; }
run_py() { UV_CACHE_DIR=/tmp/uv-cache "$UV" run python "$@" >> "$LOG_FILE" 2>&1; }

cd "$PROJECT_DIR" || exit 1

after_screener() {
    # AI 데일리 코멘터리 (claude -p 구독 인증, 하루 1회 — 이미 있으면 스킵)
    log "commentary 시작"
    if run_py scripts/daily_commentary.py; then
        log "commentary 완료"
    else
        log "commentary 실패 (대시보드는 코멘트 없이 정상 동작)"
    fi

    # 정적 사이트 빌드 → git push → Cloudflare Pages 자동 재배포
    log "site 빌드 시작"
    if run_py scripts/build_site.py; then
        git add site >> "$LOG_FILE" 2>&1
        if git diff --cached --quiet; then
            log "site 변경 없음 — 배포 생략"
        elif git commit -m "site: daily build $(date '+%Y-%m-%d')" >> "$LOG_FILE" 2>&1 \
            && git push origin main >> "$LOG_FILE" 2>&1; then
            log "site 푸시 완료 (Pages 재배포 트리거)"
        else
            log "site 커밋/푸시 실패 — 네트워크·인증 확인 필요"
        fi
    else
        log "site 빌드 실패"
    fi
}

log "screener 시작"
if run_py main.py; then
    log "screener 완료 (exit 0)"
    after_screener
else
    log "screener 실패 (exit $?) — 30분 후 1회 재시도"
    sleep 1800
    if run_py main.py; then
        log "screener 재시도 성공"
        after_screener
    else
        log "screener 재시도 실패 (exit $?)"
    fi
fi
