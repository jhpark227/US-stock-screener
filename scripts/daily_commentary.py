"""일일 AI 코멘터리 — 설명 전용 (판정은 main.py의 규칙이 결정).

판정(진입 검토/대기/스킵)과 사유는 `assign_verdicts()`가 결정론적으로 산출한다.
이 스크립트는 그 판정을 받아 '설명문'만 작성한다 — AI는 판정을 바꿀 수 없고,
데이터 밖의 정성 정보(예: 크립토 프록시, 규제 이슈)는 ⚠ 노트로만 덧붙인다.

Claude Code 헤드리스 모드(`claude -p`) 사용: API 키(토큰당 과금) 대신 구독 쿼터.
subprocess 환경에서 ANTHROPIC_API_KEY를 제거해 구독 인증을 강제한다.

실행: UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/daily_commentary.py [--force]
출력: outputs/commentary_YYYY-MM-DD.json (이미 있으면 스킵 — 하루 1회)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
CLAUDE_BIN = "/opt/homebrew/bin/claude"
EXPLAIN_CAP = 25  # AI 설명문 대상 상한 (진입 검토 우선, 남는 슬롯에 대기 점수순)

PROMPT_RULES = """\
너는 미국 주식 스크리너의 일일 코멘터리 작성자다. 판정(진입 검토/대기/스킵)과 그 사유는
백테스트 기반 규칙이 이미 결정했다 — 너의 역할은 판정을 바꾸는 것이 아니라 '설명'이다.

## 공통 문체 규칙 (market_comment·종목 comment 모두 적용)
- 명사형·구 단위로 끝맺는 개조식 ("~하는 중", "~구간", "~겹친 상태", "~반영").
  "-다/-한다" 평서문 종결과 권유형 어미("~바람직하다", "~하라")는 쓰지 마라. 마크다운(**) 금지.
- 읽는 사람이 한 번에 이해해야 한다. 압축 한자어로 뭉치지 말고 자연스러운 말로 풀어 써라
  (나쁨: "신선한 신호와 과열이 병존" / 좋음: "신호는 신선한데 RSI 과열이 겹친 상태").
  전문 지표는 첫 언급에 짧은 부연을 붙여라 (예: "시장 폭(MA60 위 종목 비율)").
- 숫자는 라인당 2개 이내 — 나머지는 방향 언어(개선/악화/횡보)로. 대시(—)는 라인당 1회 이하.
- 매일·매 종목 같은 문장 골격이 반복되지 않게 구조를 바꿔라.

## 종목별 comment 작성 규칙
- 한국어 1~2문장, 120자 이내. 판정 사유(verdict_reason)를 반복하지 말고 실행 지침을 더하라:
  진입 검토 → 리스크 관리(경고 라벨, 포지션 크기, 분할 여부)와 신호 대비 등락 해석. trigger_price는
  가장 최근 거래량 신호일 종가(발생가), ext_from_trigger는 발생가 대비 등락률이다. 등락률이 낮으면
  "발생가 근처"라는 사실만 전달하라 — 눌림 대기를 권하지 마라(되돌림 대기 전략은 백테스트상 근거 없음).
  대기 → 기다리는 조건이 충족됐는지 판단할 기준(가격대, 날짜, 신호 재발생).
- 데이터 밖의 정성 정보를 알면 "⚠" 뒤에 한 구절로 덧붙여라 (예: "⚠ 비트코인 트레저리 회사 — 사실상 크립토 레버리지").
  이는 판정 변경이 아니라 참고 노트다. 모르면 붙이지 마라.
- 수치는 구체적으로 (가격, 배수, 날짜).

## market_comment 작성 규칙 — 개조식 브리핑
아래 라벨의 라인들을 줄바꿈(\n)으로 구분해 작성하라 (라인당 1~2문장, 새 정보 없는 라벨은 생략):
오늘 핵심: 첫 줄 필수. 오늘 가장 중요한 한 가지 — '## 추세 비교' 블록의 방향(개선/악화/로테이션)과
  오늘 리스트의 성격을 묶어 서술. 이 줄만 읽어도 오늘의 그림이 잡혀야 한다.
  블록에 없는 수치를 지어내지 마라.
국면: 시장 국면 판정과 그 견고함/취약점.
포지셔닝: 진입 검토군의 섹터·테마 구성과 쏠림, 쏠림이 있으면 노출 관리 원칙.
신호 품질: '## 집계' 블록 기반 — signal_type 구성(surge+acc 유무가 기대 알파 수준을 좌우).
  용어는 surge → "거래량 급증", acc → "지속 매집", surge+acc → "급증+매집"으로 표기하라.
  신호 경과(D+N)와 RSI 과열(75+) 간 긴장. 신호가 최근이어도 과열이면 즉시 진입 대상과
  신중 접근(포지션 축소·분할) 대상을 구분하라.
유효 베팅: 클러스터 압축('## 집계'의 클러스터 대표)과 섹터 쏠림을 감안한 실질 독립 베팅 수 추정과 그 함의.
이벤트: 어닝스 대기 등 일정 리스크.

톤 견본 (내용이 아니라 문체만 참고):
"오늘 핵심: 지수는 견고하지만 참여 종목이 한 달째 줄어드는 중 — 주도권이 성장주에서 방어·실물로 넘어가는 국면이고, 오늘 리스트도 그 로테이션의 반영."
"신호 품질: 급증+매집 조합이 없어 기대치는 평범한 날. AEM 등 4종은 신호는 신선한데 RSI 과열이 겹쳐 바로 들어가기엔 부담스러운 조합."

## 출력 형식
설명 없이 아래 JSON만 출력하라. items는 입력된 전 종목을 입력 순서대로 포함:
{"market_comment": "...", "items": [{"ticker": "...", "comment": "..."}]}
"""


def build_market_summary(df: pd.DataFrame) -> str:
    """market_comment의 '신호 품질'·'유효 베팅' 라인 근거를 코드로 집계해 프롬프트에 주입."""
    lines = []
    vc = df["verdict"].value_counts()
    lines.append(f"판정 분포: 진입 검토 {vc.get('진입 검토', 0)} / 대기 {vc.get('대기', 0)} / 스킵 {vc.get('스킵', 0)}")
    entry = df[df["verdict"] == "진입 검토"]
    if entry.empty:
        return "\n".join(f"- {l}" for l in lines)

    sec = entry["sector"].value_counts()
    lines.append("진입 검토 섹터: " + ", ".join(f"{s} {n}" for s, n in sec.items()))
    sig = entry["signal_type"].value_counts()
    sig_names = {"surge": "거래량 급증(surge)", "acc": "지속 매집(acc)", "surge+acc": "급증+매집(surge+acc)"}
    sig_line = "신호 구성: " + ", ".join(f"{sig_names.get(s, s)} {n}" for s, n in sig.items())
    if "surge+acc" not in sig.index:
        sig_line += " — 최강 조합(surge+acc) 부재"
    lines.append(sig_line)
    if "rsi_14" in entry.columns and entry["rsi_14"].notna().any():
        hot = entry.loc[entry["rsi_14"] >= 75, "ticker"].tolist()
        rsi_line = f"RSI(14): 평균 {entry['rsi_14'].mean():.0f}, 75 이상 {len(hot)}개"
        if hot:
            rsi_line += f" ({', '.join(hot)})"
        lines.append(rsi_line)
    fresh = entry["days_since_trigger"]
    lines.append(f"신호 경과: D+0~2 {int((fresh <= 2).sum())}개 / D+3 이상 {int((fresh >= 3).sum())}개")
    reps = []
    for _, r in entry.iterrows():
        m = re.search(r"클러스터 대표\((\d+)종목 중\)", str(r.get("verdict_reason", "")))
        if m:
            reps.append(f"{r['ticker']}(원 클러스터 {m.group(1)}종목)")
    if reps:
        lines.append("클러스터 대표: " + ", ".join(reps) + " — 상관 클러스터 압축 결과라 실질 독립 베팅 수는 표시 종목 수보다 적음")
    return "\n".join(f"- {l}" for l in lines)


def build_trend_summary(date: str) -> str:
    """유니버스 CSV 이력으로 전일/1주/1개월 대비 시장 변화를 집계.

    필터 버전과 무관한 지표만 사용(시장 폭·RS+ 비율·국면·섹터 RS) — 후보 수·신호 구성은
    2026-08 필터 개편 전 파일과 비교할 수 없어 제외.
    """
    files = sorted(OUTPUT_DIR.glob("screener_universe_*.csv"))
    dates = [f.stem.removeprefix("screener_universe_") for f in files]
    if date not in dates:
        return "- (이력 없음)"
    idx = dates.index(date)
    offsets = [("오늘", 0), ("전일", 1), ("1주 전", 5), ("1개월 전", 21)]
    snaps = []
    for label, off in offsets:
        if idx - off < 0:
            continue
        f = files[idx - off]
        try:
            u = pd.read_csv(f, usecols=lambda c: c in {"sector", "above_ma60", "rs_spy_20d", "rsi_14"})
        except (ValueError, OSError):
            continue
        top_rs = (
            u.groupby("sector")["rs_spy_20d"].mean().sort_values(ascending=False).head(3).index.tolist()
            if {"sector", "rs_spy_20d"} <= set(u.columns) else []
        )
        snaps.append({
            "label": f"{label}({dates[idx - off]})",
            "breadth": u["above_ma60"].mean(),
            "rs_pos": (u["rs_spy_20d"] > 0).mean(),
            "rsi_med": u["rsi_14"].median() if "rsi_14" in u.columns else float("nan"),
            "top_rs": top_rs,
        })
    if len(snaps) < 2:
        return "- (비교할 이력 부족)"
    # 국면 라벨은 2026-08-22 로직 교체로 과거 파일과 정의가 달라 비교에서 제외
    lines = [
        "시장 폭(MA60 상회 비율): " + " ← ".join(f"{s['label']} {s['breadth']:.0%}" for s in snaps),
        "RS+ 비율(SPY 20d 아웃퍼폼 종목 비중): " + " ← ".join(f"{s['label']} {s['rs_pos']:.0%}" for s in snaps),
        "유니버스 RSI(14) 중앙값: " + " ← ".join(f"{s['label']} {s['rsi_med']:.0f}" for s in snaps if pd.notna(s["rsi_med"])),
        "주도 섹터(평균 RS 상위 3): " + " / ".join(f"{s['label']} {', '.join(s['top_rs'])}" for s in snaps if s["top_rs"]),
    ]
    return "\n".join(f"- {l}" for l in lines)


def build_prompt(date: str, market_state: str, rows: pd.DataFrame, df: pd.DataFrame) -> str:
    cols = [
        "ticker", "name", "sector", "verdict", "verdict_reason",
        "signal_type", "days_since_trigger", "grade", "score", "score_3m",
        "rs_spy_20d", "rs_spy_50d", "volume_ratio", "close_position", "rsi_14",
        "warnings", "next_earnings_date", "close", "trigger_price", "ext_from_trigger",
    ]
    data = rows[[c for c in cols if c in rows.columns]].round(3)
    return (
        f"{PROMPT_RULES}\n## 집계 (코드 산출, 기준일 {date}, market_state: {market_state})\n"
        + build_market_summary(df)
        + "\n\n## 추세 비교 (코드 산출 — 오늘 ← 과거 순)\n"
        + build_trend_summary(date)
        + "\n\n## 데이터 (설명 대상 종목)\n"
        + data.to_csv(index=False)
    )


def select_targets(df: pd.DataFrame) -> pd.DataFrame:
    """설명 대상: 진입 검토 전원(점수순) → 남는 슬롯에 대기(점수순). 스킵은 사유 태그로 충분."""
    entry = df[df["verdict"] == "진입 검토"].sort_values("score", ascending=False)
    wait = df[df["verdict"] == "대기"].sort_values("score", ascending=False)
    picked = pd.concat([entry, wait]).head(max(EXPLAIN_CAP, len(entry)))
    return picked


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"응답에서 JSON을 찾지 못함:\n{text[:500]}")
    return json.loads(match.group(0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="기존 코멘터리가 있어도 재생성")
    args = parser.parse_args()

    files = sorted(OUTPUT_DIR.glob("screener_candidates_*.csv"))
    if not files:
        sys.exit("후보 CSV가 없습니다. main.py를 먼저 실행하세요.")
    latest = files[-1]
    date = latest.stem.removeprefix("screener_candidates_")
    out_path = OUTPUT_DIR / f"commentary_{date}.json"
    if out_path.exists() and not args.force:
        print(f"이미 존재: {out_path} (--force로 재생성)")
        return

    df = pd.read_csv(latest)
    if "verdict" not in df.columns:
        sys.exit("candidates CSV에 verdict 컬럼이 없습니다. main.py를 최신 코드로 재실행하세요.")
    market_state = str(df["market_state"].dropna().iloc[0]) if "market_state" in df.columns else "Unknown"
    targets = select_targets(df)
    prompt = build_prompt(date, market_state, targets, df)

    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    print(f"claude -p 호출 중 (구독 인증, 설명 대상 {len(targets)}개 / 후보 {len(df)}개, {date})...")
    result = subprocess.run(
        [CLAUDE_BIN, "-p", prompt],
        capture_output=True, text=True, timeout=600, env=env,
    )
    if result.returncode != 0:
        sys.exit(f"claude 실행 실패 (exit {result.returncode}):\n{result.stderr[:1000]}")

    payload = extract_json(result.stdout)
    valid_tickers = set(targets["ticker"])
    items = [i for i in payload.get("items", []) if i.get("ticker") in valid_tickers and i.get("comment")]
    if not items:
        sys.exit(f"설명문 파싱 실패:\n{result.stdout[:500]}")

    out = {
        "date": date,
        "market_state": market_state,
        "market_comment": payload.get("market_comment", ""),
        "items": items,  # {ticker, comment} — 판정은 candidates CSV의 verdict 컬럼이 원본
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"저장: {out_path} (설명 {len(items)}개)")
    verdicts = df.set_index("ticker")["verdict"]
    for item in items:
        print(f"  [{verdicts.get(item['ticker'], '?')}] {item['ticker']}: {item['comment']}")


if __name__ == "__main__":
    main()
