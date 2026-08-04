# US Stock Screener

미국 주식의 상대강도(RS)와 거래량 이벤트를 기반으로 매일 관찰할 종목군을 좁히는 개인용 스크리너입니다. 자동매매 시스템이 아니라 관심 종목 후보를 추리는 도구입니다.

설계 프레임은 **"거래량 이벤트로 주목 → RS로 우선순위"** 입니다. 이동평균 추세나 과열 여부는 배제 조건이 아니라 등급·경고 라벨로만 표시합니다. 근거는 [백테스트](#백테스트) 섹션 참조.

## 유니버스

시가총액 기준 미국 상위 ~1000개 보통주 (`data/tickers_us1000.csv`).
yfinance screen(NMS/NYQ 상장, 시총 내림차순)으로 수집하고 Wikipedia S&P 500/400 섹터 정보를 매핑합니다.

유니버스 갱신 (월 1회 권장):

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/fetch_universe.py
```

## 스크리너 실행

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py
```

당일 캐시가 없으면 약 2분, 캐시 적중 시 약 4초 소요됩니다.
결과는 아래 경로에 저장됩니다.

```
outputs/screener_universe_YYYY-MM-DD.csv    # 전체 평가 결과 (점수 포함)
outputs/screener_candidates_YYYY-MM-DD.csv  # 필터 통과 종목만
```

## 필터 로직

### 하드 필터 (3층)

| 순서 | 층 | 조건 |
|------|------|------|
| 1 | 유동성 | 20일 평균 거래대금 ≥ $50M |
| 2 | RS 필수 | 20일 SPY 대비 RS 변화율 > 0 |
| 3 | 주목 트리거 | 당일 서지 **또는** 지속 매집 (아래) |

**당일 서지(surge)** — 거래량이 20일 평균의 1.5배 이상 5배 미만 + 당일 양봉
**지속 매집(acc)** — 최근 10일 중 매집일(가격↑ + 거래량 > 20일 평균)이 6일 이상

### signal_type

| 값 | 의미 |
|------|------|
| `surge+acc` | 서지·매집 동시 충족 — 백테스트상 가장 강한 신호 |
| `surge` | 당일 서지만 |
| `acc` | 지속 매집만 |

### 등급 (추세 컨텍스트)

**A** — 종가 > MA60 **AND** MA60이 10일 전보다 상승 (추세 순응)
**B** — 그 외 (하락·횡보 구간에서의 반등 성격)

MA 조건은 배제가 아니라 성격 구분입니다. 하락 국면에서는 B가 A보다 나은 구간도 있습니다(백테스트 참조).

### 경고 라벨 (배제 아님, `warnings` 컬럼)

| 라벨 | 조건 |
|------|------|
| `과열` | 종가 > MA20×1.25, 또는 5일 수익률 ≥ 40%, 20일 수익률 ≥ 60%, 당일 수익률 ≥ 25% |
| `거래량5x+` | 거래량 20일 평균 5배 이상 + 양봉 — 어닝스/뉴스 갭 의심 (트리거에서도 제외됨) |
| `분산N일` | 최근 10일 중 분산일(가격↓ + 거래량 > 20일 평균) 3일 이상 |

### 점수 (v2, percentile 가중합)

전 종목에 부여되어 필터 통과 여부와 무관하게 순위를 볼 수 있습니다.

| 가중치 | 피처 |
|------|------|
| 30% | SPY 대비 RS 20D |
| 25% | SPY 대비 RS 50D |
| 25% | 섹터 ETF 대비 RS 20D |
| 10% | 10일 매집일 수 |
| 10% | 거래량 추세 (5일 평균 / 20일 평균) |

2026-08 IC 분석에서 RS 계열만 유의미한 예측력을 보였고, 당일 봉 형태(`close_position`)·베이스 안정성·MA 정배열·50일 고점 근접도는 IC가 0 근처거나 음수여서 점수에서 제외했습니다.

### 매수기준가

종가가 MA20 위면 MA20×1.01, 아니면 MA60×1.01 (`buy_price`, `buy_price_basis` 컬럼). 눌림목 진입 기준선입니다.

### 시장 국면

`Confirmed Uptrend` / `Uptrend Under Pressure` / `Market in Correction` / `Unknown` 4단계.
SPY+QQQ의 MA60 위치와 기울기로 판단하며, 후보를 제거하지 않고 결과 태그로만 사용합니다.

## 대시보드

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python dashboard.py
```

브라우저에서 `http://127.0.0.1:8765` 를 엽니다. 저장된 최신 결과를 자동 로드합니다.

- **워치리스트** — 후보를 실행 관점 3그룹으로 분류: `매수가 근처`(기준가 +3% 이내) / `확장 상태`(눌림 대기) / `후순위`(점수 0.5 미만 또는 분산 5일 이상). 그룹 내 정렬은 점수 기준.
- **필터 퍼널** — Universe → Evaluated → Liquidity → RS 20D > 0 → Volume Trigger → A Grade 단계별 잔존/탈락 수.
- **전일 대비 변화** — 최근 2개 candidates CSV를 비교해 신규 진입/이탈/등급 승격·강등 표시.
- **필터·정렬** — 등급(A/B), 섹터, 시가총액 구간 필터 + 컬럼 클릭 정렬.
- **차트 모달** — 티커 클릭 시 가격/MA/거래량 차트.
- **티커 검색** — 특정 종목의 필터 통과·탈락 사유를 항목별로 확인.
- **Run** — 브라우저에서 스크리너 직접 실행 (`POST /api/run`).

API: `GET /api/latest`, `/api/diff`, `/api/ticker?symbol=`, `/api/chart?symbol=&days=`, `/api/universe`, `POST /api/run`, `GET /outputs/*`

## 백테스트

필터나 점수를 변경할 때는 **반드시** 전/후 성과를 비교합니다 (~3분).

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/backtest.py
# 옵션: --period 2y --step 5 --horizons 5 10 20 60 --min-dollar-volume 50000000
```

`compute_feature_frame()`이 전 기간 피처를 계산하므로, 과거 스냅샷 날짜의 행만 꺼내 `evaluate_row()`로 판정하면 그 시점의 스크리너 결과가 재현됩니다. 모든 피처가 backward-looking rolling이라 look-ahead가 없습니다.

결과는 `outputs/backtest/` 에 저장됩니다.

```
rows_<시작>_<종료>_step<N>.csv   # 스냅샷 원본 (판정 + 포워드 수익률)
summary_grade.csv                # 등급별 SPY 대비 초과수익
summary_decile.csv               # 점수 십분위별 초과수익
summary_ic.csv                   # 피처별 IC (Spearman)
summary_regime.csv               # 시장 국면별 등급 성과
```

### 현재 로직 성과 (2025-02 ~ 2026-05, 63개 주간 스냅샷)

20일 SPY 대비 초과수익 평균 / 승률:

| 구분 | 20d 초과수익 | 승률 | 표본 |
|------|------|------|------|
| `surge+acc` | **+4.00%** | 59.8% | 194 |
| `surge` | +2.14% | 52.7% | 1,165 |
| `acc` | +1.26% | 52.7% | 1,073 |
| 후보 전체 (A) | +1.90% | 52.5% | 1,804 |
| 후보 전체 (B) | +1.90% | 55.3% | 628 |
| 비후보 | +0.35% | 47.6% | 56,255 |

시장 국면별 (20d): `Confirmed Uptrend`에서 A가 +2.51%로 우수하고, `Market in Correction`에서는 A가 +0.50%로 약해지는 반면 B가 +2.67%를 기록합니다.

### 한계 (해석 시 유의)

- 유니버스가 "현재" 시총 상위 1000종목이라 **생존 편향**이 있습니다. 절대 수익률 추정이 아니라 필터/점수 변형 간 **상대 비교** 용도로만 사용합니다.
- `auto_adjust` 가격은 다운로드 시점 기준 조정 — 순위 비교에는 영향이 미미합니다.

## 자동화

`launchd/` 에 실행 래퍼 스크립트가 있습니다 (macOS launchd 또는 cron에 연결).

```
launchd/run_screener.sh          # main.py 실행 → launchd/screener.log
launchd/run_fetch_universe.sh    # 유니버스 갱신 → launchd/fetch_universe.log
```

crontab으로 쓸 경우 (한국 시간 기준 평일 새벽 6시, 미국 장 마감 후):

```
0 6 * * 1-5  /Users/jhpark/Documents/Claude\ Code/US-stock-screener/launchd/run_screener.sh
```

대시보드의 "마지막 실행" 표시는 `launchd/screener.log` 를 읽습니다.

## 캐시

당일 날짜 기준 pickle 캐시 (`.cache/yfinance/prices/<period>/<날짜>/<SYMBOL>.pkl`). 날짜가 바뀌면 자동 재다운로드하며, 강제 무효화가 필요하면 `.cache/yfinance/prices/` 를 삭제합니다.

## 기술 스택

- Python 3.12+, [uv](https://github.com/astral-sh/uv) 패키지 관리 (`pip install` 대신 `uv add`)
- yfinance (가격 데이터), pandas, numpy
- 대시보드: 표준 라이브러리 `ThreadingHTTPServer` — HTML/CSS/JS 전체가 `dashboard.py`의 `INDEX_HTML` 문자열에 인라인 (외부 프레임워크·템플릿 파일 없음)
