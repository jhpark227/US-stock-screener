# US Stock Screener

미국 주식의 상대강도(RS), 거래량 품질, 가격 추세, 고점 근접도를 기반으로 주도주 후보군을 필터링하는 개인용 스크리너입니다. 자동매매 시스템이 아니라 매일 관찰할 종목군을 좁히는 도구입니다.

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
outputs/screener_universe_YYYY-MM-DD.csv    # 전체 평가 결과
outputs/screener_candidates_YYYY-MM-DD.csv  # 필터 통과 종목만
```

## 대시보드

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python dashboard.py
```

브라우저에서 `http://127.0.0.1:8765` 를 엽니다.

- 저장된 최신 결과를 자동으로 로드합니다.
- 우측 상단 **Refresh** 버튼으로 최신 결과를 다시 불러올 수 있습니다.
- Grade(A/B), 섹터 필터 및 컬럼 클릭 정렬을 지원합니다.
- 티커 검색으로 특정 종목의 필터 통과/탈락 사유를 확인할 수 있습니다.

## 자동화 (crontab)

매일 장 마감 후 자동 실행 예시 (한국 시간 기준 평일 새벽 6시):

```
0 6 * * 1-5  cd /path/to/US-stock-screener && UV_CACHE_DIR=/tmp/uv-cache uv run python main.py
```

## 필터 로직

하드 필터(절대 배제 조건):

| 순서 | 필터 | 조건 |
|------|------|------|
| 1 | 유동성 | 20일 평균 거래대금 ≥ $50M |
| 2 | RS 20D > 0 | 20일 SPY 대비 RS 변화율 양수 |
| 3 | MA60 위 + 상승 | 종가 > MA60, MA60이 10일 전보다 높음 |
| 4 | 추세 구조 | 정배열(MA20 > MA60 > MA120) **또는** 최근 5일 내 MA20→MA60 골든크로스(MA60 상승 중) |
| 5 | 과열 없음 | 종가 ≤ MA20×1.25, 5일 수익률 < 40%, 20일 수익률 < 60%, 당일 수익률 < 25% |

RS Near High / 50일 고점 근접도 / 섹터 RS 양수는 하드 필터에서 제외하고 스코어/태그로만 반영합니다.

**등급 A** — 하드 필터 통과 + 거래량 품질 충족  
**등급 B** — 하드 필터 통과, 거래량 품질 미충족

**거래량 품질** — 최근 10일 내 매집일(가격↑ + 거래량 > 20일 평균) ≥ 3일 AND 분산일(가격↓ + 거래량 > 20일 평균) ≤ 1일

**점수 (percentile 가중합)** — RS 20D(20%) + RS 50D(25%) + RS 가속도 20D−50D(15%) + 섹터 RS 20D(15%) + 50일 고점 근접도(15%) + 거래량비율(10%)

**매수기준가** — 종가가 MA20 위면 MA20×1.01, 아니면 MA60×1.01 (`buy_price`, `buy_price_basis` 컬럼)

**시장 국면** — `Confirmed Uptrend` / `Uptrend Under Pressure` / `Market in Correction` / `Unknown` 4단계. SPY+QQQ의 MA60 위치와 기울기로 판단하며 후보 제거가 아니라 결과 태그로만 사용.

## 기술 스택

- Python 3.12+, [uv](https://github.com/astral-sh/uv) 패키지 관리
- yfinance (가격 데이터), pandas, numpy
- 대시보드: 표준 라이브러리 HTTP 서버 (외부 프레임워크 없음)
