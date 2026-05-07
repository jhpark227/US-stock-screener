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

| 순서 | 필터 | 조건 |
|------|------|------|
| 1 | 유동성 | 20일 평균 거래대금 ≥ $50M |
| 2 | RS 고점 근접 | SPY 대비 RS가 최근 50일 RS 최고값의 98% 이상 |
| 3 | RS 20D > 0 | 20일 SPY 대비 RS 변화율 양수 |
| 4 | 섹터 RS > 0 | 20일 섹터 ETF 대비 RS 변화율 양수 |
| 5 | MA50 위 + 상승 | 종가 > MA50, MA50이 10일 전보다 높음 |
| 6 | 50일 고점 근접 | 종가 ≥ 최근 50일 종가 최고값의 90% |
| 7 | 과열 없음 | 종가 ≤ MA20×1.25, 5일 수익률 < 40%, 20일 수익률 < 60%, 당일 수익률 < 25% |

**등급 A** — 위 조건 통과 + 거래량비율 ≥ 1.3배, 양봉, 종가위치 ≥ 60%  
**등급 B** — 위 조건 통과, 거래량 품질 미충족

**점수** — SPY RS 20D(25%) + SPY RS 50D(20%) + 섹터 RS 20D(15%) + 50일 고점 근접도(20%) + 거래량비율(15%) + 종가위치(5%) percentile 가중합

## 기술 스택

- Python 3.12+, [uv](https://github.com/astral-sh/uv) 패키지 관리
- yfinance (가격 데이터), pandas, numpy
- 대시보드: 표준 라이브러리 HTTP 서버 (외부 프레임워크 없음)
