# Trading

미국 주식의 상대강도(RS), 거래량 품질, 가격 추세, 고점 근접도를 이용해 주도주 후보군을 추리는 개인용 스크리너입니다.

## 실행

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py
```

기본 실행은 `data/tickers_top100.csv`의 주요 종목 약 100개를 대상으로 최근 2년 일봉을 내려받아 계산합니다.

결과 파일:

```text
outputs/screener_universe_YYYY-MM-DD.csv
outputs/screener_candidates_YYYY-MM-DD.csv
```

## 주요 옵션

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py --top 50
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py --period 1y
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py --min-dollar-volume 50000000
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py --tickers data/tickers_top100.csv
```

## 대시보드

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python dashboard.py
```

브라우저에서 아래 주소를 엽니다.

```text
http://127.0.0.1:8765
```

대시보드에서는 기간, 최소 거래대금, 출력 행 수를 지정하고 `Run Screener` 버튼으로 스크리너를 실행할 수 있습니다.

## 전략 문서

자세한 로직과 구현 계획은 `US_RS_VOLUME_SCREENER_PLAN.md`에 정리되어 있습니다.
