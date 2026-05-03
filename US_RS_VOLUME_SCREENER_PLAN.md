# 미국 주식 RS + 거래량 주도주 스크리너 설계

## 1. 목적

이 문서는 미국 주식에서 상대강도(Relative Strength)와 거래량을 중심으로 주도주 후보군을 필터링하는 개인용 스크리너의 로직과 구현 계획을 정리한다.

이 스크리너는 자동매매 시스템이 아니다. 직접 진입, 청산, 포지션 크기를 결정하지 않고, 매일 또는 주기적으로 관찰할 만한 종목군을 좁히는 도구로 사용한다.

## 2. 핵심 아이디어

강한 종목은 보통 시장보다 먼저 올라가고, 조정 시 덜 빠지며, 중요한 구간에서 거래량이 동반된다. 따라서 스크리너는 다음 조건을 동시에 확인한다.

- 시장 대비 상대강도가 개선되는가
- 상대강도 라인이 최근 고점 근처인가
- 가격 추세가 살아 있는가
- 가격이 최근 고점 근처에서 버티는가
- 거래량이 증가했고 종가 위치가 양호한가
- 단기 과열이나 비정상 이벤트성 움직임은 아닌가

## 3. 데이터 소스

초기 구현은 `yfinance`만 사용한다.

필요 데이터:

- 미국 개별 종목 일봉 `Open`, `High`, `Low`, `Close`, `Volume`
- 조정 가격 기준 일봉. `yfinance.download(..., auto_adjust=True)` 사용
- 벤치마크 ETF: `SPY`
- 선택 벤치마크 ETF: `QQQ`
- 섹터 ETF: `XLK`, `XLY`, `XLF`, `XLV`, `XLI`, `XLE`, `XLP`, `XLU`, `XLB`, `XLRE`, `XLC`

초기 종목 유니버스는 단순하게 사용자가 제공한 티커 리스트 또는 수동 CSV로 시작한다. 이후 Nasdaq Trader Symbol Directory 기반으로 자동 유니버스 생성을 추가한다.

## 4. 벤치마크 기준

상대강도는 상장 거래소 기준이 아니라 비교 목적 기준으로 계산한다.

기본 기준:

```text
RS_SPY = Stock_Adjusted_Close / SPY_Adjusted_Close
```

보조 기준:

```text
RS_QQQ = Stock_Adjusted_Close / QQQ_Adjusted_Close
RS_Sector = Stock_Adjusted_Close / Sector_ETF_Adjusted_Close
```

운용 원칙:

- 전체 미국 주식 스크리너: `SPY` 기준을 핵심으로 사용
- 기술주/성장주 성격 확인: `QQQ` 대비 RS를 보조로 사용
- 같은 섹터 내 진짜 주도주 확인: 섹터 ETF 대비 RS를 보조로 사용

## 5. 주요 지표

### 5.1 상대강도

```text
RS_20D = RS_today / RS_20days_ago - 1
RS_50D = RS_today / RS_50days_ago - 1
RS_Near_High = RS_today >= Max(RS, 50) * 0.98
```

`RS_Near_High`는 정확한 신고가만 보지 않고 최근 50일 RS 고점의 98% 이상이면 통과시킨다. 좋은 종목이 하루 차이로 제외되는 문제를 줄이기 위함이다.

### 5.2 가격 추세

```text
MA20 = 20일 이동평균
MA50 = 50일 이동평균

Trend_OK =
    Close > MA50
    and MA50 > MA50_10days_ago
    and Close > MA20
```

최소 버전에서는 아래 두 조건만 사용해도 된다.

```text
Close > MA50
MA50 rising
```

### 5.3 고점 근접도

```text
High_Proximity_50D = Close / High_50D
Near_50D_High = Close >= High_50D * 0.90
```

주도주 후보는 단순히 저점에서 반등하는 종목보다 최근 고점 근처에서 버티는 종목을 우선한다.

선택 지표:

```text
High_Proximity_252D = Close / High_252D
Near_52W_High = Close >= High_252D * 0.85
```

### 5.4 거래량 품질

단순 거래량 폭증보다 가격 반응을 함께 본다.

```text
Avg_Volume_20D = Average(Volume, 20)
Volume_Ratio = Volume_today / Avg_Volume_20D
Capped_Volume_Ratio = Min(Volume_Ratio, 5)

Close_Position = (Close - Low) / (High - Low)
```

거래량 품질 조건:

```text
Volume_Quality =
    Volume_Ratio >= 1.3
    and Close > Previous_Close
    and Close_Position >= 0.6
```

`High == Low`인 날은 `Close_Position`을 결측 또는 0.5로 처리한다.

### 5.5 유동성

저유동성 종목의 왜곡을 줄이기 위해 거래대금 필터를 사용한다.

```text
Avg_Dollar_Volume_20D = Average(Close * Volume, 20)
Liquidity_OK = Avg_Dollar_Volume_20D >= Min_Dollar_Volume
```

초기 기본값:

```text
Min_Dollar_Volume = 20,000,000
```

필요하면 5천만 달러 또는 1억 달러로 높인다.

### 5.6 과열 및 리스크 필터

```text
ATR_14 = Average(True Range, 14)
ATR_Ratio = ATR_14 / Close
```

과열/함정 제거 조건:

```text
Not_Overheated =
    Close <= MA20 * 1.25
    and Return_5D < 0.40
    and Daily_Return < 0.25
```

선택적으로 RSI를 과열 필터로만 사용한다.

```text
RSI_14 >= 50
RSI_14 <= 85
```

RSI는 메인 점수에 넣지 않는다. RS, 이동평균, 고점 근접도와 정보가 중복되기 때문이다.

## 6. 필터 구조

하드 필터:

```text
Liquidity_OK
RS_Near_High
RS_20D > 0
Close > MA50
MA50 rising
Close >= High_50D * 0.90
Not_Overheated
```

거래량 품질은 하드 필터로 쓸 수도 있지만, 초기에는 등급 분류에 사용한다. 좋은 주도주는 매일 거래량이 폭증하지 않을 수 있기 때문이다.

## 7. 점수화

곱셈 점수는 피한다.

기존 방식:

```text
Score = RS_Slope * Volume_Ratio
```

이 방식은 비정상적인 거래량 폭증 종목이 점수를 독식할 수 있다.

권장 방식:

```text
Score =
    0.35 * Rank(RS_SPY_20D)
  + 0.25 * Rank(RS_SPY_50D)
  + 0.20 * Rank(Close / High_50D)
  + 0.15 * Rank(Min(Volume_Ratio, 5))
  + 0.05 * Rank(Close_Position)
```

섹터 ETF 매핑을 구현한 뒤에는 아래 보조 점수를 추가할 수 있다.

```text
Score =
    0.40 * Rank(RS_SPY_20D)
  + 0.25 * Rank(RS_SPY_50D)
  + 0.20 * Rank(RS_Sector_20D)
  + 0.10 * Rank(Close / High_50D)
  + 0.05 * Rank(Min(Volume_Ratio, 5))
```

랭킹은 전체 후보군 내 percentile rank로 계산한다.

## 8. 등급 분류

최종 결과는 Top 10만 출력하지 않고 등급을 함께 제공한다.

```text
A급:
    하드 필터 통과
    and Volume_Quality == True

B급:
    하드 필터 통과
    and Volume_Quality == False

C급:
    RS_20D > 0
    and Close > MA50
    but RS_Near_High 또는 Near_50D_High 미충족
```

초기 구현에서는 A/B 등급만 만들어도 충분하다.

## 9. 출력 컬럼

```text
Ticker
Name
Sector
Close
Daily_Return
Return_5D
RS_SPY_20D
RS_SPY_50D
RS_Near_High
RS_QQQ_20D
RS_Sector_20D
Close_to_50D_High
Volume_Ratio
Capped_Volume_Ratio
Avg_Dollar_Volume_20D
Close_Position
MA20
MA50
ATR_Ratio
RSI_14
Score
Grade
Market_State
```

초기 버전에서 `Name`, `Sector`, `RS_Sector_20D`는 비워두거나 생략 가능하다.

## 10. 시장 국면 태그

시장 국면은 종목을 완전히 제거하는 필터가 아니라 결과 해석용 태그로 사용한다.

```text
Bullish:
    SPY_Close > SPY_MA50
    and SPY_MA50 > SPY_MA50_10days_ago

Neutral:
    SPY_Close near SPY_MA50

Weak:
    SPY_Close < SPY_MA50
```

약세장에서도 강한 종목은 의미가 있으므로, 시장 국면은 후보 제거보다 리스크 해석에 사용한다.

## 11. 구현 계획

### Phase 1. 최소 실행 버전

목표: 수동 티커 리스트를 입력받아 RS + 거래량 스크리너 결과 CSV를 생성한다.

작업:

1. `yfinance`, `pandas`, `numpy` 의존성 추가
2. 수동 티커 리스트 로딩
3. `SPY`, `QQQ`, 개별 종목 일봉 다운로드
4. 조정 OHLCV 기준으로 지표 계산
5. 하드 필터 적용
6. 점수 및 A/B 등급 계산
7. 결과를 `outputs/screener_YYYY-MM-DD.csv`로 저장

검증:

- `AAPL`, `MSFT`, `NVDA`, `META`, `AMD`, `TSLA`, `PLTR`, `AVGO`, `LLY`, `JPM`, `XOM` 정도의 샘플로 실행
- 결측치가 많은 종목 제외 확인
- `SPY` 기준 RS 계산이 정상인지 수동 검산

### Phase 2. 유니버스 확장

목표: 미국 주식 유니버스를 자동 또는 반자동으로 구성한다.

작업:

1. Nasdaq Trader `nasdaqlisted.txt`, `otherlisted.txt` 다운로드 지원
2. 테스트 종목, ETF, 우선주, 워런트, 유닛, 권리, 노트 제거
3. 최소 가격 필터 추가
4. 최소 거래대금 필터 적용
5. 캐시 파일 저장

기본 제외 키워드:

```text
Warrant
Right
Unit
Preferred
Depositary
Notes
```

### Phase 3. 섹터 대비 RS 추가

목표: 시장 대비 강도뿐 아니라 같은 섹터 내 상대강도를 계산한다.

작업:

1. 티커별 섹터 매핑 파일 추가
2. 섹터별 ETF 매핑 추가
3. `RS_Sector_20D`, `RS_Sector_50D` 계산
4. 점수 산식에 섹터 RS 반영

초기 섹터 ETF:

```text
Technology: XLK
Communication Services: XLC
Consumer Discretionary: XLY
Consumer Staples: XLP
Financials: XLF
Health Care: XLV
Industrials: XLI
Energy: XLE
Materials: XLB
Real Estate: XLRE
Utilities: XLU
Semiconductors: SMH 또는 SOXX
```

### Phase 4. 리포트 개선

목표: 매일 확인하기 쉬운 형태로 결과를 만든다.

작업:

1. CSV 외에 Markdown 리포트 생성
2. A급/B급 후보 분리
3. 상위 후보의 주요 지표 요약
4. 이전 실행 결과와 비교해 신규 진입/탈락 표시
5. 선택적으로 간단한 차트 이미지 생성

### Phase 5. 안정화

목표: 개인용 도구로 반복 실행 가능한 상태를 만든다.

작업:

1. 다운로드 실패 재시도
2. 로컬 캐싱
3. 결측치 및 이상치 처리
4. 설정 파일 분리
5. 기본 테스트 추가

## 12. 초기 기본 설정

```text
Lookback_Period = 252
RS_High_Window = 50
RS_Short_Window = 20
RS_Mid_Window = 50
MA_Short = 20
MA_Mid = 50
Volume_Window = 20
ATR_Window = 14
RSI_Window = 14
Min_Dollar_Volume = 20,000,000
RS_Near_High_Threshold = 0.98
Near_50D_High_Threshold = 0.90
Volume_Ratio_Min = 1.3
Volume_Ratio_Cap = 5.0
Close_Position_Min = 0.6
Max_Close_to_MA20 = 1.25
Max_Return_5D = 0.40
Max_Daily_Return = 0.25
```

## 13. 초기 의사코드

```text
Load tickers
Download OHLCV for tickers + SPY + QQQ

For each stock:
    If insufficient data: skip

    Calculate liquidity
    If liquidity below threshold: skip

    Calculate RS_SPY
    Calculate RS_20D, RS_50D, RS_Near_High

    Calculate MA20, MA50, MA50 slope
    Calculate High_50D proximity
    Calculate Volume_Ratio and Close_Position
    Calculate ATR_Ratio and optional RSI

    Apply hard filters
    If pass:
        Add to candidate list

Rank candidate metrics
Calculate Score
Assign Grade A/B
Sort by Score descending
Save output
```

## 14. 주의사항

- `yfinance`는 공식 거래소 데이터 API가 아니므로 개인용 스크리너 수준에서 사용한다.
- 실시간 자동매매나 상업용 서비스에는 적합하지 않다.
- 무료 데이터 기반 백테스트는 생존자 편향, 티커 변경, 상장폐지 종목 누락 문제가 있다.
- 스크리너 결과는 매수 신호가 아니라 관찰 후보군이다.
- 후보 종목은 차트, 뉴스, 실적 일정, 섹터 흐름을 별도로 확인한다.
