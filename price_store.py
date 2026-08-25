"""증분 parquet 가격 저장소.

구조:
    .cache/price_store/daily/<SYMBOL>.parquet   심볼당 OHLCV 이력 1개 (최대 MAX_ROWS 거래일)
    .cache/price_store/daily/_meta.json         {symbol: {"checked": 마지막 동기화 성공일, "last_date": 마지막 데이터일}}

동작:
- checked == 오늘이면 네트워크 없이 parquet만 읽는다 (전 종목 캐시 적중 시 from_cache=True).
- 아니면 마지막 저장일 - OVERLAP_DAYS 부터 증분 다운로드해 겹침 구간 Close를 대조한다.
  겹침이 허용 오차(OVERLAP_RTOL)를 벗어나면 배당/분할로 수정주가 전체가 재계산된 것이므로
  해당 종목만 자동 전체 리프레시한다 → 주기적 전체 재다운로드 잡이 필요 없다.
- 다운로드는 배치 단위 지수 백오프 재시도 후, 실패 배치를 종목 단위로 쪼개 한 번 더 시도한다.
- 그래도 실패한 종목은 저장된 (하루 이틀 뒤처진) 이력을 그대로 반환한다 — 종목 누락보다 낫다.
  저장분조차 없는 종목만 결과에서 빠진다.
- 구 pickle 캐시(.cache/yfinance/prices/)가 있으면 최초 1회 parquet로 이관해 전체 다운로드를 피하고,
  요청 심볼 전부가 저장소에 들어온 뒤 구 캐시를 삭제한다.
"""

from __future__ import annotations

import json
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_STORE_DIR = Path(".cache/price_store/daily")
LEGACY_PRICES_DIR = Path(".cache/yfinance/prices")

MAX_ROWS = 800                 # 종목당 보관 거래일 (~3.2년) — 저장소 무한 성장 방지
OVERLAP_DAYS = 7               # 증분 다운로드 시 검증용 겹침 (캘린더일)
OVERLAP_RTOL = 1e-3            # 겹침 Close 상대 오차 허용치 — 초과 시 수정주가 변경으로 판단
MAX_INCREMENTAL_LAG_DAYS = 45  # 이보다 오래 안 받은 종목은 증분 대신 전체 다운로드
RETRY_DELAYS = (0.0, 3.0, 10.0)  # 배치 재시도 간격(초) — 앞에 지터 추가
COVERAGE_WARN_RATIO = 0.95     # 신선 데이터 확보율이 이 미만이면 경고


@dataclass
class SyncReport:
    total: int = 0
    fresh: int = 0          # 오늘 이미 동기화됨 (네트워크 생략)
    incremental: int = 0    # 증분 append 성공
    full: int = 0           # 전체 다운로드 성공 (신규/수정주가 변경/장기 미갱신)
    adjusted: list[str] = field(default_factory=list)   # 수정주가 변경 감지 → 전체 리프레시된 종목
    stale: list[str] = field(default_factory=list)      # 다운로드 실패 → 저장분(구버전)으로 대체
    failed: list[str] = field(default_factory=list)     # 다운로드 실패 + 저장분도 없음 → 결과 누락
    migrated: int = 0       # 구 pickle 캐시에서 이관된 종목 수

    @property
    def from_cache(self) -> bool:
        return self.fresh == self.total

    def summary(self) -> str:
        parts = [f"신선 {self.fresh}", f"증분 {self.incremental}", f"전체 {self.full}"]
        if self.migrated:
            parts.append(f"이관 {self.migrated}")
        if self.adjusted:
            parts.append(f"수정주가 리프레시 {len(self.adjusted)}")
        if self.stale:
            parts.append(f"실패→저장분 사용 {len(self.stale)}")
        if self.failed:
            parts.append(f"누락 {len(self.failed)}")
        return f"{self.total}개 심볼: " + ", ".join(parts)


def extract_symbol_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        frame = None
        for level in range(raw.columns.nlevels):
            matched_value = None
            for value in raw.columns.get_level_values(level).unique():
                if str(value).upper() == symbol.upper():
                    matched_value = value
                    break
            if matched_value is not None:
                frame = raw.xs(matched_value, axis=1, level=level, drop_level=True)
                break
        if frame is None:
            return pd.DataFrame()
    else:
        frame = raw.copy()

    frame = frame.rename(columns={column: str(column).title() for column in frame.columns})
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    if any(column not in frame.columns for column in required_columns):
        return pd.DataFrame()

    frame = frame.loc[:, required_columns].copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.sort_index()
    frame = frame.dropna(subset=["Close"])
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def _symbol_path(store_dir: Path, symbol: str) -> Path:
    # ^VIX 같은 특수문자 심볼도 macOS 파일명으로는 유효하지만, 이식성을 위해 ^만 치환
    return store_dir / f"{symbol.replace('^', '_CARET_')}.parquet"


def _meta_path(store_dir: Path) -> Path:
    return store_dir / "_meta.json"


def _load_meta(store_dir: Path) -> dict[str, dict[str, str]]:
    path = _meta_path(store_dir)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_meta(store_dir: Path, meta: dict[str, dict[str, str]]) -> None:
    tmp = _meta_path(store_dir).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta))
    tmp.replace(_meta_path(store_dir))


def read_symbol(symbol: str, store_dir: Path = DEFAULT_STORE_DIR) -> pd.DataFrame | None:
    """네트워크 없이 저장분만 읽는다 (대시보드 차트용)."""
    path = _symbol_path(store_dir, symbol)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _write_symbol(store_dir: Path, symbol: str, frame: pd.DataFrame) -> None:
    frame = frame.tail(MAX_ROWS)
    tmp = _symbol_path(store_dir, symbol).with_suffix(".parquet.tmp")
    frame.to_parquet(tmp)
    tmp.replace(_symbol_path(store_dir, symbol))


def _migrate_legacy(symbols: list[str], store_dir: Path, meta: dict) -> int:
    """구 날짜별 pickle 캐시에서 parquet 저장소로 1회 이관 (최신 날짜 우선)."""
    if not LEGACY_PRICES_DIR.exists():
        return 0
    dated_dirs: list[tuple[str, Path]] = []
    for period_dir in LEGACY_PRICES_DIR.iterdir():
        if period_dir.is_dir():
            for dated in period_dir.iterdir():
                if dated.is_dir():
                    dated_dirs.append((dated.name, dated))
    dated_dirs.sort(reverse=True)

    migrated = 0
    wanted = [s for s in symbols if not _symbol_path(store_dir, s).exists()]
    for symbol in wanted:
        for _, dated in dated_dirs:
            pkl = dated / f"{symbol}.pkl"
            if not pkl.exists():
                continue
            try:
                with pkl.open("rb") as f:
                    frame = pickle.load(f)
            except Exception:
                continue
            if frame is None or frame.empty or "Close" not in frame.columns:
                continue
            frame.index = pd.to_datetime(frame.index).tz_localize(None)
            frame = frame.sort_index()
            frame = frame[~frame.index.duplicated(keep="last")]
            _write_symbol(store_dir, symbol, frame)
            meta[symbol] = {"checked": "", "last_date": str(frame.index[-1].date())}
            migrated += 1
            break
    return migrated


def _cleanup_legacy(symbols: list[str], store_dir: Path) -> None:
    """저장소로 이관 완료된 심볼의 구 pickle만 삭제한다 (다른 심볼의 이관 소스는 보존)."""
    if not LEGACY_PRICES_DIR.exists():
        return
    in_store = {s for s in symbols if _symbol_path(store_dir, s).exists()}
    removed = 0
    for pkl in LEGACY_PRICES_DIR.glob("*/*/*.pkl"):
        if pkl.stem in in_store:
            pkl.unlink(missing_ok=True)
            removed += 1
    # 빈 날짜/기간 디렉터리 정리 (안쪽부터)
    for directory in sorted(LEGACY_PRICES_DIR.glob("*/*"), reverse=True) + sorted(
        LEGACY_PRICES_DIR.glob("*"), reverse=True
    ) + [LEGACY_PRICES_DIR]:
        try:
            directory.rmdir()
        except OSError:
            pass
    if removed:
        print(f"  구 pickle 캐시 정리: {removed}개 파일 삭제")


def _download_with_retry(symbols: list[str], **kwargs) -> pd.DataFrame | None:
    for attempt, delay in enumerate(RETRY_DELAYS):
        if delay:
            time.sleep(delay + random.uniform(0, 1.5))
        try:
            raw = yf.download(
                symbols,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=False,
                progress=False,
                timeout=30,
                **kwargs,
            )
        except Exception:
            continue
        if raw is not None and not raw.empty:
            return raw
    return None


def _merge_incremental(
    stored: pd.DataFrame, new: pd.DataFrame
) -> tuple[pd.DataFrame | None, bool]:
    """(병합 결과, 수정주가 변경 여부). 병합 결과 None이면 전체 리프레시 필요."""
    if new.empty:
        return stored, False  # 신규 데이터 없음 (휴장 등) — 저장분 유지
    common = stored.index.intersection(new.index)
    if len(common) == 0:
        return None, False  # 겹침 없음 — 연속성 검증 불가, 전체 리프레시
    old_close = stored.loc[common, "Close"].astype(float)
    new_close = new.loc[common, "Close"].astype(float)
    rel = ((new_close - old_close).abs() / old_close.abs().clip(lower=1e-9)).max()
    if rel > OVERLAP_RTOL:
        return None, True  # 배당/분할로 과거 수정주가 재계산됨 — 전체 리프레시
    merged = pd.concat([stored.loc[stored.index < new.index.min()], new])
    return merged, False


def sync_prices(
    symbols: list[str],
    period: str = "2y",
    store_dir: Path = DEFAULT_STORE_DIR,
    force_full: bool = False,
    batch_size: int = 50,
    workers: int = 5,
) -> tuple[dict[str, pd.DataFrame], SyncReport]:
    """심볼 목록을 최신화하고 {symbol: OHLCV DataFrame}을 반환한다."""
    store_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(store_dir.parent.resolve()))
    today = date.today().isoformat()
    meta = _load_meta(store_dir)
    report = SyncReport(total=len(symbols))

    report.migrated = _migrate_legacy(symbols, store_dir, meta)

    prices: dict[str, pd.DataFrame] = {}
    need_full: list[str] = []
    need_incr: list[tuple[str, date]] = []  # (symbol, last_date)

    for symbol in symbols:
        stored = read_symbol(symbol, store_dir)
        if stored is None or stored.empty or force_full:
            need_full.append(symbol)
            continue
        info = meta.get(symbol, {})
        last_date = stored.index[-1].date()
        if info.get("checked") == today:
            prices[symbol] = stored
            report.fresh += 1
        elif (date.today() - last_date).days > MAX_INCREMENTAL_LAG_DAYS:
            need_full.append(symbol)
        else:
            need_incr.append((symbol, last_date))

    if not need_full and not need_incr:
        return prices, report

    failed_full: list[str] = []
    failed_incr: list[str] = []

    def _run_full(batch: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
        raw = _download_with_retry(batch, period=period)
        got: dict[str, pd.DataFrame] = {}
        misses: list[str] = []
        if raw is None:
            return got, list(batch)
        for symbol in batch:
            frame = extract_symbol_frame(raw, symbol)
            if frame.empty:
                misses.append(symbol)
            else:
                got[symbol] = frame
        return got, misses

    def _run_incr(batch: list[tuple[str, date]]) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
        """반환: (병합 완료 프레임, 전체 리프레시 필요 심볼, 실패 심볼)"""
        start = min(d for _, d in batch) - timedelta(days=OVERLAP_DAYS)
        raw = _download_with_retry([s for s, _ in batch], start=start.isoformat())
        got: dict[str, pd.DataFrame] = {}
        promote: list[str] = []
        misses: list[str] = []
        if raw is None:
            return got, promote, [s for s, _ in batch]
        for symbol, _ in batch:
            new = extract_symbol_frame(raw, symbol)
            stored = read_symbol(symbol, store_dir)
            if stored is None:
                promote.append(symbol)
                continue
            merged, adjusted = _merge_incremental(stored, new)
            if merged is None:
                if adjusted:
                    report.adjusted.append(symbol)
                promote.append(symbol)
            elif new.empty and merged is stored:
                misses.append(symbol)  # 응답에 이 심볼 없음 — 상장폐지/티커 변경 가능, 재시도로 확인
            else:
                got[symbol] = merged
        return got, promote, misses

    incr_batches = _chunked(sorted(need_incr, key=lambda t: t[1]), batch_size)
    n_total_batches = len(incr_batches)  # full 배치는 promote 이후 확정되므로 진행 표시는 단계별
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_incr, b): b for b in incr_batches}
        for future in as_completed(futures):
            got, promote, misses = future.result()
            for symbol, frame in got.items():
                _write_symbol(store_dir, symbol, frame)
                meta[symbol] = {"checked": today, "last_date": str(frame.index[-1].date())}
                prices[symbol] = frame.tail(MAX_ROWS)
                report.incremental += 1
            need_full.extend(promote)
            failed_incr.extend(misses)
            done += 1
            if n_total_batches:
                print(f"\r  증분 다운로드: {done}/{n_total_batches} 배치", end="", flush=True)
    if n_total_batches:
        print()

    full_batches = _chunked(sorted(set(need_full)), batch_size)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_full, b): b for b in full_batches}
        for future in as_completed(futures):
            got, misses = future.result()
            for symbol, frame in got.items():
                _write_symbol(store_dir, symbol, frame)
                meta[symbol] = {"checked": today, "last_date": str(frame.index[-1].date())}
                prices[symbol] = frame.tail(MAX_ROWS)
                report.full += 1
            failed_full.extend(misses)
            done += 1
            if full_batches:
                print(f"\r  전체 다운로드: {done}/{len(full_batches)} 배치", end="", flush=True)
    if full_batches:
        print()

    # 배치 실패분을 종목 단위로 한 번 더 (배치 내 불량 티커 1개가 나머지를 죽이는 경우 방어)
    failed_once = sorted(set(failed_full) | set(failed_incr))
    for symbol in failed_once:
        got, misses = _run_full([symbol])
        if symbol in got:
            frame = got[symbol]
            _write_symbol(store_dir, symbol, frame)
            meta[symbol] = {"checked": today, "last_date": str(frame.index[-1].date())}
            prices[symbol] = frame.tail(MAX_ROWS)
            report.full += 1
        else:
            stored = read_symbol(symbol, store_dir)
            if stored is not None and not stored.empty:
                prices[symbol] = stored  # 뒤처진 저장분으로 대체 — meta.checked는 갱신하지 않음
                report.stale.append(symbol)
            else:
                report.failed.append(symbol)

    _save_meta(store_dir, meta)
    _cleanup_legacy(symbols, store_dir)

    covered = report.fresh + report.incremental + report.full
    if report.total and covered / report.total < COVERAGE_WARN_RATIO:
        print(
            f"  ⚠ 신선 데이터 확보율 {covered}/{report.total}"
            f" ({covered / report.total:.0%}) — 네트워크/야후 상태 확인 필요"
        )
        if report.failed:
            print(f"  ⚠ 데이터 없음: {', '.join(report.failed[:20])}"
                  + (" ..." if len(report.failed) > 20 else ""))
    return prices, report


def _chunked(items: list, size: int) -> list[list]:
    return [items[index : index + size] for index in range(0, len(items), size)]
