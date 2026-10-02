#!/usr/bin/env python3
"""
build_dataset.py
----------------
Builds and caches a multi-year historical OHLCV dataset used to train the
classical-ML next-candle classifier.

Primary source: Binance public klines REST API, pulled with *pagination*
(startTime walking) so deep history (years, not the last-1000-candles the
live app fetches) is available. Falls back to CoinGecko daily aggregation
when Binance is unreachable.

Raw candles are cached to parquet under ``data/raw/{SYMBOL}_{TF}.parquet``
so a retrain does not need to re-download everything.

Usage:
    python build_dataset.py                          # all coins, 1h/4h/1d, from 2020
    python build_dataset.py --coins BTC,ETH,SOL      # subset of coins
    python build_dataset.py --tfs 1d                 # daily only
    python build_dataset.py --start 2021-06-01       # shorter history
    python build_dataset.py --refresh                # top-up existing caches to "now"

No network calls happen at prediction time - the trained model + feature
functions are fully local. This script is the only network-dependent piece
of the ML pipeline (plus retrain.py which reuses it).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Reuse the CoinGecko symbol->id table that predict.py already maintains.
from predict import KNOWN_IDS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA",
           "BNB", "AVAX", "LINK", "DOT", "LTC"]
TIMEFRAMES = ["1h", "4h", "1d"]

TF_MS: dict[str, int] = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}

DEFAULT_START_MS = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

BINANCE_HOSTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision",   # public market-data host (geo-robust)
]
COINGECKO_API = "https://api.coingecko.com/api/v3"
USER_AGENT = {"User-Agent": "crypto-dataset-builder/1.0"}

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
BINANCE_KLINE_LIMIT = 1000
REQUEST_TIMEOUT = 15
RETRY_SLEEP = 2.0          # base seconds between HTTP retries
MIN_PAGE_SLEEP = 0.15      # polite pacing between pages
MAX_PAGES = 100_000        # runaway guard (~6 years of 1m is ~3.2M pages/... no)

CANDLE_COLUMNS = ["ts", "open", "high", "low", "close", "vol"]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class FetchError(RuntimeError):
    pass


def _http_get_json(url: str, retries: int = 4) -> list | dict:
    """GET url with retry only on transport errors and 408/429/5xx."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=USER_AGENT)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code not in (408, 429) and not (500 <= e.code < 600):
                raise FetchError(f"HTTP {e.code} for {url}") from e
            last_err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
        if attempt < retries - 1:
            time.sleep(RETRY_SLEEP * (attempt + 1))
    raise FetchError(f"GET failed after {retries} tries: {last_err}")


# ---------------------------------------------------------------------------
# Binance klines (paginated)
# ---------------------------------------------------------------------------

def _binance_klines_page(host: str, symbol: str, interval: str,
                         start_ms: int, end_ms: int) -> list[list]:
    url = (f"{host}/api/v3/klines?symbol={symbol}USDT&interval={interval}"
           f"&startTime={start_ms}&endTime={end_ms}&limit={BINANCE_KLINE_LIMIT}")
    data = _http_get_json(url)
    if not isinstance(data, list):
        raise FetchError(f"unexpected klines payload from {host}")
    return data


def _row_to_candle(row: list) -> dict:
    # Binance: [openTime, open, high, low, close, volume, closeTime, ...]
    return {
        "ts": int(row[0]),
        "open": float(row[1]), "high": float(row[2]), "low": float(row[3]),
        "close": float(row[4]), "vol": float(row[5]),
    }


def fetch_binance_history(symbol: str, tf: str, start_ms: int,
                          end_ms: int | None = None,
                          progress=None) -> list[dict]:
    """Paginated deep history for one symbol/timeframe via Binance klines.

    Returns chronological candle dicts (ts/open/high/low/close/vol),
    de-duplicated, spanning [start_ms, end_ms].
    """
    interval = tf
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    step = TF_MS[tf]
    out: list[dict] = []
    seen: set[int] = set()
    cursor = start_ms
    last_err: Exception | None = None

    for host in BINANCE_HOSTS:
        out, seen = [], set()
        cursor = start_ms
        last_err = None
        try:
            while cursor <= end_ms:
                page = _binance_klines_page(host, symbol, interval, cursor, end_ms)
                if not page:
                    break
                for row in page:
                    c = _row_to_candle(row)
                    if c["ts"] not in seen:
                        seen.add(c["ts"])
                        out.append(c)
                nxt = page[-1][0] + step
                if nxt <= cursor:          # no forward progress -> done
                    break
                cursor = nxt
                time.sleep(MIN_PAGE_SLEEP)
                if progress and len(out) % 5000 < 1000:
                    progress(symbol, tf, len(out))
            if out:
                return out
        except FetchError as e:
            last_err = e
            print(f"  [{host} failed for {symbol} {tf}: {e}]", file=sys.stderr)
            continue
    raise FetchError(f"No Binance data for {symbol} {tf}: {last_err}")


# ---------------------------------------------------------------------------
# CoinGecko fallback (daily granularity only - free API constraint)
# ---------------------------------------------------------------------------

def fetch_coingecko_daily(symbol: str, days: int = 365) -> list[dict]:
    """CoinGecko fallback: daily OHLCV aggregated from market_chart.

    The free CoinGecko API only serves ~a year of daily data per request, so
    this is a coarse last-resort fallback for the ``1d`` timeframe only.
    """
    cid = KNOWN_IDS.get(symbol)
    if cid is None:
        raise FetchError(f"no CoinGecko id for {symbol}")
    data = _http_get_json(
        f"{COINGECKO_API}/coins/{cid}/market_chart?vs_currency=usd&days={days}")
    daily: dict[int, list[float]] = {}
    for ts, price in data.get("prices", []):
        day = int(ts) // 86_400_000
        if day not in daily:
            daily[day] = [price, price, price, price]
        else:
            daily[day][1] = max(daily[day][1], price)
            daily[day][2] = min(daily[day][2], price)
            daily[day][3] = price
    vmap: dict[int, float] = {}
    for ts, v in data.get("total_volumes", []):
        day = int(ts) // 86_400_000
        vmap[day] = vmap.get(day, 0.0) + float(v)
    rows = [{"ts": d * 86_400_000, "open": o, "high": h, "low": l,
             "close": c, "vol": vmap.get(d, 0.0)}
            for d, (o, h, l, c) in sorted(daily.items())]
    if not rows:
        raise FetchError(f"CoinGecko returned no data for {symbol}")
    return rows


# ---------------------------------------------------------------------------
# Local cache
# ---------------------------------------------------------------------------

def cache_path(symbol: str, tf: str) -> Path:
    return RAW_DIR / f"{symbol}_{tf}.parquet"


def save_candles(symbol: str, tf: str, candles: list[dict]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(candles, columns=CANDLE_COLUMNS).drop_duplicates("ts")
    df = df.sort_values("ts").reset_index(drop=True)
    path = cache_path(symbol, tf)
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def load_candles(symbol: str, tf: str) -> list[dict]:
    """Load raw cached candles for a symbol/timeframe as list[dict].

    Raises FileNotFoundError if the cache does not exist.
    """
    path = cache_path(symbol, tf)
    df = pd.read_parquet(path)
    return df.to_dict("records")


def has_cache(symbol: str, tf: str) -> bool:
    return cache_path(symbol, tf).exists()


def fetch_or_load(symbol: str, tf: str, start_ms: int = DEFAULT_START_MS,
                  refresh: bool = False, progress=None) -> list[dict]:
    """Return cached candles, downloading (+ merging) when missing/stale.

    ``refresh=True`` tops the cache up from its last timestamp to now - used
    by retrain.py so weekly retrains only download the new tail.
    """
    if refresh and has_cache(symbol, tf):
        existing = load_candles(symbol, tf)
        last_ts = existing[-1]["ts"]
        if last_ts < int(time.time() * 1000) - TF_MS[tf]:
            try:
                tail = fetch_binance_history(
                    symbol, tf, last_ts + TF_MS[tf], progress=progress)
                merged = existing + [c for c in tail if c["ts"] > last_ts]
                save_candles(symbol, tf, merged)
                return merged
            except FetchError as e:
                print(f"  [refresh failed for {symbol} {tf}: {e} - "
                      f"keeping existing cache]", file=sys.stderr)
                return existing
        return existing

    if has_cache(symbol, tf) and not refresh:
        return load_candles(symbol, tf)

    candles = _download(symbol, tf, start_ms, progress=progress)
    save_candles(symbol, tf, candles)
    return candles


def _download(symbol: str, tf: str, start_ms: int, progress=None) -> list[dict]:
    """Download a full range with CoinGecko as last-resort fallback."""
    last_err: Exception | None = None
    try:
        candles = fetch_binance_history(symbol, tf, start_ms, progress=progress)
        if len(candles) >= 2:
            return candles
        last_err = FetchError("Binance returned <2 candles")
    except FetchError as e:
        last_err = e
    # CoinGecko free API fallback: daily aggregation only.
    try:
        daily = fetch_coingecko_daily(symbol, days=365)
        if tf == "1d" or tf in ("4h", "1h"):
            print(f"  [CoinGecko fallback for {symbol} {tf}: daily "
                  f"aggregation only]", file=sys.stderr)
        return daily
    except FetchError as e2:
        raise FetchError(
            f"All providers failed for {symbol} {tf}: {last_err}; {e2}")


def coverage_summary() -> pd.DataFrame:
    """Rows/date range per cached (symbol, tf) - handy for reports."""
    rows = []
    for path in sorted(RAW_DIR.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["ts"])
        sym, tf = path.stem.split("_", 1)
        rows.append({
            "symbol": sym, "tf": tf, "candles": len(df),
            "first_utc": datetime.fromtimestamp(df["ts"].min() / 1000,
                                                tz=timezone.utc).date(),
            "last_utc": datetime.fromtimestamp(df["ts"].max() / 1000,
                                               tz=timezone.utc).date(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _progress_fn(symbol: str, tf: str, n: int) -> None:
    print(f"  {symbol} {tf}: {n:,} candles…", end="\r", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical OHLCV dataset cache")
    parser.add_argument("--coins", default=",".join(SYMBOLS),
                        help="comma-separated symbols (default: all)")
    parser.add_argument("--tfs", default=",".join(TIMEFRAMES),
                        help="comma-separated timeframes (default: 1h,4h,1d)")
    parser.add_argument("--start", default="2020-01-01",
                        help="history start date YYYY-MM-DD (default 2020-01-01)")
    parser.add_argument("--refresh", action="store_true",
                        help="top up existing caches to now instead of rebuilding")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.coins.split(",") if s.strip()]
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]
    for tf in tfs:
        if tf not in TF_MS:
            raise SystemExit(f"unknown timeframe {tf}; choose from {sorted(TF_MS)}")
    start_ms = int(datetime.strptime(args.start, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp() * 1000)

    print(f"Building dataset: {len(symbols)} coins x {tfs} from {args.start} "
          f"(refresh={args.refresh})")
    t0 = time.time()
    for sym in symbols:
        for tf in tfs:
            print(f"  {sym} {tf} …")
            candles = fetch_or_load(sym, tf, start_ms=start_ms,
                                    refresh=args.refresh,
                                    progress=_progress_fn)
            if candles:
                save_candles(sym, tf, candles)
                first = datetime.fromtimestamp(candles[0]["ts"] / 1000,
                                               tz=timezone.utc).date()
                last = datetime.fromtimestamp(candles[-1]["ts"] / 1000,
                                              tz=timezone.utc).date()
                print(f"  [ok] {sym} {tf}: {len(candles):,} candles "
                      f"({first} -> {last})")
            else:
                print(f"  [fail] {sym} {tf}: no data", file=sys.stderr)

    print(f"\nDone in {time.time() - t0:.0f}s. Cache: {RAW_DIR}")
    print(coverage_summary().to_string(index=False))


if __name__ == "__main__":
    main()
