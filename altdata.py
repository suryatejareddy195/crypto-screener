#!/usr/bin/env python3
"""
altdata.py
----------
Derivatives "alt-data" for the honest directional model: funding rate,
open-interest change and order-book imbalance.

Why this module is explicit about what it can and cannot validate
----------------------------------------------------------------
Not all of the data the user asked for has history. We checked the actual
venue retention before wiring anything in, and the three features are NOT
created equal:

* **Funding rate**  (``/fapi/v1/fundingRate``) - full history back to contract
  launch (2019+). This is a REAL, trainable and backtestable feature.
* **Open-interest change** (``/futures/data/openInterestHist``) - Binance only
  retains roughly the last 30 days. It can power a *live* read-out but there is
  not enough history to train or validate an edge on. It is surfaced at
  inference and clearly labelled "live-only / unvalidated".
* **Order-book imbalance** (``/fapi/v1/depth``) - a live snapshot only, no
  history at all. Same treatment: live-only, unvalidated.

Nothing here is fetched at training time for OI/OB, and nothing fabricated is
inserted into the historical feature matrix. That keeps the walk-forward
metrics honest instead of leaking a feature that only exists going forward.

Everything degrades gracefully: if the exchange is unreachable the funding
feature simply never materialises and the model trains on OHLCV only, which
the training report states explicitly.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ALT_DIR = ROOT / "data" / "alt"

# Optional credentials read from environment variables to avoid committing secrets
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "").strip()
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "").strip()

FAPI_HOST = "https://fapi.binance.com"
USER_AGENT = {"User-Agent": "crypto-screener-altdata/1.0"}
REQUEST_TIMEOUT = 15
RETRY_SLEEP = 2.0
FUNDING_LIMIT = 1000

# Binance funding is paid every 8h -> 3 payments per day.
FUNDING_PER_YEAR = 3 * 365.25

# Column names of the funding features joined into the training matrix.
FUNDING_FEATURES = ["fund_rate", "fund_sum_24h", "fund_z30", "fund_trend"]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class AltDataError(RuntimeError):
    pass


def _get_json(url: str, retries: int = 3) -> list | dict:
    headers = dict(USER_AGENT)
    if BINANCE_API_KEY:
        headers["X-MBX-APIKEY"] = BINANCE_API_KEY
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code not in (408, 429) and not (500 <= e.code < 600):
                raise AltDataError(f"HTTP {e.code} for {url}") from e
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        if attempt < retries - 1:
            time.sleep(RETRY_SLEEP * (attempt + 1))
    raise AltDataError(f"GET failed after {retries} tries: {last}")


def funding_cache_path(symbol: str) -> Path:
    return ALT_DIR / f"{symbol}_funding.parquet"


def load_cached_funding(symbol: str) -> pd.DataFrame:
    """Read the local funding cache WITHOUT any network access.

    Used at inference time so ``app.py`` stays fully offline (same guarantee
    as the OHLCV pipeline). Returns an empty frame when the cache does not
    exist or is unreadable - callers treat that as "no funding feature".
    Populate it with ``python altdata.py`` (or retrain.py) first.
    """
    path = funding_cache_path(symbol)
    if not path.exists():
        return pd.DataFrame(columns=["funding_time", "fund_rate"])
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame(columns=["funding_time", "fund_rate"])


def _perp_symbol(symbol: str) -> str:
    # All coins in the project universe trade as USDT-perpetuals; keep the
    # mapping in one place so a non-perp coin fails loudly, not silently.
    return f"{symbol.upper()}USDT"


# ---------------------------------------------------------------------------
# Funding rate (full history, paginated)
# ---------------------------------------------------------------------------

def fetch_funding_history(symbol: str, start_ms: int,
                          end_ms: int | None = None) -> pd.DataFrame:
    """Paginated funding history for ``symbol`` in ``[start_ms, end_ms]``.

    Returns a DataFrame with ``funding_time`` (ms) and ``fund_rate`` sorted
    ascending. Raises :class:`AltDataError` when the venue is unreachable or
    the symbol has no perpetual.
    """
    perp = _perp_symbol(symbol)
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    rows: list[tuple[int, float]] = []
    cursor = start_ms
    while cursor < end_ms:
        url = (f"{FAPI_HOST}/fapi/v1/fundingRate?symbol={perp}"
               f"&startTime={cursor}&endTime={end_ms}&limit={FUNDING_LIMIT}")
        page = _get_json(url)
        if not isinstance(page, list) or not page:
            break
        for item in page:
            rows.append((int(item["fundingTime"]), float(item["fundingRate"])))
        last_t = int(page[-1]["fundingTime"])
        if last_t + 1 <= cursor:          # no forward progress
            break
        cursor = last_t + 1
        if len(page) < FUNDING_LIMIT:
            break
        time.sleep(0.1)
    if not rows:
        raise AltDataError(f"no funding history for {perp}")
    df = pd.DataFrame(rows, columns=["funding_time", "fund_rate"])
    return df.drop_duplicates("funding_time").sort_values("funding_time").reset_index(drop=True)


def load_or_fetch_funding(symbol: str, start_ms: int,
                          refresh: bool = False) -> pd.DataFrame:
    """Load the cached funding history, downloading when missing/stale.

    The cache is topped up from its last funding time to now, so a retrain
    only downloads the new tail. Returns an EMPTY frame (never raises) when
    the venue cannot be reached - callers treat that as "no funding feature".
    """
    path = funding_cache_path(symbol)
    cached = pd.DataFrame(columns=["funding_time", "fund_rate"])
    if path.exists():
        try:
            cached = pd.read_parquet(path)
        except Exception:
            cached = pd.DataFrame(columns=["funding_time", "fund_rate"])
    try:
        if refresh and not cached.empty:
            start = int(cached["funding_time"].max()) + 1
        else:
            start = start_ms
        tail = fetch_funding_history(symbol, start)
        # concat only non-empty frames (avoids dtype FutureWarnings on the
        # all-empty-cache path)
        parts = [df for df in (cached, tail) if not df.empty]
        merged = (pd.concat(parts, ignore_index=True)
                  .drop_duplicates("funding_time", keep="last")
                  .sort_values("funding_time").reset_index(drop=True))
        ALT_DIR.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False, engine="pyarrow")
        return merged
    except AltDataError:
        return cached


# ---------------------------------------------------------------------------
# Causal funding feature construction
# ---------------------------------------------------------------------------

def _rolling_stats(fund_rate: np.ndarray, window: int, min_obs: int):
    """Trailing mean/std of the funding series (causal, per funding payment)."""
    s = pd.Series(fund_rate)
    mean = s.rolling(window, min_periods=min_obs).mean().to_numpy()
    std = s.rolling(window, min_periods=min_obs).std().to_numpy()
    return mean, std


def funding_features(ts_ms: np.ndarray, funding: pd.DataFrame,
                     tf_ms: int) -> dict[str, np.ndarray]:
    """Map funding payments onto candle rows, strictly causally.

    A candle with open time ``T`` and span ``tf_ms`` only "knows" funding
    payments whose ``funding_time <= T + tf_ms`` (i.e. settled by its close).
    All rolling statistics are computed over the funding series itself, so
    they too use only payments at or before that bar.

    Returns ``{}`` when there is no funding history.
    """
    n = len(ts_ms)
    if funding is None or funding.empty:
        return {}
    ft = funding["funding_time"].to_numpy(dtype=np.int64)
    fr = funding["fund_rate"].to_numpy(dtype=float)

    # index of the last funding payment settled by each candle's close time
    close_ms = ts_ms + tf_ms
    idx = np.searchsorted(ft, close_ms, side="right") - 1

    mean30, std30 = _rolling_stats(fr, window=90, min_obs=10)      # ~30 days
    # trailing 24h sum: 3 payments, via a simple running window over payments
    sum24 = pd.Series(fr).rolling(3, min_periods=1).sum().to_numpy()
    # trend: last 3 payments mean minus the 3 before that
    trend = (pd.Series(fr).rolling(3, min_periods=1).mean().to_numpy()
             - pd.Series(fr).shift(3).rolling(3, min_periods=1).mean().to_numpy())

    def take(arr: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan)
        ok = idx >= 0
        out[ok] = arr[idx[ok]]
        return out

    z = np.where(np.isfinite(std30) & (std30 > _EPS),
                 (fr - mean30) / np.maximum(std30, _EPS), np.nan)
    return {
        "fund_rate": take(fr),
        "fund_sum_24h": take(sum24),
        "fund_z30": take(z),
        "fund_trend": take(trend),
    }


def funding_coverage(df: pd.DataFrame, funding: pd.DataFrame,
                     tf_ms: int) -> float:
    """Fraction of rows that receive a non-NaN funding feature."""
    if funding is None or funding.empty or df.empty:
        return 0.0
    feats = funding_features(df["ts"].to_numpy(dtype=np.int64), funding, tf_ms)
    if not feats:
        return 0.0
    return float(np.isfinite(feats["fund_rate"]).mean())


# ---------------------------------------------------------------------------
# Open interest (recent only) and order-book imbalance (live only)
# ---------------------------------------------------------------------------

_OI_PERIOD = {"1m": "5m", "5m": "5m", "15m": "15m", "1h": "1h",
              "4h": "4h", "1d": "1d"}


def fetch_open_interest(symbol: str, tf: str, limit: int = 500) -> pd.DataFrame:
    """Recent open-interest history (venue retains only ~30 days).

    Returns columns ``ts``, ``oi`` (contracts) and ``oi_value`` (notional).
    Live-only: never joined into historical training.
    """
    period = _OI_PERIOD.get(tf, "1h")
    try:
        page = _get_json(f"{FAPI_HOST}/futures/data/openInterestHist?"
                         f"symbol={_perp_symbol(symbol)}&period={period}&limit={limit}")
    except AltDataError:
        return pd.DataFrame(columns=["ts", "oi", "oi_value"])
    if not isinstance(page, list):
        return pd.DataFrame(columns=["ts", "oi", "oi_value"])
    rows = [{"ts": int(p["timestamp"]), "oi": float(p["sumOpenInterest"]),
             "oi_value": float(p["sumOpenInterestValue"])} for p in page]
    return pd.DataFrame(rows, columns=["ts", "oi", "oi_value"])


def fetch_orderbook_imbalance(symbol: str, levels: int = 20) -> dict | None:
    """Live top-of-book imbalance: bid vs ask notional within ``levels``.

    ``imbalance = (bid - ask) / (bid + ask)`` in [-1, 1]. No history exists,
    so this is inference-time context only.
    """
    try:
        book = _get_json(f"{FAPI_HOST}/fapi/v1/depth?"
                         f"symbol={_perp_symbol(symbol)}&limit={levels}")
    except AltDataError:
        return None
    try:
        bid = sum(float(p) * float(q) for p, q in book["bids"])
        ask = sum(float(p) * float(q) for p, q in book["asks"])
    except (KeyError, TypeError, ValueError):
        return None
    tot = bid + ask
    return {
        "level_count": levels,
        "bid_notional": bid,
        "ask_notional": ask,
        "imbalance": (bid - ask) / tot if tot > 0 else 0.0,
    }


def live_alt_snapshot(symbol: str, tf: str) -> dict:
    """Everything we can fetch *right now* for the inference panel.

    Funding is the only piece with a validated history; OI change and
    order-book imbalance are returned with ``validated=False`` so the UI can
    state plainly that they carry no backtested edge.
    """
    out: dict = {"symbol": symbol, "tf": tf,
                 "funding": {"validated": True},
                 "open_interest": {"validated": False},
                 "orderbook": None}
    # live_alt_snapshot is an online diagnostic (CLI / app alt-data panel);
    # the offline inference path uses load_cached_funding() instead.
    funding = load_or_fetch_funding(symbol, start_ms=0, refresh=True)
    if not funding.empty:
        fr = funding["fund_rate"].to_numpy(dtype=float)
        mean30, std30 = _rolling_stats(fr, 90, 10)
        last = float(fr[-1])
        z = (float((fr[-1] - mean30[-1]) / std30[-1])
             if np.isfinite(std30[-1]) and std30[-1] > _EPS else None)
        out["funding"].update({
            "rate": last,
            "rate_annualized": last * FUNDING_PER_YEAR,
            "sum_24h": float(pd.Series(fr).rolling(3, min_periods=1).sum().to_numpy()[-1]),
            "z30": z,
            "last_time": int(funding["funding_time"].iloc[-1]),
            "n_history": int(len(funding)),
        })
    else:
        out["funding"]["unavailable"] = True

    oi = fetch_open_interest(symbol, tf, limit=200)
    if not oi.empty:
        vals = oi["oi"].to_numpy(dtype=float)
        chg_1 = float(vals[-1] / vals[-2] - 1.0) if len(vals) >= 2 and vals[-2] else None
        chg_24 = (float(vals[-1] / vals[-25] - 1.0)
                  if len(vals) >= 25 and vals[-25] else None)
        out["open_interest"].update({
            "value": float(oi["oi_value"].iloc[-1]),
            "change_1": chg_1, "change_24": chg_24,
            "n_history": int(len(oi)),
            "history_from": int(oi["ts"].iloc[0]),
        })
    else:
        out["open_interest"]["unavailable"] = True

    ob = fetch_orderbook_imbalance(symbol)
    if ob is not None:
        out["orderbook"] = ob
    return out


# ---------------------------------------------------------------------------
# CLI (populate/inspect the funding cache)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    from build_dataset import SYMBOLS, TF_MS

    parser = argparse.ArgumentParser(description="Derivatives alt-data cache")
    parser.add_argument("--coins", default=",".join(SYMBOLS))
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--snapshot", help="print a live alt snapshot for one coin")
    parser.add_argument("--tf", default="1d")
    args = parser.parse_args()

    if args.snapshot:
        print(json.dumps(live_alt_snapshot(args.snapshot.upper(), args.tf),
                         indent=2, default=str))
        return

    start_ms = int(datetime.strptime(args.start, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    for sym in [s.strip().upper() for s in args.coins.split(",") if s.strip()]:
        df = load_or_fetch_funding(sym, start_ms, refresh=args.refresh)
        if df.empty:
            print(f"  [fail] {sym}: no funding history")
            continue
        first = datetime.fromtimestamp(df['funding_time'].min() / 1000,
                                       tz=timezone.utc).date()
        last = datetime.fromtimestamp(df['funding_time'].max() / 1000,
                                      tz=timezone.utc).date()
        print(f"  [ok] {sym}: {len(df):,} funding rows ({first} -> {last})")


if __name__ == "__main__":
    main()
