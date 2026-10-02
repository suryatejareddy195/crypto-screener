#!/usr/bin/env python3
"""
Advanced Crypto Prediction Engine
----------------------------------
Predicts the next-candle direction with a multi-layered scoring system:

  Layer 1 – 15+ technical indicators (RSI, MACD, Stochastic, Williams %R,
            CCI, Ichimoku Cloud, VWAP, OBV, ADX, Bollinger, SMA/EMA stack,
            momentum/ROC, ATR volatility regime)
  Layer 2 – Candlestick pattern recognition (18+ patterns)
  Layer 3 – Divergence detection (bullish/bearish RSI & MACD divergences)
  Layer 4 – Volume profile analysis (OBV trend, volume climax, distribution)
  Layer 5 – Multi-timeframe confluence (higher TF trend alignment)
  Layer 6 – Regime detection (trending / ranging / volatile) with
            regime-adaptive indicator weighting
  Layer 7 – Support/resistance proximity & breakout analysis

All signals feed into a weighted composite score that adapts its weights
based on the detected market regime. Backtest evaluates historical accuracy
of the composite signal.

Data: free public OHLCV APIs with automatic fallback – Binance, Kraken,
OKX, Bybit, then CoinGecko. Real candles, so candlestick patterns are
detected properly.

Usage:
    python predict.py                          # BTC, 1d
    python predict.py --coin SOL --tf 5m       # 5-minute chart
    python predict.py --coin XRP --tf 4h --limit 500
    python predict.py --coin BTC --limit 500   # more daily candles

Honest disclaimer: even advanced technical analysis is probabilistic.
This engine aims to consistently beat a coin flip, but nothing is certain.
Use at your own risk – not financial advice.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import deque

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

API = "https://api.coingecko.com/api/v3"
USER_AGENT = {"User-Agent": "advanced-crypto-predictor/2.0"}

# --- Data-fetch tuning -----------------------------------------------------
# Per-request timeout is short so a dead provider fails fast; only 408/429/5xx
# HTTP errors (plus transport-level errors) are retried, never permanent 4xx.
REQUEST_TIMEOUT = 10          # seconds per HTTP request
REQUEST_RETRIES = 2           # total attempts per URL
REQUEST_BACKOFF = 1.0         # base seconds to sleep between retries
DATA_BUDGET_SECONDS = 30.0    # overall cap for the whole provider fallback chain
MIN_CANDLES = 70              # enough history for indicators/backtest

RETRYABLE_HTTP_STATUS = frozenset({408, 429}) | frozenset(range(500, 600))

NETWORK_ERROR_MSG = (
    "All data providers failed - network/exchange APIs unreachable. "
    "Check your internet connection or a firewall/geo-block on "
    "Binance/OKX/Bybit, then retry."
)

KNOWN_IDS: dict[str, str] = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple",
    "DOGE": "dogecoin", "BNB": "binancecoin", "ADA": "cardano",
    "AVAX": "avalanche-2", "LINK": "chainlink", "DOT": "polkadot",
    "LTC": "litecoin", "BCH": "bitcoin-cash", "NEAR": "near",
    "MATIC": "matic-network", "POL": "polygon-ecosystem-token",
    "TRX": "tron", "UNI": "uniswap", "ATOM": "cosmos",
    "XLM": "stellar", "ETC": "ethereum-classic", "FIL": "filecoin",
    "APT": "aptos", "ARB": "arbitrum", "OP": "optimism",
    "SUI": "sui", "TON": "the-open-network", "INJ": "injective-protocol",
    "ZEC": "zcash", "HYPE": "hyperliquid", "AAVE": "aave",
    "SEI": "sei-network", "PEPE": "pepe", "WIF": "dogwifcoin",
    "FET": "fetch-ai", "RENDER": "render-token", "FARTCOIN": "fartcoin",
}


class DataFetchTimeout(Exception):
    """Raised when the overall data-fetch budget (DATA_BUDGET_SECONDS) runs out."""


def fetch_json(url: str, retries: int = REQUEST_RETRIES,
               timeout: float = REQUEST_TIMEOUT,
               deadline: float | None = None) -> list | dict:
    """Fetch + parse JSON with a sane retry policy.

    Retries only on 408/429/5xx HTTP errors and transport-level errors
    (URLError/TimeoutError). Permanent 4xx errors (400/403/404/451...) fail
    immediately - retrying them just wastes the retry budget.

    ``deadline`` (monotonic seconds) caps the *cumulative* time spent across
    the whole provider fallback chain; when it passes, DataFetchTimeout is
    raised so callers can surface a clear, fast error instead of grinding.
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DataFetchTimeout(NETWORK_ERROR_MSG)
            eff_timeout = min(timeout, remaining)
        else:
            eff_timeout = timeout
        try:
            req = urllib.request.Request(url, headers=USER_AGENT)
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code not in RETRYABLE_HTTP_STATUS:
                # Permanent client error (400/403/404/451...): retrying will
                # never succeed, so fail immediately.
                raise
            last_err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
        if attempt < retries - 1:
            wait = REQUEST_BACKOFF * (attempt + 1)
            if deadline is not None:
                wait = min(wait, deadline - time.monotonic())
            if wait > 0:
                print(f"  [request failed ({last_err}), retrying in {wait:.0f}s...]",
                      file=sys.stderr)
                time.sleep(wait)
    if deadline is not None and time.monotonic() >= deadline:
        raise DataFetchTimeout(NETWORK_ERROR_MSG)
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Failed to fetch {url}")


def resolve_coin_id(arg: str, deadline: float | None = None) -> str:
    arg = arg.strip()
    if arg.lower() in KNOWN_IDS.values():
        return arg.lower()
    if arg.upper() in KNOWN_IDS:
        return KNOWN_IDS[arg.upper()]
    try:
        coins = fetch_json(f"{API}/coins/list", deadline=deadline)
        for c in coins:
            if c["symbol"].lower() == arg.lower():
                return c["id"]
        for c in coins:
            if c["name"].lower() == arg.lower():
                return c["id"]
    except Exception as e:
        print(f"  [could not resolve '{arg}' via API: {e}]", file=sys.stderr)
    raise SystemExit(
        f"Unknown coin '{arg}'. Use a CoinGecko id (e.g. bitcoin) or "
        f"a common symbol (e.g. BTC, SOL)."
    )


TF_MAP: dict[str, dict] = {
    "1m":  {"binance": "1m",  "kraken": 1,    "okx": "1m",  "bybit": "1"},
    "5m":  {"binance": "5m",  "kraken": 5,    "okx": "5m",  "bybit": "5"},
    "15m": {"binance": "15m", "kraken": 15,   "okx": "15m", "bybit": "15"},
    "1h":  {"binance": "1h",  "kraken": 60,   "okx": "1H",  "bybit": "60"},
    "4h":  {"binance": "4h",  "kraken": 240,  "okx": "4H",  "bybit": "240"},
    "1d":  {"binance": "1d",  "kraken": 1440, "okx": "1D",  "bybit": "D"},
}
MAX_CANDLES = {"binance": 1000, "kraken": 720, "okx": 300, "bybit": 1000}


def fetch_candles(sym: str, tf: str, limit: int,
                  progress=None) -> list[dict]:
    """Fetch candles with automatic provider fallback.

    ``progress`` is an optional callback ``f(message: str)`` invoked before
    each provider attempt and after each failure, so callers (e.g. the
    Streamlit UI) can surface provider-by-provider status in real time.

    The whole chain runs under a short cumulative budget
    (DATA_BUDGET_SECONDS): if the exchanges are unreachable it fails fast
    with a clear message instead of grinding through every retry.
    """
    providers = [
        ("Binance", _binance), ("Kraken", _kraken),
        ("OKX", _okx), ("Bybit", _bybit),
    ]
    # CoinGecko fallback works for every timeframe. Its free API only offers
    # daily aggregation for longer ranges, so for intraday timeframes the
    # fallback returns fewer/coarser (daily) candles rather than nothing.
    providers.append(("CoinGecko", _coingecko))
    deadline = time.monotonic() + DATA_BUDGET_SECONDS
    last_err = None
    for name, fn in providers:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if progress:
            progress(f"trying {name}…")
        try:
            candles = fn(sym, tf, limit, deadline=deadline)
            if len(candles) >= MIN_CANDLES:
                note = ""
                if name == "CoinGecko" and tf != "1d":
                    note = (" (daily aggregation - coarser than the requested "
                            f"{tf} timeframe)")
                msg = f"data source: {name} ({len(candles)} {tf} candles{note})"
                print(f"  {msg}", file=sys.stderr)
                if progress:
                    progress(msg)
                return candles
            msg = f"{name} returned only {len(candles)} candles - trying next..."
            print(f"  {msg}", file=sys.stderr)
            if progress:
                progress(msg)
        except DataFetchTimeout as e:
            last_err = e
            msg = f"{name} timed out ({e}) - stopping."
            print(f"  {msg}", file=sys.stderr)
            if progress:
                progress(msg)
            break
        except Exception as e:
            last_err = e
            msg = f"{name} unavailable ({e}) - trying next..."
            print(f"  {msg}", file=sys.stderr)
            if progress:
                progress(msg)
    raise SystemExit(NETWORK_ERROR_MSG + f"\nLast error: {last_err}")


def _to_candles(rows: list[list]) -> list[dict]:
    candles = [{
        "ts": int(r[0]), "open": float(r[1]), "high": float(r[2]),
        "low": float(r[3]), "close": float(r[4]),
        "vol": float(r[5]) if len(r) > 5 else 0.0,
    } for r in rows]
    candles.sort(key=lambda c: c["ts"])
    out, seen = [], set()
    for c in candles:
        if c["ts"] in seen:
            out[-1] = c
        else:
            out.append(c)
            seen.add(c["ts"])
    return out


def _binance(sym: str, tf: str, limit: int, deadline: float | None = None) -> list[dict]:
    n = min(limit, MAX_CANDLES["binance"])
    data = fetch_json(
        f"https://api.binance.com/api/v3/klines?symbol={sym}USDT"
        f"&interval={TF_MAP[tf]['binance']}&limit={n}",
        deadline=deadline,
    )
    return _to_candles([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in data])


def _kraken(sym: str, tf: str, limit: int, deadline: float | None = None) -> list[dict]:
    pair = "XBTUSD" if sym == "BTC" else f"{sym}USD"
    data = fetch_json(
        f"https://api.kraken.com/0/public/OHLC?pair={pair}"
        f"&interval={TF_MAP[tf]['kraken']}",
        deadline=deadline,
    )
    res = data["result"]
    key = next((k for k in res if k != "last"), None)
    if key is None:
        raise RuntimeError(f"Kraken: unknown pair {pair}")
    rows = res[key][-min(limit, MAX_CANDLES["kraken"]):]
    return _to_candles([(r[0] * 1000, r[1], r[2], r[3], r[4], r[6]) for r in rows])


def _okx(sym: str, tf: str, limit: int, deadline: float | None = None) -> list[dict]:
    n = min(limit, MAX_CANDLES["okx"])
    data = fetch_json(
        f"https://www.okx.com/api/v5/market/candles?instId={sym}-USDT"
        f"&bar={TF_MAP[tf]['okx']}&limit={n}",
        deadline=deadline,
    )
    rows = data["data"]
    return _to_candles([
        (int(r[0]), r[1], r[2], r[3], r[4], r[5]) for r in reversed(rows)
    ])


def _bybit(sym: str, tf: str, limit: int, deadline: float | None = None) -> list[dict]:
    n = min(limit, MAX_CANDLES["bybit"])
    data = fetch_json(
        f"https://api.bybit.com/v5/market/kline?category=spot"
        f"&symbol={sym}USDT&interval={TF_MAP[tf]['bybit']}&limit={n}",
        deadline=deadline,
    )
    rows = data["result"]["list"]
    return _to_candles([
        (int(r[0]), r[1], r[2], r[3], r[4], r[5]) for r in reversed(rows)
    ])


def _coingecko(sym: str, tf: str, limit: int, deadline: float | None = None) -> list[dict]:
    """Daily-granularity fallback for any timeframe.

    CoinGecko's free market_chart API only exposes intraday granularity for
    very short windows, so this aggregates everything into daily OHLCV. For
    intraday requests the fallback therefore returns fewer/coarser candles
    than requested - better than failing hard when all exchanges are down.
    """
    cid = KNOWN_IDS.get(sym) or resolve_coin_id(sym, deadline=deadline)
    data = fetch_json(
        f"{API}/coins/{cid}/market_chart?vs_currency=usd"
        f"&days={min(limit, 365)}",
        deadline=deadline,
    )
    daily: dict[int, list[float]] = {}
    for ts, p in data["prices"]:
        day = ts // 86400000
        if day not in daily:
            daily[day] = [p, p, p, p]
        else:
            daily[day][1] = max(daily[day][1], p)
            daily[day][2] = min(daily[day][2], p)
            daily[day][3] = p
    vmap: dict[int, float] = {}
    for ts, v in data.get("total_volumes", []):
        day = ts // 86400000
        vmap[day] = vmap.get(day, 0.0) + float(v)
    rows = [(d * 86400000, o, h, l, c, vmap.get(d, 0.0))
            for d, (o, h, l, c) in sorted(daily.items())]
    return _to_candles(rows)


# ---------------------------------------------------------------------------
# Core indicator functions
# ---------------------------------------------------------------------------

def _sma(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    run = 0.0
    for i, v in enumerate(values):
        run += v
        if i >= period:
            run -= values[i - period]
        if i >= period - 1:
            out[i] = run / period
    return out


def _ema(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if not values:
        return out
    k = 2.0 / (period + 1)
    prev = values[0]
    for i, v in enumerate(values):
        prev = v if i == 0 else v * k + prev * (1 - k)
        out[i] = prev
    return out


def _rsi(closes: list[float], period: int = 14) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    if len(closes) < period + 1:
        return out
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out[period] = (
        100.0 if avg_loss == 0
        else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    )
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = (
            100.0 if avg_loss == 0
            else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
        )
    return out


def _atr(candles: list[dict], period: int = 14) -> list[float | None]:
    out: list[float | None] = [None] * len(candles)
    if len(candles) < period + 1:
        return out
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr_val = sum(trs[:period]) / period
    out[period] = atr_val
    for i in range(period, len(trs)):
        atr_val = (atr_val * (period - 1) + trs[i]) / period
        out[i + 1] = atr_val
    return out


def _adx(candles: list[dict], period: int = 14):
    n = len(candles)
    out: list[float | None] = [None] * n
    plus_di: list[float | None] = [None] * n
    minus_di: list[float | None] = [None] * n
    if n < period * 2 + 1:
        return out, plus_di, minus_di
    tr, up, dn = [], [], []
    for i in range(1, n):
        h, l, ph, pl = (
            candles[i]["high"], candles[i]["low"],
            candles[i - 1]["high"], candles[i - 1]["low"],
        )
        tr.append(max(h - l, abs(h - pl), abs(l - ph)))
        up.append(h - ph if h - ph > pl - l else 0.0)
        dn.append(pl - l if pl - l > h - ph else 0.0)
    atr_val = sum(tr[:period]) / period
    up_s = sum(up[:period]) / period
    dn_s = sum(dn[:period]) / period

    def di(a: float, b: float) -> float:
        return 100.0 * a / b if b > 0 else 0.0

    dxs = []
    for i in range(period, len(tr)):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        up_s = (up_s * (period - 1) + up[i]) / period
        dn_s = (dn_s * (period - 1) + dn[i]) / period
        pdi, mdi = di(up_s, atr_val), di(dn_s, atr_val)
        plus_di[i + 1], minus_di[i + 1] = pdi, mdi
        dxs.append(
            100.0 * abs(pdi - mdi) / (pdi + mdi) if pdi + mdi > 0 else 0.0
        )
    adx_val = sum(dxs[:period]) / period
    for i in range(period, len(dxs)):
        adx_val = (adx_val * (period - 1) + dxs[i]) / period
        out[i + period + 1] = adx_val
    return out, plus_di, minus_di


def _macd(closes: list[float], fast: int = 12, slow: int = 26, sig: int = 9):
    ema_f = _ema(closes, fast)
    ema_s = _ema(closes, slow)
    line: list[float | None] = [
        None if a is None or b is None else a - b
        for a, b in zip(ema_f, ema_s)
    ]
    valid = [x for x in line if x is not None]
    sig_line = _ema(valid, sig)
    sig_line = [None] * (len(line) - len(sig_line)) + sig_line
    hist: list[float | None] = [
        None if l is None or s is None else l - s
        for l, s in zip(line, sig_line)
    ]
    return line, sig_line, hist

def _rolling_max(values: list[float], period: int) -> list[float | None]:
    """Rolling maximum over `period`, O(n) via a monotonic deque."""
    n = len(values)
    out: list[float | None] = [None] * n
    dq = deque()
    for i, v in enumerate(values):
        while dq and values[dq[-1]] <= v:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - period:
            dq.popleft()
        if i >= period - 1:
            out[i] = values[dq[0]]
    return out


def _rolling_min(values: list[float], period: int) -> list[float | None]:
    """Rolling minimum over `period`, O(n) via a monotonic deque."""
    n = len(values)
    out: list[float | None] = [None] * n
    dq = deque()
    for i, v in enumerate(values):
        while dq and values[dq[-1]] >= v:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - period:
            dq.popleft()
        if i >= period - 1:
            out[i] = values[dq[0]]
    return out


def _bollinger(closes: list[float], period: int = 20):
    """Bollinger bands - O(n): running sum + sum-of-squares instead of
    rescanning the whole window for a stddev at every position."""
    mid = _sma(closes, period)
    upper: list[float | None] = [None] * len(closes)
    lower: list[float | None] = [None] * len(closes)
    run = 0.0
    run_sq = 0.0
    for i, v in enumerate(closes):
        run += v
        run_sq += v * v
        if i >= period:
            run -= closes[i - period]
            run_sq -= closes[i - period] * closes[i - period]
        if i >= period - 1:
            m = mid[i]
            if m is None:
                continue
            var = run_sq / period - m * m
            if var < 0.0:  # guard against tiny negative float noise
                var = 0.0
            sd = math.sqrt(var)
            upper[i] = m + 2 * sd
            lower[i] = m - 2 * sd
    return upper, mid, lower


# ---------------------------------------------------------------------------
# Advanced indicators
# ---------------------------------------------------------------------------

def _stochastic(candles: list[dict], k_period: int = 14, d_period: int = 3):
    """Stochastic Oscillator %K and %D.

    Rolling high/low come from monotonic deques (O(n)) instead of scanning
    the k_period-wide window at every position.
    """
    n = len(candles)
    k_arr: list[float | None] = [None] * n
    d_arr: list[float | None] = [None] * n
    if n < k_period:
        return k_arr, d_arr
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    hh_arr = _rolling_max(highs, k_period)
    ll_arr = _rolling_min(lows, k_period)
    raw_k = []
    for i in range(k_period - 1, n):
        hh, ll = hh_arr[i], ll_arr[i]
        rng = hh - ll
        k_val = 100.0 * (candles[i]["close"] - ll) / rng if rng > 0 else 50.0
        raw_k.append(k_val)
        k_arr[i] = k_val
    # %D = 3-period average of %K; the tiny d-window is re-averaged to keep
    # the exact arithmetic (O(n * d_period), d_period is small/constant).
    if len(raw_k) >= d_period:
        for i in range(d_period - 1, len(raw_k)):
            d_arr[k_period - 1 + i] = sum(raw_k[i - d_period + 1: i + 1]) / d_period
    return k_arr, d_arr


def _williams_r(candles: list[dict], period: int = 14) -> list[float | None]:
    """Williams %R: ranges -100 to 0.

    Rolling high/low come from monotonic deques (O(n)) instead of scanning
    the period-wide window at every position.
    """
    n = len(candles)
    out: list[float | None] = [None] * n
    if n < period:
        return out
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    hh_arr = _rolling_max(highs, period)
    ll_arr = _rolling_min(lows, period)
    for i in range(period - 1, n):
        hh, ll = hh_arr[i], ll_arr[i]
        rng = hh - ll
        out[i] = -100.0 * (hh - candles[i]["close"]) / rng if rng > 0 else -50.0
    return out


def _cci(candles: list[dict], period: int = 20) -> list[float | None]:
    """Commodity Channel Index.

    O(n * period): window means come from a running sum (no per-position
    ``sum()`` rescans), and the mean-absolute-deviation term is computed over
    a fixed-width index window instead of allocating a fresh slice at every
    position. ``period`` is a small constant, so the whole series is O(n)
    for practical purposes (identical arithmetic to the textbook formula).
    """
    n = len(candles)
    out: list[float | None] = [None] * n
    if n < period:
        return out
    tp = [(c["high"] + c["low"] + c["close"]) / 3.0 for c in candles]
    run = sum(tp[:period])
    for i in range(period - 1, n):
        mean_tp = run / period
        dev = 0.0
        for j in range(i - period + 1, i + 1):
            d = tp[j] - mean_tp
            dev += d if d >= 0 else -d
        mean_dev = dev / period
        out[i] = (tp[i] - mean_tp) / (0.015 * mean_dev) if mean_dev > 0 else 0.0
        if i + 1 < n:
            run += tp[i + 1] - tp[i - period + 1]
    return out


def _ichimoku(candles: list[dict], tenkan: int = 9, kijun: int = 26,
              senkou_b: int = 52):
    """Ichimoku Cloud components.

    Window mid-points use monotonic-deque rolling high/low (O(n)) instead of
    rescanning each window at every position.
    """
    n = len(candles)
    tenkan_sen: list[float | None] = [None] * n
    kijun_sen: list[float | None] = [None] * n
    senkou_a: list[float | None] = [None] * n
    senkou_b_line: list[float | None] = [None] * n
    if n == 0:
        return tenkan_sen, kijun_sen, senkou_a, senkou_b_line
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    hh_t = _rolling_max(highs, tenkan)
    ll_t = _rolling_min(lows, tenkan)
    hh_k = _rolling_max(highs, kijun)
    ll_k = _rolling_min(lows, kijun)
    hh_s = _rolling_max(highs, senkou_b)
    ll_s = _rolling_min(lows, senkou_b)
    for i in range(tenkan - 1, n):
        tenkan_sen[i] = (hh_t[i] + ll_t[i]) / 2.0
    for i in range(kijun - 1, n):
        kijun_sen[i] = (hh_k[i] + ll_k[i]) / 2.0
    for i in range(kijun - 1, n):
        if tenkan_sen[i] is not None and kijun_sen[i] is not None:
            senkou_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2.0
    for i in range(senkou_b - 1, n):
        senkou_b_line[i] = (hh_s[i] + ll_s[i]) / 2.0
    return tenkan_sen, kijun_sen, senkou_a, senkou_b_line


def _vwap_bands(candles: list[dict], period: int = 20) -> list[float | None]:
    """Rolling VWAP over `period` candles - O(n) via running sums of
    (typical price x volume) and volume, no window rescanning."""
    n = len(candles)
    out: list[float | None] = [None] * n
    if n < period:
        return out
    run_pv = 0.0
    run_v = 0.0
    for i in range(n):
        c = candles[i]
        tp = (c["high"] + c["low"] + c["close"]) / 3.0
        v = c["vol"]
        run_pv += tp * v
        run_v += v
        if i >= period:
            prev = candles[i - period]
            tp0 = (prev["high"] + prev["low"] + prev["close"]) / 3.0
            v0 = prev["vol"]
            run_pv -= tp0 * v0
            run_v -= v0
        if i >= period - 1:
            out[i] = run_pv / run_v if run_v > 0 else candles[i]["close"]
    return out


def _obv(candles: list[dict]) -> list[float]:
    """On-Balance Volume."""
    out = [0.0] * len(candles)
    for i in range(1, len(candles)):
        if candles[i]["close"] > candles[i - 1]["close"]:
            out[i] = out[i - 1] + candles[i]["vol"]
        elif candles[i]["close"] < candles[i - 1]["close"]:
            out[i] = out[i - 1] - candles[i]["vol"]
        else:
            out[i] = out[i - 1]
    return out


def _roc(closes: list[float], period: int = 12) -> list[float | None]:
    """Rate of Change."""
    out: list[float | None] = [None] * len(closes)
    for i in range(period, len(closes)):
        if closes[i - period] != 0:
            out[i] = (closes[i] - closes[i - period]) / closes[i - period] * 100
    return out


def _slope(values: list[float]) -> float:
    """Linear regression slope, normalised by mean."""
    if len(values) < 3:
        return 0.0
    n = len(values)
    xs = list(range(n))
    mx = (n - 1) / 2.0
    my = sum(values) / n
    num = sum((x - mx) * (v - my) for x, v in zip(xs, values))
    den = sum((x - mx) ** 2 for x in xs)
    return (num / den) / my if den > 0 and my > 0 else 0.0


# ---------------------------------------------------------------------------
# Candlestick patterns
# ---------------------------------------------------------------------------

def _body(c: dict) -> float:
    return abs(c["close"] - c["open"])

def _rng(c: dict) -> float:
    return max(c["high"] - c["low"], 1e-12)

def _upper_shadow(c: dict) -> float:
    return c["high"] - max(c["open"], c["close"])

def _lower_shadow(c: dict) -> float:
    return min(c["open"], c["close"]) - c["low"]

def _bullish(c: dict) -> bool:
    return c["close"] > c["open"]

def _bearish(c: dict) -> bool:
    return c["close"] < c["open"]


def detect_patterns(candles: list[dict],
                    sma20: float | None = None) -> list[tuple[str, int, str]]:
    """Detect 18+ candlestick patterns. Returns (name, score, description).

    ``sma20`` is the precomputed SMA(20) at the last candle; pass it in to
    avoid recomputing the whole SMA(20) series just for the latest value.
    """
    if len(candles) < 5:
        return []
    found: list[tuple[str, int, str]] = []
    c1, c2, c3, c4, c5 = candles[-1], candles[-2], candles[-3], candles[-4], candles[-5]
    if sma20 is None:
        closes = [c["close"] for c in candles]
        sma20 = _sma(closes, 20)[-1]
    downtrend = sma20 is not None and c1["close"] < sma20
    uptrend = sma20 is not None and c1["close"] > sma20

    b1, r1 = _body(c1), _rng(c1)
    b2, r2 = _body(c2), _rng(c2)
    b3, r3 = _body(c3), _rng(c3)
    b4, r4 = _body(c4), _rng(c4)
    b5, r5 = _body(c5), _rng(c5)
    us1, ls1 = _upper_shadow(c1), _lower_shadow(c1)

    # --- Single-candle patterns ---
    if b1 <= 0.1 * r1:
        found.append(("Doji", 0, "indecision – market is undecided"))
    if ls1 >= 2 * b1 and us1 <= b1 and b1 <= 0.35 * r1:
        if downtrend:
            found.append(("Hammer", 1, "bullish reversal at lows (downtrend)"))
        else:
            found.append(("Hammer", 0, "hammer but no downtrend context"))
    if us1 >= 2 * b1 and ls1 <= b1 and b1 <= 0.35 * r1:
        if uptrend:
            found.append(("Shooting Star", -1, "bearish reversal at highs (uptrend)"))
        else:
            found.append(("Shooting Star", 0, "shooting star but no uptrend context"))
    if b1 >= 0.9 * r1:
        found.append(("Marubozu", 1 if _bullish(c1) else -1,
                      "strong body – decisive " + ("buying" if _bullish(c1) else "selling")))
    # Spinning top
    if b1 <= 0.3 * r1 and us1 > 0.15 * r1 and ls1 > 0.15 * r1:
        found.append(("Spinning Top", 0, "indecision with long shadows"))
    # Inverted hammer (in downtrend)
    if downtrend and us1 >= 2 * b1 and ls1 <= b1 * 0.5 and b1 <= 0.35 * r1:
        found.append(("Inverted Hammer", 1, "potential bullish reversal (downtrend)"))

    # --- Two-candle patterns ---
    if _bearish(c2) and _bullish(c1) and b1 > b2 and \
            c1["open"] <= c2["close"] and c1["close"] >= c2["open"]:
        found.append(("Bullish Engulfing", 1, "buyers overwhelmed sellers"))
    if _bullish(c2) and _bearish(c1) and b1 > b2 and \
            c1["open"] >= c2["close"] and c1["close"] <= c2["open"]:
        found.append(("Bearish Engulfing", -1, "sellers overwhelmed buyers"))
    if _bearish(c2) and b2 >= 0.5 * r2 and _bullish(c1) and \
            c1["open"] < c2["low"] and c1["close"] > c2["open"] + 0.5 * b2:
        found.append(("Piercing Line", 1, "bullish reversal after a down day"))
    if _bullish(c2) and b2 >= 0.5 * r2 and _bearish(c1) and \
            c1["open"] > c2["high"] and c1["close"] < c2["open"] - 0.5 * b2:
        found.append(("Dark Cloud Cover", -1, "bearish reversal after an up day"))
    if b1 <= 0.4 * b2 and b2 > 0:
        found.append((
            "Bullish Harami" if _bearish(c2) else "Bearish Harami",
            1 if _bearish(c2) else -1,
            "small body inside previous – reversal watch",
        ))
    if downtrend and abs(c1["low"] - c2["low"]) <= 0.1 * r1 and _bearish(c2) and _bullish(c1):
        found.append(("Tweezer Bottom", 1, "equal lows – support rejection"))
    if uptrend and abs(c1["high"] - c2["high"]) <= 0.1 * r1 and _bullish(c2) and _bearish(c1):
        found.append(("Tweezer Top", -1, "equal highs – resistance rejection"))
    # Harami Cross
    if b1 <= 0.1 * r1 and b2 > 0.5 * r2:
        found.append(("Harami Cross", 1 if _bearish(c2) else -1,
                      "doji inside large body – strong reversal watch"))
    # Kicking (Marubozu-to-Marubozu)
    if b1 >= 0.85 * r1 and b2 >= 0.85 * r2:
        if _bearish(c2) and _bullish(c1) and c1["open"] > c2["close"]:
            found.append(("Bullish Kicking", 1, "strong bullish reversal gap"))
        if _bullish(c2) and _bearish(c1) and c1["open"] < c2["close"]:
            found.append(("Bearish Kicking", -1, "strong bearish reversal gap"))

    # --- Three-candle patterns ---
    if _bearish(c3) and b3 >= 0.5 * r3 and b1 <= 0.3 * r1 and \
            _bullish(c1) and c1["close"] > (c3["open"] + c3["close"]) / 2:
        found.append(("Morning Star", 1, "3-candle bullish reversal"))
    if _bullish(c3) and b3 >= 0.5 * r3 and b1 <= 0.3 * r1 and \
            _bearish(c1) and c1["close"] < (c3["open"] + c3["close"]) / 2:
        found.append(("Evening Star", -1, "3-candle bearish reversal"))
    if _bullish(c1) and _bullish(c2) and _bullish(c3) and \
            c1["close"] > c2["close"] > c3["close"] and \
            b1 >= 0.5 * r1 and b2 >= 0.5 * r2 and b3 >= 0.5 * r3:
        found.append(("Three White Soldiers", 1, "sustained buying pressure"))
    if _bearish(c1) and _bearish(c2) and _bearish(c3) and \
            c1["close"] < c2["close"] < c3["close"] and \
            b1 >= 0.5 * r1 and b2 >= 0.5 * r2 and b3 >= 0.5 * r3:
        found.append(("Three Black Crows", -1, "sustained selling pressure"))
    # Three Inside Up / Down
    if _bearish(c3) and b3 >= 0.5 * r3 and _bearish(c2) and \
            b2 < 0.5 * b3 and _bullish(c1) and c1["close"] > c3["open"]:
        found.append(("Three Inside Up", 1, "bullish reversal confirmation"))
    if _bullish(c3) and b3 >= 0.5 * r3 and _bullish(c2) and \
            b2 < 0.5 * b3 and _bearish(c1) and c1["close"] < c3["open"]:
        found.append(("Three Inside Down", -1, "bearish reversal confirmation"))
    # Falling/Rising three methods (continuation)
    if _bearish(c5) and b5 >= 0.5 * r5:
        small_bull = all(_bullish(candles[-i]) and _body(candles[-i]) < 0.4 * _rng(candles[-i]) for i in [4, 3, 2])
        if small_bull and _bearish(c1) and c1["close"] < c5["close"]:
            found.append(("Falling Three Methods", -1, "bearish continuation"))
    if _bullish(c5) and b5 >= 0.5 * r5:
        small_bear = all(_bearish(candles[-i]) and _body(candles[-i]) < 0.4 * _rng(candles[-i]) for i in [4, 3, 2])
        if small_bear and _bullish(c1) and c1["close"] > c5["close"]:
            found.append(("Rising Three Methods", 1, "bullish continuation"))

    return found


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------

def detect_divergences(candles: list[dict], lookback: int = 30,
                       rsi: list[float | None] | None = None,
                       macd_line: list[float | None] | None = None,
                       end: int | None = None) -> list[tuple[str, int, str]]:
    """Detect bullish & bearish divergences between price and RSI/MACD.

    Precomputed full-length ``rsi`` / ``macd_line`` series can be passed in to
    avoid recomputing them here; ``end`` restricts the analysis to the prefix
    ``candles[:end + 1]`` (defaults to the last candle). Reads are causal, so
    passing full series + ``end`` is identical to computing on the slice.
    """
    if end is None:
        end = len(candles) - 1
    if end + 1 < lookback + 5:
        return []
    if rsi is None or macd_line is None:
        closes = [c["close"] for c in candles]
        if rsi is None:
            rsi = _rsi(closes, 14)
        if macd_line is None:
            macd_line = _macd(closes)[0]
    found: list[tuple[str, int, str]] = []

    start = end + 1 - lookback
    window = candles[start: end + 1]
    close_w = [c["close"] for c in window]
    rsi_w = rsi[start: end + 1]
    macd_w = macd_line[start: end + 1]

    # Find local lows (swing lows) for bullish divergence
    def find_swing_lows(vals: list[float | None], w: list[dict], n: int = 3) -> list[int]:
        lows = []
        for i in range(n, len(vals) - n):
            if vals[i] is None:
                continue
            seg = [vals[i - j] for j in range(-n, n + 1) if vals[i - j] is not None]
            if seg and vals[i] == min(seg):
                lows.append(i)
        return lows

    def find_swing_highs(vals: list[float | None], w: list[dict], n: int = 3) -> list[int]:
        highs = []
        for i in range(n, len(vals) - n):
            if vals[i] is None:
                continue
            seg = [vals[i - j] for j in range(-n, n + 1) if vals[i - j] is not None]
            if seg and vals[i] == max(seg):
                highs.append(i)
        return highs

    # Bullish divergence: price makes lower low, indicator makes higher low
    price_lows = find_swing_lows(close_w, window)
    for indicator, name in [(rsi_w, "RSI"), (macd_w, "MACD")]:
        ind_lows = find_swing_lows(indicator, window)
        if len(price_lows) >= 2 and len(ind_lows) >= 2:
            p1, p2 = price_lows[-2], price_lows[-1]
            if close_w[p2] < close_w[p1] and ind_lows[-1] > ind_lows[-2]:
                if indicator[ind_lows[-1]] is not None and indicator[ind_lows[-2]] is not None:
                    if indicator[ind_lows[-1]] > indicator[ind_lows[-2]]:
                        found.append((
                            f"Bullish {name} Divergence", 1,
                            f"price lower low but {name} higher low – reversal signal",
                        ))

    # Bearish divergence: price makes higher high, indicator makes lower high
    price_highs = find_swing_highs(close_w, window)
    for indicator, name in [(rsi_w, "RSI"), (macd_w, "MACD")]:
        ind_highs = find_swing_highs(indicator, window)
        if len(price_highs) >= 2 and len(ind_highs) >= 2:
            p1, p2 = price_highs[-2], price_highs[-1]
            if close_w[p2] > close_w[p1] and ind_highs[-1] > ind_highs[-2]:
                if indicator[ind_highs[-1]] is not None and indicator[ind_highs[-2]] is not None:
                    if indicator[ind_highs[-1]] < indicator[ind_highs[-2]]:
                        found.append((
                            f"Bearish {name} Divergence", -1,
                            f"price higher high but {name} lower high – reversal signal",
                        ))

    return found


# ---------------------------------------------------------------------------
# Support / resistance + volume profile
# ---------------------------------------------------------------------------

def swing_points(candles: list[dict], lookback: int = 3) -> tuple[list[float], list[float]]:
    window = candles[-60:]
    highs = [c["high"] for c in window]
    lows = [c["low"] for c in window]
    res, sup = [], []
    for i in range(lookback, len(window) - lookback):
        seg_h = highs[i - lookback: i + lookback + 1]
        seg_l = lows[i - lookback: i + lookback + 1]
        if highs[i] == max(seg_h):
            res.append(highs[i])
        if lows[i] == min(seg_l):
            sup.append(lows[i])
    return res, sup


def nearest_levels(price: float, res: list[float], sup: list[float]):
    above = [r for r in res if r > price * 1.001]
    below = [s for s in sup if s < price * 0.999]
    resistance = min(above) if above else (max(res) if res else None)
    support = max(below) if below else (min(sup) if sup else None)
    return resistance, support


def volume_analysis(candles: list[dict]) -> dict | None:
    vols = [c.get("vol", 0.0) for c in candles]
    if len(vols) < 30:
        return None
    recent = vols[-21:-1]
    mean20 = sum(recent) / len(recent)
    sd = statistics.stdev(recent) if len(recent) > 1 else 0.0
    z = (vols[-1] - mean20) / sd if sd > 0 else 0.0
    recent5 = sum(vols[-5:]) / 5
    prior20 = sum(vols[-25:-5]) / 20
    trend = recent5 / prior20 - 1.0 if prior20 > 0 else 0.0
    obv_arr = _obv(candles)
    obv_sl = _slope(obv_arr[-20:])

    # Volume climax detection: last volume is > 3x average
    climax = z > 3.0

    # Distribution vs accumulation: compare up-volume vs down-volume
    up_vol = sum(candles[i]["vol"] for i in range(-20, 0) if candles[i]["close"] > candles[i]["open"])
    dn_vol = sum(candles[i]["vol"] for i in range(-20, 0) if candles[i]["close"] < candles[i]["open"])
    dist_acc = (up_vol - dn_vol) / (up_vol + dn_vol) if (up_vol + dn_vol) > 0 else 0.0

    return {
        "z": z, "trend": trend, "obv_slope": obv_sl,
        "climax": climax, "distribution": dist_acc,
    }


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def detect_regime(candles: list[dict], ind: dict | None = None,
                  t: int | None = None) -> dict:
    """Detect market regime: trending, ranging, volatile, or breakout.

    ``ind`` may carry the precomputed full-length series (adx/pdi/mdi,
    bollinger, atr, closes) from :func:`precompute_indicators`; ``t`` selects
    the candle index (default: last). Reads are causal, so the value at index
    ``t`` is identical whether computed over ``candles[:t+1]`` or the full
    series - passing the arrays in just avoids the recompute.
    """
    if t is None:
        t = len(candles) - 1
    if ind is None:
        closes = [c["close"] for c in candles]
        adx_vals, pdi_vals, mdi_vals = _adx(candles)
        bb_upper, bb_mid, bb_lower = _bollinger(closes)
        atr_vals = _atr(candles)
    else:
        closes = ind["closes"]
        adx_vals = ind["adx"]
        pdi_vals = ind["pdi"]
        mdi_vals = ind["mdi"]
        bb_upper = ind["bb_upper"]
        bb_mid = ind["bb_mid"]
        bb_lower = ind["bb_lower"]
        atr_vals = ind["atr"]

    adx_v = adx_vals[t] if adx_vals[t] is not None else 20.0
    pdi_v = pdi_vals[t] if pdi_vals[t] is not None else 0.0
    mdi_v = mdi_vals[t] if mdi_vals[t] is not None else 0.0

    # Bollinger bandwidth
    if bb_upper[t] is not None and bb_lower[t] is not None and bb_mid[t] is not None:
        bandwidth = (bb_upper[t] - bb_lower[t]) / bb_mid[t] if bb_mid[t] > 0 else 0
    else:
        bandwidth = 0.05

    # ATR as % of price
    atr_pct = atr_vals[t] / closes[t] * 100 if atr_vals[t] is not None else 2.0

    # Detect regime
    regime = "ranging"
    strength = 0.0
    if adx_v > 30 and bandwidth > 0.06:
        regime = "trending"
        strength = min(adx_v / 50, 1.0)
        if pdi_v > mdi_v:
            direction = "bullish"
        else:
            direction = "bearish"
    elif adx_v < 20 and bandwidth < 0.04:
        regime = "ranging"
        strength = min((20 - adx_v) / 20, 1.0)
        direction = "neutral"
    elif atr_pct > 5.0 or bandwidth > 0.10:
        regime = "volatile"
        strength = min(atr_pct / 10, 1.0)
        direction = "neutral"
    else:
        regime = "transitional"
        strength = 0.5
        direction = "bullish" if pdi_v > mdi_v else "bearish"

    return {
        "regime": regime, "strength": strength, "direction": direction,
        "adx": adx_v, "bandwidth": bandwidth, "atr_pct": atr_pct,
    }


# ---------------------------------------------------------------------------
# Multi-timeframe confluence
# ---------------------------------------------------------------------------

def precompute_indicators(candles: list[dict]) -> dict:
    """Compute every indicator series ONCE over the full candle list.

    Returns a dict of full-length arrays aligned with ``candles``. Every
    indicator read here is causal (the value at index t depends only on
    candles[:t+1]), so reading these arrays at ``t`` is identical to what the
    old code got by recomputing every series over a truncated slice - but at
    O(n) total instead of O(n^2).
    """
    closes = [c["close"] for c in candles]
    macd_line, macd_sig, macd_hist = _macd(closes)
    bb_upper, bb_mid, bb_lower = _bollinger(closes)
    stoch_k, stoch_d = _stochastic(candles)
    tenkan, kijun, senkou_a, senkou_b = _ichimoku(candles)
    adx_vals, pdi_vals, mdi_vals = _adx(candles)
    return {
        "closes": closes,
        "rsi": _rsi(closes, 14),
        "macd_line": macd_line, "macd_sig": macd_sig,
        "macd_hist": macd_hist,
        "s20": _sma(closes, 20), "s50": _sma(closes, 50),
        "s100": _sma(closes, 100),
        "adx": adx_vals, "pdi": pdi_vals, "mdi": mdi_vals,
        "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
        "atr": _atr(candles),
        "stoch_k": stoch_k, "stoch_d": stoch_d,
        "wr": _williams_r(candles),
        "cci": _cci(candles),
        "tenkan": tenkan, "kijun": kijun,
        "senkou_a": senkou_a, "senkou_b": senkou_b,
        "vwap": _vwap_bands(candles),
        "obv": _obv(candles),
        "roc": _roc(closes),
    }


def compute_signal_for_data(candles: list[dict], ind: dict | None = None,
                            t: int | None = None) -> dict:
    """Compute signal from a list of candles (the core scoring engine).

    ``ind`` may carry the precomputed full-length indicator arrays from
    :func:`precompute_indicators`; ``t`` is the index of the "current" candle
    (defaults to the last one). When both are supplied the function reads
    values at [t] / [t-1] / [t-19:t+1] from the precomputed arrays instead of
    recomputing every indicator series over a truncated slice - this is what
    makes the backtest O(n) instead of O(n^2). ``compute_signal()`` (the live
    "current candle" call path) calls this with t = len(candles) - 1.
    """
    if ind is None:
        ind = precompute_indicators(candles)
    if t is None:
        t = len(candles) - 1
    closes = ind["closes"]
    price = closes[t]
    comp: dict[str, float] = {}

    # --- Regime detection ---
    regime = detect_regime(candles, ind=ind, t=t)
    regime_name = regime["regime"]

    # --- Regime-adaptive weights ---
    if regime_name == "trending":
        weights = {
            "patterns": 0.8, "rsi": 1.2, "macd": 1.8, "sma": 2.0,
            "adx": 1.0, "bollinger": 0.8, "slope": 1.5, "stoch": 0.6,
            "williams": 0.5, "cci": 0.8, "ichimoku": 1.5, "vwap": 0.6,
            "obv": 1.0, "roc": 1.0, "divergence": 1.5,
        }
    elif regime_name == "volatile":
        weights = {
            "patterns": 0.5, "rsi": 1.5, "macd": 0.8, "sma": 0.5,
            "adx": 0.4, "bollinger": 1.5, "slope": 0.4, "stoch": 1.0,
            "williams": 1.0, "cci": 1.2, "ichimoku": 0.6, "vwap": 1.0,
            "obv": 0.6, "roc": 0.8, "divergence": 1.2,
        }
    elif regime_name == "ranging":
        weights = {
            "patterns": 0.6, "rsi": 1.8, "macd": 0.6, "sma": 0.3,
            "adx": 0.3, "bollinger": 1.8, "slope": 0.3, "stoch": 1.8,
            "williams": 1.8, "cci": 1.2, "ichimoku": 0.5, "vwap": 1.2,
            "obv": 0.8, "roc": 0.5, "divergence": 1.2,
        }
    else:  # transitional
        weights = {
            "patterns": 0.7, "rsi": 1.3, "macd": 1.2, "sma": 1.0,
            "adx": 0.6, "bollinger": 1.0, "slope": 0.8, "stoch": 1.0,
            "williams": 0.8, "cci": 0.8, "ichimoku": 1.0, "vwap": 0.8,
            "obv": 0.8, "roc": 0.8, "divergence": 1.3,
        }

    # --- Pattern score ---
    # Patterns only ever look at the last 5 candles; the SMA(20) trend
    # context comes from the precomputed array (no full-series recompute).
    patterns = detect_patterns(candles[max(0, t - 4): t + 1],
                               sma20=ind["s20"][t])
    comp["patterns"] = sum(s for _, s, _ in patterns)

    # --- RSI ---
    rsi = ind["rsi"][t]
    if rsi is not None:
        if rsi < 25:
            comp["rsi"] = 2.0
        elif rsi < 30:
            comp["rsi"] = 1.5
        elif rsi > 75:
            comp["rsi"] = -2.0
        elif rsi > 70:
            comp["rsi"] = -1.5
        elif 40 <= rsi <= 60:
            comp["rsi"] = 0.0
        elif rsi < 40:
            comp["rsi"] = 0.5
        else:
            comp["rsi"] = -0.5
    else:
        comp["rsi"] = 0.0

    # --- MACD ---
    macd_line = ind["macd_line"]
    macd_sig = ind["macd_sig"]
    macd_hist = ind["macd_hist"]
    macd_s = 0.0
    if macd_line[t] is not None and macd_sig[t] is not None:
        if macd_line[t] > macd_sig[t]:
            macd_s = 1.0
            if macd_line[t - 1] is not None and macd_sig[t - 1] is not None and macd_line[t - 1] <= macd_sig[t - 1]:
                macd_s = 2.0  # fresh bullish crossover
        else:
            macd_s = -1.0
            if macd_line[t - 1] is not None and macd_sig[t - 1] is not None and macd_line[t - 1] >= macd_sig[t - 1]:
                macd_s = -2.0  # fresh bearish crossover
        # Histogram momentum (normalised by price)
        if macd_hist[t] is not None and macd_hist[t - 1] is not None:
            hist_accel = macd_hist[t] - macd_hist[t - 1]
            norm = hist_accel / price if price > 0 else 0
            macd_s += max(min(norm * 500, 1.0), -1.0)  # clamp to [-1, +1]
    comp["macd"] = macd_s

    # --- SMA alignment ---
    s20 = ind["s20"]
    s50 = ind["s50"]
    s100 = ind["s100"]
    sma_s = 0.0
    if s20[t] is not None and s50[t] is not None:
        if price > s20[t] > s50[t]:
            sma_s = 1.5
        elif price < s20[t] < s50[t]:
            sma_s = -1.5
        elif price > s20[t]:
            sma_s = 0.5
        elif price < s20[t]:
            sma_s = -0.5
    if s100[t] is not None and s20[t] is not None:
        if s20[t] > s100[t]:
            sma_s += 0.5
        else:
            sma_s -= 0.5
    comp["sma"] = sma_s

    # --- ADX / DI ---
    adx_vals = ind["adx"]
    pdi_vals = ind["pdi"]
    mdi_vals = ind["mdi"]
    adx_s = 0.0
    if adx_vals[t] is not None and pdi_vals[t] is not None and mdi_vals[t] is not None:
        if adx_vals[t] > 25:
            adx_s = 0.5 if pdi_vals[t] > mdi_vals[t] else -0.5
        elif adx_vals[t] < 20:
            adx_s = -0.3
        # Strong trend bonus
        if adx_vals[t] > 40:
            adx_s *= 1.5
    comp["adx"] = adx_s

    # --- Bollinger ---
    bb_upper = ind["bb_upper"]
    bb_mid = ind["bb_mid"]
    bb_lower = ind["bb_lower"]
    bb_s = 0.0
    if bb_upper[t] is not None and bb_lower[t] is not None:
        if price < bb_lower[t]:
            bb_s = 1.5  # oversold at lower band
        elif price > bb_upper[t]:
            bb_s = -1.5  # overbought at upper band
        # Mean reversion signal when near band
        bb_pos = (price - bb_lower[t]) / (bb_upper[t] - bb_lower[t]) if (bb_upper[t] - bb_lower[t]) > 0 else 0.5
        if bb_pos < 0.1:
            bb_s += 0.5
        elif bb_pos > 0.9:
            bb_s -= 0.5
    comp["bollinger"] = bb_s

    # --- Slope ---
    sl = _slope(closes[max(0, t - 19): t + 1])
    comp["slope"] = 2.0 if sl > 0.002 else (-2.0 if sl < -0.002 else
                     (0.5 if sl > 0.0005 else (-0.5 if sl < -0.0005 else 0.0)))

    # --- Stochastic ---
    stoch_k = ind["stoch_k"]
    stoch_d = ind["stoch_d"]
    stoch_s = 0.0
    if stoch_k[t] is not None and stoch_d[t] is not None:
        if stoch_k[t] < 20 and stoch_d[t] < 20:
            stoch_s = 1.5  # oversold
        elif stoch_k[t] > 80 and stoch_d[t] > 80:
            stoch_s = -1.5  # overbought
        # Bullish crossover
        if stoch_k[t] > stoch_d[t] and (stoch_k[t - 1] is not None and stoch_d[t - 1] is not None and stoch_k[t - 1] <= stoch_d[t - 1]):
            stoch_s += 1.0
        elif stoch_k[t] < stoch_d[t] and (stoch_k[t - 1] is not None and stoch_d[t - 1] is not None and stoch_k[t - 1] >= stoch_d[t - 1]):
            stoch_s -= 1.0
    comp["stoch"] = stoch_s

    # --- Williams %R ---
    wr_vals = ind["wr"]
    wr_s = 0.0
    if wr_vals[t] is not None:
        if wr_vals[t] < -80:
            wr_s = 1.0
        elif wr_vals[t] > -20:
            wr_s = -1.0
    comp["williams"] = wr_s

    # --- CCI ---
    cci_vals = ind["cci"]
    cci_s = 0.0
    if cci_vals[t] is not None:
        if cci_vals[t] < -100:
            cci_s = 1.0  # oversold
        elif cci_vals[t] > 100:
            cci_s = -1.0  # overbought
        # Extreme levels
        if cci_vals[t] < -200:
            cci_s = 2.0
        elif cci_vals[t] > 200:
            cci_s = -2.0
    comp["cci"] = cci_s

    # --- Ichimoku ---
    tenkan = ind["tenkan"]
    kijun = ind["kijun"]
    senkou_a = ind["senkou_a"]
    senkou_b = ind["senkou_b"]
    ich_s = 0.0
    if tenkan[t] is not None and kijun[t] is not None:
        # Tenkan/Kijun cross
        if tenkan[t] > kijun[t]:
            ich_s = 1.0
        else:
            ich_s = -1.0
        # Fresh cross
        if tenkan[t - 1] is not None and kijun[t - 1] is not None:
            if tenkan[t] > kijun[t] and tenkan[t - 1] <= kijun[t - 1]:
                ich_s = 1.5
            elif tenkan[t] < kijun[t] and tenkan[t - 1] >= kijun[t - 1]:
                ich_s = -1.5
    # Cloud position
    if senkou_a[t] is not None and senkou_b[t] is not None:
        cloud_top = max(senkou_a[t], senkou_b[t])
        cloud_bot = min(senkou_a[t], senkou_b[t])
        if price > cloud_top:
            ich_s += 0.5
        elif price < cloud_bot:
            ich_s -= 0.5
    comp["ichimoku"] = ich_s

    # --- VWAP ---
    vwap_vals = ind["vwap"]
    vwap_s = 0.0
    if vwap_vals[t] is not None:
        if price > vwap_vals[t] * 1.01:
            vwap_s = 0.5
        elif price < vwap_vals[t] * 0.99:
            vwap_s = -0.5
    comp["vwap"] = vwap_s

    # --- OBV ---
    obv_arr = ind["obv"]
    obv_sl = _slope(obv_arr[max(0, t - 19): t + 1])
    obv_s = 0.0
    if obv_sl > 0.002:
        obv_s = 1.0
    elif obv_sl < -0.002:
        obv_s = -1.0
    comp["obv"] = obv_s

    # --- ROC ---
    roc_vals = ind["roc"]
    roc_s = 0.0
    if roc_vals[t] is not None:
        if roc_vals[t] > 5:
            roc_s = 1.0
        elif roc_vals[t] > 2:
            roc_s = 0.5
        elif roc_vals[t] < -5:
            roc_s = -1.0
        elif roc_vals[t] < -2:
            roc_s = -0.5
    comp["roc"] = roc_s

    # --- Divergences ---
    divergences = detect_divergences(candles, rsi=ind["rsi"],
                                     macd_line=ind["macd_line"], end=t)
    div_s = sum(s for _, s, _ in divergences)
    comp["divergence"] = div_s

    # --- Weighted composite ---
    weighted_total = sum(comp.get(k, 0) * weights.get(k, 1.0) for k in weights)
    max_possible = sum(abs(weights[k]) * 2.0 for k in weights)  # theoretical max

    # Normalise score to roughly [-10, 10] range
    normalised = weighted_total / (max_possible / 20) if max_possible > 0 else 0.0

    # Direction & confidence
    if normalised >= 3.0:
        direction = "UP"
        conf_val = min(abs(normalised) / 10.0, 1.0)
    elif normalised <= -3.0:
        direction = "DOWN"
        conf_val = min(abs(normalised) / 10.0, 1.0)
    else:
        direction = "SIDEWAYS"
        conf_val = max(0.0, 1.0 - abs(normalised) / 3.0) * 0.5

    if conf_val > 0.75:
        conf = "very high"
    elif conf_val > 0.55:
        conf = "high"
    elif conf_val > 0.35:
        conf = "medium"
    else:
        conf = "low"

    return {
        "total": normalised,
        "raw_total": weighted_total,
        "direction": direction,
        "confidence": conf,
        "confidence_value": conf_val,
        "components": comp,
        "weights": weights,
        "regime": regime,
        "patterns": patterns,
        "divergences": divergences,
    }


# ---------------------------------------------------------------------------
# Public API (used by app.py)
# ---------------------------------------------------------------------------

# Expose core helpers so app.py can import them
sma = _sma
ema = _ema
rsi_series = _rsi
atr_series = _atr
adx_series = _adx
macd_series = _macd
bollinger = _bollinger
stochastic = _stochastic
williams_r = _williams_r
cci_series = _cci
ichimoku = _ichimoku
vwap_bands = _vwap_bands
obv_series = _obv
roc_series = _roc
slope = _slope


def compute_signal(candles: list[dict]) -> dict:
    return compute_signal_for_data(candles)


def backtest(candles: list[dict], warmup: int = 60) -> dict:
    """Evaluate signal accuracy over trailing history.

    All indicator series are precomputed ONCE over the full candle list
    (O(n)); each historical step then reads precomputed values at index t
    instead of recomputing every indicator over a truncated slice, so the
    whole backtest is O(n) instead of O(n^2).
    """
    ind = precompute_indicators(candles)
    t0 = time.perf_counter()
    bull_hit = bull_total = bear_hit = bear_total = 0
    correct = total = 0
    results = []
    for t in range(warmup, len(candles) - 1):
        sig = compute_signal_for_data(candles, ind=ind, t=t)
        ret = candles[t + 1]["close"] / candles[t]["close"] - 1.0
        results.append((sig["direction"], ret))
        total += 1
        if sig["direction"] == "UP":
            bull_total += 1
            if ret > 0:
                bull_hit += 1
                correct += 1
        elif sig["direction"] == "DOWN":
            bear_total += 1
            if ret < 0:
                bear_hit += 1
                correct += 1
        else:
            # Sideways: count as correct if within ±0.5%
            if abs(ret) < 0.005:
                correct += 1
    dt = time.perf_counter() - t0
    print(f"  [dev] backtest: {total} signals over {len(candles)} candles "
          f"in {dt:.3f}s", file=sys.stderr)
    return {
        "bull": (bull_hit / bull_total) if bull_total else None,
        "bear": (bear_hit / bear_total) if bear_total else None,
        "bull_n": bull_total, "bear_n": bear_total,
        "overall": (correct / total) if total else None,
        "overall_n": total,
        "n": len(results),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced crypto prediction engine")
    parser.add_argument("--coin", default="BTC",
                        help="coin symbol (e.g. BTC, SOL, XRP; default BTC)")
    parser.add_argument("--tf", choices=list(TF_MAP), default="1d",
                        help="timeframe: 1m, 5m, 15m, 1h, 4h, 1d (default 1d)")
    parser.add_argument("--limit", type=int, default=300,
                        help="candles to fetch (default 300, max 1000)")
    args = parser.parse_args()
    limit = min(max(args.limit, 80), 1000)

    coin = args.coin.strip().upper()
    print(f"Fetching up to {limit} {args.tf} candles for {coin}...")
    candles = fetch_candles(coin, args.tf, limit)
    if len(candles) < 70:
        raise SystemExit(f"Only {len(candles)} candles fetched - need at least 70.")

    closes = [c["close"] for c in candles]
    price = closes[-1]
    atr = _atr(candles)[-1]
    res, sup = swing_points(candles)
    resistance, support = nearest_levels(price, res, sup)

    sig = compute_signal(candles)
    bt = backtest(candles)
    regime = sig["regime"]

    print()
    print(f"{'=' * 60}")
    print(f"  ADVANCED CRYPTO PREDICTOR – {coin} {args.tf}")
    print(f"{'=' * 60}")
    print(f"  Current price:       ${price:,.4f}")
    print(f"  Data window:         {len(candles)} {args.tf} candles")
    print(f"  ATR(14) volatility:  ${atr:,.4f} ({atr / price * 100:.1f}% of price)")

    # Regime
    regime_icons = {"trending": "[UP]", "ranging": "[--]", "volatile": "[!!]", "transitional": "[~~]"}
    print(f"  Market regime:       {regime_icons.get(regime['regime'], '?')} "
          f"{regime['regime'].upper()} (ADX {regime['adx']:.0f}, "
          f"BB width {regime['bandwidth']:.3f})")
    print()

    # Candlestick patterns
    patterns = sig["patterns"]
    if patterns:
        print("Candlestick patterns:")
        for name, s, desc in patterns:
            tag = "BULL" if s > 0 else ("BEAR" if s < 0 else "neut")
            print(f"  [{tag:>4}] {name}: {desc}")
    print()

    # Divergences
    divergences = sig["divergences"]
    if divergences:
        print("Divergences detected:")
        for name, s, desc in divergences:
            tag = "BULL" if s > 0 else "BEAR"
            print(f"  [{tag:>4}] {name}: {desc}")
    print()

    # Indicators
    rsi = rsi_series(closes, 14)[-1]
    macd_line, macd_sig, macd_hist = macd_series(closes)
    s20, s50 = sma(closes, 20), sma(closes, 50)
    e20, e50, e200 = ema(closes, 20), ema(closes, 50), ema(closes, 200)
    adx, pdi, mdi = adx_series(candles)
    bb_upper, bb_mid, bb_lower = bollinger(closes)
    stoch_k, stoch_d = stochastic(candles)
    wr = williams_r(candles)
    cci = cci_series(candles)
    tenkan, kijun, senkou_a, senkou_b = ichimoku(candles)
    vwap = vwap_bands(candles)

    print("Indicator dashboard:")
    print(f"  RSI(14):          {rsi:.0f}" if rsi is not None else "  RSI(14):          n/a")
    print(f"  Stochastic K/D:   {stoch_k[-1]:.0f}/{stoch_d[-1]:.0f}" if stoch_k[-1] is not None else "  Stochastic:       n/a")
    print(f"  Williams %R:      {wr[-1]:.0f}" if wr[-1] is not None else "  Williams %R:      n/a")
    print(f"  CCI(20):          {cci[-1]:.0f}" if cci[-1] is not None else "  CCI(20):          n/a")
    print(f"  MACD:             {'bullish' if macd_hist[-1] and macd_hist[-1] > 0 else 'bearish'} "
          f"(hist {macd_hist[-1]:+.4f})" if macd_hist[-1] is not None else "  MACD:             n/a")
    if adx[-1] is not None:
        trend_state = "trending" if adx[-1] > 25 else "choppy" if adx[-1] < 20 else "transitional"
        print(f"  ADX(14):          {adx[-1]:.0f} ({trend_state})  +DI {pdi[-1]:.0f} / -DI {mdi[-1]:.0f}")
    if tenkan[-1] is not None and kijun[-1] is not None:
        cloud_bias = "above cloud" if price > max(tenkan[-1] or 0, kijun[-1] or 0) else "below cloud"
        print(f"  Ichimoku:         Tenkan {tenkan[-1]:,.2f} / Kijun {kijun[-1]:,.2f} ({cloud_bias})")
    if s20[-1] is not None and s50[-1] is not None:
        trend = "uptrend" if price > s20[-1] > s50[-1] else "downtrend" if price < s20[-1] < s50[-1] else "mixed"
        print(f"  SMA alignment:    {trend} (SMA20 ${s20[-1]:,.4f}, SMA50 ${s50[-1]:,.4f})")
    if vwap[-1] is not None:
        vwap_rel = "above" if price > vwap[-1] else "below"
        print(f"  VWAP(20):         ${vwap[-1]:,.4f} ({vwap_rel})")
    va = volume_analysis(candles)
    if va:
        vol_tag = "rising" if va["trend"] > 0.1 else "falling" if va["trend"] < -0.1 else "flat"
        print(f"  Volume:           {vol_tag} ({va['trend'] * 100:+.0f}%)  OBV {'rising' if va['obv_slope'] > 0.001 else 'falling' if va['obv_slope'] < -0.001 else 'flat'}"
              + ("  [CLIMAX]" if va["climax"] else ""))
    print()

    # Key levels
    print("Key levels:")
    print(f"  Resistance:      ${resistance:,.4f}" if resistance else "  Resistance:      none near")
    print(f"  Support:         ${support:,.4f}" if support else "  Support:         none near")
    if atr:
        print(f"  Expected range:  ${price - atr:,.4f} – ${price + atr:,.4f} (+/- 1 ATR)")
    print()

    # Signal breakdown
    print("Signal breakdown (weighted by regime):")
    labels = {
        "patterns": "Patterns", "rsi": "RSI", "macd": "MACD",
        "sma": "SMA trend", "adx": "ADX/DI", "bollinger": "Bollinger",
        "slope": "Slope", "stoch": "Stochastic", "williams": "Williams %R",
        "cci": "CCI", "ichimoku": "Ichimoku", "vwap": "VWAP",
        "obv": "OBV", "roc": "ROC", "divergence": "Divergence",
    }
    for k, v in sig["components"].items():
        w = sig["weights"].get(k, 1.0)
        print(f"  {labels.get(k, k):<12} {v:+.2f}  (wt {w:.1f})  =  {v * w:+.2f}")
    print(f"  {'WEIGHTED':<12} {sig['raw_total']:+.2f}  ->  normalised: {sig['total']:+.2f}")
    print()

    # Verdict
    print(f"{'=' * 60}")
    print(f"  >>> PREDICTION: {sig['direction']}  (confidence: {sig['confidence']}) <<<")
    print(f"{'=' * 60}")
    if sig["direction"] == "SIDEWAYS":
        print("  Expect range-bound action; no strong directional bias.")
    elif sig["direction"] == "UP":
        print(f"  Bullish bias. Watch resistance at ${resistance:,.4f}" if resistance else "  Bullish bias.")
    else:
        print(f"  Bearish bias. Watch support at ${support:,.4f}" if support else "  Bearish bias.")
    print()

    # Backtest
    print("Backtest (signal accuracy over trailing period):")
    if bt["bull_n"]:
        print(f"  Bullish calls: {bt['bull_n']}, accuracy: {bt['bull'] * 100:.0f}%")
    if bt["bear_n"]:
        print(f"  Bearish calls: {bt['bear_n']}, accuracy: {bt['bear'] * 100:.0f}%")
    if bt["overall"] is not None:
        print(f"  Overall accuracy: {bt['overall'] * 100:.0f}% ({bt['overall_n']} signals)")
    print()
    print("Disclaimer: advanced TA is probabilistic, not predictive.")
    print("Past backtest performance does not guarantee future results.")


if __name__ == "__main__":
    main()
