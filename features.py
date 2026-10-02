#!/usr/bin/env python3
"""
features.py
-----------
Causal feature engineering + labeling for the ML next-candle classifier.

STRICT no-lookahead rule
------------------------
The feature row at candle index ``t`` uses ONLY information from candles
``[0 .. t]`` - never candle ``t+1`` or later. Implementation strategy:

* Every indicator series is computed ONCE over the whole history with
  ``predict.precompute_indicators()``. Every value in those arrays is causal
  (the value at index ``t`` depends only on ``candles[:t+1]``), and we only
  *read* index ``t`` / ``[t-k .. t]`` - identical to the O(n) backtest that
  predict.py already uses.
* Higher-timeframe context bars are aggregated from the *same* candle stream,
  and a higher-TF bar only enters the picture at ``t`` once it has fully
  closed (its close time <= close time of candle ``t``).
* A self-test (``python features.py --selftest``) proves the rule: features
  at a fixed ``t`` are unchanged after the future candles are corrupted.

Labeling (volatility-adjusted deadzone)
---------------------------------------
* forward return ``fwd_ret = close[t+1]/close[t] - 1``
* deadzone ``= 0.5 * ATR(14)[t]/close[t]``  -> tiny noise moves are NOT
  labeled as directional
* UP if ``fwd_ret > +deadzone``, DOWN if ``< -deadzone``, else SIDEWAYS.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import predict as engine
import altdata
from build_dataset import SYMBOLS, TIMEFRAMES, TF_MS, load_candles

# ---------------------------------------------------------------------------
# Timeframe wiring
# ---------------------------------------------------------------------------

# Higher timeframes aggregated from the base stream for MTF context.
# A "1h" model sees 4h + 1d context; a "4h" model sees 1d context; a "1d"
# model has no higher timeframe available in-app (features stay NaN -> the
# model simply learns to ignore them; they are dropped from its feature list).
HIGHER_TFS: dict[str, list[str]] = {"1h": ["4h", "1d"], "4h": ["1d"], "1d": []}

FEATURES_DIR = Path(__file__).resolve().parent / "data" / "features"

# Skip the first `WARMUP` bars of every series: enough room for SMA(100),
# Ichimoku(52), ADX(28), S/R swing detection, 60-bar windows etc.
WARMUP = 160

EPS = 1e-12


def _f(x) -> float:
    """None -> NaN so downstream pandas/numpy treat warmup as missing."""
    return float("nan") if x is None else float(x)


# ---------------------------------------------------------------------------
# Higher-timeframe aggregation
# ---------------------------------------------------------------------------

def aggregate_candles(candles: list[dict], target_ms: int) -> list[dict]:
    """Aggregate a base candle stream into higher-TF OHLCV candles.

    Bins are aligned to UTC epoch boundaries (identical to Binance bars).
    Only *complete* bins are returned: a bin counts as complete when it
    contains the full ``target_ms/base_ms`` candle count, so a trailing
    partial (still-forming) bar is never included.
    """
    if not candles:
        return []
    base_ms = candles[1]["ts"] - candles[0]["ts"] if len(candles) > 1 else TF_MS["1h"]
    if base_ms <= 0:
        base_ms = TF_MS["1h"]
    per_bin = max(1, round(target_ms / base_ms))
    out: list[dict] = []
    i = 0
    n = len(candles)
    while i < n:
        bin_start = (candles[i]["ts"] // target_ms) * target_ms
        o = h = l = c = candles[i]["close"]
        v = 0.0
        cnt = 0
        j = i
        while j < n and candles[j]["ts"] < bin_start + target_ms:
            row = candles[j]
            if cnt == 0:
                o = row["open"]
            h = row["high"] if cnt == 0 or row["high"] > h else h
            l = row["low"] if cnt == 0 or row["low"] < l else l
            c = row["close"]
            v += row.get("vol", 0.0)
            cnt += 1
            j += 1
        i = j
        if cnt == per_bin:      # only completed bins
            out.append({"ts": bin_start, "open": o, "high": h,
                        "low": l, "close": c, "vol": v})
    return out


def _ht_context(ht_candles: list[dict], target_ms: int) -> dict[str, np.ndarray]:
    """Indicator arrays on an aggregated higher-TF stream.

    Each array value is causal *per bar*; callers additionally restrict which
    bar index is visible at base-candle ``t`` via close-time alignment.
    """
    n = len(ht_candles)
    if n == 0:
        return {}
    closes = np.array([c["close"] for c in ht_candles], dtype=float)
    nan = np.full(n, np.nan)
    rsi = np.array([_f(x) for x in engine._rsi(list(closes), 14)], dtype=float)
    adx, pdi, mdi = engine._adx(ht_candles)
    adx_a = np.array([_f(x) for x in adx], dtype=float)
    pdi_a = np.array([_f(x) for x in pdi], dtype=float)
    mdi_a = np.array([_f(x) for x in mdi], dtype=float)
    atr_a = np.array([_f(x) for x in engine._atr(ht_candles, 14)], dtype=float)
    s50 = np.array([_f(x) for x in engine._sma(list(closes), 50)], dtype=float)
    di = np.where(np.isfinite(pdi_a) & np.isfinite(mdi_a), pdi_a - mdi_a, nan)
    return {
        "rsi": rsi,
        # trend direction: how far price sits above/below the 50-bar mean
        "trend": np.where(np.isfinite(s50),
                          (closes - s50) / np.maximum(s50, EPS), nan),
        "adx": adx_a,
        "di": di,
        "atr_pct": np.where(np.isfinite(atr_a), atr_a / np.maximum(closes, EPS), nan),
        # close time (ms) of each bar = bin open time + bar span
        "close_times": np.array([c["ts"] + target_ms for c in ht_candles], dtype=np.int64),
    }


def feature_names() -> list[str]:
    """Deterministic feature column order (must match live inference)."""
    return [
        # returns / momentum
        "ret_1", "ret_2", "ret_4", "ret_8", "ret_16",
        # realized volatility (per-bar log-return std over windows)
        "rv_7", "rv_14", "rv_30",
        # volatility regime
        "atr_pct", "bb_width", "bb_pos",
        # oscillators & trend
        "rsi14", "stoch_k", "stoch_d", "stoch_kd", "willr", "cci",
        "macd_line_rel", "macd_hist_rel", "macd_accel_rel", "slope20",
        "sma20_dist", "sma50_dist", "sma100_dist",
        "sma20_50_rel", "sma50_100_rel",
        # ichimoku / vwap
        "cloud_top_dist", "cloud_bot_dist", "tenkan_kijun_rel", "vwap_dist",
        # volume
        "vol_rel20", "vol_trend", "obv_slope", "updown_ratio",
        # candle shape
        "body_ratio", "up_wick_ratio", "low_wick_ratio",
        "close_pos_range", "body_dir",
        # pattern / divergence / structure
        "pat_score", "pat_score_5", "bull_div", "bear_div",
        "dist_support", "dist_resistance",
        # time-of-day / day-of-week (UTC)
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        # ---- derivatives alt-data: funding rate (trainable, backtestable) ----
        # Binance funding has full history, so these are honest features.
        # Open interest / order-book imbalance are live-only (see altdata.py):
        # they are surfaced at inference, never fabricated into history.
        "fund_rate", "fund_sum_24h", "fund_z30", "fund_trend",
        # ---- higher-timeframe context (all-NaN when not applicable) ----
        "htf1_rsi", "htf1_trend", "htf1_adx", "htf1_di", "htf1_atr_pct",
        "htf2_rsi", "htf2_trend", "htf2_adx", "htf2_di", "htf2_atr_pct",
    ]


def build_feature_frame(candles: list[dict], tf: str, symbol: str,
                        include_labels: bool = True,
                        min_t: int | None = None) -> pd.DataFrame:
    """Compute the full causal feature frame for one (symbol, tf) series.

    Row ``t`` holds the state known at the close of candle ``t`` and the
    label for candle ``t+1``. Returns a DataFrame with columns:
    ts, symbol, tf, <feature names>, and (when include_labels) fwd_ret,
    deadzone_frac, label.
    """
    if min_t is None:
        min_t = WARMUP
    n = len(candles)
    if n < min_t + 2:
        return pd.DataFrame()

    ind = engine.precompute_indicators(candles)
    closes = np.array(ind["closes"], dtype=float)
    ts_arr = np.array([c["ts"] for c in candles], dtype=np.int64)

    # ---- derivatives alt-data: funding (causal) ---------------------------
    # Loaded from the local parquet cache only (never a network call here).
    # Empty history -> all funding features stay NaN, which downstream
    # imputation/GBM handles; the cache is topped up by altdata.py / retrain.
    funding_df = altdata.load_cached_funding(symbol)
    fund_feats = altdata.funding_features(ts_arr, funding_df, TF_MS[tf])
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows = np.array([c["low"] for c in candles], dtype=float)
    opens = np.array([c["open"] for c in candles], dtype=float)
    vols = np.array([c.get("vol", 0.0) for c in candles], dtype=float)

    def arr(key: str) -> np.ndarray:
        return np.array([_f(x) for x in ind[key]], dtype=float)

    rsi = arr("rsi")
    macd_l, macd_h = arr("macd_line"), arr("macd_hist")
    s20, s50, s100 = arr("s20"), arr("s50"), arr("s100")
    bb_u, bb_m, bb_l = arr("bb_upper"), arr("bb_mid"), arr("bb_lower")
    atr = arr("atr")
    sk, sd = arr("stoch_k"), arr("stoch_d")
    wr = arr("wr")
    cci = arr("cci")
    tk, kj = arr("tenkan"), arr("kijun")
    sa, sb = arr("senkou_a"), arr("senkou_b")
    vwap = arr("vwap")
    obv = arr("obv")
    adx_v = arr("adx")  # noqa: F841  (kept for future use)

    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(closes[1:] / np.maximum(closes[:-1], EPS))

    # ---- higher timeframe context (aggregated from the same stream) --------
    base_ms = TF_MS[tf]
    ht_data: dict[str, dict] = {}
    if tf in HIGHER_TFS:
        for i, htf in enumerate(HIGHER_TFS[tf], start=1):
            ht_candles = aggregate_candles(candles, TF_MS[htf])
            ctx = _ht_context(ht_candles, TF_MS[htf])
            if ctx:
                close_times = ctx.pop("close_times")
                # last ht bar with close time <= base close time of candle t
                ctx["_idx"] = np.searchsorted(close_times, ts_arr + base_ms,
                                              side="right") - 1
                ht_data[f"htf{i}"] = ctx

    # ---- vectorized helpers --------------------------------------------------
    def roll_ret(k: int) -> np.ndarray:
        out = np.full(n, np.nan)
        out[k:] = np.log(closes[k:] / np.maximum(closes[:-k], EPS))
        return out

    def roll_std(k: int) -> np.ndarray:
        return pd.Series(log_ret).rolling(k).std().to_numpy()

    ret = {k: roll_ret(k) for k in (1, 2, 4, 8, 16)}
    rv = {k: roll_std(k) for k in (7, 14, 30)}

    vser = pd.Series(vols)
    vol_ma20 = vser.rolling(20).mean().to_numpy()
    vol_ma5 = vser.rolling(5).mean().to_numpy()
    vol_ma_prev20 = vser.shift(5).rolling(20).mean().to_numpy()

    obv_slope = np.full(n, np.nan)
    for t in range(19, n):
        obv_slope[t] = engine._slope(list(obv[t - 19: t + 1]))

    # ---- per-index candlestick pattern score (computed once, O(n*5)) --------
    pat = np.zeros(n)
    rsi_list = list(rsi)
    macd_list = list(macd_l)
    for u in range(4, n):
        sma_u = float(s20[u]) if np.isfinite(s20[u]) else None
        pat[u] = sum(s for _, s, _ in engine.detect_patterns(
            candles[max(0, u - 4): u + 1], sma20=sma_u))
    pat_cum = np.concatenate([[0.0], np.cumsum(pat)])

    # ---- swing support/resistance (causal, confirmed pivots) -----------------
    # A pivot at index j needs look = 3 bars on each side, so it is only
    # "confirmed" once j + 3 <= t. Mirrors predict.swing_points semantics:
    # resistance = nearest confirmed swing high above price, falling back to
    # the highest recent swing high; support symmetric.
    look = 3
    is_max = np.zeros(n, dtype=bool)
    is_min = np.zeros(n, dtype=bool)
    for i in range(look, n - look):
        is_max[i] = highs[i] >= highs[i - look: i + look + 1].max()
        is_min[i] = lows[i] <= lows[i - look: i + look + 1].min()
    dist_sup = np.full(n, np.nan)
    dist_res = np.full(n, np.nan)
    for t in range(max(60, 2 * look + 1), n):
        price = closes[t]
        lo = max(t - 59, look)
        res, sup = [], []
        for j in range(lo, t - look + 1):
            if is_max[j]:
                res.append(highs[j])
            if is_min[j]:
                sup.append(lows[j])
        above = [r for r in res if r > price * 1.001]
        below = [s for s in sup if s < price * 0.999]
        resistance = min(above) if above else (max(res) if res else None)
        support = max(below) if below else (min(sup) if sup else None)
        if resistance is not None and price > 0:
            dist_res[t] = (resistance - price) / price
        if support is not None and price > 0:
            dist_sup[t] = (price - support) / price

    # ---- per-row assembly -----------------------------------------------------
    rows: list[dict] = []
    last_idx = n - 2 if include_labels else n - 1
    up_vol = np.where(closes > opens, vols, 0.0)
    dn_vol = np.where(closes < opens, vols, 0.0)

    for t in range(min_t, last_idx + 1):
        c_t = closes[t]
        rng = highs[t] - lows[t]
        body = abs(closes[t] - opens[t])
        row: dict = {"ts": int(ts_arr[t]), "symbol": symbol, "tf": tf}

        row["ret_1"] = ret[1][t]
        row["ret_2"] = ret[2][t]
        row["ret_4"] = ret[4][t]
        row["ret_8"] = ret[8][t]
        row["ret_16"] = ret[16][t]
        row["rv_7"] = rv[7][t]
        row["rv_14"] = rv[14][t]
        row["rv_30"] = rv[30][t]
        row["atr_pct"] = atr[t] / c_t if c_t > 0 else np.nan
        row["bb_width"] = ((bb_u[t] - bb_l[t]) / bb_m[t]
                           if np.isfinite(bb_u[t]) and bb_m[t] > 0 else np.nan)
        row["bb_pos"] = ((c_t - bb_l[t]) / (bb_u[t] - bb_l[t])
                         if np.isfinite(bb_u[t]) and bb_u[t] > bb_l[t] else np.nan)
        row["rsi14"] = rsi[t]
        row["stoch_k"] = sk[t]
        row["stoch_d"] = sd[t]
        row["stoch_kd"] = (sk[t] - sd[t]
                           if np.isfinite(sk[t]) and np.isfinite(sd[t]) else np.nan)
        row["willr"] = wr[t] / 100.0 if np.isfinite(wr[t]) else np.nan
        row["cci"] = cci[t]
        row["macd_line_rel"] = macd_l[t] / c_t if c_t > 0 else np.nan
        row["macd_hist_rel"] = macd_h[t] / c_t if c_t > 0 else np.nan
        row["macd_accel_rel"] = ((macd_h[t] - macd_h[t - 1]) / c_t
                                 if (t >= 1 and c_t > 0 and np.isfinite(macd_h[t])
                                     and np.isfinite(macd_h[t - 1])) else np.nan)
        row["slope20"] = engine._slope(list(closes[t - 19: t + 1]))
        row["sma20_dist"] = (c_t - s20[t]) / c_t if (np.isfinite(s20[t]) and c_t > 0) else np.nan
        row["sma50_dist"] = (c_t - s50[t]) / c_t if (np.isfinite(s50[t]) and c_t > 0) else np.nan
        row["sma100_dist"] = (c_t - s100[t]) / c_t if (np.isfinite(s100[t]) and c_t > 0) else np.nan
        row["sma20_50_rel"] = ((s20[t] - s50[t]) / c_t
                               if (np.isfinite(s20[t]) and np.isfinite(s50[t]) and c_t > 0) else np.nan)
        row["sma50_100_rel"] = ((s50[t] - s100[t]) / c_t
                                if (np.isfinite(s50[t]) and np.isfinite(s100[t]) and c_t > 0) else np.nan)
        if np.isfinite(sa[t]) and np.isfinite(sb[t]):
            cloud_top, cloud_bot = max(sa[t], sb[t]), min(sa[t], sb[t])
        else:
            cloud_top = cloud_bot = np.nan
        row["cloud_top_dist"] = (c_t - cloud_top) / c_t if (np.isfinite(cloud_top) and c_t > 0) else np.nan
        row["cloud_bot_dist"] = (c_t - cloud_bot) / c_t if (np.isfinite(cloud_bot) and c_t > 0) else np.nan
        row["tenkan_kijun_rel"] = ((tk[t] - kj[t]) / c_t
                                   if (np.isfinite(tk[t]) and np.isfinite(kj[t]) and c_t > 0) else np.nan)
        row["vwap_dist"] = (c_t - vwap[t]) / c_t if (np.isfinite(vwap[t]) and c_t > 0) else np.nan
        row["vol_rel20"] = (vols[t] / vol_ma20[t] - 1.0
                            if np.isfinite(vol_ma20[t]) and vol_ma20[t] > 0 else np.nan)
        row["vol_trend"] = ((vol_ma5[t] / vol_ma_prev20[t] - 1.0)
                            if (np.isfinite(vol_ma5[t]) and np.isfinite(vol_ma_prev20[t])
                                and vol_ma_prev20[t] > 0) else np.nan)
        row["obv_slope"] = obv_slope[t]
        uv, dv = float(up_vol[t - 19: t + 1].sum()), float(dn_vol[t - 19: t + 1].sum())
        row["updown_ratio"] = (uv - dv) / (uv + dv) if (uv + dv) > 0 else 0.0

        if rng > 0:
            row["body_ratio"] = body / rng
            row["up_wick_ratio"] = (highs[t] - max(opens[t], closes[t])) / rng
            row["low_wick_ratio"] = (min(opens[t], closes[t]) - lows[t]) / rng
            row["close_pos_range"] = (closes[t] - lows[t]) / rng
        else:
            row["body_ratio"] = np.nan
            row["up_wick_ratio"] = np.nan
            row["low_wick_ratio"] = np.nan
            row["close_pos_range"] = 0.5
        row["body_dir"] = (closes[t] - opens[t]) / opens[t] if opens[t] > 0 else np.nan

        row["pat_score"] = pat[t]
        row["pat_score_5"] = pat_cum[t + 1] - pat_cum[max(0, t - 4)]

        divs = engine.detect_divergences(candles, rsi=rsi_list,
                                         macd_line=macd_list, end=t)
        row["bull_div"] = 1.0 if any("Bullish" in d[0] for d in divs) else 0.0
        row["bear_div"] = 1.0 if any("Bearish" in d[0] for d in divs) else 0.0
        row["dist_support"] = dist_sup[t]
        row["dist_resistance"] = dist_res[t]

        dt = datetime.fromtimestamp(ts_arr[t] / 1000, tz=timezone.utc)
        frac = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
        row["hour_sin"] = math.sin(2 * math.pi * frac)
        row["hour_cos"] = math.cos(2 * math.pi * frac)
        frac_dow = dt.weekday() / 7.0
        row["dow_sin"] = math.sin(2 * math.pi * frac_dow)
        row["dow_cos"] = math.cos(2 * math.pi * frac_dow)

        # funding features: value of the last funding payment settled by the
        # close of candle t (see altdata.funding_features for the alignment).
        for k in altdata.FUNDING_FEATURES:
            row[k] = float(fund_feats[k][t]) if k in fund_feats else np.nan

        for i in (1, 2):
            key = f"htf{i}"
            sub = ht_data.get(key)
            if sub is not None:
                j = int(sub["_idx"][t])
                if j < 0:
                    vals = dict.fromkeys(("rsi", "trend", "adx", "di", "atr_pct"), np.nan)
                else:
                    vals = {k2: float(sub[k2][j])
                            for k2 in ("rsi", "trend", "adx", "di", "atr_pct")}
            else:
                vals = dict.fromkeys(("rsi", "trend", "adx", "di", "atr_pct"), np.nan)
            for k2 in ("rsi", "trend", "adx", "di", "atr_pct"):
                row[f"{key}_{k2}"] = vals[k2]

        if include_labels and t <= n - 2:
            fwd = closes[t + 1] / closes[t] - 1.0 if closes[t] > 0 else np.nan
            dead = 0.5 * atr[t] / closes[t] if (closes[t] > 0 and np.isfinite(atr[t])) else np.nan
            if np.isfinite(fwd) and np.isfinite(dead):
                row["fwd_ret"] = fwd
                row["deadzone_frac"] = dead
                row["label"] = "UP" if fwd > dead else ("DOWN" if fwd < -dead else "SIDEWAYS")
            else:
                row["fwd_ret"] = np.nan
                row["deadzone_frac"] = np.nan
                row["label"] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    meta = ["ts", "symbol", "tf"]
    label_cols = ["fwd_ret", "deadzone_frac", "label"]
    cols = meta + feature_names() + label_cols
    return df[[c for c in cols if c in df.columns]]


def label_distribution(df: pd.DataFrame) -> pd.Series:
    return df["label"].value_counts(normalize=True)


# ---------------------------------------------------------------------------
# No-lookahead self test
# ---------------------------------------------------------------------------

def _synth_candles(n: int = 600, seed: int = 7, step_ms: int = 3_600_000) -> list[dict]:
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    candles = []
    for i in range(n):
        o = closes[i - 1] if i else closes[i]
        c = closes[i]
        h = max(o, c) * (1 + abs(rng.normal(0, 0.004)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.004)))
        candles.append({"ts": int(1_700_000_000_000 + i * step_ms),
                        "open": float(o), "high": float(h), "low": float(l),
                        "close": float(c), "vol": float(rng.uniform(50, 5000))})
    return candles


def selftest() -> None:
    """Prove no feature at row t depends on candles after t."""
    print("Running no-lookahead self-test…")
    for tf, step in (("1h", 3_600_000), ("4h", 14_400_000), ("1d", 86_400_000)):
        candles = _synth_candles(700, step_ms=step)
        df_a = build_feature_frame(candles, tf, "BTC", include_labels=True)
        assert not df_a.empty, f"{tf}: empty frame"

        t_test = 300
        future = [dict(c) for c in candles]
        rng = np.random.default_rng(99)
        for i in range(t_test + 1, len(future)):
            c = dict(future[i])
            scale = rng.uniform(0.4, 2.5)
            c["open"] *= scale
            c["close"] *= scale
            c["high"] = max(c["open"], c["close"]) * 1.02
            c["low"] = min(c["open"], c["close"]) * 0.98
            future[i] = c
        df_b = build_feature_frame(future, tf, "BTC", include_labels=True)

        row_a = df_a[df_a["ts"] == candles[t_test]["ts"]].iloc[0]
        row_b = df_b[df_b["ts"] == future[t_test]["ts"]].iloc[0]
        for col in feature_names():
            va, vb = row_a[col], row_b[col]
            ok = (pd.isna(va) and pd.isna(vb)) or (
                not pd.isna(va) and not pd.isna(vb)
                and abs(float(va) - float(vb)) < 1e-9)
            assert ok, (f"{tf} col {col} changed after future corruption: "
                        f"{va} -> {vb}")
        if not altdata.funding_cache_path("BTC").exists():
            for col in feature_names():
                if col.startswith("fund_"):
                    assert df_a[col].isna().all(), (
                        f"{col} must be all-NaN when no funding cache exists")
        print(f"  [ok] {tf}: {len(df_a)} rows; features at t={t_test} unchanged "
              f"after future corruption")

    # funding alignment self-test: a payment settling exactly at a bar's
    # close time is known AT that close (inclusive semantics, matches Binance
    # where funding lands on candle boundaries), so bar t carries fr[t].
    step = 3_600_000
    n = 100
    candles = _synth_candles(n, seed=3, step_ms=step)
    ft = np.array([candles[i]["ts"] + step for i in range(n)], dtype=np.int64)
    fr = np.array([0.0001 * (i + 1) for i in range(n)], dtype=float)
    funding = pd.DataFrame({"funding_time": ft, "fund_rate": fr})
    ts_arr = np.array([c["ts"] for c in candles], dtype=np.int64)
    feats = altdata.funding_features(ts_arr, funding, step)
    assert int(np.isfinite(feats["fund_rate"]).sum()) == n, (
        "every bar's close knows the payment settled at that instant")
    assert feats["fund_rate"][0] == fr[0] and feats["fund_rate"][5] == fr[5], (
        "bar t must carry the payment settled by its close")

    # stricter: payments settling one full bar LATER must never leak backwards
    ft_late = ft + step                      # payment i settles at close of bar i+1
    feats_late = altdata.funding_features(
        ts_arr, pd.DataFrame({"funding_time": ft_late, "fund_rate": fr}), step)
    assert feats_late["fund_rate"][0] != feats_late["fund_rate"][0], (
        "bar 0 closes before the first late payment -> must be NaN")
    assert feats_late["fund_rate"][1] == fr[0] and feats_late["fund_rate"][5] == fr[4], (
        "bar t must carry the last payment settled strictly before its close")
    print("  [ok] funding: causal settlement alignment verified")

    # labeling sanity: monotonic strong drift -> everything must be UP.
    # Drift per bar (1.0) is 50x the ATR (~0.02), far beyond the deadzone.
    n = 400
    candles = [{"ts": int(1_700_000_000_000 + i * 86_400_000),
                "open": 100 + i * 1.0, "high": 100 + i * 1.0 + 0.02,
                "low": 100 + i * 1.0 - 0.02, "close": 100 + (i + 1) * 1.0,
                "vol": 1000.0} for i in range(n)]
    df = build_feature_frame(candles, "1d", "BTC")
    non_null = df[df["label"].notna()]
    assert (non_null["label"] == "UP").all(), "monotonic drift should label all UP"
    print(f"  [ok] labeling: monotonic drift -> 100% UP ({len(non_null)} rows)")

    df2 = build_feature_frame(_synth_candles(2000), "1d", "BTC")
    print(f"  [ok] noise data label mix: {label_distribution(df2).to_dict()}")
    print("Self-test passed.")


# ---------------------------------------------------------------------------
# Feature cache builder (per timeframe, all coins pooled)
# ---------------------------------------------------------------------------

def _prefetch_funding(symbols: list[str]) -> None:
    """Top up the funding cache for all symbols, sequentially, in the parent
    process BEFORE the parallel feature build starts.

    Doing it here (not inside the workers) avoids several processes writing
    the same funding parquet concurrently, and keeps network access in one
    place. On failure we fall back to whatever is cached - features just
    carry fewer non-NaN funding values.
    """
    for sym in symbols:
        try:
            df = altdata.load_or_fetch_funding(sym, start_ms=0, refresh=True)
            if df.empty:
                print(f"  [warn] {sym}: no funding history - features will be "
                      f"NaN (offline or no perp)", file=sys.stderr)
            else:
                last = datetime.fromtimestamp(df["funding_time"].max() / 1000,
                                              tz=timezone.utc)
                print(f"  [ok] {sym}: funding cache to {last:%Y-%m-%d} "
                      f"({len(df):,} payments)")
        except Exception as e:
            print(f"  [warn] {sym}: funding top-up failed ({e}) - using cache",
                  file=sys.stderr)


def _build_one_pair(args) -> tuple[str, str, pd.DataFrame]:
    """Worker for the parallel cache builder (picklable top-level fn)."""
    symbol, tf = args
    # Funding was already topped up by _prefetch_funding(); the feature build
    # itself is strictly cache-only (no network in workers).
    candles = load_candles(symbol, tf)
    df = build_feature_frame(candles, tf, symbol, include_labels=True)
    df = df[df["label"].notna()]
    return symbol, tf, df


def build_feature_cache(tfs: list[str] | None = None,
                        symbols: list[str] | None = None) -> dict[str, Path]:
    """Build data/features/features_{tf}.parquet from the raw candle cache.

    Returns {tf: path}. Rows whose label is missing (last bar / warmup) are
    dropped - a supervised label must exist. Per-(coin, tf) frames are built
    in parallel across CPU cores.
    """
    import concurrent.futures
    import os

    tfs = tfs or TIMEFRAMES
    symbols = symbols or SYMBOLS
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    print("topping up funding caches (sequential, network)…")
    _prefetch_funding(symbols)
    pairs = [(sym, tf) for tf in tfs for sym in symbols]
    done: dict[tuple[str, str], pd.DataFrame] = {}
    workers = min(8, os.cpu_count() or 1)
    print(f"building feature frames ({len(pairs)} coin x tf pairs, "
          f"{workers} workers)...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_build_one_pair, p): p for p in pairs}
        for fut in concurrent.futures.as_completed(futs):
            p = futs[fut]
            try:
                symbol, tf, df = fut.result()
            except FileNotFoundError:
                print(f"  [skip {p[0]} {p[1]}: no raw cache - run "
                      f"build_dataset.py first]", file=sys.stderr)
                continue
            except Exception as e:
                print(f"  [fail {p[0]} {p[1]}: {e}]", file=sys.stderr)
                continue
            done[p] = df
            print(f"  [ok] {p[0]} {p[1]}: {len(df):,} labeled rows")

    out: dict[str, Path] = {}
    for tf in tfs:
        frames = [df for (sym, _tf), df in done.items() if _tf == tf]
        if frames:
            all_df = pd.concat(frames, ignore_index=True)
            path = FEATURES_DIR / f"features_{tf}.parquet"
            all_df.to_parquet(path, index=False, engine="pyarrow")
            out[tf] = path
            print(f"[ok] {tf}: {len(all_df):,} rows -> {path}")
    return out


def load_feature_cache(tf: str) -> pd.DataFrame:
    path = FEATURES_DIR / f"features_{tf}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"no feature cache at {path} - run `python features.py --cache` "
            f"(run `python build_dataset.py` first if raw candles are missing)")
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal ML features for crypto candles")
    parser.add_argument("--cache", action="store_true",
                        help="build per-timeframe feature caches from raw data")
    parser.add_argument("--tfs", default=",".join(TIMEFRAMES))
    parser.add_argument("--coins", default=",".join(SYMBOLS))
    parser.add_argument("--selftest", action="store_true",
                        help="run the no-lookahead + labeling self test")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.cache:
        tfs = [t for t in args.tfs.split(",") if t]
        syms = [s.strip().upper() for s in args.coins.split(",") if s.strip()]
        build_feature_cache(tfs=tfs, symbols=syms)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
