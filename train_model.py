#!/usr/bin/env python3
"""
train_model.py
--------------
Trains and honestly evaluates the classical-ML DIRECTIONAL PROBABILITY model
(calibrated P(UP) and P(DOWN) for the next candle) that replaces the hand-tuned
rule-based scoring in predict.py.

Framing (deliberately honest about what is achievable)
------------------------------------------------------
A 70-80% 3-class accuracy on daily crypto bars from OHLCV + public TA is NOT
achievable - that is an information-theoretic ceiling, not a tuning problem -
and raw 3-class accuracy is further inflated by majority-class collapse (the
SIDEWAYS class is ~2/3 of bars). So this pipeline does not chase accuracy:

1. Target: two BINARY probabilities, calibrated - P(UP) and P(DOWN) for the
   next candle (N=1 bar horizon). The 3-class view is derived from them for
   reference only.
2. Primary metrics: Brier score / Brier skill score vs climatology and
   ROC-AUC on the held-out test set. Accuracy is reported only as a
   reference, with the majority-class baseline next to it.
3. Confidence gate: a directional call is surfaced only when the calibrated
   probability sits in the TOP ~20% (top quintile) of *historical*
   confidence, measured on walk-forward out-of-fold probabilities.
   Otherwise the honest output is "no clear edge".
4. The success metric is the FEE-ADJUSTED BACKTEST: following only the
   gated calls, sized by fractional Kelly (half-Kelly, capped), must show a
   positive Sharpe net of fees on the held-out test set. That - not
   classification accuracy - decides whether this is useful.
5. The verdict is tiered and reported with explicit reasons:
     POSITIVE NET EDGE        -> gated calls made money after fees on test
     SIGNAL BUT NOT TRADEABLE -> model beats climatology but the gated
                                 strategy does not survive fees
     NOT DEPLOYABLE           -> no signal at all (worse than climatology)

Methodology (strict, unchanged from the previous version):

1. NEVER random k-fold on time series. Validation is **walk-forward
   (expanding window)**: train on data up to block N, predict block N+1,
   roll forward. Every metric derived from these folds is out-of-sample.
2. The most recent ~18% of history is a **final untouched test set**,
   evaluated ONCE after all tuning, never during development.
3. Calibration: isotonic regressors per direction are fit on the
   walk-forward out-of-fold probabilities (data disjoint from the test set).
4. Feature hygiene: features are strictly causal (see features.py --selftest).
   Funding-rate features (Binance, full history) are trainable; open
   interest / order-book imbalance have no history and are NEVER fabricated
   into the training matrix (see altdata.py).

Artifacts are saved versioned by date under models/ and the newest is
registered in models/manifest.json so app.py always loads the latest.
Artifacts with schema 2 ("lightgbm-binary-updown") power the new UI; older
multiclass artifacts still load and are labelled legacy by the app.

Usage:
    python train_model.py --tf 1d
    python train_model.py --all-tfs
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

import features as feat
import predict as engine
from build_dataset import SYMBOLS, TIMEFRAMES, load_candles
from mlutils import IsotonicBinary, IsotonicMulticlass, StandardScalerImp

MODEL_DIR = Path(__file__).resolve().parent / "models"
MANIFEST = MODEL_DIR / "manifest.json"
CLASSES = ["DOWN", "SIDEWAYS", "UP"]     # fixed label order (legacy 3-class view)
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}

# Walk-forward fold sizes per timeframe (coarser for heavy intraday data so
# the number of fits stays tractable).
WF_BLOCK_MONTHS = {"1m": 6, "5m": 6, "15m": 6, "1h": 6, "4h": 3, "1d": 1}

# Cap per-fold training rows (random sample *within the past only* - safe
# for time series) so walk-forward stays fast on hundreds of thousands of
# rows while remaining leak-free.
MAX_TRAIN_SAMPLES = 150_000
MIN_TRAIN_ROWS = 2_000
MIN_VALID_ROWS = 300

TEST_FRACTION = 0.18          # final untouched test set = last ~18% of time
BARS_PER_YEAR = {"1h": 24 * 365.25, "4h": 6 * 365.25, "1d": 365.25}

# Confidence gate: surface a directional call only when the calibrated
# probability is in the top CONF_TOP_FRAC of *historical* (OOF) confidence.
CONF_TOP_FRAC = 0.20

# Fractional Kelly sizing for the backtest: position = min(kelly_fraction *
# full-Kelly, kelly_cap). Full Kelly for a binary bet at even odds is 2p-1
# (an approximation for a one-bar directional position; documented as such).
KELLY_FRACTION = 0.5          # half-Kelly
KELLY_CAP = 0.25              # never risk more than 25% notional per call
MIN_TRADES_FOR_EDGE = 30      # pooled trades needed before a Sharpe means much

# "coin" as a categorical feature: one-hot over SYMBOLS with a fixed
# reference level (BTC dropped). Generalises across coins with limited data.
REFERENCE_COIN = "BTC"
COIN_COLUMNS = [f"coin_{c}" for c in SYMBOLS if c != REFERENCE_COIN]

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    n_estimators=600,
    num_leaves=31,
    min_child_samples=250,
    reg_alpha=0.1,
    reg_lambda=0.5,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    n_jobs=-1,
    verbosity=-1,
    random_state=42,
)


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------

def _to_period(ts_ms: pd.Series) -> pd.Series:
    return pd.to_datetime(ts_ms, unit="ms", utc=True).dt.to_period("M")


def prepare_frame(tf: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load/sort the feature cache and add the coin one-hot feature columns.

    Coin dummy names are a *fixed* function of SYMBOLS (not of which coins
    happen to be in the frame), so training, test and live inference always
    produce identical column sets; an unseen coin maps to all-zeros.
    """
    if df is None:
        df = feat.load_feature_cache(tf)
    df = df.sort_values(["ts", "symbol"]).reset_index(drop=True)
    df = df[df["label"].notna()].copy()
    for c in SYMBOLS:
        df[f"coin_{c}"] = (df["symbol"] == c).astype(np.float32)
    df = df.drop(columns=[f"coin_{REFERENCE_COIN}"])
    return df


def feature_columns_for_tf(df: pd.DataFrame) -> list[str]:
    """Feature names after dropping all-NaN / constant columns.

    Decided on the training+validation history only, persisted in the model
    artifact, and applied identically at inference time. All-NaN funding
    columns (no funding cache available) simply drop out here.
    """
    cand = feat.feature_names() + COIN_COLUMNS
    keep = []
    for c in cand:
        if c not in df.columns:
            continue
        col = df[c]
        if col.isna().all():
            continue
        if col.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return keep


def _x(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Calibration (helper classes live in mlutils.py so pickles load cleanly)
# ---------------------------------------------------------------------------

def ece_curve(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10):
    """Binary confidence calibration: bin by predicted probability.

    Returns (ECE, reliability curve as [(bin_center, empirical_rate, n)]).
    """
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    curve, counts = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo == 0.0:
            m = (proba >= lo) & (proba <= hi)
        else:
            m = (proba > lo) & (proba <= hi)
        n = int(m.sum())
        counts.append(n)
        curve.append(float(y_true[m].mean()) if n else float("nan"))
    centers = (bins[:-1] + bins[1:]) / 2
    weights = np.array(counts, dtype=float)
    err = np.abs(np.array(curve) - centers)
    ece = float(np.nansum(weights * err) / max(weights.sum(), 1))
    return ece, list(zip(centers.tolist(), [float(c) for c in curve], counts))


# ---------------------------------------------------------------------------
# Walk-forward validation (binary, both directions)
# ---------------------------------------------------------------------------

def walk_forward(df: pd.DataFrame, tf: str, cols: list[str]):
    """Expanding-window walk-forward for the two binary targets.

    Returns (P_oof_up, P_oof_dn, y_up, y_dn, folds) restricted to valid rows.
    """
    df = df.copy()
    df["_period"] = _to_period(df["ts"])
    months = sorted(df["_period"].unique())
    block = WF_BLOCK_MONTHS.get(tf, 3)
    blocks = [months[i:i + block] for i in range(0, len(months), block)]

    y_up_all = (df["label"] == "UP").astype(np.int8).to_numpy()
    y_dn_all = (df["label"] == "DOWN").astype(np.int8).to_numpy()
    P_up = np.full(len(df), np.nan)
    P_dn = np.full(len(df), np.nan)
    folds = []
    rng = np.random.default_rng(0)

    for blk in blocks[1:]:
        tr_mask = df["_period"] < blk[0]          # expanding: ALL prior data
        va_mask = df["_period"].isin(blk)
        if tr_mask.sum() < MIN_TRAIN_ROWS or va_mask.sum() < MIN_VALID_ROWS:
            continue
        tr_df = df[tr_mask]
        if len(tr_df) > MAX_TRAIN_SAMPLES:
            idx = rng.choice(len(tr_df), MAX_TRAIN_SAMPLES, replace=False)
            tr_df = tr_df.iloc[idx]
        Xtr = _x(tr_df, cols)
        ytr_up = y_up_all[tr_df.index.to_numpy()]
        ytr_dn = y_dn_all[tr_df.index.to_numpy()]
        Xva = _x(df[va_mask], cols)
        rows = df[va_mask].index.to_numpy()
        for target, ytr in (("up", ytr_up), ("dn", ytr_dn)):
            clf = lgb.LGBMClassifier(**LGB_PARAMS)
            clf.fit(Xtr, ytr)
            P = clf.predict_proba(Xva)[:, 1]
            if target == "up":
                P_up[rows] = P
            else:
                P_dn[rows] = P
        folds.append({"block": str(blk[0]),
                      "train_rows": int(tr_mask.sum()),
                      "valid_rows": int(va_mask.sum())})
        print(f"  fold {str(blk[0])}: train {int(tr_mask.sum()):,} -> valid "
              f"{int(va_mask.sum()):,}")
    valid = np.isfinite(P_up) & np.isfinite(P_dn)
    return P_up[valid], P_dn[valid], y_up_all[valid], y_dn_all[valid], folds


# ---------------------------------------------------------------------------
# Confidence gate (top quintile of historical confidence)
# ---------------------------------------------------------------------------

def oof_percentile(P_oof: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Mid-rank percentile (0..100) of each ``p`` within the OOF distribution.

    Isotonic calibration yields heavily tied (step-function) probabilities,
    so a plain threshold can admit far more than the intended top fraction.
    Ranking by percentile with mid-rank tie handling gives exactly the
    'top ~20% of historical confidence' semantics regardless of ties.
    """
    P_sorted = np.sort(np.asarray(P_oof, dtype=float))
    n = len(P_sorted)
    p = np.asarray(p, dtype=float)
    less = np.searchsorted(P_sorted, p, side="left")
    equal = np.searchsorted(P_sorted, p, side="right") - less
    return (less + 0.5 * equal) / max(n, 1) * 100.0


def fit_gate(P_oof_up_cal: np.ndarray, P_oof_dn_cal: np.ndarray,
             y_up_oof: np.ndarray, y_dn_oof: np.ndarray,
             top_frac: float = CONF_TOP_FRAC) -> dict:
    """Confidence gate: fire when the probability sits in the top
    ``top_frac`` of historical (out-of-fold) confidence.

    Computed on walk-forward out-of-fold calibrated probabilities - i.e. from
    history the model could have known - never on the test set. The OOF
    probability arrays are stored in the gate so test rows and live
    inference can be ranked against the same historical reference.
    Also stores the historical base rates, which anchor the Kelly sizing.
    """
    pct_thr = 100.0 * (1.0 - top_frac)
    up_thr = float(np.quantile(P_oof_up_cal, 1.0 - top_frac))
    dn_thr = float(np.quantile(P_oof_dn_cal, 1.0 - top_frac))
    up_pct_self = oof_percentile(P_oof_up_cal, P_oof_up_cal)
    dn_pct_self = oof_percentile(P_oof_dn_cal, P_oof_dn_cal)
    return {
        "top_frac": float(top_frac),
        "pct_thr": float(pct_thr),
        "up_thr": up_thr,
        "dn_thr": dn_thr,
        "base_up": float(y_up_oof.mean()),
        "base_dn": float(y_dn_oof.mean()),
        "oof_rate_up": float((up_pct_self >= pct_thr).mean()),
        "oof_rate_dn": float((dn_pct_self >= pct_thr).mean()),
        "oof_n": int(len(P_oof_up_cal)),
        "_oof_p_up": np.asarray(P_oof_up_cal, dtype=float),
        "_oof_p_dn": np.asarray(P_oof_dn_cal, dtype=float),
    }


def gate_quality(y_up: np.ndarray, y_dn: np.ndarray, p_up: np.ndarray,
                 p_dn: np.ndarray, gate: dict) -> dict:
    """How the gate performs on a probability set (used for OOF and for test).

    Reports, per direction: n calls, realized hit rate, mean predicted prob
    (calibration check at the gate), base rate, and the uplift over base.
    """
    out = {}
    for key, y, p, oof_p in (("up", y_up, p_up, gate["_oof_p_up"]),
                             ("dn", y_dn, p_dn, gate["_oof_p_dn"])):
        m = oof_percentile(oof_p, p) >= gate["pct_thr"]
        base = float(y.mean()) if len(y) else float("nan")
        if m.sum():
            hit = float(y[m].mean())
            mean_p = float(p[m].mean())
        else:
            hit = mean_p = float("nan")
        out[key] = {
            "thr": gate[f"{key}_thr"],
            "pct_thr": gate["pct_thr"],
            "n_calls": int(m.sum()),
            "rate_of_rows": float(m.mean()) if len(p) else 0.0,
            "hit_rate": hit,
            "mean_predicted": mean_p,
            "base_rate": base,
            "uplift": hit - base if np.isfinite(hit) else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------
# OLD rule engine (reference only)
# ---------------------------------------------------------------------------

def rule_engine_predictions(df_test: pd.DataFrame, tf: str) -> pd.Series:
    """Old rule-based engine direction (UP/DOWN/SIDEWAYS) per test row.

    Runs predict.compute_signal_for_data() at each test candle index using
    precomputed (causal) indicator arrays - the code path the live app used
    before the ML upgrade. Kept purely as a reference comparison.
    """
    out = pd.Series(index=df_test.index, dtype=object)
    for symbol, grp in df_test.groupby("symbol"):
        candles = load_candles(symbol, tf)
        ind = engine.precompute_indicators(candles)
        ts_to_i = {c["ts"]: i for i, c in enumerate(candles)}
        for ts, idx in zip(grp["ts"], grp.index):
            i = ts_to_i.get(int(ts))
            if i is None:
                continue
            sig = engine.compute_signal_for_data(candles, ind=ind, t=i)
            out.loc[idx] = sig["direction"]
    return out


# ---------------------------------------------------------------------------
# Fee-adjusted backtest with fractional-Kelly sizing (THE success metric)
# ---------------------------------------------------------------------------

def kelly_backtest(df_test: pd.DataFrame, p_up: np.ndarray, p_dn: np.ndarray,
                   gate: dict, tf: str,
                   fees=(0.0005, 0.001),
                   kelly_fraction: float = KELLY_FRACTION,
                   kelly_cap: float = KELLY_CAP) -> dict:
    """Follow ONLY the high-confidence calls, sized by fractional Kelly.

    For every test bar: enter UP when p_up >= up_thr, DOWN when
    p_dn >= dn_thr (if both fire, take the more confident side). Hold one
    bar (the label horizon).

    Position size: the event is rare (base rate q ~ 0.17 under the
    deadzone label), so the even-odds Kelly formula 2p-1 does not apply
    (it would be negative and every position would size to zero - this
    actually happened in the first run). Instead we size as a Kelly bet at
    the fair odds implied by the historical base rate q:
        f* = (p - q) / (1 - q)
    i.e. the edge over climatology, normalised by the odds structure.
    Position = min(kelly_fraction * f*, kelly_cap), floored at 0.

    Fees (per side, round trip = 2x) are charged proportionally to size.
    Returns pooled-portfolio and per-symbol stats: total return, Sharpe
    (annualized over ALL test bars, including flat ones), max drawdown,
    plus half/half robustness splits and buy & hold benchmarks.
    """
    dfc = df_test.reset_index(drop=True)
    fwd = dfc["fwd_ret"].to_numpy(dtype=float)
    short_ret = -fwd / (1.0 + fwd)

    take_up = oof_percentile(gate["_oof_p_up"], p_up) >= gate["pct_thr"]
    take_dn = oof_percentile(gate["_oof_p_dn"], p_dn) >= gate["pct_thr"]
    both = take_up & take_dn
    direction = np.zeros(len(dfc), dtype=int)
    direction[take_up & ~take_dn] = 1
    direction[take_dn & ~take_up] = -1
    direction[both] = np.where(p_up[both] >= p_dn[both], 1, -1)
    take = direction != 0

    q_up = float(gate.get("base_up", 0.5))
    q_dn = float(gate.get("base_dn", 0.5))
    # NOTE: sizing uses the gate's historical base rates (walk-forward OOF),
    # never the test-set rates - no peeking.
    f_full = np.where(
        direction == 1, (p_up - q_up) / max(1.0 - q_up, 1e-9),
        np.where(direction == -1, (p_dn - q_dn) / max(1.0 - q_dn, 1e-9), 0.0))
    size = np.minimum(kelly_fraction * np.clip(f_full, 0.0, 1.0), kelly_cap)
    gross = np.where(direction == 1, fwd, np.where(direction == -1, short_ret, 0.0))

    per_year = BARS_PER_YEAR.get(tf, 365.25)
    n = len(dfc)
    ts = dfc["ts"].to_numpy()
    sym = dfc["symbol"].to_numpy()

    # pooled portfolio: equal-weight mean across coins at each timestamp
    order = np.argsort(ts, kind="stable")
    ts_sorted = ts[order]
    uniq_ts, first_idx = np.unique(ts_sorted, return_index=True)
    bar_groups = np.split(order, first_idx[1:])

    def pooled_series(net: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(timestamps, mean return per timestamp) across symbols."""
        t_out = np.empty(len(uniq_ts), dtype=np.int64)
        r_out = np.empty(len(uniq_ts), dtype=float)
        for i, g in enumerate(bar_groups):
            t_out[i] = ts[g[0]]
            r_out[i] = net[g].mean()
        return t_out, r_out

    def equity_stats(seq: np.ndarray, per_year: float) -> dict:
        eq = np.cumprod(1.0 + seq)
        peak = np.maximum.accumulate(eq)
        sharpe = float(seq.mean() / seq.std() * np.sqrt(per_year)) \
            if len(seq) > 1 and seq.std() > 0 else 0.0
        return {
            "total_return": float(eq[-1] - 1.0) if len(eq) else 0.0,
            "sharpe": sharpe,
            "max_drawdown": float(((eq - peak) / peak).min()) if len(eq) else 0.0,
        }

    result: dict = {
        "_benchmark": {
            "n_bars": int(n),
            "signal_rate": float(take.mean()),
            "n_double_signals": int(both.sum()),
            "kelly_fraction": float(kelly_fraction),
            "kelly_cap": float(kelly_cap),
            "base_up": q_up,
            "base_dn": q_dn,
            "mean_size_when_trading": float(size[take].mean()) if take.sum() else 0.0,
        },
    }

    # buy & hold benchmarks on the same window (pooled equal weight + per symbol)
    _, bh_ret = pooled_series(fwd)
    result["_benchmark"]["pooled_buy_hold"] = equity_stats(bh_ret, per_year)

    symbols = dfc["symbol"].unique()
    result["_benchmark"]["pooled_buy_hold_per_symbol"] = {}
    for s in symbols:
        m = sym == s
        result["_benchmark"]["pooled_buy_hold_per_symbol"][s] = \
            equity_stats(fwd[m], per_year)

    trades_gross = gross[take]
    for fee in fees:
        net_trade = size * (gross - 2.0 * fee)
        seq = np.where(take, net_trade, 0.0)
        trades = net_trade[take]
        t_out, r_out = pooled_series(seq)
        pooled = equity_stats(r_out, per_year)

        # robustness: first half vs second half of the pooled timeline
        half = len(r_out) // 2
        pooled["sharpe_first_half"] = equity_stats(r_out[:half], per_year)["sharpe"]
        pooled["sharpe_second_half"] = equity_stats(r_out[half:], per_year)["sharpe"]

        per_symbol = {}
        for s in symbols:
            m = sym == s
            per_symbol[s] = equity_stats(seq[m], per_year)

        result[f"fee_{fee * 100:.2f}pct"] = {
            "pooled": pooled,
            "per_symbol": per_symbol,
            "n_trades": int(take.sum()),
            "n_trades_up": int((direction == 1).sum()),
            "n_trades_down": int((direction == -1).sum()),
            "win_rate": float((trades > 0).mean()) if len(trades) else 0.0,
            "avg_ret_per_trade": float(trades.mean()) if len(trades) else 0.0,
            "sum_ret_per_trade": float(trades.sum()),
        }

        # full-notional comparison (what you'd get without Kelly sizing)
        seq_full = np.where(take, gross - 2.0 * fee, 0.0)
        _, r_full = pooled_series(seq_full)
        result[f"fee_{fee * 100:.2f}pct"]["pooled_full_notional"] = \
            equity_stats(r_full, per_year)
        result[f"fee_{fee * 100:.2f}pct"]["_trades_gross_sum"] = \
            float(trades_gross.sum()) if len(trades_gross) else 0.0

    return result


# ---------------------------------------------------------------------------
# Binary evaluation (primary metrics)
# ---------------------------------------------------------------------------

def evaluate_binary(y: np.ndarray, p_cal: np.ndarray, p_raw: np.ndarray,
                    base_rate: float, name: str) -> dict:
    """Honest probability metrics for one direction on the held-out test set."""
    y = np.asarray(y, dtype=int)
    p_cal = np.asarray(p_cal, dtype=float)
    p_raw = np.asarray(p_raw, dtype=float)
    brier = float(brier_score_loss(y, p_cal))
    brier_ref = float(base_rate * (1.0 - base_rate))   # climatology Brier
    bss = float(1.0 - brier / brier_ref) if brier_ref > 0 else float("nan")
    try:
        auc = float(roc_auc_score(y, p_cal))
    except ValueError:
        auc = float("nan")
    try:
        auc_raw = float(roc_auc_score(y, p_raw))
    except ValueError:
        auc_raw = float("nan")
    ece, curve = ece_curve(y, p_cal)
    hit05 = float((p_cal >= 0.5).mean())
    return {
        "name": name,
        "n": int(len(y)),
        "base_rate": float(base_rate),
        "mean_predicted": float(p_cal.mean()),
        "brier": brier,
        "brier_climatology": brier_ref,
        "bss": bss,
        "roc_auc": auc,
        "roc_auc_raw": auc_raw,
        "ece": ece,
        "calibration_curve": curve,
        "frac_above_50": hit05,
        "hit_rate_when_above_50": (float(y[p_cal >= 0.5].mean())
                                   if (p_cal >= 0.5).any() else float("nan")),
    }


def legacy_3class_reference(labels_true: pd.Series,
                            p_up: np.ndarray, p_dn: np.ndarray) -> dict:
    """3-class view derived from the two binary probabilities - REFERENCE ONLY.

    P(SIDEWAYS) = 1 - P(UP) - P(DOWN). Accuracy here is dominated by the
    majority class and is shown only so the old framing can be compared.
    """
    p_side = np.clip(1.0 - p_up - p_dn, 0.0, 1.0)
    P = np.column_stack([p_dn, p_side, p_up])
    yhat = np.array(CLASSES)[P.argmax(axis=1)]
    y_true = labels_true.to_numpy()
    acc = float((yhat == y_true).mean())
    majority = labels_true.mode()[0]
    return {
        "accuracy": acc,
        "majority_class": majority,
        "majority_acc": float((labels_true == majority).mean()),
        "confusion_matrix": confusion_matrix(labels_true, yhat,
                                             labels=CLASSES).tolist(),
    }


# ---------------------------------------------------------------------------
# Full train + evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(tf: str, rebuild_features: bool = True) -> dict:
    t0 = time.time()
    print(f"\n{'=' * 70}\nTRAIN tf={tf}  (binary calibrated P(UP)/P(DOWN))\n{'=' * 70}")
    if rebuild_features and not (feat.FEATURES_DIR / f"features_{tf}.parquet").exists():
        feat.build_feature_cache(tfs=[tf])

    df_all = feat.load_feature_cache(tf).sort_values("ts").reset_index(drop=True)
    print(f"total labeled rows: {len(df_all):,}  span: "
          f"{datetime.fromtimestamp(df_all['ts'].min() / 1000, tz=timezone.utc).date()} "
          f"-> {datetime.fromtimestamp(df_all['ts'].max() / 1000, tz=timezone.utc).date()}")

    dist = df_all["label"].value_counts()
    print("label balance (all history): " + ", ".join(
        f"{k}={v:,} ({v / len(df_all):.1%})" for k, v in dist.items()))

    # ---- final untouched test split (by time) ----
    span = df_all["ts"].max() - df_all["ts"].min()
    test_cut = df_all["ts"].max() - TEST_FRACTION * span
    trainval = df_all[df_all["ts"] < test_cut].copy()
    test_df = df_all[df_all["ts"] >= test_cut].copy()
    print(f"\ntest cut: {datetime.fromtimestamp(test_cut / 1000, tz=timezone.utc)}  "
          f"trainval {len(trainval):,} rows, test {len(test_df):,} rows "
          f"({len(test_df) / len(df_all):.1%})")

    # ---- features (columns decided on trainval ONLY) ----
    trainval = prepare_frame(tf, trainval)
    test_df = prepare_frame(tf, test_df)
    cols = feature_columns_for_tf(trainval)
    meta_cols = ["ts", "symbol", "label", "fwd_ret"]
    test_df = test_df[[c for c in cols + meta_cols if c in test_df.columns]]
    fund_cols_in = [c for c in cols if c.startswith("fund_")]
    print(f"features kept: {len(cols)} (incl. funding: "
          f"{fund_cols_in if fund_cols_in else 'none - no funding cache'})")

    # ---- walk-forward validation (expanding window, out-of-sample) ----
    print("\nwalk-forward validation…")
    P_oof_up_raw, P_oof_dn_raw, y_oof_up, y_oof_dn, folds = walk_forward(
        trainval, tf, cols)
    print(f"OOF rows: {len(y_oof_up):,}  (UP base rate {y_oof_up.mean():.3f}, "
          f"DOWN base rate {y_oof_dn.mean():.3f})")

    # ---- calibrators fit on OOF probs (disjoint from the test set) ----
    cal_up = IsotonicBinary().fit(P_oof_up_raw, y_oof_up)
    cal_dn = IsotonicBinary().fit(P_oof_dn_raw, y_oof_dn)
    P_oof_up = cal_up.predict(P_oof_up_raw)
    P_oof_dn = cal_dn.predict(P_oof_dn_raw)

    # OOF skill (out-of-sample by construction)
    oof_up = evaluate_binary(y_oof_up, P_oof_up, P_oof_up_raw,
                             base_rate=float(y_oof_up.mean()), name="UP (OOF)")
    oof_dn = evaluate_binary(y_oof_dn, P_oof_dn, P_oof_dn_raw,
                             base_rate=float(y_oof_dn.mean()), name="DOWN (OOF)")
    print(f"OOF  UP : Brier {oof_up['brier']:.4f} (clim {oof_up['brier_climatology']:.4f}, "
          f"BSS {oof_up['bss']:+.3f})  AUC {oof_up['roc_auc']:.3f}  ECE {oof_up['ece']:.3f}")
    print(f"OOF  DN : Brier {oof_dn['brier']:.4f} (clim {oof_dn['brier_climatology']:.4f}, "
          f"BSS {oof_dn['bss']:+.3f})  AUC {oof_dn['roc_auc']:.3f}  ECE {oof_dn['ece']:.3f}")

    # ---- confidence gate from historical (OOF) confidence ----
    gate = fit_gate(P_oof_up, P_oof_dn, y_oof_up, y_oof_dn)
    oof_gate_quality = gate_quality(y_oof_up, y_oof_dn, P_oof_up, P_oof_dn, gate)
    print(f"\nconfidence gate (top {gate['top_frac']:.0%} of historical confidence):")
    print(f"  UP  thr {gate['up_thr']:.3f} -> OOF calls {oof_gate_quality['up']['n_calls']:,} "
          f"hit {oof_gate_quality['up']['hit_rate']:.3f} vs base "
          f"{oof_gate_quality['up']['base_rate']:.3f} "
          f"(uplift {oof_gate_quality['up']['uplift']:+.3f})")
    print(f"  DN  thr {gate['dn_thr']:.3f} -> OOF calls {oof_gate_quality['dn']['n_calls']:,} "
          f"hit {oof_gate_quality['dn']['hit_rate']:.3f} vs base "
          f"{oof_gate_quality['dn']['base_rate']:.3f} "
          f"(uplift {oof_gate_quality['dn']['uplift']:+.3f})")

    # ---- final models on full trainval ----
    print("\nfitting final LightGBM (UP / DOWN) + logistic baselines on trainval…")
    Xtr = _x(trainval, cols)
    ytr_up = (trainval["label"] == "UP").astype(int).to_numpy()
    ytr_dn = (trainval["label"] == "DOWN").astype(int).to_numpy()
    Xte = _x(test_df, cols)
    yte_up = (test_df["label"] == "UP").astype(int).to_numpy()
    yte_dn = (test_df["label"] == "DOWN").astype(int).to_numpy()

    lgb_up = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_up.fit(Xtr, ytr_up)
    lgb_dn = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_dn.fit(Xtr, ytr_dn)
    p_up_raw = lgb_up.predict_proba(Xte)[:, 1]
    p_dn_raw = lgb_dn.predict_proba(Xte)[:, 1]
    p_up = cal_up.predict(p_up_raw)
    p_dn = cal_dn.predict(p_dn_raw)

    log_up = Pipeline([("impute", SimpleImputer(strategy="median")),
                       ("scale", StandardScalerImp()),
                       ("lr", LogisticRegression(max_iter=3000, random_state=42))])
    log_dn = Pipeline([("impute", SimpleImputer(strategy="median")),
                       ("scale", StandardScalerImp()),
                       ("lr", LogisticRegression(max_iter=3000, random_state=42))])
    log_up.fit(Xtr, ytr_up)
    log_dn.fit(Xtr, ytr_dn)
    p_up_log = np.clip(log_up.predict_proba(Xte)[:, 1], 0.0, 1.0)
    p_dn_log = np.clip(log_dn.predict_proba(Xte)[:, 1], 0.0, 1.0)

    # ---- honest test evaluation (ONCE) ----
    base_up = float(ytr_up.mean())   # climatology = trainval base rate
    base_dn = float(ytr_dn.mean())
    metrics = {
        "up": evaluate_binary(yte_up, p_up, p_up_raw, base_up, "P(UP) calibrated"),
        "dn": evaluate_binary(yte_dn, p_dn, p_dn_raw, base_dn, "P(DOWN) calibrated"),
    }
    metrics_logistic = {
        "up": evaluate_binary(yte_up, p_up_log, p_up_log, base_up,
                              "P(UP) logistic baseline"),
        "dn": evaluate_binary(yte_dn, p_dn_log, p_dn_log, base_dn,
                              "P(DOWN) logistic baseline"),
    }

    # ---- confidence-gate quality on the test set (the deployable rule) ----
    test_gate_quality = gate_quality(yte_up, yte_dn, p_up, p_dn, gate)

    # ---- 3-class reference (secondary; majority-collapse caveat) ----
    ref3 = legacy_3class_reference(test_df["label"], p_up, p_dn)

    # ---- OLD rule engine (reference only) ----
    print("\ncomputing OLD rule-based engine predictions on the test set (reference)…")
    rule_dir = rule_engine_predictions(test_df, tf)
    mask = rule_dir.notna()
    old_rule_acc = float((rule_dir[mask] == test_df.loc[mask, "label"]).mean()) \
        if mask.sum() else None

    # ---- fee-adjusted fractional-Kelly backtest (the number that matters) ----
    print("\nfractional-Kelly backtest on held-out test set (gated calls only)…")
    trading = kelly_backtest(test_df, p_up, p_dn, gate, tf)

    # ---- tiered verdict with explicit reasons ----
    has_signal = ((metrics["up"]["bss"] > 0 or metrics["dn"]["bss"] > 0)
                  and np.nanmean([metrics["up"]["roc_auc"],
                                  metrics["dn"]["roc_auc"]]) > 0.5)
    fee05 = trading["fee_0.05pct"]["pooled"]
    fee10 = trading["fee_0.10pct"]["pooled"]
    n_trades = trading["fee_0.05pct"]["n_trades"]
    reasons: list[str] = []
    if not has_signal:
        reasons.append("no probability skill: BSS <= 0 vs climatology and/or "
                       "AUC <= 0.5 on the held-out test set")
    if fee05["sharpe"] <= 0:
        reasons.append(f"pooled Sharpe <= 0 at 0.05% per-side fees "
                       f"({fee05['sharpe']:+.2f})")
    if fee10["sharpe"] <= 0:
        reasons.append(f"pooled Sharpe <= 0 at 0.10% per-side fees "
                       f"({fee10['sharpe']:+.2f})")
    if n_trades < MIN_TRADES_FOR_EDGE:
        reasons.append(f"too few gated trades on test ({n_trades}) for a "
                       f"meaningful Sharpe (need >= {MIN_TRADES_FOR_EDGE})")
    if has_signal and not reasons:
        verdict = "POSITIVE NET EDGE"
        reasons.append("calibrated probabilities beat climatology (BSS > 0) and "
                       "the gated fractional-Kelly strategy shows a positive "
                       "pooled Sharpe net of fees on the held-out test set")
    elif has_signal:
        verdict = "SIGNAL BUT NOT TRADEABLE"
        reasons.insert(0, "model beats climatology, but following only the "
                          "gated calls does not survive fees on the test set")
    else:
        verdict = "NOT DEPLOYABLE"
    print(f"\n>>> verdict on held-out test: {verdict}")
    for r in reasons:
        print(f"    - {r}")

    baselines = {
        "climatology_brier_up": metrics["up"]["brier_climatology"],
        "climatology_brier_dn": metrics["dn"]["brier_climatology"],
        "base_rate_up": base_up,
        "base_rate_dn": base_dn,
        "brier_always_05_up": float(brier_score_loss(yte_up, np.full(len(yte_up), 0.5))),
        "brier_always_05_dn": float(brier_score_loss(yte_dn, np.full(len(yte_dn), 0.5))),
        "reference_3class": ref3,
        "reference_3class_old_rule_acc": old_rule_acc,
        "reference_3class_old_rule_rows": int(mask.sum()),
        "note": "3-class accuracy is reference-only: the SIDEWAYS majority "
                "inflates it (majority-class baseline shown beside it).",
    }

    artifact = {
        "schema": 2,
        "kind": "lightgbm-binary-updown",
        "tf": tf,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_start": str(trainval["ts"].min()),
        "train_end": str(trainval["ts"].max()),
        "test_start": str(test_df["ts"].min()),
        "test_end": str(test_df["ts"].max()),
        "classes": CLASSES,
        "feature_columns": cols,
        "n_train_rows": int(len(trainval)),
        "n_test_rows": int(len(test_df)),
        "params": dict(LGB_PARAMS),
        "model_up": lgb_up,
        "model_dn": lgb_dn,
        "calibrator_up": cal_up,
        "calibrator_dn": cal_dn,
        "logistic_baseline_up": log_up,
        "logistic_baseline_dn": log_dn,
        "gate": gate,
        "oof_gate_quality": oof_gate_quality,
        "oof_metrics": {"up": oof_up, "dn": oof_dn},
        "metrics": metrics,
        "metrics_logistic": metrics_logistic,
        "test_gate_quality": test_gate_quality,
        "baselines": baselines,
        "trading": trading,
        "walk_forward": {"n_oof": int(len(y_oof_up)), "folds": folds},
        "funding_features_used": fund_cols_in,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "label_balance_all": {k: int(v) for k, v in dist.items()},
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"model_{tf}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.joblib"
    joblib.dump(artifact, MODEL_DIR / fname, compress=3)
    _update_manifest(tf, fname)
    print(f"\nartifact saved: {MODEL_DIR / fname}  (total {time.time() - t0:.0f}s)")
    print_report(artifact)
    return artifact


def _update_manifest(tf: str, fname: str) -> None:
    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
    manifest.setdefault("latest", {})[tf] = fname
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def load_latest(tf: str) -> dict | None:
    """Load the latest trained artifact for a timeframe (or None)."""
    if not MANIFEST.exists():
        return None
    try:
        manifest = json.loads(MANIFEST.read_text())
        fname = manifest.get("latest", {}).get(tf)
    except (json.JSONDecodeError, OSError):
        return None
    if not fname:
        return None
    path = MODEL_DIR / fname
    if not path.exists():
        return None
    return joblib.load(path)


def latest_meta(tf: str) -> dict | None:
    art = load_latest(tf)
    if art is None:
        return None
    keys = ("tf", "trained_at_utc", "train_start", "train_end",
            "test_start", "test_end", "n_train_rows", "n_test_rows",
            "metrics", "baselines", "verdict", "trading")
    return {k: art[k] for k in keys if k in art}


def manifest_filename(tf: str) -> str | None:
    """Fresh (uncached) read of the registered artifact name for a tf."""
    if not MANIFEST.exists():
        return None
    try:
        manifest = json.loads(MANIFEST.read_text())
        fname = manifest.get("latest", {}).get(tf)
    except (json.JSONDecodeError, OSError):
        return None
    if fname and (MODEL_DIR / fname).exists():
        return fname
    return None


def load_artifact_file(tf: str, fname: str) -> dict | None:
    path = MODEL_DIR / fname
    return joblib.load(path) if path.exists() else None


def repair_artifacts() -> int:
    """Re-dump any artifact whose pickles reference __main__ (early builds
    pickled IsotonicMulticlass from a script run so joblib recorded the class
    under the __main__ module). Returns count fixed.
    """
    import sys
    import types
    from mlutils import IsotonicMulticlass as _Real

    fixed = 0
    for path in sorted(MODEL_DIR.glob("model_*.joblib")):
        try:
            joblib.load(path)
            continue                      # loads fine already
        except Exception:
            pass
        try:
            saved_main = sys.modules.get("__main__")
            standin = types.ModuleType("__main__")
            standin.IsotonicMulticlass = _Real
            sys.modules["__main__"] = standin   # satisfy pickle lookups
            try:
                art = joblib.load(path)
            finally:
                if saved_main is not None:
                    sys.modules["__main__"] = saved_main
                else:
                    del sys.modules["__main__"]
            tmp = path.with_suffix(".tmp")
            joblib.dump(art, tmp, compress=3)
            tmp.replace(path)
            fixed += 1
            print(f"  repaired {path.name}")
        except Exception as e:
            print(f"  could not repair {path.name}: {e}", file=sys.stderr)
    return fixed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(art: dict) -> None:
    m = art["metrics"]
    gate = art["gate"]
    tgq = art["test_gate_quality"]
    trading = art["trading"]
    print("\n" + "=" * 70)
    print("HELD-OUT TEST REPORT  (primary metrics = probability quality)")
    print("=" * 70)
    print(f"\ntest rows: {art['n_test_rows']:,}   "
          f"base rates: UP {m['up']['base_rate']:.3f}, DOWN {m['dn']['base_rate']:.3f}")
    for key, lbl in (("up", "P(UP)"), ("dn", "P(DOWN)")):
        r = m[key]
        print(f"\n  {lbl} ({r['name']}):")
        print(f"    Brier:        {r['brier']:.4f}   "
              f"(climatology {r['brier_climatology']:.4f} -> BSS {r['bss']:+.3f})")
        print(f"    ROC-AUC:      {r['roc_auc']:.3f}   (raw pre-calibration {r['roc_auc_raw']:.3f})")
        print(f"    ECE:          {r['ece']:.3f}   mean predicted {r['mean_predicted']:.3f}")
        print(f"    logistic:     Brier {art['metrics_logistic'][key]['brier']:.4f} "
              f"AUC {art['metrics_logistic'][key]['roc_auc']:.3f}")
    lr = art["baselines"]["reference_3class"]
    print(f"\n  3-class reference (secondary, majority-inflated): accuracy "
          f"{lr['accuracy']:.3f} vs always-'{lr['majority_class']}' "
          f"{lr['majority_acc']:.3f}")
    if art["baselines"].get("reference_3class_old_rule_acc") is not None:
        print(f"  old rule engine 3-class accuracy (reference): "
              f"{art['baselines']['reference_3class_old_rule_acc']:.3f} "
              f"({art['baselines']['reference_3class_old_rule_rows']:,} rows)")

    print(f"\nConfidence gate: top {gate['top_frac']:.0%} of historical confidence")
    for key, lbl in (("up", "UP"), ("dn", "DOWN")):
        q = tgq[key]
        print(f"  {lbl:<4} thr {q['thr']:.3f} -> {q['n_calls']:,} calls "
              f"({q['rate_of_rows']:.1%} of test rows)  hit {q['hit_rate']:.3f} "
              f"vs base {q['base_rate']:.3f} (uplift {q['uplift']:+.3f})  "
              f"mean p {q['mean_predicted']:.3f}")

    print("\nFee-adjusted fractional-Kelly backtest "
          f"(half-Kelly capped {trading['_benchmark']['kelly_cap']:.0%}, "
          f"signal rate {trading['_benchmark']['signal_rate']:.1%}, "
          f"base rates UP {trading['_benchmark']['base_up']:.3f} / "
          f"DOWN {trading['_benchmark']['base_dn']:.3f}):")
    bh = trading["_benchmark"]["pooled_buy_hold"]
    print(f"  pooled buy & hold (same window): total {bh['total_return']:+.1%}, "
          f"sharpe {bh['sharpe']:+.2f}, maxDD {bh['max_drawdown']:.1%}")
    for fee_key, r in trading.items():
        if not fee_key.startswith("fee_"):
            continue
        p = r["pooled"]
        print(f"  {fee_key}: trades {r['n_trades']:,} (UP {r['n_trades_up']} / "
              f"DN {r['n_trades_down']})  win {r['win_rate']:.1%}  "
              f"avg/trade {r['avg_ret_per_trade']:+.4%}")
        print(f"      pooled: total {p['total_return']:+.1%}  sharpe {p['sharpe']:+.2f} "
              f"(halves {p['sharpe_first_half']:+.2f} / {p['sharpe_second_half']:+.2f})  "
              f"maxDD {p['max_drawdown']:.1%}")
        pf = r["pooled_full_notional"]
        print(f"      full-notional comparison: sharpe {pf['sharpe']:+.2f}")

    print(f"\nVERDICT: {art['verdict']}")
    for reason in art["verdict_reasons"]:
        print(f"  - {reason}")


# ---------------------------------------------------------------------------
# Live inference (used by app.py - fully local, no network at predict time)
# ---------------------------------------------------------------------------

def _binary_from_frame(art: dict, row_X: np.ndarray) -> tuple[float, float]:
    p_up_raw = float(art["model_up"].predict_proba(row_X)[0, 1])
    p_dn_raw = float(art["model_dn"].predict_proba(row_X)[0, 1])
    return p_up_raw, p_dn_raw


def live_predict(candles: list[dict], tf: str, symbol: str,
                 artifact: dict | None = None) -> dict | None:
    """Predict the next candle from live candles using the latest artifact.

    Schema-2 artifacts (binary up/down) return calibrated P(UP) / P(DOWN),
    the derived 3-class view, and the confidence gate verdict: a directional
    call ONLY when the probability is in the top quintile of historical
    confidence, otherwise the honest "NO CLEAR EDGE".

    Legacy multiclass artifacts (schema 1) still work and are marked
    ``legacy=True`` so the UI can label them as the old framing.
    """
    art = artifact if artifact is not None else load_latest(tf)
    if art is None:
        return None
    frame = feat.build_feature_frame(candles, tf, symbol, include_labels=False)
    if frame.empty:
        raise ValueError(
            f"Not enough candles for ML features (need >= {feat.WARMUP + 1}, "
            f"got {len(candles)}). Increase the candle count.")
    row = frame.iloc[[-1]].copy()
    for c in SYMBOLS:
        row[f"coin_{c}"] = (row["symbol"] == c).astype(np.float32)
    row = row.drop(columns=[f"coin_{REFERENCE_COIN}"])
    cols = art["feature_columns"]
    missing = [c for c in cols if c not in row.columns]
    if missing:
        raise ValueError(f"model expects features not computed: {missing}")
    X = row[cols].to_numpy(dtype=np.float32)

    if art.get("kind") == "lightgbm-binary-updown":
        p_up_raw, p_dn_raw = _binary_from_frame(art, X)
        p_up = float(art["calibrator_up"].predict([p_up_raw])[0])
        p_dn = float(art["calibrator_dn"].predict([p_dn_raw])[0])
        p_side = float(np.clip(1.0 - p_up - p_dn, 0.0, 1.0))

        gate = art["gate"]
        up_pct = float(oof_percentile(gate["_oof_p_up"], np.array([p_up]))[0]) \
            if gate.get("_oof_p_up") is not None else None
        dn_pct = float(oof_percentile(gate["_oof_p_dn"], np.array([p_dn]))[0]) \
            if gate.get("_oof_p_dn") is not None else None
        pct_thr = gate.get("pct_thr", 80.0)
        up_fires = up_pct is not None and up_pct >= pct_thr
        dn_fires = dn_pct is not None and dn_pct >= pct_thr
        if up_fires and (not dn_fires or p_up >= p_dn):
            direction, edge_p = "UP", p_up
        elif dn_fires:
            direction, edge_p = "DOWN", p_dn
        else:
            direction, edge_p = "NO CLEAR EDGE", max(p_up, p_dn)

        return {
            "schema": 2,
            "tf": tf,
            "symbol": symbol,
            "ts": int(row["ts"].iloc[0]),
            "p_up": p_up,
            "p_dn": p_dn,
            "p_sideways": p_side,
            "proba_raw": {"UP": p_up_raw, "DOWN": p_dn_raw},
            # legacy-shaped fields so downstream code can render either artifact
            "classes": list(art["classes"]),
            "proba": {"UP": p_up, "DOWN": p_dn, "SIDEWAYS": p_side},
            "prediction": direction,
            "confidence": float(edge_p),
            "gate": {
                "up_thr": gate["up_thr"], "dn_thr": gate["dn_thr"],
                "pct_thr": pct_thr, "top_frac": gate["top_frac"],
                "up_fires": bool(up_fires), "dn_fires": bool(dn_fires),
                "up_pctile": up_pct, "dn_pctile": dn_pct,
            },
            "funding_features_used": art.get("funding_features_used", []),
            "meta": {k: art[k] for k in (
                "tf", "trained_at_utc", "train_start", "train_end", "test_start",
                "test_end", "n_train_rows", "n_test_rows", "metrics",
                "baselines", "verdict", "verdict_reasons", "trading",
                "test_gate_quality", "oof_metrics", "funding_features_used") if k in art},
        }

    # ---- legacy multiclass artifact (schema 1) ----
    P_raw = art["model"].predict_proba(X)
    P_cal = art["calibrator"].predict(P_raw)
    classes = list(art["classes"])
    probs = {c: float(p) for c, p in zip(classes, P_cal[0])}
    pred = classes[int(P_cal[0].argmax())]
    return {
        "schema": 1,
        "legacy": True,
        "tf": tf,
        "symbol": symbol,
        "ts": int(row["ts"].iloc[0]),
        "classes": classes,
        "proba_raw": {c: float(p) for c, p in zip(classes, P_raw[0])},
        "proba": probs,
        "prediction": pred,
        "confidence": float(P_cal[0].max()),
        "meta": {k: art[k] for k in (
            "tf", "trained_at_utc", "train_start", "train_end", "test_start",
            "test_end", "n_train_rows", "n_test_rows", "metrics",
            "baselines", "verdict", "trading") if k in art},
    }


def available_timeframes() -> list[str]:
    """Timeframes that have a trained model artifact registered."""
    if not MANIFEST.exists():
        return []
    try:
        manifest = json.loads(MANIFEST.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return [tf for tf, fn in manifest.get("latest", {}).items()
            if (MODEL_DIR / fn).exists()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate the directional-probability model")
    parser.add_argument("--tf", default=None, help="timeframe to train (1h/4h/1d)")
    parser.add_argument("--all-tfs", action="store_true",
                        help="train every timeframe with cached data")
    parser.add_argument("--no-feature-build", action="store_true",
                        help="do not auto-build the feature cache")
    args = parser.parse_args()

    if args.all_tfs:
        tfs = TIMEFRAMES
    elif args.tf:
        tfs = [args.tf]
    else:
        parser.print_help()
        return
    for tf in tfs:
        train_and_evaluate(tf, rebuild_features=not args.no_feature_build)


if __name__ == "__main__":
    main()
