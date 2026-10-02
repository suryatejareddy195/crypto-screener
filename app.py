#!/usr/bin/env python3
"""
Advanced Crypto Predictor – Web UI
-----------------------------------
Streamlit front-end for the advanced prediction engine in ``predict.py``.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import altdata
import predict as engine
import train_model as mlmod

st.set_page_config(
    page_title="Advanced Crypto Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DIR_COLORS = {"UP": "#16a34a", "DOWN": "#dc2626", "SIDEWAYS": "#6b7280",
              "NO CLEAR EDGE": "#6b7280"}
DIR_ICONS = {"UP": "▲", "DOWN": "▼", "SIDEWAYS": "◆", "NO CLEAR EDGE": "∅"}
REGIME_COLORS = {
    "trending": "#3b82f6", "ranging": "#6b7280",
    "volatile": "#f59e0b", "transitional": "#8b5cf6",
}


# ---------------------------------------------------------------------------
# ML model loading (cached once per artifact)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_model_artifact(tf: str, fname: str):
    return mlmod.load_artifact_file(tf, fname)


def model_artifact(tf: str):
    """Latest trained artifact for ``tf`` (or None). Reads the manifest fresh
    each call so a retrain is picked up on the next app rerun without a
    restart, while the (potentially large) model itself stays cached."""
    fname = mlmod.manifest_filename(tf)
    if not fname:
        return None
    return _load_model_artifact(tf, fname)


# ---------------------------------------------------------------------------
# Prediction engine wrapper
# ---------------------------------------------------------------------------

def run_prediction(coin: str, tf: str, limit: int, progress=None) -> dict:
    """Run the full prediction pipeline.

    ``progress`` is an optional callback (message: str) that receives
    provider-by-provider status while candles are being fetched, so the UI
    can show live progress instead of nothing until the whole chain finishes.

    The classical rule-based signal is always computed (shown as the
    "Classic TA view" comparison). When a trained ML model exists for the
    chosen timeframe, the ML verdict (calibrated probabilities) is computed
    from the same candles and returned as ``ml``.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            candles = engine.fetch_candles(coin, tf, limit, progress=progress)
    except SystemExit as e:
        raise RuntimeError(str(e)) from e
    if len(candles) < 70:
        raise RuntimeError(f"Only {len(candles)} candles fetched - need at least 70.")

    closes = [c["close"] for c in candles]
    price = closes[-1]
    atr_val = engine.atr_series(candles)[-1]
    res, sup = engine.swing_points(candles)
    resistance, support = engine.nearest_levels(price, res, sup)

    sig = engine.compute_signal(candles)
    bt = engine.backtest(candles)
    regime = sig["regime"]
    patterns = sig["patterns"]
    divergences = sig["divergences"]

    # Indicator snapshots
    rsi = engine.rsi_series(closes, 14)[-1]
    line, sig_l, hist = engine.macd_series(closes)
    s20, s50 = engine.sma(closes, 20), engine.sma(closes, 50)
    s100 = engine.sma(closes, 100)
    e20, e50, e200 = engine.ema(closes, 20), engine.ema(closes, 50), engine.ema(closes, 200)
    adx, pdi, mdi = engine.adx_series(candles)
    upper, mid, lower = engine.bollinger(closes)
    stoch_k, stoch_d = engine.stochastic(candles)
    wr = engine.williams_r(candles)
    cci = engine.cci_series(candles)
    tenkan, kijun, senkou_a, senkou_b = engine.ichimoku(candles)
    vwap = engine.vwap_bands(candles)
    obv = engine.obv_series(candles)
    roc = engine.roc_series(closes)
    atr_arr = engine.atr_series(candles)
    sl = engine.slope(closes[-20:])
    va = engine.volume_analysis(candles)

    # Full series for charting
    sma20_full = engine.sma(closes, 20)
    sma50_full = engine.sma(closes, 50)
    bb_upper_full, bb_mid_full, bb_lower_full = engine.bollinger(closes)

    ind = {
        "rsi": rsi,
        "macd_line": line[-1], "macd_sig": sig_l[-1], "macd_hist": hist[-1],
        "sma20": s20[-1], "sma50": s50[-1], "sma100": s100[-1] if s100[-1] is not None else None,
        "ema20": e20[-1], "ema50": e50[-1], "ema200": e200[-1],
        "adx": adx[-1], "pdi": pdi[-1], "mdi": mdi[-1],
        "boll_upper": upper[-1], "boll_mid": mid[-1], "boll_lower": lower[-1],
        "stoch_k": stoch_k[-1], "stoch_d": stoch_d[-1],
        "williams_r": wr[-1],
        "cci": cci[-1],
        "tenkan": tenkan[-1], "kijun": kijun[-1],
        "senkou_a": senkou_a[-1], "senkou_b": senkou_b[-1],
        "vwap": vwap[-1],
        "slope": sl,
        "volume": va,
        "roc": roc[-1],
        # Full series for charts
        "sma20_full": sma20_full,
        "sma50_full": sma50_full,
        "bb_upper_full": bb_upper_full,
        "bb_mid_full": bb_mid_full,
        "bb_lower_full": bb_lower_full,
    }

    # ---- derivatives alt-data snapshot ----
    # Funding = validated (full history, trained on). OI change & order-book
    # imbalance are live-only diagnostics (no history to validate) and the
    # panel labels them as such.
    alt = None
    try:
        alt = altdata.live_alt_snapshot(coin, tf)
    except Exception as e:
        alt = {"error": str(e)}

    # ---- ML verdict (when a trained model exists for this timeframe) ----
    ml = None
    artifact = model_artifact(tf)
    if artifact is not None:
        try:
            ml = mlmod.live_predict(candles, tf, coin, artifact=artifact)
            if ml is None:
                ml = {"error": "model artifact failed to predict"}
        except ValueError as e:
            ml = {"error": str(e)}
        except Exception as e:  # never let ML crash the whole app
            ml = {"error": f"ML inference failed: {e}"}

    return {
        "coin": coin, "tf": tf, "limit": limit,
        "candles": candles,
        "source_msg": buf.getvalue().strip(),
        "price": price, "atr": atr_val,
        "atr_pct": atr_val / price * 100 if price else 0.0,
        "resistance": resistance, "support": support,
        "sig": sig, "bt": bt, "patterns": patterns,
        "divergences": divergences, "regime": regime, "ind": ind,
        "ml": ml, "alt": alt,
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def verdict_banner(sig: dict, tf: str) -> str:
    direction = sig["direction"]
    conf = sig["confidence"].capitalize()
    color = DIR_COLORS[direction]
    icon = DIR_ICONS[direction]
    bg = {"UP": "#dcfce7", "DOWN": "#fee2e2", "SIDEWAYS": "#f3f4f6"}[direction]
    return f"""
    <div style="padding:20px 24px;border-radius:14px;background:{bg};
                border:1px solid {color}44;text-align:center;">
      <div style="font-size:14px;letter-spacing:2px;color:#64748b;">
        NEXT {tf.upper()} CANDLE PREDICTION</div>
      <div style="font-size:44px;font-weight:800;color:{color};
                  margin:4px 0 2px;">{icon} {direction}</div>
      <div style="font-size:16px;color:#475569;">
        Confidence: <b>{conf}</b> ({sig['confidence_value'] * 100:.0f}%) &nbsp;·&nbsp;
        Score: <b>{sig['total']:+.2f}</b></div>
    </div>
    """


def regime_badge(regime: dict) -> str:
    r = regime["regime"]
    color = REGIME_COLORS.get(r, "#6b7280")
    icons = {"trending": "📈", "ranging": "📊", "volatile": "🌪️", "transitional": "🔄"}
    direction_str = regime["direction"].capitalize() if regime["direction"] != "neutral" else ""
    return f"""
    <div style="display:inline-block;padding:8px 16px;border-radius:20px;
                background:{color}18;border:1px solid {color}55;
                font-size:14px;color:{color};font-weight:600;">
      {icons.get(r, '')} {r.upper()} {f'· {direction_str}' if direction_str else ''}
    </div>
    """


def build_chart(candles: list[dict], ind: dict) -> go.Figure:
    ts = [datetime.fromtimestamp(c["ts"] / 1000) for c in candles]
    o = [c["open"] for c in candles]
    h = [c["high"] for c in candles]
    l = [c["low"] for c in candles]
    cl = [c["close"] for c in candles]
    vols = [c.get("vol", 0.0) for c in candles]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=("Price & Indicators", "Volume", "RSI / Stochastic"),
    )

    fig.add_trace(go.Candlestick(
        x=ts, open=o, high=h, low=l, close=cl, name="OHLC",
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626"),
        row=1, col=1)

    # Overlays
    for values, name, color, width, dash in [
        (ind.get("sma20_full"), "SMA20", "#f59e0b", 1.2, "solid"),
        (ind.get("sma50_full"), "SMA50", "#3b82f6", 1.2, "solid"),
        (ind.get("bb_upper_full"), "BB upper", "#94a3b8", 0.8, "dash"),
        (ind.get("bb_lower_full"), "BB lower", "#94a3b8", 0.8, "dash"),
        (ind.get("bb_mid_full"), "BB mid", "#94a3b8", 0.6, "dot"),
    ]:
        if values is None:
            continue
        fig.add_trace(go.Scatter(
            x=ts, y=values, name=name, mode="lines",
            line=dict(color=color, width=width, dash=dash)),
            row=1, col=1)

    # Volume
    vol_colors = ["#22c55e" if c["close"] >= c["open"] else "#ef4444" for c in candles]
    fig.add_trace(go.Bar(x=ts, y=vols, name="Volume",
                         marker_color=vol_colors, opacity=0.75), row=2, col=1)

    # RSI
    rsi_vals = engine.rsi_series(cl, 14)
    fig.add_trace(go.Scatter(
        x=ts, y=rsi_vals, name="RSI(14)", mode="lines",
        line=dict(color="#8b5cf6", width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(220,38,38,0.4)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(22,163,74,0.4)", row=3, col=1)

    fig.update_layout(
        height=700, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
        template="plotly_dark",
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="Osc", range=[0, 100], row=3, col=1)
    return fig


def render_patterns(patterns: list[tuple[str, int, str]]) -> None:
    if not patterns:
        st.info("No candlestick patterns detected.")
        return
    bias_map = {1: "▲ Bullish", -1: "▼ Bearish", 0: "◆ Neutral"}
    df = pd.DataFrame(
        [{"Pattern": n, "Bias": bias_map[s], "Description": d}
         for n, s, d in patterns])
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_divergences(divergences: list[tuple[str, int, str]]) -> None:
    if not divergences:
        st.info("No divergences detected.")
        return
    bias_map = {1: "▲ Bullish", -1: "▼ Bearish"}
    df = pd.DataFrame(
        [{"Divergence": n, "Bias": bias_map.get(s, "?"), "Description": d}
         for n, s, d in divergences])
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_indicators(ind: dict) -> None:
    c1, c2, c3 = st.columns(3)

    rsi = ind["rsi"]
    if rsi is not None:
        delta = "oversold" if rsi < 30 else ("overbought" if rsi > 70 else "neutral")
    else:
        delta = None
    c1.metric("RSI(14)", f"{rsi:.0f}" if rsi is not None else "n/a", delta)

    macd = ind["macd_hist"]
    if macd is not None:
        macd_delta = "bullish" if macd > 0 else "bearish"
    else:
        macd_delta = None
    c2.metric("MACD histogram", f"{macd:+.4f}" if macd is not None else "n/a", macd_delta)

    adx = ind["adx"]
    if adx is not None:
        adx_state = "trending" if adx > 25 else "choppy/ranging" if adx < 20 else "transitional"
    else:
        adx_state = None
    c3.metric("ADX(14)", f"{adx:.0f}" if adx is not None else "n/a",
              f"{adx_state} · +DI {ind['pdi']:.0f} / -DI {ind['mdi']:.0f}" if adx_state else None)

    c4, c5, c6 = st.columns(3)

    sk = ind["stoch_k"]
    sd = ind["stoch_d"]
    if sk is not None:
        stoch_state = "oversold" if sk < 20 else ("overbought" if sk > 80 else "neutral")
    else:
        stoch_state = None
    c4.metric("Stochastic %K/%D", f"{sk:.0f}/{sd:.0f}" if sk is not None else "n/a", stoch_state)

    wr = ind["williams_r"]
    if wr is not None:
        wr_state = "oversold" if wr < -80 else ("overbought" if wr > -20 else "neutral")
    else:
        wr_state = None
    c5.metric("Williams %R", f"{wr:.0f}" if wr is not None else "n/a", wr_state)

    cci = ind["cci"]
    if cci is not None:
        cci_state = "oversold" if cci < -100 else ("overbought" if cci > 100 else "neutral")
    else:
        cci_state = None
    c6.metric("CCI(20)", f"{cci:.0f}" if cci is not None else "n/a", cci_state)

    c7, c8, c9 = st.columns(3)

    if ind["sma20"] is not None and ind["sma50"] is not None:
        trend = "uptrend" if ind["sma20"] > ind["sma50"] else "downtrend"
        c7.metric("SMA alignment", trend, f"20: ${ind['sma20']:,.2f} / 50: ${ind['sma50']:,.2f}")
    else:
        c7.metric("SMA alignment", "n/a")

    sl = ind["slope"]
    sl_lbl = "rising" if sl > 0.001 else "falling" if sl < -0.001 else "flat"
    c8.metric("20c slope", sl_lbl, f"{sl * 100:+.3f}%/candle")

    roc = ind.get("roc")
    if roc is not None:
        roc_lbl = "rising" if roc > 2 else ("falling" if roc < -2 else "flat")
        c9.metric("ROC(12)", f"{roc:+.1f}%", roc_lbl)
    else:
        c9.metric("ROC(12)", "n/a")

    # Ichimoku
    c10, c11, c12 = st.columns(3)
    if ind["tenkan"] is not None and ind["kijun"] is not None:
        tk_bias = "bullish" if ind["tenkan"] > ind["kijun"] else "bearish"
        c10.metric("Ichimoku Tenkan/Kijun", tk_bias, f"{ind['tenkan']:,.2f} / {ind['kijun']:,.2f}")
    else:
        c10.metric("Ichimoku Tenkan/Kijun", "n/a")

    if ind["senkou_a"] is not None and ind["senkou_b"] is not None:
        cloud_top = max(ind["senkou_a"], ind["senkou_b"])
        cloud_bot = min(ind["senkou_a"], ind["senkou_b"])
        price = ind.get("sma20", 0)  # approximate
        c11.metric("Cloud", f"{cloud_bot:,.2f} – {cloud_top:,.2f}")
    else:
        c11.metric("Cloud", "n/a")

    if ind["vwap"] is not None:
        c12.metric("VWAP(20)", f"${ind['vwap']:,.2f}")
    else:
        c12.metric("VWAP(20)", "n/a")

    # Volume
    va = ind["volume"]
    if va:
        vol_lbl = "rising" if va["trend"] > 0.1 else "falling" if va["trend"] < -0.1 else "flat"
        st.metric("Volume", vol_lbl, f"5c/20c {va['trend'] * 100:+.0f}%  ·  OBV {'rising' if va['obv_slope'] > 0.001 else 'falling' if va['obv_slope'] < -0.001 else 'flat'}"
                  + ("  ⚡ CLIMAX" if va.get("climax") else ""))
    else:
        st.metric("Volume", "n/a")


def render_signal_breakdown(sig: dict) -> None:
    labels = {
        "patterns": "Patterns", "rsi": "RSI", "macd": "MACD",
        "sma": "SMA trend", "adx": "ADX/DI", "bollinger": "Bollinger",
        "slope": "Slope", "stoch": "Stochastic", "williams": "Williams %R",
        "cci": "CCI", "ichimoku": "Ichimoku", "vwap": "VWAP",
        "obv": "OBV", "roc": "ROC", "divergence": "Divergence",
    }
    rows = []
    for k, v in sig["components"].items():
        w = sig["weights"].get(k, 1.0)
        rows.append({
            "Signal": labels.get(k, k),
            "Raw Score": v,
            "Weight": w,
            "Weighted": v * w,
        })
    rows.append({
        "Signal": "**TOTAL**",
        "Raw Score": sig["raw_total"],
        "Weight": None,
        "Weighted": sig["total"],
    })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        column_config={
            "Raw Score": st.column_config.ProgressColumn(
                "Raw Score", min_value=-6, max_value=6, format="%.2f"),
            "Weight": st.column_config.NumberColumn("Weight", format="%.1f"),
            "Weighted": st.column_config.ProgressColumn(
                "Weighted", min_value=-15, max_value=15, format="%.2f"),
        },
        use_container_width=True, hide_index=True)


def render_backtest(bt: dict) -> None:
    c1, c2, c3 = st.columns(3)
    if bt["bull_n"]:
        c1.metric("Bullish accuracy", f"{bt['bull'] * 100:.0f}%",
                   f"{bt['bull_n']} calls")
    else:
        c1.metric("Bullish accuracy", "—", "no signals")
    if bt["bear_n"]:
        c2.metric("Bearish accuracy", f"{bt['bear'] * 100:.0f}%",
                   f"{bt['bear_n']} calls")
    else:
        c2.metric("Bearish accuracy", "—", "no signals")
    if bt["overall"] is not None:
        c3.metric("Overall accuracy", f"{bt['overall'] * 100:.0f}%",
                   f"{bt['overall_n']} total signals")
    else:
        c3.metric("Overall accuracy", "—")

    st.caption("For each historical candle, the signal was computed from all data "
               "up to that point and compared against the next candle's actual direction. "
               "Bullish = predicted UP and price went up; Bearish = predicted DOWN "
               "and price went down; Sideways counted correct if within ±0.5%.")


def ml_verdict_banner(ml: dict, tf: str) -> str:
    """Primary verdict banner driven by the calibrated probabilities.

    Schema 2 (binary up/down): a directional call ONLY when the calibrated
    probability is in the top ~20% of historical confidence; otherwise the
    honest "NO CLEAR EDGE". Legacy multiclass artifacts keep the old banner.
    """
    direction = ml["prediction"]
    conf = ml["confidence"]
    color = DIR_COLORS.get(direction, "#6b7280")
    icon = DIR_ICONS.get(direction, "∅")
    bg = {"UP": "#dcfce7", "DOWN": "#fee2e2",
          "SIDEWAYS": "#f3f4f6", "NO CLEAR EDGE": "#f3f4f6"}[direction]
    if ml.get("schema") == 2:
        gate = ml.get("gate") or {}
        top_pct = gate.get("top_frac", 0.2) * 100
        pct_thr = gate.get("pct_thr", 80.0)
        tf_label = (f"NEXT {tf.upper()} CANDLE · ML MODEL (LightGBM, calibrated)"
                    f" · GATE: TOP {top_pct:.0f}% OF HISTORICAL CONFIDENCE")
        if direction == "NO CLEAR EDGE":
            up_p, dn_p = gate.get("up_pctile"), gate.get("dn_pctile")
            pct_txt = ""
            if up_p is not None and dn_p is not None:
                pct_txt = (f"({up_p:.0f}th / {dn_p:.0f}th percentile of historical "
                           f"confidence) ")
            sub = (f"P(UP) {ml['p_up'] * 100:.1f}% · P(DOWN) {ml['p_dn'] * 100:.1f}% "
                   f"{pct_txt}— below the top-{top_pct:.0f}% historical-confidence "
                   f"gate (≥ {pct_thr:.0f}th percentile). No directional call. "
                   f"'I don't know' beats a fake label.")
        else:
            pct = gate.get("up_pctile") if direction == "UP" else gate.get("dn_pctile")
            pct_txt = f", {pct:.0f}th percentile of historical confidence" if pct else ""
            sub = (f"Calibrated P({direction}) = {conf * 100:.1f}%{pct_txt} — "
                   f"in the top {top_pct:.0f}% (gate ≥ {pct_thr:.0f}th percentile)")
        return f"""
        <div style="padding:20px 24px;border-radius:14px;background:{bg};
                    border:1px solid {color}44;text-align:center;">
          <div style="font-size:13px;letter-spacing:2px;color:#64748b;">{tf_label}</div>
          <div style="font-size:44px;font-weight:800;color:{color};
                      margin:4px 0 2px;">{icon} {direction}</div>
          <div style="font-size:15px;color:#475569;">{sub}</div>
        </div>
        """
    # legacy multiclass banner
    return f"""
    <div style="padding:20px 24px;border-radius:14px;background:{bg};
                border:1px solid {color}44;text-align:center;">
      <div style="font-size:14px;letter-spacing:2px;color:#64748b;">
        NEXT {tf.upper()} CANDLE · ML MODEL (LightGBM, calibrated · LEGACY 3-CLASS)</div>
      <div style="font-size:44px;font-weight:800;color:{color};
                  margin:4px 0 2px;">{icon} {direction}</div>
      <div style="font-size:16px;color:#475569;">
        Calibrated confidence: <b>{conf * 100:.0f}%</b> of the model's top class
        (class = argmax of the three calibrated probabilities below)</div>
    </div>
    """


def render_ml_probabilities(ml: dict) -> None:
    """Calibrated probabilities - the real, honest numbers."""
    if ml.get("schema") == 2:
        st.markdown("#### Calibrated directional probabilities")
        c1, c2, c3 = st.columns(3)
        for col, cls, p in ((c1, "UP", ml["p_up"]), (c2, "DOWN", ml["p_dn"]),
                            (c3, "SIDEWAYS (derived)", ml["p_sideways"])):
            color = DIR_COLORS.get(cls, "#6b7280")
            col.markdown(
                f"<div style='text-align:center;font-size:15px;font-weight:700;"
                f"color:{color};'>{DIR_ICONS.get(cls, '◆')} {cls}</div>",
                unsafe_allow_html=True)
            col.progress(min(p, 1.0))
            col.caption(f"{p * 100:.1f}%")
        gate = ml.get("gate") or {}
        up_pct, dn_pct = gate.get("up_pctile"), gate.get("dn_pctile")
        pct_txt = ""
        if up_pct is not None:
            pct_txt += f" · P(UP) is at the {up_pct:.0f}th percentile of historical confidence"
        if dn_pct is not None:
            pct_txt += f" · P(DOWN) at the {dn_pct:.0f}th"
        st.caption(
            "UP = next candle moves more than +0.5 × ATR(14) · DOWN = less than "
            "−0.5 × ATR(14) · SIDEWAYS = within ±0.5 × ATR(14). Isotonic-calibrated "
            "binary probabilities; the SIDEWAYS figure is derived "
            "(1 − P(UP) − P(DOWN)), not independently modelled."
            + (pct_txt + " (walk-forward out-of-fold reference distribution)." if pct_txt else ""))
        return
    # legacy
    st.markdown("#### Model probabilities (calibrated · legacy multiclass)")
    cols = st.columns(len(ml["classes"]))
    for col, cls in zip(cols, ml["classes"]):
        p = ml["proba"][cls]
        color = DIR_COLORS[cls]
        col.markdown(
            f"<div style='text-align:center;font-size:15px;font-weight:700;"
            f"color:{color};'>{DIR_ICONS[cls]} {cls}</div>",
            unsafe_allow_html=True)
        col.progress(min(p, 1.0))
        col.caption(f"{p * 100:.1f}%")
    cls_label = """
    UP = next candle moves more than +0.5 × ATR(14) · DOWN = less than
    −0.5 × ATR(14) · SIDEWAYS = within ±0.5 × ATR(14). Probabilities come
    from the trained model after per-class isotonic calibration.
    """
    st.caption(cls_label)


def _fmt_ms(ms: str) -> str:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=datetime.now().astimezone().tzinfo)\
            .strftime("%Y-%m-%d")
    except Exception:
        return ms


def render_model_info(ml: dict) -> None:
    """Traceable model info - every number on screen maps to a real metric."""
    meta = ml.get("meta") or {}
    is_binary = ml.get("schema") == 2
    metrics = meta.get("metrics") or {}
    baselines = meta.get("baselines") or {}
    verdict = meta.get("verdict") or "?"
    reasons = meta.get("verdict_reasons") or []

    with st.expander("🤖 Model info & validation (traceable numbers)", expanded=False):
        # ---- verdict banner with explicit reasons (never hidden) ----
        if verdict == "POSITIVE NET EDGE":
            st.success("Verdict: **POSITIVE NET EDGE** — gated calls made money "
                       "after fees on the held-out test set.")
        elif verdict == "SIGNAL BUT NOT TRADEABLE":
            st.warning("Verdict: **SIGNAL BUT NOT TRADEABLE** — the model beats "
                       "climatology, but following only the gated calls does "
                       "not survive fees on the test set.")
        elif verdict == "DEPLOYABLE":
            st.info("Verdict (legacy schema): DEPLOYABLE — beat coin-flip / "
                    "majority-class / old rule engine on the held-out test set. "
                    "Note this legacy framing did not test fee-adjusted trading.")
        else:
            st.error("Verdict: **NOT DEPLOYABLE** — no exploitable edge on the "
                     "held-out test set.")
        for r in reasons:
            st.caption(f"• {r}")
        if ml.get("schema") == 1:
            st.caption("⚠️ Legacy 3-class artifact: accuracy was the old primary "
                       "metric and the SIDEWAYS majority inflates it. Retrain to "
                       "get the binary P(UP)/P(DOWN) framing with Brier/BSS/AUC "
                       "and the fee-adjusted Kelly backtest.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last retrain", (meta.get("trained_at_utc") or "?")[:10])
        c2.metric("Training data", f"{_fmt_ms(str(meta['train_start']))} → "
                  f"{_fmt_ms(str(meta['train_end']))}" if meta.get("train_start") else "?")
        c3.metric("Out-of-sample test", f"{_fmt_ms(str(meta['test_start']))} → "
                  f"{_fmt_ms(str(meta['test_end']))}" if meta.get("test_start") else "?")
        c4.metric("Test rows", f"{meta.get('n_test_rows', 0):,}")

        fund_used = meta.get("funding_features_used")
        if fund_used is not None:
            st.caption("Funding-rate features in this model: "
                       + (", ".join(fund_used) if fund_used
                          else "none (no funding cache at train time)"))

        if is_binary and "up" in (metrics or {}):
            st.markdown("**Primary metrics — calibrated directional probabilities "
                        "on the held-out test set**")
            hdr = st.columns(5)
            hdr[0].markdown("<div style='font-size:12px;color:#64748b;'>METRIC</div>",
                            unsafe_allow_html=True)
            hdr[1].markdown("<div style='text-align:center;font-size:12px;color:#64748b;'>P(UP)</div>", unsafe_allow_html=True)
            hdr[2].markdown("<div style='text-align:center;font-size:12px;color:#64748b;'>P(DOWN)</div>", unsafe_allow_html=True)
            hdr[3].markdown("<div style='text-align:center;font-size:12px;color:#64748b;'>logistic UP</div>", unsafe_allow_html=True)
            hdr[4].markdown("<div style='text-align:center;font-size:12px;color:#64748b;'>logistic DN</div>", unsafe_allow_html=True)
            m_up, m_dn = metrics["up"], metrics["dn"]
            lg = meta.get("metrics_logistic") or {}
            rows = [
                ("Brier (lower=better)", f"{m_up['brier']:.4f}", f"{m_dn['brier']:.4f}",
                 f"{(lg.get('up') or {}).get('brier', float('nan')):.4f}",
                 f"{(lg.get('dn') or {}).get('brier', float('nan')):.4f}"),
                ("Brier climatology", f"{m_up['brier_climatology']:.4f}",
                 f"{m_dn['brier_climatology']:.4f}", "—", "—"),
                ("BSS vs climatology", f"{m_up['bss']:+.3f}", f"{m_dn['bss']:+.3f}",
                 "—", "—"),
                ("ROC-AUC", f"{m_up['roc_auc']:.3f}", f"{m_dn['roc_auc']:.3f}",
                 f"{(lg.get('up') or {}).get('roc_auc', float('nan')):.3f}",
                 f"{(lg.get('dn') or {}).get('roc_auc', float('nan')):.3f}"),
                ("ECE (calibration)", f"{m_up['ece']:.3f}", f"{m_dn['ece']:.3f}", "—", "—"),
                ("Base rate (test)", f"{m_up['base_rate']:.3f}", f"{m_dn['base_rate']:.3f}",
                 "—", "—"),
            ]
            dfm = pd.DataFrame(rows, columns=["Metric", "P(UP)", "P(DOWN)",
                                              "logistic UP", "logistic DN"])
            st.dataframe(dfm, use_container_width=True, hide_index=True)
            oof = meta.get("oof_metrics") or {}
            if oof:
                st.caption(f"Walk-forward OOF reference: UP BSS "
                           f"{(oof.get('up') or {}).get('bss', float('nan')):+.3f}, "
                           f"AUC {(oof.get('up') or {}).get('roc_auc', float('nan')):.3f} · "
                           f"DOWN BSS {(oof.get('dn') or {}).get('bss', float('nan')):+.3f}, "
                           f"AUC {(oof.get('dn') or {}).get('roc_auc', float('nan')):.3f}")

            # confidence-gate quality on test
            tgq = meta.get("test_gate_quality") or {}
            if tgq:
                st.markdown("**Confidence gate on the held-out test set** "
                            "(top ~20% of historical confidence)")
                g1, g2 = st.columns(2)
                for col, key, lbl in ((g1, "up", "UP calls"), (g2, "dn", "DOWN calls")):
                    q = tgq.get(key) or {}
                    if not q:
                        continue
                    col.metric(
                        lbl,
                        f"{q.get('hit_rate', float('nan')):.1%} hit"
                        if np.isfinite(q.get("hit_rate", float("nan"))) else "no calls",
                        f"{q.get('n_calls', 0):,} calls · base {q.get('base_rate', 0):.1%} "
                        f"· uplift {q.get('uplift', 0):+.1%}")
                st.caption("Hit rate at the gate vs the unconditional base rate. "
                           "The uplift — not raw accuracy — is the model's edge.")

            # 3-class reference, labelled as reference-only
            ref3 = baselines.get("reference_3class") or {}
            if ref3:
                st.markdown("**Reference only — legacy 3-class view** "
                            "(majority-inflated; not the success metric)")
                r1, r2, r3 = st.columns(3)
                r1.metric("3-class accuracy", f"{ref3.get('accuracy', 0):.1%}")
                r2.metric(f"Always '{ref3.get('majority_class', '?')}'",
                          f"{ref3.get('majority_acc', 0):.1%}")
                old = baselines.get("reference_3class_old_rule_acc")
                r3.metric("Old rule engine (3-class)",
                          f"{old:.1%}" if old is not None else "n/a")
        elif metrics and "accuracy" in metrics:
            # legacy multiclass metrics
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Test accuracy (legacy)", f"{metrics.get('accuracy', 0):.1%}")
            m2.metric("ECE", f"{metrics.get('ece', 0):.3f}")
            m3.metric("Multiclass Brier", f"{metrics.get('brier_multi', 0):.4f}")
            aucs = metrics.get("roc_auc_ovr") or []
            if aucs:
                m4.metric("AUC UP / DOWN", f"{aucs[2]:.2f} / {aucs[0]:.2f}")
            m5.metric("Confidence floor*", f"{1 - metrics.get('ece', 0):.0%}")
            classes = ml["classes"]
            dfp = pd.DataFrame({
                "Class": classes,
                "Precision": [f"{p:.1%}" for p in metrics.get("per_class_precision", [])],
                "Recall": [f"{r:.1%}" for r in metrics.get("per_class_recall", [])],
                "F1": [f"{f:.1%}" for f in metrics.get("per_class_f1", [])],
            })
            st.dataframe(dfp, use_container_width=True, hide_index=True)
            if baselines:
                b1, b2, b3 = st.columns(3)
                b1.metric("Coin flip", f"{baselines.get('coin_flip_acc', 0):.1%}")
                b2.metric(f"Majority class ({baselines.get('majority_class', '?')})",
                          f"{baselines.get('majority_acc', 0):.1%}")
                old = baselines.get("old_rule_acc")
                b3.metric("Old rule-based engine", f"{old:.1%}" if old is not None else "n/a")
            st.caption("*‘Confidence floor’ = 1 − ECE: how often a 70%-confident "
                       "prediction is right ~70% of the time (perfect = 1.0).")

        render_kelly_backtest(meta.get("trading"))

        if not metrics:
            st.caption("No validation metrics in this artifact.")


def render_kelly_backtest(trading: dict | None) -> None:
    """Fee-adjusted fractional-Kelly backtest - THE deployability metric."""
    if not trading or not any(k.startswith("fee_") for k in trading):
        return
    bench = trading.get("_benchmark") or {}
    st.markdown("**Fee-adjusted backtest (gated calls only, half-Kelly capped "
                f"{bench.get('kelly_cap', 0.25):.0%})** — the deployability metric")
    bh = bench.get("pooled_buy_hold") or {}
    st.caption(
        f"Signal rate {bench.get('signal_rate', 0):.1%} of test bars · "
        f"mean size when trading {bench.get('mean_size_when_trading', 0):.1%} of "
        f"notional · pooled equal-weight buy & hold same window: "
        f"total {bh.get('total_return', 0):+.1%}, Sharpe {bh.get('sharpe', 0):+.2f}, "
        f"maxDD {bh.get('max_drawdown', 0):.1%}")
    rows = []
    for fee_key, r in trading.items():
        if not fee_key.startswith("fee_"):
            continue
        p = r.get("pooled") or {}
        pf = r.get("pooled_full_notional") or {}
        rows.append({
            "Fees/side": fee_key.replace("fee_", "").replace("pct", "%"),
            "Trades": f"{r.get('n_trades', 0):,}",
            "Win rate": f"{r.get('win_rate', 0):.1%}",
            "Avg/trade": f"{r.get('avg_ret_per_trade', 0):+.4%}",
            "Total": f"{p.get('total_return', 0):+.1%}",
            "Sharpe": f"{p.get('sharpe', 0):+.2f}",
            "Sharpe 1H/2H": f"{p.get('sharpe_first_half', 0):+.2f} / "
                            f"{p.get('sharpe_second_half', 0):+.2f}",
            "Max DD": f"{p.get('max_drawdown', 0):.1%}",
            "Sharpe (full notional)": f"{pf.get('sharpe', 0):+.2f}",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Kelly sizing = min(half-Kelly × (2p−1), cap) per call; fees charged "
               "proportionally to size. A positive Sharpe here — not accuracy — is "
               "the only number that decides whether this is useful.")


def render_alt_data(alt: dict | None) -> None:
    """Derivatives panel with explicit validated / live-only labelling."""
    if not alt or alt.get("error"):
        if alt:
            st.caption(f"Alt-data unavailable: {alt.get('error')}")
        return
    with st.expander("📡 Derivatives context (funding · open interest · order book)",
                     expanded=False):
        f = alt.get("funding") or {}
        c1, c2, c3 = st.columns(3)
        if f.get("rate") is not None:
            c1.metric("Funding rate (8h)", f"{f['rate'] * 100:+.4f}%",
                      f"annualised {f.get('rate_annualized', 0) * 100:+.1f}%")
            z = f.get("z30")
            c1.caption((f"30d z-score {z:+.2f}" if z is not None else "z-score n/a")
                       + " · ✅ validated: full history, trained on")
        else:
            c1.metric("Funding rate", "n/a")
            c1.caption("no funding cache — run `python altdata.py`")

        oi = alt.get("open_interest") or {}
        if oi.get("value") is not None:
            c2.metric("Open interest", f"${oi['value'] / 1e9:,.2f}B",
                      f"24h change {oi.get('change_24', 0) * 100:+.2f}%")
            c2.caption("⚠️ live-only: ~30d venue history, not trained on")
        else:
            c2.metric("Open interest", "n/a")
            c2.caption("⚠️ live-only (unavailable now)")

        ob = alt.get("orderbook") or {}
        if ob.get("imbalance") is not None:
            c3.metric("Order-book imbalance", f"{ob['imbalance']:+.3f}",
                      "bid-heavy" if ob["imbalance"] > 0 else "ask-heavy")
            c3.caption("⚠️ live-only snapshot: no history exists, not trained on")
        else:
            c3.metric("Order-book imbalance", "n/a")
            c3.caption("⚠️ live-only (unavailable now)")
        st.caption("Only funding rate has history deep enough to train and "
                   "backtest on. OI change and order-book imbalance are shown as "
                   "context and carry **no validated edge**.")


def render_results(r: dict) -> None:
    coin, tf = r["coin"], r["tf"]
    price, atr_val = r["price"], r["atr"]
    sig = r["sig"]
    regime = r["regime"]
    ml = r.get("ml")
    ml_active = isinstance(ml, dict) and "prediction" in ml

    if ml_active:
        st.markdown(ml_verdict_banner(ml, tf), unsafe_allow_html=True)
        st.write("")
        render_ml_probabilities(ml)
        render_model_info(ml)
    elif ml is not None and "error" in ml:
        st.warning(f"⚠️ ML model unavailable for this input: {ml['error']}")
        st.markdown(
            "Showing the classic rule-based verdict below as fallback.")
        st.write("")
        st.markdown(verdict_banner(sig, tf), unsafe_allow_html=True)
    else:
        st.markdown(verdict_banner(sig, tf), unsafe_allow_html=True)
        st.write("")
        st.caption("No trained ML model for this timeframe yet - verdict comes "
                   "from the classic rule-based engine. Train one with "
                   "`python train_model.py --tf <tf>` (or retrain.py; run "
                   "`python altdata.py` first so funding features are available).")
    st.write("")
    st.markdown(regime_badge(regime), unsafe_allow_html=True)
    st.write("")
    render_alt_data(r.get("alt"))
    st.write("")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current price", f"${price:,.4f}")
    c2.metric("ATR(14)", f"${atr_val:,.4f}", f"{r['atr_pct']:.1f}% of price")
    c3.metric("Expected range", f"${price - atr_val:,.4f} – ${price + atr_val:,.4f}")
    c4.metric("Resistance", f"${r['resistance']:,.4f}" if r["resistance"] else "—")
    c5.metric("Support", f"${r['support']:,.4f}" if r["support"] else "—")

    st.plotly_chart(build_chart(r["candles"], r["ind"]), use_container_width=True)

    if ml_active:
        st.caption("The tabs below are the **classic rule-based TA view** - kept "
                   "for transparency so you can compare it against the ML verdict.")
    tab_ind, tab_pat, tab_div, tab_sig, tab_bt, tab_src = st.tabs([
        "📊 Indicators", "🕯️ Patterns", "🔀 Divergences",
        "🎯 Classic signal breakdown", "📈 Classic backtest", "📡 Data source",
    ])

    with tab_ind:
        render_indicators(r["ind"])
    with tab_pat:
        render_patterns(r["patterns"])
    with tab_div:
        render_divergences(r["divergences"])
    with tab_sig:
        st.caption("OLD rule-based scoring engine (kept as comparison - the ML "
                   "model above is the primary verdict). Positive scores = "
                   "bullish, negative = bearish. Weights adapt to regime.")
        render_signal_breakdown(sig)
    with tab_bt:
        st.caption("Classic rule-based engine backtest (pre-ML).")
        render_backtest(r["bt"])
    with tab_src:
        if r["source_msg"]:
            st.code(r["source_msg"])
        else:
            st.caption("No provider messages captured.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔬 Advanced Crypto Predictor")
    st.caption("Calibrated directional probabilities (LightGBM) · surfaced only "
               "in the top ~20% of historical confidence, otherwise 'no clear "
               "edge' · fee-adjusted Kelly backtest is the deployability metric "
               "· classic TA kept as comparison")

    coin_choices = list(engine.KNOWN_IDS.keys())
    coin_choice = st.selectbox("Coin", coin_choices + ["Custom…"], index=0)
    if coin_choice == "Custom…":
        coin = st.text_input("Symbol or CoinGecko id", value="PEPE",
                             help="e.g. BTC, SOL, XRP, or a CoinGecko id "
                                  "like pepe, dogwifcoin").strip().upper()
    else:
        coin = coin_choice

    tf = st.selectbox("Timeframe", list(engine.TF_MAP.keys()), index=5,
                      help="Trained models exist for 1h / 4h / 1d; other "
                           "timeframes fall back to the classic rule engine. "
                           "ML features need >= ~170 candles.")
    limit = st.slider("Candles to fetch", 80, 1000, 300, step=20,
                      help="More candles = longer backtest, slower fetch. "
                           "Use >= 250 so the ML model has enough history.")
    run_btn = st.button("🔮 Predict next candle (ML + classic TA)",
                        type="primary", use_container_width=True)

    st.divider()
    st.caption("Inference is fully local (scikit-learn / LightGBM, no API calls "
               "at prediction time). Data providers (auto-fallback): Binance → "
               "Kraken → OKX → Bybit → CoinGecko.")

if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.params = None
    st.session_state.error = None

params = (coin, tf, limit)

# Live provider-by-provider status (updates while the fetch runs, so the
# user sees "trying Binance… trying Kraken…" instead of a blank wait).
fetch_status = st.empty()

if run_btn or st.session_state.results is None:
    def _fetch_progress(msg: str) -> None:
        fetch_status.caption(f"📡 {msg}")

    with st.spinner(f"Fetching {limit} {tf} candles for {coin}…"):
        try:
            st.session_state.results = run_prediction(
                coin, tf, limit, progress=_fetch_progress)
            st.session_state.params = params
            st.session_state.error = None
        except Exception as e:
            st.session_state.results = None
            st.session_state.error = str(e)
        finally:
            fetch_status.empty()

if st.session_state.error:
    st.error(f"⚠️ Prediction failed: {st.session_state.error}")
    st.caption("Check the coin symbol / network connection and try again.")
    st.stop()

if st.session_state.results is not None:
    if st.session_state.params != params:
        st.warning("Settings changed — press 'Predict next candle' to update.")
    render_results(st.session_state.results)

st.divider()
st.caption("⚠️ **Disclaimer:** a 70–80% 3-class accuracy on daily crypto bars is "
           "not achievable from OHLCV + public TA/derivatives data — that is an "
           "information ceiling, not a tuning problem, so this app does not chase "
           "it. It outputs calibrated P(UP)/P(DOWN), surfaces a directional call "
           "only in the top ~20% of historical confidence, says **NO CLEAR EDGE** "
           "otherwise, and reports Brier/BSS/AUC plus a fee-adjusted Kelly "
           "backtest as the only deployability test. An 'I don't know' here is a "
           "feature, not a bug. Past backtest performance does not guarantee "
           "future results. Not financial advice.")
