#!/usr/bin/env python3
"""
retrain.py
----------
Refresh + retrain the ML model for one or more timeframes.

Crypto regimes drift, so a model trained once decays. This script:
  1. tops up the raw candle cache to "now" (only downloads the new tail),
  2. rebuilds the feature cache,
  3. retrains (walk-forward validation + final model) and saves a NEW
     date-versioned artifact under models/,
  4. points models/manifest.json at the new artifact so app.py picks it up.

Schedule it (e.g. weekly) with cron / Task Scheduler:

    # weekly, Mondays 04:00 UTC
    0 4 * * 1  cd /path/to/crypto-screener && python retrain.py --all

Usage:
    python retrain.py --tf 1d
    python retrain.py --all
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

from build_dataset import SYMBOLS, TIMEFRAMES, fetch_or_load, save_candles
import altdata
import features as feat
import train_model as tm


def refresh_raw(tfs: list[str], symbols: list[str]) -> None:
    print("Refreshing raw candle cache to now…")
    for sym in symbols:
        for tf in tfs:
            candles = fetch_or_load(sym, tf, refresh=True)
            if candles:
                save_candles(sym, tf, candles)
                last = datetime.fromtimestamp(candles[-1]["ts"] / 1000,
                                              tz=timezone.utc)
                print(f"  {sym} {tf}: {len(candles):,} candles, latest {last}")
            else:
                print(f"  [fail] {sym} {tf}: no candles", file=sys.stderr)


def refresh_funding(symbols: list[str]) -> int:
    """Top up the funding-rate cache (the only alt-data with full history).

    Open interest / order-book imbalance are live-only (no venue history) and
    are intentionally NOT cached into the training path - see altdata.py.
    Returns the number of symbols with a usable funding cache.
    """
    print("Refreshing funding-rate cache (validated alt-data)…")
    ok = 0
    for sym in symbols:
        try:
            df = altdata.load_or_fetch_funding(sym, start_ms=0, refresh=True)
        except Exception as e:
            print(f"  [fail] {sym}: {e}", file=sys.stderr)
            continue
        if df.empty:
            print(f"  [fail] {sym}: no funding history", file=sys.stderr)
            continue
        ok += 1
        last = datetime.fromtimestamp(df["funding_time"].max() / 1000,
                                      tz=timezone.utc)
        print(f"  {sym}: {len(df):,} funding payments, latest {last}")
    print(f"funding cache usable for {ok}/{len(symbols)} symbols")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh data and retrain the model")
    parser.add_argument("--tf", default=None, help="timeframe to retrain (1h/4h/1d)")
    parser.add_argument("--all", action="store_true", help="retrain all timeframes")
    parser.add_argument("--no-refresh", action="store_true",
                        help="skip the data refresh (retrain on cached data)")
    args = parser.parse_args()

    if args.all:
        tfs = TIMEFRAMES
    elif args.tf:
        tfs = [args.tf]
    else:
        parser.print_help()
        return
    if args.tf and args.tf not in TIMEFRAMES:
        raise SystemExit(f"unknown tf {args.tf}; choose from {TIMEFRAMES}")

    t0 = time.time()
    if not args.no_refresh:
        refresh_raw(tfs, SYMBOLS)
        refresh_funding(SYMBOLS)
    for tf in tfs:
        feat.build_feature_cache(tfs=[tf])
        tm.train_and_evaluate(tf, rebuild_features=False)
    print(f"\nRetrain complete in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
