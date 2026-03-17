"""
Data quality validation module.

Replaces a Great Expectations dependency with a lightweight,
zero-config validation suite that runs as part of the Airflow pipeline.

Each check returns {"name": str, "passed": bool, "detail": str}.
The suite fails fast: the Airflow task raises ValueError on any failure,
preventing bad data from flowing into the feature store.

Interview talking point:
  "I designed a validation layer between raw ingestion and feature computation.
   Every dataset must pass 7 checks before features are calculated — this prevents
   garbage-in-garbage-out and makes pipeline failures explicit rather than silent."
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


# ── Individual checks ─────────────────────────────────────────────────────────

def check_no_nulls(df: pd.DataFrame) -> dict:
    null_counts = df[["Open", "High", "Low", "Close", "Volume"]].isnull().sum()
    total_nulls = int(null_counts.sum())
    return {
        "name":   "no_null_values",
        "passed": total_nulls == 0,
        "detail": f"{total_nulls} null values found" if total_nulls else "ok",
    }


def check_positive_close(df: pd.DataFrame) -> dict:
    bad = int((df["Close"] <= 0).sum())
    return {
        "name":   "close_positive",
        "passed": bad == 0,
        "detail": f"{bad} non-positive Close prices" if bad else "ok",
    }


def check_positive_volume(df: pd.DataFrame) -> dict:
    bad = int((df["Volume"] <= 0).sum())
    return {
        "name":   "volume_positive",
        "passed": bad == 0,
        "detail": f"{bad} non-positive Volume values" if bad else "ok",
    }


def check_monotonic_dates(df: pd.DataFrame) -> dict:
    is_mono = df.index.is_monotonic_increasing
    return {
        "name":   "dates_monotonic",
        "passed": bool(is_mono),
        "detail": "ok" if is_mono else "date index is not monotonically increasing",
    }


def check_no_duplicate_dates(df: pd.DataFrame) -> dict:
    dupes = int(df.index.duplicated().sum())
    return {
        "name":   "no_duplicate_dates",
        "passed": dupes == 0,
        "detail": f"{dupes} duplicate dates" if dupes else "ok",
    }


def check_min_rows(df: pd.DataFrame, min_rows: int = 200) -> dict:
    n = len(df)
    return {
        "name":   f"min_{min_rows}_rows",
        "passed": n >= min_rows,
        "detail": f"only {n} rows (need ≥ {min_rows})" if n < min_rows else f"{n} rows ok",
    }


def check_ohlc_consistency(df: pd.DataFrame) -> dict:
    """High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close."""
    bad_high = int(((df["High"] < df["Low"]) |
                    (df["High"] < df["Open"]) |
                    (df["High"] < df["Close"])).sum())
    bad_low  = int(((df["Low"] > df["Open"]) |
                    (df["Low"] > df["Close"])).sum())
    bad = bad_high + bad_low
    return {
        "name":   "ohlc_consistency",
        "passed": bad == 0,
        "detail": f"{bad} OHLC inconsistencies" if bad else "ok",
    }


def check_no_price_spikes(df: pd.DataFrame, max_daily_move: float = 0.30) -> dict:
    """Flag single-day moves > 30% as likely data errors."""
    pct = df["Close"].pct_change().abs()
    spikes = int((pct > max_daily_move).sum())
    return {
        "name":   f"no_spikes_over_{int(max_daily_move*100)}pct",
        "passed": spikes == 0,
        "detail": f"{spikes} days with >{max_daily_move*100:.0f}% move" if spikes else "ok",
    }


# ── Validation suite ──────────────────────────────────────────────────────────

def validate_raw_data(df: pd.DataFrame, ticker: str = "") -> dict:
    """
    Run full raw data validation suite.
    Returns a result dict with 'passed' (bool) and 'checks' (list of check results).
    """
    checks = [
        check_no_nulls(df),
        check_positive_close(df),
        check_positive_volume(df),
        check_monotonic_dates(df),
        check_no_duplicate_dates(df),
        check_min_rows(df),
        check_ohlc_consistency(df),
        check_no_price_spikes(df),
    ]
    all_passed = all(c["passed"] for c in checks)
    failed     = [c["name"] for c in checks if not c["passed"]]

    return {
        "ticker":     ticker,
        "timestamp":  datetime.now().isoformat(),
        "passed":     all_passed,
        "n_checks":   len(checks),
        "n_failed":   len(failed),
        "failed":     failed,
        "checks":     checks,
    }


def validate_features(df: pd.DataFrame, ticker: str = "") -> dict:
    """
    Validate computed feature DataFrame.
    Checks that all expected feature columns exist and have no NaN values.
    """
    required_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "MACD_line", "MACD_signal", "MACD_hist",
        "RSI", "BB_upper", "BB_mid", "BB_lower", "ATR",
    ]
    checks = []

    # Check all columns exist
    missing = [c for c in required_cols if c not in df.columns]
    checks.append({
        "name":   "all_feature_columns_present",
        "passed": len(missing) == 0,
        "detail": f"missing: {missing}" if missing else "ok",
    })

    # Check no NaN in features (add_all() should have handled warmup)
    if not missing:
        null_counts = df[required_cols].isnull().sum()
        total_nulls = int(null_counts.sum())
        checks.append({
            "name":   "no_null_features",
            "passed": total_nulls == 0,
            "detail": f"{total_nulls} null values in features" if total_nulls else "ok",
        })

    # Check RSI is in [0, 100]
    if "RSI" in df.columns:
        bad_rsi = int(((df["RSI"] < 0) | (df["RSI"] > 100)).sum())
        checks.append({
            "name":   "rsi_range_valid",
            "passed": bad_rsi == 0,
            "detail": f"{bad_rsi} RSI values out of [0,100]" if bad_rsi else "ok",
        })

    # Check BB ordering: upper >= mid >= lower
    if all(c in df.columns for c in ["BB_upper", "BB_mid", "BB_lower"]):
        bad_bb = int(((df["BB_upper"] < df["BB_mid"]) |
                      (df["BB_mid"] < df["BB_lower"])).sum())
        checks.append({
            "name":   "bollinger_band_ordering",
            "passed": bad_bb == 0,
            "detail": f"{bad_bb} rows with invalid BB ordering" if bad_bb else "ok",
        })

    all_passed = all(c["passed"] for c in checks)
    return {
        "ticker":    ticker,
        "timestamp": datetime.now().isoformat(),
        "passed":    all_passed,
        "checks":    checks,
    }


# ── CLI for quick validation ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from src.data.downloader import download
    from src.features.indicators import add_all

    ticker = sys.argv[1] if len(sys.argv) > 1 else "^TWII"
    print(f"Validating {ticker}...")

    df_raw = download(ticker, start="2020-01-01")
    raw_result = validate_raw_data(df_raw, ticker)
    print(f"\nRaw data: {'PASSED' if raw_result['passed'] else 'FAILED'}")
    for c in raw_result["checks"]:
        status = "✓" if c["passed"] else "✗"
        print(f"  {status} {c['name']}: {c['detail']}")

    df_feat = add_all(df_raw)
    feat_result = validate_features(df_feat, ticker)
    print(f"\nFeatures: {'PASSED' if feat_result['passed'] else 'FAILED'}")
    for c in feat_result["checks"]:
        status = "✓" if c["passed"] else "✗"
        print(f"  {status} {c['name']}: {c['detail']}")
