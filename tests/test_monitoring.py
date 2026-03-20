"""Tests for data quality and drift monitoring modules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.monitoring.data_quality import (
    validate_raw_data, validate_features,
    check_no_nulls, check_positive_close, check_ohlc_consistency,
    check_no_price_spikes, check_min_rows, check_positive_volume,
)
from src.monitoring.drift import compute_psi, compute_psi_report


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_ohlcv():
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.abs(close) + 10
    return pd.DataFrame({
        "Open":   close * 0.999,
        "High":   close * 1.01,
        "Low":    close * 0.99,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


# ── Data quality: raw ─────────────────────────────────────────────────────────

class TestRawValidation:

    def test_clean_data_passes_all_checks(self, clean_ohlcv):
        result = validate_raw_data(clean_ohlcv, "TEST")
        assert result["passed"] is True
        assert result["n_failed"] == 0

    def test_nulls_detected(self, clean_ohlcv):
        df = clean_ohlcv.copy()
        df.iloc[5, df.columns.get_loc("Close")] = np.nan
        r = check_no_nulls(df)
        assert r["passed"] is False

    def test_negative_close_detected(self, clean_ohlcv):
        df = clean_ohlcv.copy()
        df.iloc[0, df.columns.get_loc("Close")] = -1.0
        r = check_positive_close(df)
        assert r["passed"] is False

    def test_zero_volume_allowed(self, clean_ohlcv):
        """Index tickers have zero volume — should pass."""
        df = clean_ohlcv.copy()
        df["Volume"] = 0.0
        r = check_positive_volume(df)
        assert r["passed"] is True

    def test_negative_volume_detected(self, clean_ohlcv):
        df = clean_ohlcv.copy()
        df.iloc[0, df.columns.get_loc("Volume")] = -1.0
        r = check_positive_volume(df)
        assert r["passed"] is False

    def test_ohlc_inconsistency_detected(self, clean_ohlcv):
        df = clean_ohlcv.copy()
        # Make High less than Close on one row
        df.iloc[0, df.columns.get_loc("High")] = df.iloc[0]["Close"] * 0.5
        r = check_ohlc_consistency(df)
        assert r["passed"] is False

    def test_too_few_rows_detected(self, clean_ohlcv):
        r = check_min_rows(clean_ohlcv.iloc[:10], min_rows=200)
        assert r["passed"] is False

    def test_price_spike_detected(self, clean_ohlcv):
        df = clean_ohlcv.copy()
        df.iloc[10, df.columns.get_loc("Close")] = df.iloc[9]["Close"] * 5
        r = check_no_price_spikes(df, max_daily_move=0.30)
        assert r["passed"] is False

    def test_result_has_required_keys(self, clean_ohlcv):
        result = validate_raw_data(clean_ohlcv)
        for key in ["passed", "n_checks", "n_failed", "checks", "timestamp"]:
            assert key in result


# ── Data quality: features ────────────────────────────────────────────────────

class TestFeatureValidation:

    def test_valid_features_pass(self, clean_ohlcv):
        from src.features.indicators import add_all
        df = add_all(clean_ohlcv)
        result = validate_features(df, "TEST")
        assert result["passed"] is True

    def test_missing_column_detected(self, clean_ohlcv):
        from src.features.indicators import add_all
        df = add_all(clean_ohlcv).drop(columns=["RSI"])
        result = validate_features(df, "TEST")
        assert result["passed"] is False

    def test_null_feature_detected(self, clean_ohlcv):
        from src.features.indicators import add_all
        df = add_all(clean_ohlcv)
        df.iloc[0, df.columns.get_loc("MACD_hist")] = np.nan
        result = validate_features(df, "TEST")
        assert result["passed"] is False


# ── PSI drift ─────────────────────────────────────────────────────────────────

class TestPSI:

    def test_identical_distributions_near_zero(self):
        x = np.random.normal(0, 1, 500)
        psi = compute_psi(x, x.copy())
        assert psi < 0.01

    def test_shifted_distribution_high_psi(self):
        ref = np.random.normal(0, 1, 500)
        cur = np.random.normal(5, 1, 500)  # large shift
        psi = compute_psi(ref, cur)
        assert psi > 0.20

    def test_psi_non_negative(self):
        ref = np.random.normal(0, 1, 200)
        cur = np.random.normal(0.5, 1.2, 200)
        psi = compute_psi(ref, cur)
        assert psi >= 0

    def test_psi_report_stable_label(self, clean_ohlcv):
        from src.features.indicators import add_all
        df = add_all(clean_ohlcv)
        # Same data → should be stable
        report = compute_psi_report(df, df.copy(), ticker="TEST")
        assert report["status"] == "stable"
        assert report["max_psi"] < 0.10

    def test_psi_report_keys(self, clean_ohlcv):
        from src.features.indicators import add_all
        df = add_all(clean_ohlcv)
        report = compute_psi_report(df, df.copy())
        for key in ["status", "max_psi", "drifted", "feature_psi", "timestamp"]:
            assert key in report

    def test_drifted_status_on_shifted_data(self, clean_ohlcv):
        from src.features.indicators import add_all
        df_ref = add_all(clean_ohlcv)
        # Simulate a major regime change — multiply Close by 3
        df_shifted = clean_ohlcv.copy()
        df_shifted["Close"] *= 3
        df_shifted["High"]  *= 3
        df_shifted["Low"]   *= 3
        df_shifted["Open"]  *= 3
        df_cur = add_all(df_shifted)
        report = compute_psi_report(df_ref, df_cur, ticker="TEST")
        assert report["status"] in ("moderate", "drifted")
