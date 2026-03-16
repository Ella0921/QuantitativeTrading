"""
Unit tests for data downloader and feature engineering.
Uses offline mocking — no actual network calls.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data.downloader import get_train_test_split
from src.features.indicators import (
    add_macd, add_rsi, add_bollinger_bands, add_atr, add_all, to_macd_series
)


# ── Fixture: mock OHLCV DataFrame ─────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv():
    n = 120
    np.random.seed(42)
    dates  = pd.date_range("2020-01-01", periods=n, freq="B")
    close  = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "Open":   close * 0.999,
        "High":   close * 1.01,
        "Low":    close * 0.99,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


# ── get_train_test_split ──────────────────────────────────────────────────────

class TestTrainTestSplit:

    def test_basic_split(self, sample_ohlcv):
        train, test = get_train_test_split(sample_ohlcv, train_end="2020-03-31")
        assert len(train) > 0
        assert len(test) > 0
        assert train.index[-1] <= pd.Timestamp("2020-03-31")
        assert test.index[0] >= pd.Timestamp("2020-03-31")

    def test_no_overlap(self, sample_ohlcv):
        train, test = get_train_test_split(
            sample_ohlcv,
            train_end="2020-03-31",
            test_start="2020-04-01",
        )
        assert train.index[-1] < test.index[0]

    def test_original_df_unchanged(self, sample_ohlcv):
        original_len = len(sample_ohlcv)
        get_train_test_split(sample_ohlcv, "2020-03-31")
        assert len(sample_ohlcv) == original_len


# ── Indicators ────────────────────────────────────────────────────────────────

class TestIndicators:

    def test_add_macd_columns(self, sample_ohlcv):
        df = add_macd(sample_ohlcv)
        for col in ["MACD_line", "MACD_signal", "MACD_hist"]:
            assert col in df.columns

    def test_add_macd_no_mutation(self, sample_ohlcv):
        """add_macd should not mutate the input DataFrame."""
        original_cols = set(sample_ohlcv.columns)
        add_macd(sample_ohlcv)
        assert set(sample_ohlcv.columns) == original_cols

    def test_add_rsi_range(self, sample_ohlcv):
        df = add_rsi(sample_ohlcv)
        valid = df["RSI"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_add_bollinger_bands_relationship(self, sample_ohlcv):
        df = add_bollinger_bands(sample_ohlcv)
        valid = df[["BB_upper", "BB_mid", "BB_lower"]].dropna()
        assert (valid["BB_upper"] >= valid["BB_mid"]).all()
        assert (valid["BB_mid"]   >= valid["BB_lower"]).all()

    def test_add_atr_positive(self, sample_ohlcv):
        df = add_atr(sample_ohlcv)
        valid = df["ATR"].dropna()
        assert (valid > 0).all()

    def test_add_all_drops_nans(self, sample_ohlcv):
        df = add_all(sample_ohlcv)
        assert df.isna().sum().sum() == 0

    def test_add_all_shorter_than_input(self, sample_ohlcv):
        """add_all drops warmup rows, so result must be shorter."""
        df = add_all(sample_ohlcv)
        assert len(df) < len(sample_ohlcv)

    def test_to_macd_series_length(self, sample_ohlcv):
        prices = sample_ohlcv["Close"].tolist()
        macd   = to_macd_series(prices)
        assert len(macd) == len(prices)

    def test_to_macd_series_all_finite(self, sample_ohlcv):
        prices = sample_ohlcv["Close"].tolist()
        macd   = to_macd_series(prices)
        assert all(np.isfinite(v) for v in macd)

    def test_macd_converges_on_flat_prices(self):
        flat   = [100.0] * 200
        macd   = to_macd_series(flat)
        # After enough flat prices, MACD histogram should be near zero
        assert abs(macd[-1]) < 1e-8
