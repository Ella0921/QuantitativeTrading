"""
Unit tests for the backtest engine and feature indicators.
Run: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine, BacktestResult
from src.features.indicators import to_macd_series, add_macd, add_rsi


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_prices():
    """100 days of flat price at 100."""
    return [100.0] * 100


@pytest.fixture
def trending_prices():
    """100 days of linearly increasing price from 100 to 200."""
    return list(np.linspace(100, 200, 100))


@pytest.fixture
def sample_df():
    n = 100
    dates = pd.date_range("2020-01-01", periods=n)
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "Open": price,
        "High": price * 1.01,
        "Low": price * 0.99,
        "Close": price,
        "Volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


# ── BacktestEngine tests ──────────────────────────────────────────────────────

class TestBacktestEngine:

    def test_no_trades_returns_flat_portfolio(self, flat_prices):
        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(flat_prices, buy_indices=[], sell_indices=[])
        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_values) == len(flat_prices)

    def test_buy_and_hold_return_calculation(self, trending_prices):
        bnh = BacktestEngine.buy_and_hold_return(trending_prices)
        assert abs(bnh - 100.0) < 0.1  # 100→200 = 100% return

    def test_metrics_keys_present(self, trending_prices):
        engine = BacktestEngine()
        buys = [10, 30, 50]
        sells = [20, 40, 60]
        result = engine.run(trending_prices, buys, sells)
        for key in ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct"]:
            assert key in result.metrics, f"Missing metric: {key}"

    def test_stop_loss_triggers_sell(self):
        """Price drops sharply — stop loss should fire."""
        prices = [100.0] * 10 + [80.0] * 10  # 20% drop triggers 5% stop
        engine = BacktestEngine(initial_capital=1_000_000, stop_loss_pct=0.05)
        result = engine.run(prices, buy_indices=[5], sell_indices=[])
        stop_trades = [t for t in result.trades if t["type"] == "stop_loss"]
        assert len(stop_trades) >= 1, "Stop-loss should have triggered"

    def test_insufficient_capital_prevents_buy(self, trending_prices):
        """With very small capital, buy should be skipped."""
        engine = BacktestEngine(initial_capital=1.0)  # tiny capital
        result = engine.run(trending_prices, buy_indices=list(range(50)), sell_indices=[])
        buy_trades = [t for t in result.trades if t["type"] == "buy"]
        assert len(buy_trades) == 0

    def test_portfolio_length_matches_prices(self, trending_prices):
        engine = BacktestEngine()
        result = engine.run(trending_prices, buy_indices=[5, 20], sell_indices=[30, 50])
        assert len(result.portfolio_values) == len(trending_prices)

    def test_summary_string_contains_metrics(self, trending_prices):
        engine = BacktestEngine()
        result = engine.run(trending_prices, [10], [50], ticker="TEST", model_name="DQN")
        summary = result.summary()
        assert "TEST" in summary
        assert "Sharpe" in summary
        assert "Win rate" in summary


# ── Indicators tests ──────────────────────────────────────────────────────────

class TestIndicators:

    def test_macd_returns_list_same_length(self, trending_prices):
        macd = to_macd_series(trending_prices)
        assert len(macd) == len(trending_prices)

    def test_macd_values_are_finite(self, trending_prices):
        macd = to_macd_series(trending_prices)
        assert all(np.isfinite(v) for v in macd)

    def test_add_macd_adds_columns(self, sample_df):
        df = add_macd(sample_df)
        for col in ["MACD_line", "MACD_signal", "MACD_hist"]:
            assert col in df.columns

    def test_add_rsi_range(self, sample_df):
        df = add_rsi(sample_df)
        valid = df["RSI"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd_flat_prices_near_zero(self, flat_prices):
        macd = to_macd_series(flat_prices)
        # Flat prices → MACD should converge to near zero
        assert abs(macd[-1]) < 1e-6
