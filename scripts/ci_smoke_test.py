"""
CI smoke test — validates core pipeline modules without network calls.
All imports at the top to satisfy E402, imports used in assertions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.features.indicators import add_all, to_macd_series
from src.monitoring.data_quality import validate_raw_data
from src.monitoring.drift import compute_psi

# DAG imports — validates the DAG files are importable (syntax + dependency check)
from dags.stock_pipeline import task_compute_features  # noqa: F401
from dags.retrain_pipeline import task_should_retrain  # noqa: F401

# ── Data quality ──────────────────────────────────────────────────────────────
dates = pd.date_range("2020-01-01", periods=300, freq="B")
close = np.abs(100 + np.cumsum(np.random.randn(300)))
df = pd.DataFrame({
    "Open":   close * 0.999,
    "High":   close * 1.01,
    "Low":    close * 0.99,
    "Close":  close,
    "Volume": np.ones(300) * 1e6,
}, index=dates)

result = validate_raw_data(df, "TEST")
assert result["passed"], f"Quality check failed: {result['failed']}"
print("Data quality smoke test: PASSED")

# ── PSI ───────────────────────────────────────────────────────────────────────
psi = compute_psi(close[:150], close[150:])
assert psi >= 0
print("PSI smoke test: PASSED")

# ── Backtest engine ───────────────────────────────────────────────────────────
prices = close.tolist()
macd   = to_macd_series(prices)
engine = BacktestEngine(initial_capital=100_000)
bt     = engine.run(prices, buy_indices=[10, 50], sell_indices=[30, 80])
assert "sharpe_ratio" in bt.metrics
print("Backtest engine smoke test: PASSED")

# ── Feature indicators ────────────────────────────────────────────────────────
df_feat = add_all(df)
assert "MACD_hist" in df_feat.columns
assert "RSI" in df_feat.columns
print("Feature indicators smoke test: PASSED")

# ── DAG imports ───────────────────────────────────────────────────────────────
print("DAG imports smoke test: PASSED")

print("\nAll smoke tests passed.")
