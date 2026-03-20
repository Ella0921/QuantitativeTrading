"""
CI smoke test — validates core pipeline modules without network calls.
Run by GitHub Actions in the smoke job.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# ── Data quality ──────────────────────────────────────────────────────────────
from src.monitoring.data_quality import validate_raw_data
from src.monitoring.drift import compute_psi

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

psi = compute_psi(close[:150], close[150:])
assert psi >= 0

print("Data quality smoke test: PASSED")

# ── DAG imports ───────────────────────────────────────────────────────────────
from dags.stock_pipeline import (
    task_download_raw,
    task_validate_quality,
    task_compute_features,
)
from dags.retrain_pipeline import task_should_retrain

print("DAG imports smoke test: PASSED")

# ── Backtest engine ───────────────────────────────────────────────────────────
from src.backtest.engine import BacktestEngine
from src.features.indicators import to_macd_series

prices = close.tolist()
macd   = to_macd_series(prices)
engine = BacktestEngine(initial_capital=100_000)
result = engine.run(prices, buy_indices=[10, 50], sell_indices=[30, 80])
assert "sharpe_ratio" in result.metrics

print("Backtest engine smoke test: PASSED")
print("\nAll smoke tests passed.")
