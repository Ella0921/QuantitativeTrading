"""Container smoke test — verifies the image can import and run core modules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.backtest.engine import BacktestEngine
from src.features.indicators import to_macd_series
from src.monitoring.data_quality import validate_raw_data

prices = (100 + np.cumsum(np.random.randn(50))).tolist()
macd   = to_macd_series(prices)
assert len(macd) == len(prices)

engine = BacktestEngine()
result = engine.run(prices, [5], [20])
assert "sharpe_ratio" in result.metrics

print("Container smoke test passed")
