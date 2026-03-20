"""Container smoke test — run inside Docker to verify the image is functional."""
from src.backtest.engine import BacktestEngine
from src.features.indicators import to_macd_series
from src.monitoring.data_quality import validate_raw_data

print("Container smoke test passed")
