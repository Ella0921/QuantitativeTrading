"""
Data downloader with Parquet caching.
Replaces the ad-hoc yf.download() calls scattered across the original codebase.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(ticker: str, start: str, end: str) -> Path:
    key = f"{ticker}_{start}_{end}".replace("^", "").replace(".", "_")
    return CACHE_DIR / f"{key}.parquet"


def download(
    ticker: str,
    start: str = "2016-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker, with Parquet cache.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    Index is DatetimeIndex.
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    cache_file = _cache_path(ticker, start, end)

    if use_cache and cache_file.exists():
        logger.info(f"Loading {ticker} from cache: {cache_file}")
        return pd.read_parquet(cache_file)

    logger.info(f"Downloading {ticker} from {start} to {end}")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Flatten multi-level columns if present (yfinance >= 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"

    if use_cache:
        df.to_parquet(cache_file)
        logger.info(f"Cached to {cache_file}")

    return df


def download_multiple(
    tickers: list[str],
    start: str = "2016-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download data for multiple tickers, returns {ticker: DataFrame}."""
    result = {}
    for ticker in tickers:
        try:
            result[ticker] = download(ticker, start, end, use_cache)
        except Exception as e:
            logger.warning(f"Failed to download {ticker}: {e}")
    return result


def get_train_test_split(
    df: pd.DataFrame,
    train_end: str,
    test_start: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/test by date."""
    test_start = test_start or train_end
    df_train = df.loc[:train_end].copy()
    df_test = df.loc[test_start:].copy()
    return df_train, df_test
