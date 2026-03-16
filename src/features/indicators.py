"""
Technical indicator computation.
Extracted and cleaned from the original project.py to_MACD() and scattered logic.
"""

import pandas as pd
import numpy as np


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Add MACD columns to DataFrame.
    Matches the original to_MACD() logic exactly for backward compatibility.
    Adds: MACD_line, MACD_signal, MACD_hist
    """
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    sig = dif.ewm(span=signal, adjust=False).mean()
    df["MACD_line"] = dif
    df["MACD_signal"] = sig
    df["MACD_hist"] = dif - sig
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI column."""
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Band columns: BB_upper, BB_mid, BB_lower."""
    df = df.copy()
    mid = df["Close"].rolling(period).mean()
    band = df["Close"].rolling(period).std()
    df["BB_upper"] = mid + std * band
    df["BB_mid"] = mid
    df["BB_lower"] = mid - std * band
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range — used in position sizing and stop-loss."""
    df = df.copy()
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean()
    return df


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all indicators and drop rows with NaN (warmup period)."""
    df = add_macd(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df.dropna(inplace=True)
    return df


def to_macd_series(close: list | pd.Series) -> list:
    """
    Backward-compatible wrapper matching the original to_MACD() return format.
    Returns MACD histogram as a plain list.
    """
    s = pd.Series(close)
    ema_fast = s.ewm(span=12, adjust=False).mean()
    ema_slow = s.ewm(span=26, adjust=False).mean()
    dif = ema_fast - ema_slow
    macd_sig = dif.ewm(span=9, adjust=False).mean()
    return (dif - macd_sig).tolist()
