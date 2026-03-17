"""
Baseline trading strategies for benchmark comparison.

These are used to demonstrate that the DQN + CNN models
outperform (or at least compete with) standard approaches.

All strategies share the same interface:
    generate_signals(prices, **kwargs) -> (buy_indices, sell_indices)

This makes them drop-in replacements in BacktestEngine.run().
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── 1. Buy & Hold ─────────────────────────────────────────────────────────────

def buy_and_hold_signals(prices: list) -> tuple[list[int], list[int]]:
    """Buy on day 0, sell on last day. The simplest possible baseline."""
    return [0], [len(prices) - 1]


# ── 2. SMA Crossover ─────────────────────────────────────────────────────────

def sma_crossover_signals(
    prices: list,
    fast: int = 5,
    slow: int = 20,
) -> tuple[list[int], list[int]]:
    """
    Classic moving average crossover strategy.
    Buy when fast SMA crosses above slow SMA.
    Sell when fast SMA crosses below slow SMA.
    """
    s = pd.Series(prices)
    fast_ma = s.rolling(fast).mean()
    slow_ma = s.rolling(slow).mean()

    buys, sells = [], []
    position = False

    for i in range(slow, len(prices)):
        prev_above = fast_ma.iloc[i - 1] > slow_ma.iloc[i - 1]
        curr_above = fast_ma.iloc[i] > slow_ma.iloc[i]

        if not prev_above and curr_above and not position:
            buys.append(i)
            position = True
        elif prev_above and not curr_above and position:
            sells.append(i)
            position = False

    # Close any open position at end
    if position and sells and buys:
        if buys[-1] > (sells[-1] if sells else -1):
            sells.append(len(prices) - 1)

    return buys, sells


# ── 3. MACD + RSI Rule-based ──────────────────────────────────────────────────

def macd_rsi_signals(
    df: pd.DataFrame,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
) -> tuple[list[int], list[int]]:
    """
    Rule-based strategy combining MACD and RSI.

    Buy conditions (all must be true):
      - MACD histogram turns positive (crosses zero from below)
      - RSI is below overbought threshold (not already overextended)

    Sell conditions (any):
      - MACD histogram turns negative
      - RSI exceeds overbought threshold

    Requires df with columns: MACD_hist, RSI (from indicators.add_all())
    """
    assert "MACD_hist" in df.columns and "RSI" in df.columns, \
        "DataFrame must have MACD_hist and RSI columns. Run add_all() first."

    buys, sells = [], []
    position = False

    hist = df["MACD_hist"].values
    rsi  = df["RSI"].values

    for i in range(1, len(df)):
        macd_cross_up   = hist[i - 1] <= 0 and hist[i] > 0
        macd_cross_down = hist[i - 1] >= 0 and hist[i] < 0
        rsi_ok_buy      = rsi[i] < rsi_overbought
        rsi_sell        = rsi[i] > rsi_overbought

        if macd_cross_up and rsi_ok_buy and not position:
            buys.append(i)
            position = True
        elif (macd_cross_down or rsi_sell) and position:
            sells.append(i)
            position = False

    if position:
        sells.append(len(df) - 1)

    return buys, sells


# ── 4. RSI Mean Reversion ─────────────────────────────────────────────────────

def rsi_mean_reversion_signals(
    df: pd.DataFrame,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> tuple[list[int], list[int]]:
    """
    Simple RSI mean-reversion strategy.
    Buy when RSI crosses back above oversold level.
    Sell when RSI crosses above overbought level.
    """
    assert "RSI" in df.columns
    rsi = df["RSI"].values
    buys, sells = [], []
    position = False

    for i in range(1, len(df)):
        was_oversold = rsi[i - 1] < oversold
        now_above    = rsi[i] >= oversold
        is_overbought = rsi[i] > overbought

        if was_oversold and now_above and not position:
            buys.append(i)
            position = True
        elif is_overbought and position:
            sells.append(i)
            position = False

    if position:
        sells.append(len(df) - 1)

    return buys, sells


# ── Strategy registry ─────────────────────────────────────────────────────────

BASELINE_NAMES = {
    "buy_and_hold":      "Buy & Hold",
    "sma_crossover":     "SMA Crossover (5/20)",
    "macd_rsi":          "MACD + RSI Rules",
    "rsi_mean_reversion":"RSI Mean Reversion",
}
