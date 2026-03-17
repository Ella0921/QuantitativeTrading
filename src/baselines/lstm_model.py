"""
LSTM baseline model for benchmark comparison.

Architecture: single LSTM layer → Dense(1) → sigmoid
Task: predict whether next-day return is positive (binary classification)
Trading rule: buy if predicted probability > threshold, sell otherwise

Kept intentionally simple — the goal is a fair comparison baseline,
not a production LSTM trading system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path


def build_lstm(window: int = 20, n_features: int = 5) -> tf.keras.Model:
    """
    Simple LSTM: (window, n_features) → sigmoid probability of next-day up move.
    Features: Close, MACD_hist, RSI, BB_upper-Close (distance), Volume_norm
    """
    inp = tf.keras.Input(shape=(window, n_features))
    x   = tf.keras.layers.LSTM(64, return_sequences=False)(inp)
    x   = tf.keras.layers.Dropout(0.2)(x)
    x   = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out, name="lstm_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_lstm_features(df: pd.DataFrame, window: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays for LSTM training/evaluation.

    Features (normalised within each window):
      0: Close price (pct change)
      1: MACD histogram
      2: RSI / 100
      3: (BB_upper - Close) / Close  — distance from upper band
      4: Volume (log-normalised)

    y: 1 if next-day close > today's close, else 0
    """
    df = df.copy()
    df["close_ret"]  = df["Close"].pct_change()
    df["bb_dist"]    = (df["BB_upper"] - df["Close"]) / df["Close"]
    df["rsi_norm"]   = df["RSI"] / 100.0
    df["vol_norm"]   = np.log1p(df["Volume"]) / np.log1p(df["Volume"]).mean()
    df["next_up"]    = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    feature_cols = ["close_ret", "MACD_hist", "rsi_norm", "bb_dist", "vol_norm"]
    data   = df[feature_cols].values
    labels = df["next_up"].values

    X, y = [], []
    for i in range(window, len(data) - 1):
        X.append(data[i - window: i])
        y.append(labels[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LSTMAgent:
    """
    Thin wrapper that trains an LSTM and generates buy/sell signals.
    Matches the same interface as DQNAgent for drop-in comparison.
    """

    def __init__(self, window: int = 20, threshold: float = 0.55):
        self.window    = window
        self.threshold = threshold
        self.model     = build_lstm(window=window)

    def train(
        self,
        df_train: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 32,
        verbose: int = 0,
    ) -> dict:
        X, y = prepare_lstm_features(df_train, self.window)
        split = int(len(X) * 0.9)
        history = self.model.fit(
            X[:split], y[:split],
            validation_data=(X[split:], y[split:]),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        val_acc = max(history.history.get("val_accuracy", [0]))
        return {"val_accuracy": val_acc, "epochs": epochs}

    def backtest(self, df_test: pd.DataFrame) -> tuple[list[int], list[int]]:
        """
        Generate buy/sell signal indices from the test DataFrame.
        Returns (buy_indices, sell_indices) aligned to df_test index positions.
        """
        X, _ = prepare_lstm_features(df_test, self.window)
        if len(X) == 0:
            return [], []

        probs    = self.model.predict(X, verbose=0).flatten()
        offset   = self.window  # first prediction is at position `window`
        buys, sells = [], []
        position = False

        for i, prob in enumerate(probs):
            idx = i + offset
            if prob > self.threshold and not position:
                buys.append(idx)
                position = True
            elif prob < (1 - self.threshold) and position:
                sells.append(idx)
                position = False

        if position:
            sells.append(len(df_test) - 1)

        return buys, sells

    def save(self, path: str | Path) -> None:
        path = str(path)
        if not path.endswith(".keras"):
            path += ".keras"
        self.model.save(path)

    def load(self, path: str | Path) -> None:
        path = str(path)
        if not path.endswith(".keras"):
            path += ".keras"
        self.model = tf.keras.models.load_model(path)
