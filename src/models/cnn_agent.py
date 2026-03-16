"""
CNN Q-Network — rewritten for TF2 / Keras.

Original: train.py + convNN.py using tf.compat.v1.Session, tf.compat.v1.placeholder,
          tf.compat.v1.layers.batch_normalization (deprecated), tf.compat.v1.train.Saver.
New:      tf.keras functional API, standard Keras training, model.save().

Architecture preserved from original:
  Input (32x32x1) → BN → Conv(5,16) → Conv(5,16) → MaxPool(2,2)
                 → BN → Conv(5,32) → Conv(5,32) → MaxPool(2,2)
                 → Flatten → FC(32,BN,ReLU) → FC(3)

Action space: 3 actions — Long (0), Neutral (1), Short (2)
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf


# ── Model ─────────────────────────────────────────────────────────────────────

def build_cnn_model(
    input_size: int = 32,
    num_actions: int = 3,
    filter_size: int = 5,
) -> tf.keras.Model:
    """
    Replaces ConstructCNN.QValue() with a clean Keras functional model.
    BatchNorm after each conv block, as in original.
    """
    inp = tf.keras.Input(shape=(input_size, input_size, 1))

    # Normalise input
    x = tf.keras.layers.LayerNormalization()(inp)

    # Block 1
    x = tf.keras.layers.Conv2D(16, filter_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(16, filter_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(32, filter_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(32, filter_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # Fully connected
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    out = tf.keras.layers.Dense(num_actions)(x)

    return tf.keras.Model(inputs=inp, outputs=out, name="cnn_qnet")


# ── Data preprocessing (matches original process() logic in main.py) ─────────

def encode_ohlcv_to_matrix(
    prices: np.ndarray,
    volumes: np.ndarray,
    matrix_size: int = 32,
) -> np.ndarray:
    """
    Convert price+volume arrays to a 32x32 binary image matrix.
    Preserves the original encoding from DataPPRL / main.py process().

    Args:
        prices: 1-D array of length matrix_size
        volumes: 1-D array of length matrix_size
        matrix_size: width/height of output matrix (default 32)

    Returns:
        (matrix_size, matrix_size) int32 array
    """
    assert len(prices) == matrix_size and len(volumes) == matrix_size

    matrix = np.zeros((matrix_size, matrix_size), dtype=np.int32)

    vol_min, vol_max = volumes.min(), volumes.max()
    price_min, price_max = prices.min(), prices.max()

    def _scale(arr, lo, hi, target=14):
        if hi == lo:
            return np.full_like(arr, 7, dtype=float)
        return (arr - lo) / (hi - lo) * target

    vol_scaled = np.rint(_scale(volumes, vol_min, vol_max)).astype(int)
    price_scaled = np.rint(_scale(prices, price_min, price_max)).astype(int)

    for j in range(matrix_size):
        row_v = int((vol_scaled[j] - 14) * -1)
        row_p = int(((price_scaled[j] - 14) * -1) + 17)
        row_v = max(0, min(matrix_size - 1, row_v))
        row_p = max(0, min(matrix_size - 1, row_p))
        matrix[row_v, j] = 1
        matrix[row_p, j] = 1

    return matrix


def prepare_inference_matrix(ticker: str, matrix_size: int = 32) -> np.ndarray:
    """
    Download last 3 months of data and encode the most recent window.
    Used for live signal prediction (replaces process() in original main.py).
    """
    df = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close", "Volume"]].dropna().tail(matrix_size)
    if len(df) < matrix_size:
        raise ValueError(f"Not enough data for {ticker} (got {len(df)}, need {matrix_size})")

    return encode_ohlcv_to_matrix(
        df["Close"].values.astype(float),
        df["Volume"].values.astype(float),
        matrix_size,
    )


def prepare_training_matrices(
    df: pd.DataFrame,
    matrix_size: int = 32,
) -> np.ndarray:
    """
    Slide a window over a full historical DataFrame to produce training matrices.
    Returns shape (N_windows, matrix_size, matrix_size).
    """
    prices = df["Close"].values.astype(float)
    volumes = df["Volume"].values.astype(float)
    n = len(prices)
    matrices = []

    for i in range(n - matrix_size + 1):
        m = encode_ohlcv_to_matrix(
            prices[i: i + matrix_size],
            volumes[i: i + matrix_size],
            matrix_size,
        )
        matrices.append(m)

    return np.array(matrices)  # (N, 32, 32)


# ── Agent ─────────────────────────────────────────────────────────────────────

class CNNAgent:
    """
    CNN Q-Network agent for generating Long / Neutral / Short signals.

    Maps to original train.py trainModel but with:
    - Keras training loop instead of manual Session management
    - Clean replay buffer using numpy arrays
    - model.save() / model.load() instead of tf.compat.v1.train.Saver
    """

    ACTIONS = {0: "Long", 1: "Neutral", 2: "Short"}

    def __init__(
        self,
        matrix_size: int = 32,
        num_actions: int = 3,
        filter_size: int = 5,
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.1,
        batch_size: int = 32,
        memory_size: int = 1000,
    ):
        self.matrix_size = matrix_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.model = build_cnn_model(matrix_size, num_actions, filter_size)
        self.target_model = build_cnn_model(matrix_size, num_actions, filter_size)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
        )

        # Replay buffer
        self._buf_s = []
        self._buf_a = []
        self._buf_r = []
        self._buf_ns = []
        self._memory_size = memory_size

        self._update_target()

    def _update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def _remember(self, s, a, r, ns):
        self._buf_s.append(s)
        self._buf_a.append(a)
        self._buf_r.append(r)
        self._buf_ns.append(ns)
        if len(self._buf_s) > self._memory_size:
            self._buf_s.pop(0)
            self._buf_a.pop(0)
            self._buf_r.pop(0)
            self._buf_ns.pop(0)

    def predict(self, matrix: np.ndarray) -> int:
        """Return action index for a single (32,32) input matrix."""
        x = matrix.reshape(1, self.matrix_size, self.matrix_size, 1).astype(np.float32)
        q_values = self.model(x, training=False).numpy()[0]
        return int(np.argmax(q_values))

    def predict_signal(self, matrix: np.ndarray) -> dict:
        """Return human-readable signal dict with Q-values."""
        x = matrix.reshape(1, self.matrix_size, self.matrix_size, 1).astype(np.float32)
        q_values = self.model(x, training=False).numpy()[0]
        action_idx = int(np.argmax(q_values))
        return {
            "action": action_idx,
            "signal": self.ACTIONS[action_idx],
            "q_values": {self.ACTIONS[i]: float(q_values[i]) for i in range(self.num_actions)},
        }

    def train(
        self,
        matrices: np.ndarray,
        returns: np.ndarray,
        max_iter: int = 5000,
        target_update_interval: int = 1000,
        log_interval: int = 100,
    ) -> list[float]:
        """
        Train on pre-processed matrix sequence.

        matrices: (N, 32, 32) array
        returns:  (N,) daily return array (used to compute reward)
        """
        losses = []
        b = 0

        while b < max_iter:
            idx = np.random.randint(1, len(matrices) - 1)
            s = matrices[idx - 1]
            ns = matrices[idx]

            # Epsilon-greedy action
            if np.random.rand() <= self.epsilon:
                a = np.zeros(self.num_actions, dtype=np.int32)
                a[np.random.randint(self.num_actions)] = 1
            else:
                x = s.reshape(1, self.matrix_size, self.matrix_size, 1).astype(np.float32)
                q = self.model(x, training=False).numpy()[0]
                a = np.zeros(self.num_actions, dtype=np.int32)
                a[int(np.argmax(q))] = 1

            # Reward: action_value * daily_return - transaction_penalty
            pre_act = 1 - np.argmax(a)
            reward = pre_act * returns[idx]

            self._remember(s, a, reward, ns)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= 0.999999

            # Train when buffer full
            if len(self._buf_s) >= self._memory_size and b % 10 == 0:
                idxs = np.random.choice(len(self._buf_s), self.batch_size, replace=False)
                S = np.array([self._buf_s[i] for i in idxs]).reshape(
                    self.batch_size, self.matrix_size, self.matrix_size, 1
                ).astype(np.float32)
                NS = np.array([self._buf_ns[i] for i in idxs]).reshape(
                    self.batch_size, self.matrix_size, self.matrix_size, 1
                ).astype(np.float32)
                A = np.array([self._buf_a[i] for i in idxs])
                R = np.array([self._buf_r[i] for i in idxs])

                q_next = self.target_model(NS, training=False).numpy()
                targets = R + self.gamma * q_next.max(axis=1)

                q_curr = self.model(S, training=False).numpy()
                for k in range(self.batch_size):
                    q_curr[k][np.argmax(A[k])] = targets[k]

                loss = self.model.train_on_batch(S, q_curr)
                losses.append(float(loss))

                if b % (target_update_interval * 10) == 0:
                    self._update_target()

                if b % (log_interval * 10) == 0:
                    print(f"Iter {b:>6} | Loss: {loss:.6f} | ε: {self.epsilon:.4f}")

            b += 1

        return losses

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        print(f"CNN model saved → {path}")

    def load(self, path: str | Path) -> None:
        self.model = tf.keras.models.load_model(str(path))
        self._update_target()
        print(f"CNN model loaded ← {path}")
