"""
DQN Agent — rewritten for TF2 / Keras.

Original: project.py using tf.compat.v1.InteractiveSession + tf.placeholder
New: tf.keras.Model subclassing, standard fit loop, no Session/Graph API.

Key changes vs original:
  - Removed tf.reset_default_graph() / tf.InteractiveSession()
  - Replaced tf.placeholder + sess.run() with model.predict() / model.train_on_batch()
  - Model save/load via model.save() / tf.keras.models.load_model()
  - Added MLflow logging hooks
  - Action size kept at 19 (multi-unit buy/sell) to preserve original strategy logic
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.features.indicators import to_macd_series


# ── Neural network ──────────────────────────────────────────────────────────

class DQNModel(tf.keras.Model):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(256, activation="relu")
        self.d2 = tf.keras.layers.Dense(128, activation="relu")
        self.out = tf.keras.layers.Dense(action_size)

    def call(self, x, training=False):
        return self.out(self.d2(self.d1(x)))


# ── Agent ────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Deep Q-Network trading agent.

    action_size=19: action 0 = do-nothing,
                    1..9  = buy N units,
                    10..18 = sell N-9 units
    """

    def __init__(
        self,
        state_size: int = 30,
        action_size: int = 19,
        batch_size: int = 32,
        gamma: float = 0.95,
        epsilon: float = 0.5,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.999,
        learning_rate: float = 1e-5,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.action_size_half = action_size // 2
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory: deque = deque(maxlen=1000)

        self.model = DQNModel(state_size, action_size)
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss="mse",
        )
        # Build weights by calling with dummy input
        self.model(np.zeros((1, state_size)))

    # ── State extraction ─────────────────────────────────────────────────────

    @staticmethod
    def _get_state(series: list, t: int, window: int) -> np.ndarray:
        """Return 1-D state vector of length window (price/macd differences)."""
        d = t - window
        if d >= 0:
            block = series[d: t + 1]
        else:
            block = [-d] * [series[0]] + series[: t + 1]
        return np.array([[block[i + 1] - block[i] for i in range(window)]])

    # ── Action ───────────────────────────────────────────────────────────────

    def act(self, state: np.ndarray) -> int:
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        return int(np.argmax(self.model(state, training=False)[0]))

    def greedy_act(self, state: np.ndarray) -> int:
        return int(np.argmax(self.model(state, training=False)[0]))

    # ── Replay ───────────────────────────────────────────────────────────────

    def replay(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states = np.vstack([s for s, *_ in batch])
        next_states = np.vstack([ns for _, _, _, ns, _ in batch])

        q_current = self.model(states, training=False).numpy()
        q_next = self.model(next_states, training=False).numpy()

        for i, (_, action, reward, _, done) in enumerate(batch):
            target = reward
            if not done:
                target += self.gamma * np.amax(q_next[i])
            q_current[i][action] = target

        loss = self.model.train_on_batch(states, q_current)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss)

    # ── Train ────────────────────────────────────────────────────────────────

    def train(
        self,
        prices: list,
        macd: list | None = None,
        iterations: int = 200,
        initial_money: float = 1_000_000,
        checkpoint: int = 10,
        mlflow_run=None,
    ) -> dict:
        """
        Train the agent on historical price data.
        Returns a dict with loss history and final portfolio value.
        """
        series = macd if macd is not None else prices
        losses = []

        for epoch in range(iterations):
            capital = initial_money
            inventory: list[float] = []
            holding = 0
            state = self._get_state(series, 0, self.state_size)

            for t in range(len(prices) - 1):
                action = self.act(state)
                next_state = self._get_state(series, t + 1, self.state_size)
                price = prices[t]

                # Buy N units
                if 0 < action <= self.action_size_half:
                    n = action
                    if capital >= n * price and t < len(prices) - self.action_size_half:
                        inventory.extend([price] * n)
                        capital -= n * price
                        holding += n

                # Sell N units
                elif action > self.action_size_half and holding >= action - self.action_size_half:
                    n = action - self.action_size_half
                    for _ in range(n):
                        inventory.pop(0)
                        capital += price
                    holding -= n

                reward = (capital - initial_money) / initial_money
                done = capital < initial_money
                self.memory.append((state, action, reward, next_state, done))
                state = next_state

                loss = self.replay()

            portfolio_value = capital + holding * prices[-1]
            total_return = (portfolio_value - initial_money) / initial_money * 100
            losses.append(loss)

            if (epoch + 1) % checkpoint == 0:
                print(
                    f"Epoch {epoch+1:>4}/{iterations} | "
                    f"Return: {total_return:+.2f}% | "
                    f"Loss: {loss:.6f} | "
                    f"Capital: {capital:,.0f}"
                )
                if mlflow_run:
                    import mlflow
                    mlflow.log_metrics(
                        {"loss": loss, "return_pct": total_return},
                        step=epoch + 1,
                    )

        return {"losses": losses, "final_return_pct": total_return}

    # ── Backtest (inference only) ─────────────────────────────────────────────

    def backtest(
        self,
        prices: list,
        macd: list | None = None,
    ) -> tuple[list, list, list]:
        """
        Run greedy inference on test data.
        Returns (buy_indices, sell_indices, portfolio_values).
        """
        series = macd if macd is not None else prices
        capital = 1_000_000.0
        inventory: list[float] = []
        holding = 0
        buys, sells, portfolio = [], [], []
        state = self._get_state(series, 0, self.state_size)

        for t in range(len(prices) - 1):
            action = self.greedy_act(state)
            price = prices[t]

            if 0 < action <= self.action_size_half:
                n = action
                if capital >= n * price:
                    inventory.extend([price] * n)
                    capital -= n * price
                    holding += n
                    buys.append(t)

            elif action > self.action_size_half:
                n = action - self.action_size_half
                if holding >= n:
                    for _ in range(n):
                        inventory.pop(0)
                        capital += price
                    holding -= n
                    sells.append(t)

            portfolio.append(capital + holding * price)
            state = self._get_state(series, t + 1, self.state_size)

        return buys, sells, portfolio

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        print(f"Model saved → {path}")

    def load(self, path: str | Path) -> None:
        self.model = tf.keras.models.load_model(str(path))
        print(f"Model loaded ← {path}")

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def from_prices(
        cls,
        train_prices: list,
        use_macd: bool = True,
        **kwargs,
    ) -> "DQNAgent":
        """Create and train an agent from price data in one call."""
        macd = to_macd_series(train_prices) if use_macd else None
        agent = cls(**kwargs)
        agent.train(train_prices, macd=macd)
        return agent
