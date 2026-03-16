"""
Ensemble agent — combines DQN and CNN signals.

Strategy options:
  - "vote"     : majority vote (DQN buy/sell + CNN long/short must agree)
  - "dqn_only" : use DQN signal, filter by CNN confidence
  - "cnn_gate" : only trade when CNN is not Neutral

This is the most interview-worthy module — it shows you can think about
combining signals rather than treating each model in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.dqn_agent import DQNAgent
from src.models.cnn_agent import CNNAgent, prepare_training_matrices
from src.features.indicators import to_macd_series


@dataclass
class EnsembleSignal:
    day: int
    dqn_action: int          # 0=hold, 1-9=buy N, 10-18=sell N
    cnn_action: int          # 0=Long, 1=Neutral, 2=Short
    cnn_confidence: float    # max Q-value from CNN
    final_signal: str        # "buy" | "sell" | "hold"
    reason: str


class EnsembleAgent:
    """
    Combines DQN (position sizing) with CNN (directional filter).

    The CNN acts as a regime filter:
      - CNN says Long  → allow DQN buy signals through
      - CNN says Short → allow DQN sell signals through
      - CNN says Neutral → suppress both (stay flat)

    This reduces overtrading in sideways markets, which is the main
    failure mode of the original DQN-only strategy.
    """

    def __init__(
        self,
        strategy: str = "cnn_gate",
        cnn_confidence_threshold: float = 0.0,
    ):
        assert strategy in ("vote", "dqn_only", "cnn_gate"), \
            f"Unknown strategy: {strategy}"
        self.strategy = strategy
        self.cnn_confidence_threshold = cnn_confidence_threshold
        self.dqn: DQNAgent | None = None
        self.cnn: CNNAgent | None = None

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(self, dqn_path: str, cnn_path: str) -> None:
        self.dqn = DQNAgent()
        self.dqn.load(dqn_path)
        self.cnn = CNNAgent()
        self.cnn.load(cnn_path)
        print(f"Ensemble loaded — DQN: {dqn_path}  CNN: {cnn_path}")

    # ── Signal generation ─────────────────────────────────────────────────────

    def _combine(
        self,
        dqn_action: int,
        cnn_action: int,
        cnn_q: dict[str, float],
    ) -> tuple[str, str]:
        """
        Returns (final_signal, reason).
        dqn_action: 0=hold, 1-9=buy, 10-18=sell
        cnn_action: 0=Long, 1=Neutral, 2=Short
        """
        dqn_half = self.dqn.action_size_half  # 9
        is_dqn_buy  = 0 < dqn_action <= dqn_half
        is_dqn_sell = dqn_action > dqn_half
        is_dqn_hold = dqn_action == 0

        cnn_conf = max(cnn_q.values())
        is_confident = cnn_conf >= self.cnn_confidence_threshold

        if self.strategy == "cnn_gate":
            # CNN must agree on direction; Neutral = suppress
            if cnn_action == 1:  # Neutral
                return "hold", "CNN neutral — suppressed"
            if is_dqn_buy and cnn_action == 0:  # both say long
                return "buy", "DQN buy + CNN long"
            if is_dqn_sell and cnn_action == 2:  # both say short
                return "sell", "DQN sell + CNN short"
            if is_dqn_hold:
                return "hold", "DQN hold"
            return "hold", f"DQN/CNN disagree (DQN={'buy' if is_dqn_buy else 'sell'}, CNN={['Long','Neutral','Short'][cnn_action]})"

        elif self.strategy == "vote":
            votes_long  = (is_dqn_buy  and is_confident) + (cnn_action == 0)
            votes_short = (is_dqn_sell and is_confident) + (cnn_action == 2)
            if votes_long >= 2:
                return "buy", f"Majority vote long ({votes_long}/2)"
            if votes_short >= 2:
                return "sell", f"Majority vote short ({votes_short}/2)"
            return "hold", "No majority"

        else:  # dqn_only — use DQN, filter by CNN confidence
            if not is_confident:
                return "hold", f"CNN confidence too low ({cnn_conf:.3f})"
            if is_dqn_buy:
                return "buy", "DQN buy (CNN confident)"
            if is_dqn_sell:
                return "sell", "DQN sell (CNN confident)"
            return "hold", "DQN hold"

    # ── Full backtest signal generation ───────────────────────────────────────

    def generate_signals(
        self,
        df: pd.DataFrame,
    ) -> list[EnsembleSignal]:
        """
        Generate ensemble signals for every day in df.
        df must have columns: Open, High, Low, Close, Volume (after add_all).
        """
        assert self.dqn is not None and self.cnn is not None, \
            "Call load() first."

        prices  = df["Close"].tolist()
        macd    = to_macd_series(prices)
        matrices = prepare_training_matrices(df)
        n_mat    = len(matrices)

        signals = []
        for t in range(len(prices) - 1):
            # DQN state + action
            dqn_state  = DQNAgent._get_state(macd, t, self.dqn.state_size)
            dqn_action = self.dqn.greedy_act(dqn_state)

            # CNN matrix (align index — matrices start at window=32)
            mat_idx = t - (len(prices) - n_mat)
            if mat_idx < 0:
                cnn_action, cnn_q = 1, {"Long": 0.0, "Neutral": 1.0, "Short": 0.0}
            else:
                cnn_result = self.cnn.predict_signal(matrices[mat_idx])
                cnn_action = cnn_result["action"]
                cnn_q      = cnn_result["q_values"]

            final, reason = self._combine(dqn_action, cnn_action, cnn_q)

            signals.append(EnsembleSignal(
                day=t,
                dqn_action=dqn_action,
                cnn_action=cnn_action,
                cnn_confidence=max(cnn_q.values()),
                final_signal=final,
                reason=reason,
            ))

        return signals

    def signals_to_indices(
        self,
        signals: list[EnsembleSignal],
    ) -> tuple[list[int], list[int]]:
        """Convert signal list → (buy_indices, sell_indices) for BacktestEngine."""
        buys  = [s.day for s in signals if s.final_signal == "buy"]
        sells = [s.day for s in signals if s.final_signal == "sell"]
        return buys, sells

    # ── Signal summary ────────────────────────────────────────────────────────

    @staticmethod
    def signal_summary(signals: list[EnsembleSignal]) -> str:
        total = len(signals)
        buys  = sum(1 for s in signals if s.final_signal == "buy")
        sells = sum(1 for s in signals if s.final_signal == "sell")
        holds = total - buys - sells
        suppressed = sum(1 for s in signals if "suppressed" in s.reason)
        return (
            f"Signals over {total} days: "
            f"{buys} buy  |  {sells} sell  |  {holds} hold  "
            f"({suppressed} suppressed by CNN filter)"
        )
