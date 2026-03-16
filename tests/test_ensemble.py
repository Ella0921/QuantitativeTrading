"""Tests for the Ensemble agent signal combination logic."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ensemble import EnsembleAgent, EnsembleSignal


# ── Fixture: mock Q-values ────────────────────────────────────────────────────

LONG_Q    = {"Long": 1.0, "Neutral": 0.1, "Short": 0.0}
NEUTRAL_Q = {"Long": 0.3, "Neutral": 0.9, "Short": 0.2}
SHORT_Q   = {"Long": 0.0, "Neutral": 0.1, "Short": 1.0}

CNN_LONG    = 0
CNN_NEUTRAL = 1
CNN_SHORT   = 2


# ── cnn_gate strategy ─────────────────────────────────────────────────────────

class TestCnnGate:

    def setup_method(self):
        self.agent = EnsembleAgent(strategy="cnn_gate")
        # Patch in a minimal DQN so action_size_half is available
        from src.models.dqn_agent import DQNAgent
        self.agent.dqn = DQNAgent()

    def test_buy_when_dqn_buy_and_cnn_long(self):
        signal, reason = self.agent._combine(1, CNN_LONG, LONG_Q)
        assert signal == "buy"

    def test_sell_when_dqn_sell_and_cnn_short(self):
        signal, reason = self.agent._combine(10, CNN_SHORT, SHORT_Q)
        assert signal == "sell"

    def test_hold_when_cnn_neutral(self):
        signal, reason = self.agent._combine(1, CNN_NEUTRAL, NEUTRAL_Q)
        assert signal == "hold"
        assert "neutral" in reason.lower()

    def test_hold_when_dqn_buy_but_cnn_short(self):
        """Disagreement → hold."""
        signal, reason = self.agent._combine(1, CNN_SHORT, SHORT_Q)
        assert signal == "hold"
        assert "disagree" in reason.lower()

    def test_hold_when_dqn_sell_but_cnn_long(self):
        signal, reason = self.agent._combine(10, CNN_LONG, LONG_Q)
        assert signal == "hold"

    def test_hold_when_dqn_hold(self):
        signal, reason = self.agent._combine(0, CNN_LONG, LONG_Q)
        assert signal == "hold"


# ── vote strategy ─────────────────────────────────────────────────────────────

class TestVoteStrategy:

    def setup_method(self):
        self.agent = EnsembleAgent(strategy="vote")
        from src.models.dqn_agent import DQNAgent
        self.agent.dqn = DQNAgent()

    def test_buy_with_two_long_votes(self):
        signal, reason = self.agent._combine(1, CNN_LONG, LONG_Q)
        assert signal == "buy"

    def test_sell_with_two_short_votes(self):
        signal, reason = self.agent._combine(10, CNN_SHORT, SHORT_Q)
        assert signal == "sell"

    def test_hold_with_no_majority(self):
        # DQN buy, CNN short → no majority
        signal, reason = self.agent._combine(1, CNN_SHORT, SHORT_Q)
        assert signal == "hold"


# ── signals_to_indices ────────────────────────────────────────────────────────

class TestSignalsToIndices:

    def test_split_buy_sell_hold(self):
        agent = EnsembleAgent()
        signals = [
            EnsembleSignal(0, 1, 0, 0.9, "buy",  "test"),
            EnsembleSignal(1, 0, 1, 0.3, "hold", "test"),
            EnsembleSignal(2, 10, 2, 0.8, "sell", "test"),
            EnsembleSignal(3, 1, 0, 0.9, "buy",  "test"),
        ]
        buys, sells = agent.signals_to_indices(signals)
        assert buys  == [0, 3]
        assert sells == [2]

    def test_all_hold(self):
        agent = EnsembleAgent()
        signals = [EnsembleSignal(i, 0, 1, 0.3, "hold", "test") for i in range(5)]
        buys, sells = agent.signals_to_indices(signals)
        assert buys == [] and sells == []


# ── signal_summary ────────────────────────────────────────────────────────────

class TestSignalSummary:

    def test_summary_counts(self):
        signals = [
            EnsembleSignal(0, 1, 0, 0.9, "buy",  "DQN buy + CNN long"),
            EnsembleSignal(1, 0, 1, 0.3, "hold", "CNN neutral — suppressed"),
            EnsembleSignal(2, 10, 2, 0.8, "sell", "DQN sell + CNN short"),
            EnsembleSignal(3, 0, 1, 0.3, "hold", "CNN neutral — suppressed"),
        ]
        summary = EnsembleAgent.signal_summary(signals)
        assert "1 buy" in summary
        assert "1 sell" in summary
        assert "2 suppressed" in summary
