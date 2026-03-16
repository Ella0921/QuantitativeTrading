"""
Unit tests for DQN Agent and CNN Agent.
Fast tests only — no actual training iterations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import tempfile

from src.models.dqn_agent import DQNAgent
from src.models.cnn_agent import CNNAgent, build_cnn_model, encode_ohlcv_to_matrix


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class TestDQNAgent:

    def test_init_default_params(self):
        agent = DQNAgent()
        assert agent.state_size == 30
        assert agent.action_size == 19
        assert agent.action_size_half == 9

    def test_model_output_shape(self):
        agent = DQNAgent(state_size=30, action_size=19)
        import numpy as np
        x = np.zeros((1, 30))
        out = agent.model(x)
        assert out.shape == (1, 19)

    def test_act_returns_valid_action(self):
        agent = DQNAgent()
        state = np.zeros((1, 30))
        action = agent.act(state)
        assert 0 <= action < 19

    def test_greedy_act_deterministic(self):
        """Greedy act should return same action for same state."""
        agent = DQNAgent()
        agent.epsilon = 0  # no exploration
        state = np.random.randn(1, 30)
        a1 = agent.greedy_act(state)
        a2 = agent.greedy_act(state)
        assert a1 == a2

    def test_get_state_shape(self):
        prices = list(np.linspace(100, 200, 100))
        state = DQNAgent._get_state(prices, t=50, window=30)
        assert state.shape == (1, 30)

    def test_get_state_early_padding(self):
        """State at t=5 with window=30 should pad with first value."""
        prices = list(np.linspace(100, 200, 100))
        state = DQNAgent._get_state(prices, t=5, window=30)
        assert state.shape == (1, 30)

    def test_backtest_returns_correct_types(self):
        agent = DQNAgent()
        prices = list(np.linspace(100, 150, 50))
        buys, sells, portfolio = agent.backtest(prices)
        assert isinstance(buys, list)
        assert isinstance(sells, list)
        assert isinstance(portfolio, list)
        assert len(portfolio) == len(prices) - 1

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "dqn_test")
            agent = DQNAgent()
            original_weights = [w.numpy().copy() for w in agent.model.weights]
            agent.save(path)

            agent2 = DQNAgent()
            agent2.load(path)
            loaded_weights = [w.numpy() for w in agent2.model.weights]

            for ow, lw in zip(original_weights, loaded_weights):
                np.testing.assert_array_equal(ow, lw)

    def test_epsilon_decay_during_replay(self):
        agent = DQNAgent(epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.9)
        # Fill memory above batch_size
        for _ in range(64):
            s = np.zeros((1, 30))
            agent.memory.append((s, 0, 0.0, s, False))
        initial_eps = agent.epsilon
        agent.replay()
        assert agent.epsilon < initial_eps


# ── CNN Agent ─────────────────────────────────────────────────────────────────

class TestCNNAgent:

    def test_build_model_output_shape(self):
        model = build_cnn_model(input_size=32, num_actions=3)
        import tensorflow as tf
        x = tf.zeros((1, 32, 32, 1))
        out = model(x)
        assert out.shape == (1, 3)

    def test_encode_matrix_shape(self):
        prices  = np.linspace(100, 200, 32)
        volumes = np.random.randint(1_000_000, 5_000_000, 32).astype(float)
        matrix  = encode_ohlcv_to_matrix(prices, volumes)
        assert matrix.shape == (32, 32)
        assert matrix.dtype == np.int32

    def test_encode_matrix_binary(self):
        """Matrix should contain only 0s and 1s."""
        prices  = np.linspace(100, 200, 32)
        volumes = np.ones(32) * 1_000_000
        matrix  = encode_ohlcv_to_matrix(prices, volumes)
        unique  = np.unique(matrix)
        assert set(unique).issubset({0, 1})

    def test_encode_flat_prices(self):
        """Flat prices should not raise."""
        prices  = np.ones(32) * 100.0
        volumes = np.ones(32) * 1_000_000
        matrix  = encode_ohlcv_to_matrix(prices, volumes)
        assert matrix.shape == (32, 32)

    def test_predict_returns_valid_action(self):
        agent  = CNNAgent()
        matrix = np.random.randint(0, 2, (32, 32)).astype(np.float32)
        action = agent.predict(matrix)
        assert action in (0, 1, 2)

    def test_predict_signal_keys(self):
        agent  = CNNAgent()
        matrix = np.random.randint(0, 2, (32, 32)).astype(np.float32)
        result = agent.predict_signal(matrix)
        assert "action" in result
        assert "signal" in result
        assert "q_values" in result
        assert result["signal"] in ("Long", "Neutral", "Short")

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "cnn_test")
            agent = CNNAgent()
            agent.save(path)

            agent2 = CNNAgent()
            agent2.load(path)

            matrix = np.random.randint(0, 2, (32, 32)).astype(np.float32)
            a1 = agent.predict(matrix)
            a2 = agent2.predict(matrix)
            assert a1 == a2
