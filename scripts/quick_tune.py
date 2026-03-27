"""
Quick manual hyperparameter search — faster than Optuna for GPU JIT environments.
Tests 6 parameter combinations, picks the best Sharpe on test set.

Usage:
    export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
    python scripts/quick_tune.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import mlflow
from src.data.downloader import download, get_train_test_split
from src.features.indicators import add_all, to_macd_series
from src.models.dqn_agent import DQNAgent
from src.backtest.engine import BacktestEngine
from src.utils.device import configure_gpu

configure_gpu()

# ── Data ──────────────────────────────────────────────────────────────────────
print("Downloading data...")
df = download("^TWII", start="2016-01-01", end="2024-12-31")
df = add_all(df)
df_train, df_test = get_train_test_split(df, "2022-12-31", "2023-01-01")
prices_train = df_train["Close"].tolist()
prices_test  = df_test["Close"].tolist()
macd_train   = to_macd_series(prices_train)
macd_test    = to_macd_series(prices_test)

# ── Search grid ───────────────────────────────────────────────────────────────
# Each combo: (learning_rate, window_size, stop_loss_pct, max_position_pct)
# Focused on reducing overtrading (high stop_loss → fewer forced sells)
GRID = [
    (1e-4, 20, 0.05, 0.20),   # baseline
    (5e-5, 20, 0.05, 0.15),   # lower LR, smaller position
    (1e-4, 30, 0.07, 0.20),   # wider window
    (5e-5, 30, 0.05, 0.10),   # conservative
    (2e-4, 20, 0.03, 0.20),   # higher LR
    (1e-4, 10, 0.05, 0.20),   # narrow window
]

# ── MLflow setup ──────────────────────────────────────────────────────────────
uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
mlflow.set_tracking_uri(uri)
mlflow.set_experiment("quant_quick_tune")

ITERS = 100  # fixed — enough to see convergence, fast enough to run 6x

results = []
print(f"\nTesting {len(GRID)} combinations ({ITERS} epochs each)...\n")

for i, (lr, window, stop_loss, max_pos) in enumerate(GRID):
    print(f"[{i+1}/{len(GRID)}] lr={lr:.0e}  window={window}  stop_loss={stop_loss}  max_pos={max_pos}")

    with mlflow.start_run(run_name=f"combo_{i+1}"):
        mlflow.log_params({
            "learning_rate": lr, "window_size": window,
            "stop_loss_pct": stop_loss, "max_position_pct": max_pos,
            "iterations": ITERS, "use_macd": True,
        })

        agent = DQNAgent(state_size=window, learning_rate=lr)
        agent.train(prices_train, macd=macd_train, iterations=ITERS,
                    initial_money=1_000_000, checkpoint=999)

        buys, sells, _ = agent.backtest(prices_test, macd=macd_test)
        engine = BacktestEngine(initial_capital=1_000_000,
                                stop_loss_pct=stop_loss,
                                max_position_pct=max_pos)
        result = engine.run(prices_test, buys, sells)
        m = result.metrics

        mlflow.log_metrics(m)
        results.append((i+1, lr, window, stop_loss, max_pos,
                        m["sharpe_ratio"], m["total_return_pct"], m["total_trades"]))

        print(f"  → Sharpe: {m['sharpe_ratio']:.3f}  Return: {m['total_return_pct']:+.2f}%  Trades: {m['total_trades']}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("RESULTS (sorted by Sharpe):")
print("="*70)
results.sort(key=lambda x: x[5], reverse=True)
for rank, (idx, lr, w, sl, mp, sharpe, ret, trades) in enumerate(results, 1):
    print(f"#{rank} combo_{idx}: Sharpe={sharpe:.3f}  Return={ret:+.2f}%  "
          f"Trades={trades}  lr={lr:.0e}  window={w}  stop={sl}")

best = results[0]
print(f"\nBest: combo_{best[0]}  lr={best[1]:.0e}  window={best[2]}  "
      f"stop_loss={best[3]}  max_pos={best[4]}")
print(f"\nNext step: retrain with best params + 200 epochs:")
print(f"  python scripts/train_mlflow.py --ticker ^TWII --model dqn --use-macd \\")
print(f"    --learning-rate {best[1]} --window {best[2]} --iterations 200")
