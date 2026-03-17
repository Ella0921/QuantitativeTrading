"""
Training script with full MLflow experiment tracking.

Usage:
    python scripts/train_mlflow.py --ticker ^TWII --model dqn
    python scripts/train_mlflow.py --ticker 2330.TW --model cnn --iterations 300

MLflow UI:
    mlflow ui --port 5000
    open http://localhost:5000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import mlflow.tensorflow
import pandas as pd

from src.data.downloader import download, get_train_test_split
from src.features.indicators import add_all, to_macd_series
from src.models.dqn_agent import DQNAgent
from src.models.cnn_agent import CNNAgent, prepare_training_matrices
from src.backtest.engine import BacktestEngine


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train a trading agent with MLflow tracking")
    p.add_argument("--ticker",       default="^TWII",      help="Yahoo Finance ticker")
    p.add_argument("--model",        default="dqn",        choices=["dqn", "cnn"])
    p.add_argument("--train-start",  default="2016-01-01")
    p.add_argument("--train-end",    default="2022-12-31")
    p.add_argument("--test-start",   default="2023-01-01")
    p.add_argument("--test-end",     default="2024-12-31")
    p.add_argument("--iterations",   default=200, type=int)
    p.add_argument("--lr",           default=1e-5, type=float, help="Learning rate")
    p.add_argument("--window",       default=30,  type=int,   help="DQN state window size")
    p.add_argument("--use-macd",     action="store_true",     help="Use MACD features (DQN)")
    p.add_argument("--initial-money",default=1_000_000, type=float)
    p.add_argument("--stop-loss",    default=0.05, type=float)
    p.add_argument("--max-position", default=0.20, type=float)
    p.add_argument("--mlflow-uri",   default="mlruns",       help="MLflow tracking URI")
    p.add_argument("--experiment",   default=None,           help="MLflow experiment name")
    return p.parse_args()


# ── DQN training run ──────────────────────────────────────────────────────────

def run_dqn(args, df_train: pd.DataFrame, df_test: pd.DataFrame):
    prices_train = df_train["Close"].tolist()
    prices_test  = df_test["Close"].tolist()
    macd_train   = to_macd_series(prices_train) if args.use_macd else None
    macd_test    = to_macd_series(prices_test)  if args.use_macd else None

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "ticker":       args.ticker,
            "model":        "dqn",
            "train_start":  args.train_start,
            "train_end":    args.train_end,
            "test_start":   args.test_start,
            "test_end":     args.test_end,
            "iterations":   args.iterations,
            "learning_rate": args.lr,
            "window_size":  args.window,
            "use_macd":     args.use_macd,
            "initial_money": args.initial_money,
            "stop_loss_pct": args.stop_loss,
            "max_position_pct": args.max_position,
        })

        # Train
        print(f"\n── Training DQN on {args.ticker} ({args.train_start} → {args.train_end}) ──")
        agent = DQNAgent(
            state_size=args.window,
            learning_rate=args.lr,
        )
        train_results = agent.train(
            prices_train,
            macd=macd_train,
            iterations=args.iterations,
            initial_money=args.initial_money,
            checkpoint=max(1, args.iterations // 10),
        )

        # Log training metrics
        for i, loss in enumerate(train_results["losses"]):
            mlflow.log_metric("train_loss", loss, step=i)
        mlflow.log_metric("train_final_return_pct", train_results["final_return_pct"])

        # Save model
        save_path = f"models/dqn_{args.ticker.replace('^','').replace('.','_')}"
        agent.save(save_path)
        mlflow.log_artifact(save_path, artifact_path="model")

        # Backtest on test set
        print(f"\n── Backtesting on {args.ticker} ({args.test_start} → {args.test_end}) ──")
        buys, sells, portfolio = agent.backtest(prices_test, macd=macd_test)

        engine = BacktestEngine(
            initial_capital=args.initial_money,
            stop_loss_pct=args.stop_loss,
            max_position_pct=args.max_position,
        )
        result = engine.run(prices_test, buys, sells, ticker=args.ticker, model_name="DQN")
        bnh    = engine.buy_and_hold_return(prices_test)

        # Log backtest metrics
        mlflow.log_metrics({
            **result.metrics,
            "buy_and_hold_return_pct": bnh,
            "num_buy_signals":  len(buys),
            "num_sell_signals": len(sells),
        })

        # Log model with signature
        mlflow.tensorflow.log_model(agent.model, artifact_path="tf_model")

        print("\n" + result.summary())
        print(f"Buy-and-hold baseline: {bnh:+.2f}%")
        print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")

        return result


# ── CNN training run ──────────────────────────────────────────────────────────

def run_cnn(args, df_train: pd.DataFrame, df_test: pd.DataFrame):
    matrices_train = prepare_training_matrices(df_train)
    returns_train  = df_train["Close"].pct_change().fillna(0).values * 100

    with mlflow.start_run():
        mlflow.log_params({
            "ticker":        args.ticker,
            "model":         "cnn",
            "train_start":   args.train_start,
            "train_end":     args.train_end,
            "learning_rate": args.lr,
            "max_iter":      args.iterations * 25,  # CNN uses more inner iters
        })

        print(f"\n── Training CNN on {args.ticker} ({args.train_start} → {args.train_end}) ──")
        agent = CNNAgent(learning_rate=args.lr)
        losses = agent.train(
            matrices_train,
            returns=returns_train[: len(matrices_train)],
            max_iter=args.iterations * 25,
            log_interval=100,
        )

        for i, loss in enumerate(losses):
            mlflow.log_metric("train_loss", loss, step=i * 10)

        save_path = f"models/cnn_{args.ticker.replace('^','').replace('.','_')}"
        agent.save(save_path)
        mlflow.log_artifact(save_path, artifact_path="model")
        mlflow.tensorflow.log_model(agent.model, artifact_path="tf_model")

        print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    from src.utils.device import configure_gpu
    configure_gpu()

    # Setup MLflow — use SQLite backend to avoid FileStore deprecation warning
    uri = args.mlflow_uri
    if not uri.startswith(("sqlite://", "postgresql://", "mysql://", "http://", "https://")):
        uri = f"sqlite:///{uri}.db"
    mlflow.set_tracking_uri(uri)
    exp_name = args.experiment or f"quant_{args.model}_{args.ticker.replace('^','').replace('.','_')}"
    mlflow.set_experiment(exp_name)

    # Download data
    print(f"Downloading {args.ticker}...")
    df_full  = download(args.ticker, start=args.train_start, end=args.test_end)
    df_full  = add_all(df_full)
    df_train, df_test = get_train_test_split(df_full, args.train_end, args.test_start)

    print(f"Train: {len(df_train)} days | Test: {len(df_test)} days")

    if args.model == "dqn":
        run_dqn(args, df_train, df_test)
    else:
        run_cnn(args, df_train, df_test)


if __name__ == "__main__":
    main()
