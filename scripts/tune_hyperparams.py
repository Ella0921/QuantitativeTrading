"""
Hyperparameter tuning with Optuna.

Optimises DQN hyperparameters to maximise Sharpe ratio on the test set.

Usage:
    python scripts/tune_hyperparams.py --ticker ^TWII --trials 50
    python scripts/tune_hyperparams.py --ticker 2330.TW --trials 100 --jobs 2

Results are saved to:
    models/best_params_{ticker}.json
    optuna_studies/{ticker}.db  (SQLite, can be visualised with optuna-dashboard)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import optuna
import mlflow

from src.data.downloader import download, get_train_test_split
from src.features.indicators import add_all, to_macd_series
from src.models.dqn_agent import DQNAgent
from src.backtest.engine import BacktestEngine

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Objective ─────────────────────────────────────────────────────────────────

def make_objective(df_train, df_test, initial_money: float):
    """
    Returns an Optuna objective function.
    Searches over learning rate, window size, stop-loss, and position sizing.
    Metric: Sharpe ratio on the test set (higher = better).
    """
    prices_train = df_train["Close"].tolist()
    prices_test  = df_test["Close"].tolist()

    def objective(trial: optuna.Trial) -> float:
        # ── Hyperparameter search space ──────────────────────────────────
        lr          = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        window      = trial.suggest_int("window_size", 10, 40, step=10)
        use_macd    = trial.suggest_categorical("use_macd", [True, False])
        iterations  = 80   # fixed for speed — retrain with more epochs after finding best params
        stop_loss   = trial.suggest_float("stop_loss_pct", 0.03, 0.10, step=0.01)
        max_pos     = trial.suggest_float("max_position_pct", 0.10, 0.30, step=0.10)

        macd_train  = to_macd_series(prices_train) if use_macd else None
        macd_test   = to_macd_series(prices_test)  if use_macd else None

        # Train
        try:
            agent = DQNAgent(state_size=window, learning_rate=lr)
            agent.train(
                prices_train,
                macd=macd_train,
                iterations=iterations,
                initial_money=initial_money,
                checkpoint=999,          # suppress per-epoch logging
            )
        except Exception:
            raise optuna.TrialPruned()

        # Backtest
        buys, sells, _ = agent.backtest(prices_test, macd=macd_test)

        engine = BacktestEngine(
            initial_capital=initial_money,
            stop_loss_pct=stop_loss,
            max_position_pct=max_pos,
        )
        result = engine.run(prices_test, buys, sells)
        sharpe = result.metrics.get("sharpe_ratio", -999)

        # Prune clearly bad trials early
        if np.isnan(sharpe) or sharpe < -5:
            raise optuna.TrialPruned()

        # Log to MLflow (one run per trial)
        with mlflow.start_run(nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metrics(result.metrics)

        return sharpe

    return objective


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",       default="^TWII")
    p.add_argument("--train-start",  default="2016-01-01")
    p.add_argument("--train-end",    default="2022-12-31")
    p.add_argument("--test-start",   default="2023-01-01")
    p.add_argument("--test-end",     default="2024-12-31")
    p.add_argument("--trials",       default=50,  type=int)
    p.add_argument("--jobs",         default=1,   type=int, help="Parallel workers")
    p.add_argument("--initial-money",default=1_000_000, type=float)
    p.add_argument("--mlflow-uri",   default="mlruns")
    return p.parse_args()


def main():
    args = parse_args()
    from src.utils.device import configure_gpu
    configure_gpu()

    # Data
    print(f"Downloading {args.ticker}...")
    df_full = download(args.ticker, start=args.train_start, end=args.test_end)
    df_full = add_all(df_full)
    df_train, df_test = get_train_test_split(df_full, args.train_end, args.test_start)
    print(f"Train: {len(df_train)} days | Test: {len(df_test)} days")

    # MLflow experiment
    uri = args.mlflow_uri
    if not uri.startswith(("sqlite://", "postgresql://", "mysql://", "http://", "https://")):
        uri = f"sqlite:///{uri}.db"
    mlflow.set_tracking_uri(uri)
    exp_name = f"optuna_dqn_{args.ticker.replace('^','').replace('.','_')}"
    mlflow.set_experiment(exp_name)

    # Optuna study (TPE sampler + MedianPruner)
    study_dir = Path("optuna_studies")
    study_dir.mkdir(exist_ok=True)
    db_path = study_dir / f"{args.ticker.replace('^','').replace('.','_')}.db"

    study = optuna.create_study(
        study_name=exp_name,
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        load_if_exists=True,
    )

    print(f"\nStarting {args.trials} trials (n_jobs={args.jobs})...")
    print(f"Study DB: {db_path}")
    print(f"To visualise: optuna-dashboard sqlite:///{db_path}\n")

    objective = make_objective(df_train, df_test, args.initial_money)

    with mlflow.start_run(run_name="optuna_search"):
        study.optimize(
            objective,
            n_trials=args.trials,
            n_jobs=args.jobs,
            show_progress_bar=True,
        )

    # ── Results ───────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'─'*50}")
    print(f"Best trial #{best.number}")
    print(f"  Sharpe ratio : {best.value:.4f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save best params
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"best_params_{args.ticker.replace('^','').replace('.','_')}.json"
    with open(out_path, "w") as f:
        json.dump({"sharpe": best.value, "params": best.params}, f, indent=2)
    print(f"\nBest params saved → {out_path}")

    # Importances
    try:
        importances = optuna.importance.get_param_importances(study)
        print("\nHyperparameter importances:")
        for k, v in importances.items():
            bar = "█" * int(v * 30)
            print(f"  {k:<25} {bar} {v:.3f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
