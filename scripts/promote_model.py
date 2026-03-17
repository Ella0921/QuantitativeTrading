"""
Model promotion script.

Compares the Staging model against the current Production model
in MLflow registry, and promotes if the Staging model has a
higher Sharpe ratio (by at least THRESHOLD).

This script is called by the retrain Airflow DAG, but can also
be run manually after a training run.

Usage:
    python scripts/promote_model.py
    python scripts/promote_model.py --threshold 0.05 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

THRESHOLD    = 0.05   # Staging must beat Production by this Sharpe margin
MODEL_NAME   = "dqn_agent"
TICKER       = "^TWII"
TEST_START   = "2023-01-01"
TEST_END     = "2024-12-31"


def get_model_sharpe(stage: str, client, ticker: str) -> tuple[float, str]:
    """
    Load a model from MLflow registry, run backtest, return Sharpe ratio.
    Returns (sharpe, version).
    """
    import mlflow.tensorflow
    from src.data.downloader import download, get_train_test_split
    from src.features.indicators import add_all, to_macd_series
    from src.models.dqn_agent import DQNAgent
    from src.backtest.engine import BacktestEngine

    versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
    if not versions:
        return None, None

    version = versions[0].version

    # Check if eval metrics already logged (avoid re-running backtest)
    run = client.get_run(versions[0].run_id)
    cached_sharpe = run.data.metrics.get("eval_sharpe_ratio")
    if cached_sharpe is not None:
        return float(cached_sharpe), version

    # Run backtest to get Sharpe
    tf_model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/{stage}")
    agent    = DQNAgent()
    agent.model = tf_model

    df = download(ticker, start=TEST_START, end=TEST_END)
    df = add_all(df)
    prices = df["Close"].tolist()
    macd   = to_macd_series(prices)

    buys, sells, _ = agent.backtest(prices, macd=macd)
    engine  = BacktestEngine(initial_capital=1_000_000,
                             stop_loss_pct=0.05, max_position_pct=0.20)
    result  = engine.run(prices, buys, sells, ticker=ticker)
    sharpe  = result.metrics["sharpe_ratio"]

    # Cache back to MLflow
    import mlflow
    with mlflow.start_run(run_id=versions[0].run_id):
        mlflow.log_metric("eval_sharpe_ratio", sharpe)

    return float(sharpe), version


def promote(dry_run: bool = False, threshold: float = THRESHOLD):
    import mlflow

    uri = "sqlite:///mlruns.db"
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()

    print(f"\nModel registry: {MODEL_NAME}")
    print(f"Promotion threshold: Sharpe improvement > {threshold}")
    print(f"Test period: {TEST_START} → {TEST_END}")
    print(f"Ticker: {TICKER}\n")

    # ── Get Staging model ─────────────────────────────────────────────────────
    staging_sharpe, staging_version = get_model_sharpe("Staging", client, TICKER)
    if staging_sharpe is None:
        print("❌ No Staging model found. Train a model first:")
        print("   python scripts/train_mlflow.py --ticker ^TWII --model dqn")
        return

    print(f"Staging  v{staging_version}: Sharpe = {staging_sharpe:.4f}")

    # ── Get Production model ──────────────────────────────────────────────────
    prod_sharpe, prod_version = get_model_sharpe("Production", client, TICKER)
    if prod_sharpe is None:
        print("No Production model exists → promoting Staging unconditionally")
        if not dry_run:
            client.transition_model_version_stage(
                name=MODEL_NAME, version=staging_version, stage="Production"
            )
            print(f"✅ v{staging_version} promoted to Production")
        else:
            print(f"[dry-run] Would promote v{staging_version} to Production")
        return

    print(f"Production v{prod_version}: Sharpe = {prod_sharpe:.4f}")

    # ── Compare ───────────────────────────────────────────────────────────────
    improvement = staging_sharpe - prod_sharpe
    print(f"\nImprovement: {improvement:+.4f} (threshold: {threshold})")

    if improvement >= threshold:
        print(f"\n✅ Promoting v{staging_version} to Production "
              f"(Sharpe {prod_sharpe:.4f} → {staging_sharpe:.4f})")
        if not dry_run:
            client.transition_model_version_stage(
                name=MODEL_NAME, version=prod_version, stage="Archived"
            )
            client.transition_model_version_stage(
                name=MODEL_NAME, version=staging_version, stage="Production"
            )
            print(f"   v{prod_version} → Archived")
            print(f"   v{staging_version} → Production")
    else:
        print(f"\n⏸  Not promoting — improvement ({improvement:+.4f}) "
              f"below threshold ({threshold})")
        print(f"   Production v{prod_version} remains unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", default=THRESHOLD, type=float)
    parser.add_argument("--dry-run",   action="store_true",
                        help="Show what would happen without making changes")
    parser.add_argument("--ticker",    default=TICKER)
    args = parser.parse_args()

    TICKER = args.ticker
    promote(dry_run=args.dry_run, threshold=args.threshold)
