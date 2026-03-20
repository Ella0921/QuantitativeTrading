"""
Weekly model retraining pipeline — Airflow DAG.

Schedule: every Sunday at 02:00 UTC

Pipeline tasks:
    check_drift_trigger  → skip retrain if PSI < 0.1 (stable)
    train_candidate      → train new DQN on latest features, log to MLflow
    evaluate_candidate   → backtest on holdout, compute Sharpe/MDD/WinRate
    promote_if_better    → compare vs production model, promote if Sharpe ↑
    notify               → log result summary

Design decisions worth explaining in interviews:
  - We only retrain if drift is detected OR it's been > 4 weeks since last train
  - Promotion is automatic but guarded: new model must beat prod by > 0.05 Sharpe
  - All experiments tracked in MLflow so every promotion has an audit trail

To run locally:
    python dags/retrain_pipeline.py --run-local
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator, ShortCircuitOperator
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

PROMOTION_SHARPE_THRESHOLD = 0.05   # new model must beat prod by this margin
MIN_RETRAIN_INTERVAL_DAYS  = 7
DRIFT_PSI_TRIGGER          = 0.10   # retrain if any feature PSI exceeds this
TICKER                     = "^TWII"
TRAIN_START                = "2016-01-01"


# ── Task functions ────────────────────────────────────────────────────────────

def task_should_retrain(**context) -> bool:
    """
    ShortCircuit: return False to skip downstream tasks if no retrain needed.

    Retrain if:
      (a) any feature's PSI >= DRIFT_PSI_TRIGGER in the latest drift log, OR
      (b) no model has been trained in the last MIN_RETRAIN_INTERVAL_DAYS
    """
    import json
    import glob
    from datetime import datetime

    log_dir = Path("data/logs")
    drift_logs = sorted(glob.glob(str(log_dir / "drift_TWII_*.json")))

    if drift_logs:
        latest_log = json.load(open(drift_logs[-1]))
        max_psi = latest_log.get("max_psi", 0)
        if max_psi >= DRIFT_PSI_TRIGGER:
            print(f"[retrain] Drift detected (max PSI={max_psi:.4f}) → retraining")
            return True

    # Check last training date from MLflow
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("dqn_agent", stages=["Production", "Staging"])
        if versions:
            last_ts = max(v.creation_timestamp for v in versions) / 1000
            days_since = (datetime.now().timestamp() - last_ts) / 86400
            if days_since > MIN_RETRAIN_INTERVAL_DAYS:
                print(f"[retrain] {days_since:.0f} days since last train → retraining")
                return True
            print(f"[retrain] Model fresh ({days_since:.0f}d), no drift → skipping")
            return False
    except Exception:
        pass

    print("[retrain] No existing model found → retraining")
    return True


def task_train_candidate(**context) -> str:
    """
    Train a new DQN candidate on the latest versioned features.
    Logs everything to MLflow and returns the run_id.
    """
    import mlflow
    import pandas as pd
    from datetime import datetime

    from src.features.indicators import to_macd_series
    from src.models.dqn_agent import DQNAgent
    from src.data.downloader import download, get_train_test_split
    from src.features.indicators import add_all

    feat_path = Path("data/features") / TICKER.replace("^","").replace(".","_") / "latest.parquet"
    if feat_path.exists():
        df = pd.read_parquet(feat_path)
    else:
        df = download(TICKER, start=TRAIN_START)
        df = add_all(df)

    train_end  = "2022-12-31"
    df_train, _ = get_train_test_split(df, train_end)
    prices = df_train["Close"].tolist()
    macd   = to_macd_series(prices)

    mlflow.set_experiment("dqn_weekly_retrain")

    with mlflow.start_run(run_name=f"candidate_{datetime.now().strftime('%Y%m%d')}") as run:
        mlflow.log_params({
            "ticker":      TICKER,
            "train_end":   train_end,
            "use_macd":    True,
            "iterations":  200,
            "trigger":     "scheduled_retrain",
        })

        agent = DQNAgent()
        result = agent.train(prices, macd=macd, iterations=200,
                             initial_money=1_000_000, checkpoint=200)

        mlflow.log_metrics({
            "train_final_return_pct": result["final_return_pct"],
        })

        model_path = f"models/candidate_{run.info.run_id[:8]}"
        agent.save(model_path)
        mlflow.tensorflow.log_model(agent.model, "model")

        # Register in MLflow model registry as Staging
        model_uri  = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(model_uri, "dqn_agent")
        client     = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="dqn_agent",
            version=registered.version,
            stage="Staging",
        )

        print(f"[train] Candidate registered: version {registered.version} → Staging")
        return run.info.run_id


def task_evaluate_candidate(**context) -> dict:
    """
    Backtest the Staging model on the holdout test set.
    Logs metrics back to MLflow and returns the metrics dict.
    """
    import mlflow
    import mlflow.tensorflow
    from src.data.downloader import download, get_train_test_split
    from src.features.indicators import add_all, to_macd_series
    from src.models.dqn_agent import DQNAgent
    from src.backtest.engine import BacktestEngine

    run_id = context["task_instance"].xcom_pull(task_ids="train_candidate")

    df      = download(TICKER, start="2016-01-01", end="2024-12-31")
    df      = add_all(df)
    _, df_test = get_train_test_split(df, "2022-12-31", "2023-01-01")
    prices  = df_test["Close"].tolist()
    macd    = to_macd_series(prices)

    # Load staging model
    client   = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("dqn_agent", stages=["Staging"])
    if not versions:
        raise RuntimeError("No Staging model found")

    model_uri = "models:/dqn_agent/Staging"
    tf_model = mlflow.tensorflow.load_model(model_uri)

    agent = DQNAgent()
    agent.model = tf_model

    buys, sells, _ = agent.backtest(prices, macd=macd)
    engine  = BacktestEngine(initial_capital=1_000_000,
                             stop_loss_pct=0.05, max_position_pct=0.20)
    result  = engine.run(prices, buys, sells, ticker=TICKER)
    metrics = result.metrics

    # Log evaluation metrics back to the training run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})

    print(f"[evaluate] Staging — Sharpe: {metrics['sharpe_ratio']:.4f}  "
          f"Return: {metrics['total_return_pct']:+.2f}%  "
          f"MDD: {metrics['max_drawdown_pct']:.2f}%")
    return metrics


def task_promote_if_better(**context) -> dict:
    """
    Compare Staging vs Production model.
    Promote Staging → Production if Sharpe ratio improves by PROMOTION_SHARPE_THRESHOLD.

    If no Production model exists, promote unconditionally.
    Archives the old Production model as Archived (keeps audit trail).
    """
    import mlflow
    from src.data.downloader import download, get_train_test_split
    from src.features.indicators import add_all, to_macd_series
    from src.models.dqn_agent import DQNAgent
    from src.backtest.engine import BacktestEngine

    candidate_metrics = context["task_instance"].xcom_pull(task_ids="evaluate_candidate")
    candidate_sharpe  = candidate_metrics["sharpe_ratio"]

    client = mlflow.tracking.MlflowClient()

    # Get current production model's Sharpe (if exists)
    prod_versions = client.get_latest_versions("dqn_agent", stages=["Production"])
    if prod_versions:
        prod_run_id = prod_versions[0].run_id
        prod_run    = client.get_run(prod_run_id)
        prod_sharpe = prod_run.data.metrics.get("eval_sharpe_ratio", 0.0)

        gap = candidate_sharpe - prod_sharpe
        print(f"[promote] Candidate Sharpe: {candidate_sharpe:.4f}  "
              f"Production Sharpe: {prod_sharpe:.4f}  "
              f"Gap: {gap:+.4f}  Threshold: {PROMOTION_SHARPE_THRESHOLD}")

        if gap < PROMOTION_SHARPE_THRESHOLD:
            print("[promote] Not promoting — improvement below threshold")
            return {"promoted": False, "reason": "below_threshold",
                    "candidate_sharpe": candidate_sharpe, "prod_sharpe": prod_sharpe}

        # Archive old production
        client.transition_model_version_stage(
            name="dqn_agent",
            version=prod_versions[0].version,
            stage="Archived",
        )
    else:
        print("[promote] No existing Production model — promoting unconditionally")

    # Promote staging → production
    staging_versions = client.get_latest_versions("dqn_agent", stages=["Staging"])
    client.transition_model_version_stage(
        name="dqn_agent",
        version=staging_versions[0].version,
        stage="Production",
    )
    print(f"[promote] ✅ Version {staging_versions[0].version} promoted to Production")
    return {"promoted": True, "new_sharpe": candidate_sharpe}


# ── Airflow DAG ───────────────────────────────────────────────────────────────

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="model_retrain_pipeline",
        description="Weekly model retrain with drift-triggered scheduling and auto-promotion",
        schedule_interval="0 2 * * 0",   # Sunday 02:00 UTC
        start_date=days_ago(7),
        catchup=False,
        default_args={
            "owner": "quant-pipeline",
            "retries": 1,
            "retry_delay": timedelta(minutes=10),
        },
        tags=["quant", "training", "mlops"],
    ) as dag:

        should_retrain = ShortCircuitOperator(
            task_id="check_drift_trigger",
            python_callable=task_should_retrain,
        )
        train = PythonOperator(
            task_id="train_candidate",
            python_callable=task_train_candidate,
        )
        evaluate = PythonOperator(
            task_id="evaluate_candidate",
            python_callable=task_evaluate_candidate,
        )
        promote = PythonOperator(
            task_id="promote_if_better",
            python_callable=task_promote_if_better,
        )

        should_retrain >> train >> evaluate >> promote


# ── Local execution ───────────────────────────────────────────────────────────

def run_local():
    print("\n=== Running retrain pipeline locally ===\n")
    should = task_should_retrain()
    if not should:
        print("No retrain needed.")
        return
    run_id = task_train_candidate()
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    run_local()
