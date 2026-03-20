"""
Daily stock data pipeline — Airflow DAG.

Schedule: weekdays at 18:00 Asia/Taipei (after market close)

Pipeline tasks:
    download_raw      → fetch OHLCV from yfinance, save to data/raw/
    validate_quality  → run Great Expectations suite, fail fast on bad data
    compute_features  → calculate MACD/RSI/BB/ATR, save versioned Parquet
    check_feature_drift → PSI check vs last week's features, alert if drifting

This DAG is the entry point of the entire pipeline.
Every downstream component (training, serving) reads from data/features/,
never from data/raw/ — enforcing a clean separation between ingestion and use.

To run locally without Airflow:
    python dags/stock_pipeline.py --run-local --ticker ^TWII
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ── Airflow imports (graceful fallback for local execution) ───────────────────
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.downloader import download
from src.features.indicators import add_all
from src.monitoring.data_quality import validate_raw_data

TICKERS    = ["^TWII", "2330.TW", "2317.TW", "2454.TW"]
RAW_DIR    = Path("data/raw")
FEAT_DIR   = Path("data/features")
LOG_DIR    = Path("data/logs")

for d in [RAW_DIR, FEAT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Task functions ────────────────────────────────────────────────────────────

def task_download_raw(ticker: str, execution_date: str | None = None) -> dict:
    """
    Download latest OHLCV data and save as Parquet in data/raw/.

    Uses a 2-year rolling window so the file doesn't grow unbounded.
    Idempotent: re-running for the same date overwrites the same file.
    """
    end   = execution_date or datetime.today().strftime("%Y-%m-%d")
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=730)).strftime("%Y-%m-%d")

    df = download(ticker, start=start, end=end, use_cache=False)  # always fresh

    out = RAW_DIR / f"{ticker.replace('^','').replace('.','_')}_{end}.parquet"
    df.to_parquet(out)

    print(f"[download] {ticker}: {len(df)} rows → {out}")
    return {"ticker": ticker, "rows": len(df), "path": str(out)}


def task_validate_quality(ticker: str, execution_date: str | None = None) -> dict:
    """
    Run data quality checks on the raw Parquet file.

    Checks:
      - No missing values in OHLCV columns
      - Close > 0 (no zero/negative prices)
      - Date index is monotonically increasing
      - Volume > 0
      - No duplicate dates
      - Row count > 200 (enough data for meaningful features)

    Raises ValueError on failure — Airflow marks the task as FAILED
    and halts downstream tasks.
    """
    end  = execution_date or datetime.today().strftime("%Y-%m-%d")
    path = RAW_DIR / f"{ticker.replace('^','').replace('.','_')}_{end}.parquet"

    import pandas as pd
    df = pd.read_parquet(path)

    result = validate_raw_data(df, ticker=ticker)

    log_path = LOG_DIR / f"quality_{ticker.replace('^','').replace('.','_')}_{end}.json"
    import json
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)

    if not result["passed"]:
        failed = [c["name"] for c in result["checks"] if not c["passed"]]
        raise ValueError(f"Data quality FAILED for {ticker}: {failed}")

    print(f"[quality] {ticker}: all {len(result['checks'])} checks passed")
    return result


def task_compute_features(ticker: str, execution_date: str | None = None) -> dict:
    """
    Compute technical indicators and save a versioned feature Parquet.

    Output path: data/features/{ticker}/{YYYY-MM-DD}.parquet
    The date in the filename = the execution date = the latest data date.
    This versioning allows:
      - Reproducible training (pin to a specific feature date)
      - Rollback if a bad feature version is deployed
      - Audit trail for feature changes
    """
    import pandas as pd

    end  = execution_date or datetime.today().strftime("%Y-%m-%d")
    path = RAW_DIR / f"{ticker.replace('^','').replace('.','_')}_{end}.parquet"

    df = pd.read_parquet(path)
    df = add_all(df)  # MACD, RSI, BB, ATR — dropna handled inside

    out_dir = FEAT_DIR / ticker.replace("^","").replace(".","_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{end}.parquet"
    df.to_parquet(out)

    # Also write a "latest" symlink-style pointer file for easy serving access
    latest = out_dir / "latest.parquet"
    df.to_parquet(latest)

    print(f"[features] {ticker}: {df.shape[1]} features, {len(df)} rows → {out}")
    return {"ticker": ticker, "features": df.shape[1], "rows": len(df), "path": str(out)}


def task_check_feature_drift(ticker: str, execution_date: str | None = None) -> dict:
    """
    Compare today's feature distribution vs 7 days ago using PSI.

    PSI thresholds (standard):
      PSI < 0.1  → no significant drift
      PSI < 0.2  → moderate drift, monitor
      PSI >= 0.2 → significant drift, investigate / retrain

    Logs the result but does not fail the DAG (drift is an alert, not an error).
    """
    import pandas as pd

    end      = execution_date or datetime.today().strftime("%Y-%m-%d")
    prev_end = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")

    feat_dir = FEAT_DIR / ticker.replace("^","").replace(".","_")
    curr_path = feat_dir / f"{end}.parquet"
    prev_path = feat_dir / f"{prev_end}.parquet"

    if not prev_path.exists():
        print(f"[drift] {ticker}: no previous features at {prev_path}, skipping")
        return {"ticker": ticker, "skipped": True}

    from src.monitoring.drift import compute_psi_report
    curr = pd.read_parquet(curr_path)
    prev = pd.read_parquet(prev_path)

    report = compute_psi_report(prev, curr, ticker=ticker)

    log_path = LOG_DIR / f"drift_{ticker.replace('^','').replace('.','_')}_{end}.json"
    import json
    with open(log_path, "w") as f:
        json.dump(report, f, indent=2)

    drifted = [f for f, psi in report["feature_psi"].items() if psi >= 0.2]
    if drifted:
        print(f"[drift] ⚠️  {ticker}: significant drift in {drifted} — consider retraining")
    else:
        print(f"[drift] {ticker}: feature distributions stable (max PSI: {report['max_psi']:.4f})")

    return report


# ── Airflow DAG definition ────────────────────────────────────────────────────

if AIRFLOW_AVAILABLE:
    default_args = {
        "owner":            "quant-pipeline",
        "retries":          2,
        "retry_delay":      timedelta(minutes=5),
        "email_on_failure": False,
    }

    with DAG(
        dag_id="stock_daily_pipeline",
        description="Daily OHLCV ingestion → quality check → feature computation → drift monitor",
        schedule_interval="0 10 * * 1-5",  # 18:00 Asia/Taipei = 10:00 UTC, weekdays
        start_date=days_ago(1),
        catchup=False,
        default_args=default_args,
        tags=["quant", "ingestion", "features"],
    ) as dag:

        for _ticker in TICKERS:
            t_safe = _ticker.replace("^", "").replace(".", "_")

            t1 = PythonOperator(
                task_id=f"download_raw_{t_safe}",
                python_callable=task_download_raw,
                op_kwargs={"ticker": _ticker,
                           "execution_date": "{{ ds }}"},
            )
            t2 = PythonOperator(
                task_id=f"validate_quality_{t_safe}",
                python_callable=task_validate_quality,
                op_kwargs={"ticker": _ticker,
                           "execution_date": "{{ ds }}"},
            )
            t3 = PythonOperator(
                task_id=f"compute_features_{t_safe}",
                python_callable=task_compute_features,
                op_kwargs={"ticker": _ticker,
                           "execution_date": "{{ ds }}"},
            )
            t4 = PythonOperator(
                task_id=f"check_drift_{t_safe}",
                python_callable=task_check_feature_drift,
                op_kwargs={"ticker": _ticker,
                           "execution_date": "{{ ds }}"},
            )

            # Task dependency: download → validate → features → drift
            t1 >> t2 >> t3 >> t4


# ── Local execution (no Airflow needed) ──────────────────────────────────────

def run_local(ticker: str):
    """Run the full pipeline locally for a single ticker."""
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"\n{'='*50}")
    print(f"Running pipeline locally: {ticker} ({today})")
    print('='*50)

    print("\n[1/4] Downloading raw data...")
    task_download_raw(ticker, today)

    print("\n[2/4] Validating data quality...")
    task_validate_quality(ticker, today)

    print("\n[3/4] Computing features...")
    task_compute_features(ticker, today)

    print("\n[4/4] Checking feature drift...")
    task_check_feature_drift(ticker, today)

    print(f"\n✅ Pipeline complete for {ticker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-local", action="store_true")
    parser.add_argument("--ticker", default="^TWII")
    args = parser.parse_args()

    if args.run_local:
        run_local(args.ticker)
    else:
        print("Add --run-local to run without Airflow.")
        print("For Airflow: place this file in your $AIRFLOW_HOME/dags/ directory.")
