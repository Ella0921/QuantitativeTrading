"""
Feature and prediction drift monitoring using PSI
(Population Stability Index).

PSI is the standard metric used in financial ML to detect
when a model's input distribution has shifted enough to
warrant retraining.

Thresholds (industry standard):
  PSI < 0.10  → stable, no action needed
  PSI < 0.20  → moderate shift, monitor closely
  PSI >= 0.20 → significant shift, retrain recommended

Interview talking point:
  "I added a drift monitoring layer that runs weekly. It computes PSI
   for each feature comparing the current week's distribution against
   the previous week. If any feature's PSI exceeds 0.2, the retrain
   DAG is triggered automatically. This closes the feedback loop
   between model performance degradation and the training pipeline."
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


# ── PSI computation ───────────────────────────────────────────────────────────

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute PSI between a reference and current distribution.

    PSI = sum((current% - reference%) * ln(current% / reference%))

    Args:
        reference: 1-D array of reference period values
        current:   1-D array of current period values
        n_bins:    number of equal-width buckets
        epsilon:   small constant to avoid log(0)

    Returns:
        PSI score (float)
    """
    # Build bins from reference distribution
    ref_clean = reference[np.isfinite(reference)]
    cur_clean = current[np.isfinite(current)]

    if len(ref_clean) < 10 or len(cur_clean) < 10:
        return 0.0  # not enough data to compute PSI

    bins = np.percentile(ref_clean, np.linspace(0, 100, n_bins + 1))
    bins[0]  -= epsilon
    bins[-1] += epsilon

    ref_counts, _ = np.histogram(ref_clean, bins=bins)
    cur_counts, _ = np.histogram(cur_clean, bins=bins)

    ref_pct = (ref_counts / len(ref_clean)) + epsilon
    cur_pct = (cur_counts / len(cur_clean)) + epsilon

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 6)


def compute_psi_report(
    reference_df: pd.DataFrame,
    current_df:   pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    ticker: str = "",
) -> dict:
    """
    Compute PSI for all feature columns between two DataFrames.

    Returns a dict with:
      - feature_psi: {col: psi_score}
      - max_psi: the highest PSI across all features
      - drifted: list of features with PSI >= 0.2
      - status: "stable" | "moderate" | "drifted"
    """
    if feature_cols is None:
        feature_cols = [
            c for c in reference_df.columns
            if c not in ("Open", "High", "Low", "Volume")
            and reference_df[c].dtype in (np.float64, np.float32, float)
        ]

    feature_psi = {}
    for col in feature_cols:
        if col in reference_df.columns and col in current_df.columns:
            psi = compute_psi(
                reference_df[col].values,
                current_df[col].values,
            )
            feature_psi[col] = psi

    max_psi  = max(feature_psi.values()) if feature_psi else 0.0
    drifted  = [f for f, p in feature_psi.items() if p >= 0.20]
    moderate = [f for f, p in feature_psi.items() if 0.10 <= p < 0.20]

    if drifted:
        status = "drifted"
    elif moderate:
        status = "moderate"
    else:
        status = "stable"

    return {
        "ticker":      ticker,
        "timestamp":   datetime.now().isoformat(),
        "status":      status,
        "max_psi":     round(max_psi, 6),
        "drifted":     drifted,
        "moderate":    moderate,
        "feature_psi": feature_psi,
        "n_features":  len(feature_psi),
        "ref_rows":    len(reference_df),
        "cur_rows":    len(current_df),
    }


# ── Prediction drift ──────────────────────────────────────────────────────────

def compute_prediction_drift(
    ref_predictions: np.ndarray,
    cur_predictions: np.ndarray,
    label: str = "signal",
) -> dict:
    """
    Monitor drift in model output distribution.

    For classification (Long/Neutral/Short), computes the shift
    in class proportions between reference and current periods.

    Returns PSI of the output distribution + proportion breakdown.
    """
    unique = np.unique(np.concatenate([ref_predictions, cur_predictions]))

    ref_props = {int(k): float(np.mean(ref_predictions == k)) for k in unique}
    cur_props = {int(k): float(np.mean(cur_predictions == k)) for k in unique}

    psi = compute_psi(
        ref_predictions.astype(float),
        cur_predictions.astype(float),
        n_bins=len(unique),
    )

    return {
        "label":          label,
        "psi":            round(psi, 6),
        "status":         "drifted" if psi >= 0.20 else "moderate" if psi >= 0.10 else "stable",
        "ref_proportions": ref_props,
        "cur_proportions": cur_props,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from src.data.downloader import download
    from src.features.indicators import add_all

    ticker = sys.argv[1] if len(sys.argv) > 1 else "^TWII"
    df = download(ticker, start="2022-01-01")
    df = add_all(df)

    mid   = len(df) // 2
    ref   = df.iloc[:mid]
    curr  = df.iloc[mid:]
    report = compute_psi_report(ref, curr, ticker=ticker)

    print(f"\nDrift report for {ticker}")
    print(f"Status : {report['status'].upper()}")
    print(f"Max PSI: {report['max_psi']:.4f}")
    if report["drifted"]:
        print(f"Drifted features: {report['drifted']}")
    print("\nPer-feature PSI:")
    for feat, psi in sorted(report["feature_psi"].items(), key=lambda x: -x[1]):
        flag = " ⚠️" if psi >= 0.20 else " ●" if psi >= 0.10 else ""
        print(f"  {feat:<20} {psi:.4f}{flag}")
