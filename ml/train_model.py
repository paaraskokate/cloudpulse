"""Train the Random Forest CPU forecaster with a strict chronological split.

Time-series data must NEVER be randomly shuffled for train/test splitting:
test samples must occur strictly after training samples. This module also
computes a persistence baseline (predicting 'same as last observed value')
so the model has a fair comparison point.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ml.features import add_features, infer_interval_minutes, load_series, make_supervised

ROOT = Path(__file__).resolve().parents[1]


def chronological_split(n: int, train_frac: float = 0.8) -> int:
    """Boundary index so that train = [0, split) and test = [split, n)."""
    if not 0.1 <= train_frac < 1.0:
        raise ValueError("train_frac must be within [0.1, 1.0).")
    split = int(n * train_frac)
    return max(1, min(n - 1, split))


def compute_metrics(y_true, y_pred) -> tuple[float, float]:
    """MAE and RMSE between two equally sized arrays."""
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse


def train_model(
    input_csv: str | Path,
    model_out: str | Path | None = None,
    report_out: str | Path | None = None,
    train_frac: float = 0.8,
    seed: int = 42,
) -> dict:
    """Train on the early part of the series, test on the later part, save bundle."""
    df = load_series(input_csv)
    feat = add_features(df)
    X, y, times, names = make_supervised(feat)
    if len(y) < 25:
        raise ValueError(
            f"Not enough supervised samples ({len(y)}). Fetch a longer history window "
            f"(e.g. 6+ hours at a 5-minute period)."
        )

    split = chronological_split(len(y), train_frac)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Persistence baseline on the SAME test samples: predict y at t from y at t-1.
    lag1_te = X_te[:, names.index("lag_1")]
    persistence_mae, persistence_rmse = compute_metrics(y_te, lag1_te)

    model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae, rmse = compute_metrics(y_te, preds)

    interval = infer_interval_minutes(df) or 5.0
    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "persistence_mae": round(persistence_mae, 4),
        "persistence_rmse": round(persistence_rmse, 4),
        "n_train": int(split),
        "n_test": int(len(y) - split),
        "train_start_utc": str(times[0]),
        "train_end_utc": str(times[split - 1]),
        "test_start_utc": str(times[split]),
        "test_end_utc": str(times[-1]),
        "forecast_interval_minutes": interval,
        "n_features": len(names),
    }
    bundle = {
        "model": model,
        "feature_names": names,
        "model_type": "RandomForestRegressor",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_csv),
        "metrics": metrics,
        "forecast_interval_minutes": interval,
    }

    if model_out is not None:
        model_out = Path(model_out)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_out)
    if report_out is not None:
        report_out = Path(report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return bundle


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the CloudPulse CPU forecaster.")
    p.add_argument("--input", default=str(ROOT / "data" / "synthetic_cpu.csv"),
                   help="CSV with columns timestamp,cpu (synthetic or real CloudWatch export).")
    p.add_argument("--model-out", default=str(ROOT / "models" / "rf_cpu.joblib"))
    p.add_argument("--report-out", default=str(ROOT / "models" / "training_report.json"))
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    bundle = train_model(args.input, args.model_out, args.report_out, args.train_frac, args.seed)
    m = bundle["metrics"]
    print("Training complete.")
    print(f"  Source      : {args.input}")
    print(f"  Train/test  : {m['n_train']} / {m['n_test']} samples (chronological split)")
    print(f"  MAE         : {m['mae']:.3f}   RMSE: {m['rmse']:.3f}")
    print(f"  Baseline MAE: {m['persistence_mae']:.3f}   RMSE: {m['persistence_rmse']:.3f}")
    if m["mae"] > m["persistence_mae"]:
        print("  NOTE: model does not beat the persistence baseline on this data.")
    else:
        print("  Model beats the persistence baseline on this data.")
    print(f"  Model saved : {args.model_out}")
    print(f"  Report saved: {args.report_out}")
    sys.exit(0)
