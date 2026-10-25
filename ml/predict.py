"""Load a trained CloudPulse model and forecast CPU for the next interval."""
from __future__ import annotations

from pathlib import Path

import joblib

from ml.features import build_latest_features


def load_model(path: str | Path) -> dict:
    """Load the joblib bundle written by ml.train_model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Train one first:  python -m ml.train_model --input data/synthetic_cpu.csv"
        )
    bundle = joblib.load(path)
    for key in ("model", "feature_names"):
        if key not in bundle:
            raise ValueError(f"Invalid model bundle: '{key}' missing in {path}.")
    return bundle


def predict_next(bundle: dict, history_df) -> dict:
    """Forecast the next interval from the end of a (timestamp, cpu) frame.

    Returns {"predicted_cpu", "forecast_time", "interval_minutes"}.
    Values are clamped to the physically possible 0..100 % range.
    """
    X_row, forecast_time, cols = build_latest_features(history_df)
    if list(cols) != list(bundle["feature_names"]):
        raise ValueError(
            "Feature mismatch between model and data pipeline — retrain the model:\n"
            "  python -m ml.train_model --input data/yours.csv --model-out models/rf_cpu.joblib"
        )
    pred = float(bundle["model"].predict(X_row)[0])
    return {
        "predicted_cpu": max(0.0, min(100.0, pred)),
        "forecast_time": forecast_time,
        "interval_minutes": bundle.get("forecast_interval_minutes"),
    }
