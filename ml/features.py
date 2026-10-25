"""Feature engineering for CloudPulse CPU forecasting.

Given a chronologically ordered series of (timestamp, cpu) samples, builds
lag features, rolling statistics and calendar features so that a model can
forecast the CPU of the *next* monitoring interval (default 5 minutes).

No future leakage: every feature at row t is computed only from cpu values
strictly before t (lags) or from the window ending at t-1 (rolling stats).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# With 5-minute samples: lag 1 = 5 min ago, lag 6 = 30 min ago, lag 12 = 1 h ago.
LAGS = (1, 2, 3, 6, 12)
ROLL_WINDOWS = (3, 12)  # 15 minutes, 60 minutes


def load_series(path: str | Path) -> pd.DataFrame:
    """Load and validate a (timestamp, cpu) CSV. Timestamps are normalised to UTC."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    for col in ("timestamp", "cpu"):
        if col not in df.columns:
            raise ValueError(f"CSV must contain a '{col}' column. Found columns: {list(df.columns)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["cpu"] = pd.to_numeric(df["cpu"], errors="coerce")
    df = df.dropna(subset=["timestamp", "cpu"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid (timestamp, cpu) rows found in the file.")
    out_of_range = int(((df["cpu"] < 0) | (df["cpu"] > 100)).sum())
    if out_of_range:
        df["cpu"] = df["cpu"].clip(0.0, 100.0)
    return df[["timestamp", "cpu"]]


def infer_interval_minutes(df: pd.DataFrame) -> float | None:
    """Median sampling interval in minutes (e.g. 5.0 for standard EC2 metrics)."""
    diffs = df["timestamp"].diff().dt.total_seconds().dropna()
    if diffs.empty:
        return None
    return float(round(diffs.median() / 60.0, 3))


def add_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = LAGS,
    roll_windows: tuple[int, ...] = ROLL_WINDOWS,
    target: str = "cpu",
) -> pd.DataFrame:
    """Return a frame with lag / rolling / calendar feature columns added."""
    out = df.sort_values("timestamp").reset_index(drop=True).copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target].shift(lag)
    shifted = out[target].shift(1)
    for w in roll_windows:
        roll = shifted.rolling(window=w, min_periods=w)
        out[f"roll_mean_{w}"] = roll.mean()
        out[f"roll_std_{w}"] = roll.std()
        out[f"roll_min_{w}"] = roll.min()
        out[f"roll_max_{w}"] = roll.max()
    # Short-term trend: last observed change (t-1 minus t-2).
    out["delta_1"] = out[target].shift(1) - out[target].shift(2)
    # Calendar features sinusoidal patterns follow local time-of-day.
    out["hour"] = out["timestamp"].dt.hour
    out["minute"] = out["timestamp"].dt.minute
    out["dayofweek"] = out["timestamp"].dt.dayofweek
    return out


def feature_columns(df_feat: pd.DataFrame) -> list[str]:
    """All model input columns (everything except timestamp and the target)."""
    return [c for c in df_feat.columns if c not in ("timestamp", "cpu")]


def make_supervised(df_feat: pd.DataFrame):
    """Drop warm-up rows and return (X, y, timestamps, feature_names)."""
    cols = feature_columns(df_feat)
    valid = df_feat.dropna(subset=cols)
    X = valid[cols].to_numpy(dtype=float)
    y = valid["cpu"].to_numpy(dtype=float)
    times = pd.DatetimeIndex(valid["timestamp"])
    return X, y, times, cols


def min_history_rows(lags: tuple[int, ...] = LAGS, roll_windows: tuple[int, ...] = ROLL_WINDOWS) -> int:
    """Minimum number of history samples needed before features become usable."""
    return int(max(max(lags) + 1, max(roll_windows) + 2))  # 12+1, 12+2 -> 14


def build_latest_features(
    history_df: pd.DataFrame,
    lags: tuple[int, ...] = LAGS,
    roll_windows: tuple[int, ...] = ROLL_WINDOWS,
):
    """Feature vector for forecasting the NEXT interval from the newest sample.

    Returns (X_row [1 x n_features], forecast_timestamp, feature_names).
    """
    need = min_history_rows(lags, roll_windows)
    if len(history_df) < need:
        raise ValueError(
            f"Not enough history to build forecasting features: {len(history_df)} rows available, "
            f"at least {need} required. Fetch a longer window (e.g. 6 hours @ 5-minute period = 72 rows)."
        )
    feat = add_features(history_df, lags=lags, roll_windows=roll_windows)
    cols = feature_columns(feat)
    valid = feat.dropna(subset=cols)
    if valid.empty:
        raise ValueError("Every feature row is incomplete — history window is too short or irregular.")
    last = valid.iloc[[-1]]
    X = last[cols].to_numpy(dtype=float)
    interval = infer_interval_minutes(history_df) or 5.0
    forecast_time = pd.Timestamp(last["timestamp"].iloc[0]) + pd.Timedelta(minutes=interval)
    return X, forecast_time, cols
