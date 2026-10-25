"""Deterministic synthetic CPU series for offline development and tests.

USE ONLY for clearly-labelled synthetic experiments — never present this
data as real AWS measurements. The pattern: office-hours daily cycle
(peak ~ early afternoon) + small evening bump + noise + a few overload
spikes per day so thresholds and the fallback path get exercised.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def generate_series(days: float = 3, interval_minutes: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(max(2, round(days * 24 * 60 / interval_minutes)))
    start = pd.Timestamp("2026-01-05 00:00:00")  # naive UTC; loader localises
    stamps = pd.date_range(start, periods=n, freq=f"{int(interval_minutes)}min")
    hour = stamps.hour.to_numpy() + stamps.minute.to_numpy() / 60.0

    daily = 32.0 * np.exp(-((hour - 14.0) ** 2) / (2 * 4.5**2))   # midday peak
    evening = 10.0 * np.exp(-((hour - 20.5) ** 2) / (2 * 1.8**2))  # evening bump
    cpu = 22.0 + daily + evening + rng.normal(0.0, 2.5, n)

    # Overload bursts: a few per day, amplitude 30..60, lasting 2..5 samples.
    for _ in range(max(1, int(round(days * 4)))):
        i0 = int(rng.integers(int(0.1 * n), int(0.95 * n)))
        amp = float(rng.uniform(30.0, 60.0))
        dur = int(rng.integers(2, 6))
        for k in range(dur):
            idx = i0 + k
            if idx < n:
                cpu[idx] = min(100.0, cpu[idx] + amp * np.exp(-k / max(1, dur - 1)))

    cpu = np.clip(cpu, 0.0, 100.0)
    return pd.DataFrame({"timestamp": stamps, "cpu": np.round(cpu, 2)})


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate the labelled synthetic CPU CSV.")
    p.add_argument("--days", type=float, default=3.0)
    p.add_argument("--interval-minutes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(ROOT / "data" / "synthetic_cpu.csv"))
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_series(args.days, args.interval_minutes, args.seed)
    df.to_csv(out, index=False)
    print(f"Synthetic series written: {out}")
    print(f"  samples: {len(df)}, span: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    print("  Label this data SYNTHETIC in the dashboard — it is not real AWS telemetry.")
