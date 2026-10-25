"""Compare fixed / reactive / predictive policies on the SAME chronological samples.

Method (matches the implementation guide):
  * predictions are one-step walk-forward on chronologically ordered samples
    (no random train/test splitting anywhere in this pipeline);
  * all three policies are stepped through the identical rows, so their
    timestamps align exactly and the comparison is fair;
  * metrics reported: forecast quality (MAE/RMSE vs persistence baseline),
    average recommended capacity, instance-hours under each policy,
    responsiveness (samples between an alert crossing and a scale-up
    recommendation), safety counters, and ESTIMATED cost from
    evaluation/cost_assumptions.json (clearly labelled estimates).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from controller.allocator import Allocator, CapacityBounds
from controller.policy import POLICIES, REACTIVE_ALERT_CPU, REACTIVE_LOW_CPU
from ml.features import add_features, infer_interval_minutes, load_series, make_supervised
from ml.train_model import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
COST_FILE = Path(__file__).resolve().parent / "cost_assumptions.json"


def default_hourly_rate() -> float:
    try:
        cfg = json.loads(COST_FILE.read_text(encoding="utf-8"))
        return float(cfg.get("default_hourly_rate_usd", 0.0058))
    except Exception:
        return 0.0058


def build_rows(df: pd.DataFrame, model_bundle: dict | None = None) -> list[dict]:
    """Walk-forward one-step predictions over the supervised rows.

    With model_bundle=None the 'predictive' policy forecasts with a
    persistence baseline (predicted = previous observed value); if a trained
    bundle is supplied it is used and validated against the feature names.
    """
    feat = add_features(df)
    X, y, times, names = make_supervised(feat)
    if len(y) < 15:
        raise ValueError(
            f"Only {len(y)} usable samples — load more history (>= ~15 supervised rows) "
            "before comparing policies."
        )
    preds: np.ndarray
    model_used = False
    if model_bundle is not None:
        if list(names) != list(model_bundle.get("feature_names", [])):
            raise ValueError(
                "Model features do not match this pipeline — retrain the model:\n"
                "  python -m ml.train_model --input <csv>"
            )
        preds = np.asarray(model_bundle["model"].predict(X), dtype=float)
        model_used = True
    else:
        preds = y  # placeholder; persistence below uses previous observed value

    rows: list[dict] = []
    for i in range(len(y)):
        observed = float(y[i])
        persistence = float(y[i - 1]) if i > 0 else observed
        rows.append({
            "timestamp": times[i],
            "observed": observed,
            "predicted": float(preds[i]) if model_used else persistence,
            "persistence_predicted": persistence,
        })
    return rows


def simulate_policy(policy_name: str, rows: list[dict], bounds: CapacityBounds, **policy_kwargs) -> list[dict]:
    """Step one policy through every row, updating capacity as it goes.

    The per-policy `last_action_at`/`now` pair lets the predictive policy's
    cooldown behave exactly like the guarded live controller (5 min default).
    """
    step = POLICIES[policy_name]
    current = bounds.min_capacity
    consecutive_low = 0
    recent: list[float] = []
    last_action_at = None
    out: list[dict] = []
    for r in rows:
        step_now = r["timestamp"].to_pydatetime() if hasattr(r["timestamp"], "to_pydatetime") else r["timestamp"]
        d = step(
            r["observed"],
            r["predicted"],
            current=current,
            bounds=bounds,
            consecutive_low=consecutive_low,
            recent_observations=list(recent),
            last_action_at=last_action_at,
            now=step_now,
            **policy_kwargs,
        )
        d["timestamp"] = r["timestamp"]
        d["observed"] = r["observed"]
        d["predicted"] = r["predicted"]
        out.append(d)
        current = int(d["recommended"])
        recent.append(r["observed"])
        consecutive_low = consecutive_low + 1 if r["observed"] <= REACTIVE_LOW_CPU else 0
        if d["action"] != "hold":
            last_action_at = step_now
    return out


def compare_on_dataframe(
    df: pd.DataFrame,
    model_bundle: dict | None = None,
    bounds: CapacityBounds | None = None,
    interval_minutes: float | None = None,
    rate_per_hour: float | None = None,
    allocator: Allocator | None = None,
):
    """Run all three policies. Returns (decisions_df, summary, options)."""
    bounds = bounds or CapacityBounds()
    interval = float(interval_minutes) if interval_minutes else (infer_interval_minutes(df) or 5.0)
    rate = float(rate_per_hour) if rate_per_hour else default_hourly_rate()
    rows = build_rows(df, model_bundle)

    frames: list[dict] = []
    for name in ("fixed", "reactive", "predictive"):
        frames.extend(simulate_policy(name, rows, bounds, allocator=allocator))
    decisions = pd.DataFrame(frames)

    # Forecast quality on the SAME samples shared by all policies.
    y_true = np.array([r["observed"] for r in rows], dtype=float)
    y_model = np.array([r["predicted"] for r in rows], dtype=float)
    y_pers = np.array([r["persistence_predicted"] for r in rows], dtype=float)
    forecast_mae, forecast_rmse = compute_metrics(y_true, y_model)
    persistence_mae, persistence_rmse = compute_metrics(y_true, y_pers)

    summary: dict[str, dict] = {}
    for policy, g in decisions.groupby("policy", sort=False):
        rec = g["recommended"].astype(float)
        instance_hours = float((rec * interval / 60.0).sum())
        avg_cap = float(rec.mean())
        summary[policy] = {
            "samples": int(len(g)),
            "avg_recommended_capacity": round(avg_cap, 3),
            "instance_hours": round(instance_hours, 3),
            "estimated_cost_usd": round(instance_hours * rate, 4),
            "scale_up_events": int((g["action"] == "scale_up").sum()),
            "scale_down_events": int((g["action"] == "scale_down").sum()),
            "bounds_hits": int(g["bounds_hit"].astype(bool).sum()),
            "fallback_events": int(g["fallback_triggered"].astype(bool).sum()) if policy == "predictive" else 0,
            "median_samples_to_react": _median_reaction_samples(g, alert=REACTIVE_ALERT_CPU),
        }

    options = {
        "interval_minutes": interval,
        "rate_usd_per_instance_hour": rate,
        "cost_is_estimate": True,
        "model_used": model_bundle is not None,
        "forecast_mae": round(forecast_mae, 4),
        "forecast_rmse": round(forecast_rmse, 4),
        "persistence_mae": round(persistence_mae, 4),
        "persistence_rmse": round(persistence_rmse, 4),
        "bounds": {"min": bounds.min_capacity, "max": bounds.max_capacity},
        "alert_cpu": REACTIVE_ALERT_CPU,
    }
    return decisions, summary, options


def _median_reaction_samples(policy_df: pd.DataFrame, alert: float) -> float | None:
    """Median samples between an alert crossing (prev<alert<=cur) and a scale_up."""
    lags: list[int] = []
    actions = policy_df["action"].tolist()
    observed = policy_df["observed"].tolist()
    for i in range(1, len(observed)):
        if observed[i - 1] < alert <= observed[i]:
            for j in range(i, min(i + 5, len(observed))):
                if actions[j] == "scale_up":
                    lags.append(j - i)
                    break
    return float(np.median(lags)) if lags else None


def _load_model_optional(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    from ml.predict import load_model

    return load_model(p)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare fixed/reactive/predictive policies.")
    p.add_argument("--input", default=str(ROOT / "data" / "synthetic_cpu.csv"))
    p.add_argument("--model", default=str(ROOT / "models" / "rf_cpu.joblib"),
                   help="trained model path; omitted file => persistence forecast")
    p.add_argument("--bounds-min", type=int, default=1)
    p.add_argument("--bounds-max", type=int, default=2)
    p.add_argument("--rate", type=float, default=None, help="USD per instance-hour (estimates)")
    p.add_argument("--out", default=str(ROOT / "evaluation" / "policy_summary.json"))
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    df = load_series(args.input)
    model = _load_model_optional(args.model)
    decisions, summary, options = compare_on_dataframe(
        df,
        model_bundle=model,
        bounds=CapacityBounds(args.bounds_min, args.bounds_max),
        rate_per_hour=args.rate,
    )
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "options": options, "summary": summary}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    table = pd.DataFrame(summary).T[
        ["avg_recommended_capacity", "instance_hours", "estimated_cost_usd",
         "scale_up_events", "scale_down_events", "bounds_hits", "fallback_events",
         "median_samples_to_react"]
    ]
    print(f"Policies compared on {options['interval_minutes']:.0f}-minute samples over "
          f"{len(decisions) // 3} rows per policy.")
    print(f"Bounds {options['bounds']}, alert CPU {options['alert_cpu']:.0f}%  (cost = ESTIMATE).")
    print(f"Forecast MAE {options['forecast_mae']:.3f} vs persistence {options['persistence_mae']:.3f}")
    print(table.to_string())
    print(f"\nFull JSON report: {out}")
