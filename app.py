"""CloudPulse — Streamlit dashboard.

Two explicit data modes (never mixed or mislabelled):
  * Synthetic (offline) — local CSV, labelled SYNTHETic, no AWS calls.
  * AWS CloudWatch (real) — real EC2 CPU metrics via a read-only boto3 query.

Live Auto Scaling is guarded: disabled unless CLOUDPULSE_ENABLE_LIVE_SCALING
is set to "true", requires a separate scaling profile, an explicit confirmation
checkbox, and every attempt (dry-run, blocked, applied, error) is written to
the shared audit log. Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
_r = MODEL_DIR / "rf_real.joblib"
DEFAULT_MODEL = _r if _r.exists() else (MODEL_DIR / "rf_cpu.joblib")
DEFAULT_SYNTH = DATA_DIR / "synthetic_cpu.csv"
BILLING_URL = "https://console.aws.amazon.com/billing/"

try:  # optional: auto-load non-secret config from .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from cloud.aws_metrics import (  # noqa: E402
    ALLOWED_STATISTICS,
    EmptyMetricDataError,
    AWSQueryError,
    export_with_metadata,
    newest_sample_age_minutes,
)
from cloud.aws_scaling import (  # noqa: E402
    ASGNotFoundError,
    AutoScalingController,
    LiveScalingNotEnabledError,
    live_scaling_enabled,
)
from controller.audit import DEFAULT_AUDIT_PATH, read_rows  # noqa: E402
from controller.allocator import Allocator, CapacityBounds  # noqa: E402
from evaluation.compare_policies import compare_on_dataframe  # noqa: E402
from ml.features import load_series  # noqa: E402
from ml.predict import load_model, predict_next  # noqa: E402
from ml.train_model import train_model  # noqa: E402

st.set_page_config(page_title="CloudPulse", page_icon="📡", layout="wide")

ss = st.session_state
ss.setdefault("data_source", "Synthetic (offline)")
ss.setdefault("aws_df", None)
ss.setdefault("aws_error", None)
ss.setdefault("current_capacity", 1)
ss.setdefault("just_logged", None)


# --------------------------------------------------------------------- cached
@st.cache_data(ttl=180, show_spinner="Querying CloudWatch (read-only)…")
def fetch_cpu_cached(instance_id, region, profile, hours, period, statistic) -> pd.DataFrame:
    from cloud.aws_metrics import fetch_history_frame

    return fetch_history_frame(
        instance_id=instance_id, region=region or None,
        profile=profile, hours=hours, period=period, statistic=statistic,
    )


@st.cache_resource(show_spinner=False)
def load_model_cached(path: str):
    return load_model(path)


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("CloudPulse 📡")
    st.caption("Predictive workload monitoring & capacity recommendation.")

    st.subheader("1 · Data source")
    source = st.radio("Mode (always clearly labelled)", ["Synthetic (offline)", "AWS CloudWatch (real)"], key="data_source")

    df: pd.DataFrame | None = None
    if source == "Synthetic (offline)":
        synth_path = st.text_input("Synthetic CSV path", str(DEFAULT_SYNTH))
        if st.button("Load synthetic CSV", use_container_width=True):
            try:
                ss["synth_df"] = load_series(synth_path)
            except Exception as exc:
                ss["synth_df"] = None
                st.error(f"Could not load CSV: {exc}")
        candidate = ss.get("synth_df")
        if candidate is not None:
            df = candidate
            st.success(f"Loaded SYNTHETIC data · {len(df):,} rows.")
    else:
        st.subheader("AWS (read-only)")
        aws_profile = st.text_input("AWS profile", os.getenv("CLOUDPULSE_PROFILE", "cloudpulse"))
        aws_region = st.text_input("Region", os.getenv("CLOUDPULSE_REGION", ""),
                                   placeholder="empty = auto from profile (recommended)")
        aws_instance = st.text_input("EC2 instance ID", os.getenv("CLOUDPULSE_INSTANCE_ID", ""),
                                     placeholder="i-0123456789abcdef0")
        hours = st.slider("History window (hours)", 1, 48, int(os.getenv("CLOUDPULSE_HISTORY_HOURS", "6")))
        statistic = st.selectbox("Metric statistic", list(ALLOWED_STATISTICS), index=0)

        if st.button("Fetch real metrics", type="primary", use_container_width=True):
            if not aws_instance.strip():
                ss["aws_error"] = "Enter an EC2 instance ID first (looks like i-0abc123...)."
                ss["aws_df"] = None
            else:
                try:
                    ss["aws_df"] = fetch_cpu_cached(
                        aws_instance.strip(), (aws_region.strip() or None), aws_profile.strip(),
                        hours, 300, statistic,
                    )
                    ss["aws_error"] = None
                except (EmptyMetricDataError, AWSQueryError, ValueError, OSError) as exc:
                    ss["aws_df"] = None
                    ss["aws_error"] = str(exc)
        if ss.get("aws_error"):
            st.error(ss["aws_error"])
            st.info("You can continue in **Synthetic (offline)** mode — it stays clearly labelled as synthetic.")
        df = ss.get("aws_df")
        if df is not None:
            st.success(f"Loaded AWS data · {len(df):,} real datapoints.")

    st.subheader("2 · Model")
    model_path = st.text_input("Model file", str(DEFAULT_MODEL))
    train_frac = st.slider("Chronological train fraction", 0.5, 0.9, 0.8)
    if st.button("Retrain model on loaded data", use_container_width=True):
        if df is None:
            st.error("Load some data first (synthetic or AWS).")
        else:
            if source == "Synthetic (offline)":
                input_csv = Path(synth_path)
            else:
                input_csv, _meta = export_with_metadata(
                    df, DATA_DIR / "aws_session_cpu.csv",
                    instance_id=aws_instance, region=aws_region, hours=hours,
                    period=300, statistic=statistic, source="aws-cloudwatch-session",
                )
            try:
                bundle = train_model(input_csv, model_path, MODEL_DIR / "training_report.json",
                                     train_frac=train_frac)
                st.cache_resource.clear()
                st.success(f"Trained ✓  MAE {bundle['metrics']['mae']:.2f} "
                           f"(baseline {bundle['metrics']['persistence_mae']:.2f})")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    st.subheader("3 · Controller safety")
    bounds_min = st.number_input("Min capacity", 1, 10, int(os.getenv("CLOUDPULSE_MIN_CAPACITY", "1")))
    bounds_max = st.number_input("Max capacity", 1, 10, int(os.getenv("CLOUDPULSE_MAX_CAPACITY", "2")))
    ss["current_capacity"] = st.number_input("Current capacity (context)", 1, 10, int(ss["current_capacity"]))
    fallback_thr = st.slider("Fallback threshold (observed CPU %)", 50, 100, int(os.getenv("CLOUDPULSE_FALLBACK_CPU_THRESHOLD", "85")))
    scale_up_thr = st.slider("Scale-up threshold (forecast CPU %)", 20, 99, 70)
    scale_down_thr = st.slider("Scale-down threshold (CPU %)", 0, 50, 30)
    con_low = st.slider("Consecutive low readings to scale down", 1, 12, 4)
    cooldown_min = st.slider("Cooldown (minutes)", 0, 30, 5)

    st.subheader("4 · Live scaling (optional)")
    st.caption(f"Env flag CLOUDPULSE_ENABLE_LIVE_SCALING = {os.getenv('CLOUDPULSE_ENABLE_LIVE_SCALING', '<unset>')}")
    asg_name = st.text_input("ASG name", os.getenv("CLOUDPULSE_ASG", ""), disabled=source != "AWS CloudWatch (real)")
    scaling_profile = st.text_input("Scaling profile (separate!)", os.getenv("CLOUDPULSE_SCALING_PROFILE", "cloudpulse-scaler"))

    st.divider()
    st.caption("💰 Dashboard queries are read-only. It cannot create EC2 instances for you; "
               "check [AWS Billing](%s) after enabling anything billable." % BILLING_URL)

# validate bounds pairing
if bounds_max < bounds_min:
    st.error("Max capacity must be >= Min capacity — fix the controller bounds in the sidebar.")
    st.stop()

# ----------------------------------------------------------------- main header
st.title("CloudPulse 📡 — predictive workload monitoring")
mode_chip = "🟢 AWS CloudWatch — REAL data" if source == "AWS CloudWatch (real)" else "🟪 SYNTHETIC data (offline)"
if df is None:
    st.info(f"**Mode:** {mode_chip} — load data in the sidebar first "
            "(synthetic CSV, or 'Fetch real metrics' in AWS mode).")
    st.stop()
st.markdown(f"**Mode:** {mode_chip}")

if source == "AWS CloudWatch (real)":
    age = newest_sample_age_minutes(df)
    if age > 20:
        st.warning(f"Newest real sample is {age:.0f} minutes old — click 'Fetch real metrics' to refresh.")

try:
    bundle = load_model_cached(model_path)
except FileNotFoundError:
    bundle = None
except Exception as exc:
    bundle = None
    st.error(f"Could not load model: {exc}")

# prediction (or persistence fallback, always labelled)
predicted, forecast_time, used_persistence = None, None, False
if bundle is not None:
    try:
        pf = predict_next(bundle, df)
        predicted, forecast_time = pf["predicted_cpu"], pf["forecast_time"]
    except ValueError as exc:
        if "Not enough history" in str(exc):
            st.warning(f"Model ready, but not enough rows to forecast: {exc}")
        else:
            st.warning(f"Forecast unavailable: {exc}")
if predicted is None:
    if len(df) >= 2:
        predicted = float(df["cpu"].iloc[-2])  # explicit persistence fallback
        forecast_time = df["timestamp"].iloc[-1] + (df["timestamp"].iloc[-1] - df["timestamp"].iloc[-2])
        used_persistence = True

observed_now = float(df["cpu"].iloc[-1])
newest_ts = df["timestamp"].max()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Newest sample (UTC)", newest_ts.strftime("%d %b %H:%M"),
          f"{newest_sample_age_minutes(df):.0f} min ago")
c2.metric("Observed CPU (latest)", f"{observed_now:.1f}%")
c3.metric(f"Forecast ({'persistence' if used_persistence else 'RF model'})",
          f"{predicted:.1f}%" if predicted is not None else "—",
          f"for {forecast_time.strftime('%H:%M') if forecast_time is not None else ''}")
c4.metric("Current capacity", f"{int(ss['current_capacity'])} instance(s)")

with st.expander("CPU history (real or synthetic, as labelled)", expanded=True):
    st.line_chart(df.set_index("timestamp")["cpu"])
    if source == "AWS CloudWatch (real)" and bundle is not None:
        st.caption(f"Model: {bundle.get('model_type')} · trained {bundle.get('trained_at_utc', '')[:19]} · "
                   f"MAE {bundle['metrics']['mae']:.3f} / RMSE {bundle['metrics']['rmse']:.3f} vs persistence "
                   f"MAE {bundle['metrics']['persistence_mae']:.3f}")

# ------------------------------------------------------------ guarded decision
st.subheader("Guarded allocation controller")
allocator = Allocator(
    bounds=CapacityBounds(int(bounds_min), int(bounds_max)),
    fallback_cpu=float(fallback_thr),
    scale_up_cpu=float(scale_up_thr),
    scale_down_cpu=float(scale_down_thr),
    consecutive_low_required=int(con_low),
    cooldown_seconds=int(cooldown_min) * 60,
)
last_live_at = None
for row in read_rows(DEFAULT_AUDIT_PATH, limit=100):
    if row.get("mode") == "live_scaling" and row.get("result") == "applied":
        try:
            last_live_at = datetime.fromisoformat(str(row["timestamp"]))
            break
        except ValueError:
            continue

decision = allocator.decide(
    observed_cpu=observed_now,
    predicted_cpu=predicted,
    current_capacity=int(ss["current_capacity"]),
    recent_observations=df["cpu"].tail(30).tolist()[::-1],
    last_action_at=last_live_at,
    data_source="aws" if source.startswith("AWS") else "synthetic",
)

d1, d2, d3 = st.columns([2, 1, 1])
d1.markdown(f"**Recommended desired capacity: {decision.recommended_capacity}** "
            f"(`{decision.action}` · from {int(ss['current_capacity'])})")
d1.caption(decision.reason)
if decision.fallback_triggered:
    d1.error("Safety fallback ACTIVE — observed CPU above threshold; prediction overridden.")
if decision.bounds_hit:
    d1.warning(f"Bounds hit: raw prediction wanted {decision.unbounded_target}; clamped to "
               f"{decision.recommended_capacity} (min {bounds_min} / max {bounds_max}).")
if decision.cooldown_active:
    d1.info("Cooldown active — scale-down postponed.")
d2.caption("Dry Run indicator")
d2.success("DRY RUN (default)")
d3.caption("Live status")
live_on = live_scaling_enabled()
if live_on:
    d3.error("⚠ LIVE SCALING ENABLED")
else:
    d3.info("Live scaling OFF (safe — dry-run mode)")

b1, b2, b3 = st.columns(3)
if b1.button("Record decision (dry-run, no API)", use_container_width=True):
    allocator.log_decision(decision, DEFAULT_AUDIT_PATH)
    ss["just_logged"] = "Dry-run decision recorded in data/audit_log.jsonl — no cloud resource changed."
if b3.button("Adopt recommendation as current capacity", use_container_width=True, type="primary"):
    ss["current_capacity"] = int(decision.recommended_capacity)
    st.rerun()
if ss.get("just_logged"):
    st.info(ss.pop("just_logged"))

# ------------------------------------------------------------- live scaling UI
st.markdown("### Optional live Auto Scaling — explicitly guarded")
if source != "AWS CloudWatch (real)":
    st.caption("Live scaling applies to a real EC2 Auto Scaling Group; switch to AWS mode to use it.")
if not live_on:
    st.warning("Live changes are blocked. To enable: set `CLOUDPULSE_ENABLE_LIVE_SCALING=true` in your "
               "environment, use the separate scaling profile, understand charges, and confirm below.")
confirm = st.checkbox("I verified costs/permissions and confirm a live ASG change", disabled=not live_on)
requested = st.number_input("Requested desired capacity", 1, 10, int(decision.recommended_capacity))
if st.button("Apply live change to ASG", disabled=(not live_on) or (not asg_name.strip()) or (not confirm),
             use_container_width=True):
    try:
        ctl = AutoScalingController(
            asg_name.strip(), region=aws_region, profile=scaling_profile,
            min_capacity=int(bounds_min), max_capacity=int(bounds_max),
            cooldown_seconds=int(cooldown_min) * 60,
        )
        info = ctl.describe_asg()
        st.write("ASG state before change:", info)
        row = ctl.apply_capacity(
            int(requested), reason=decision.reason, observed_cpu=observed_now,
            predicted_cpu=predicted, data_source=source, live=True, confirm_live=True,
        )
        st.success(f"AWS accepted: desired capacity → {row['final_capacity']} "
                   f"(was {row.get('old_desired_capacity')}). Check Activity history in the console.")
    except (ASGNotFoundError, LiveScalingNotEnabledError, RuntimeError, ValueError) as exc:
        st.error(f"Live change refused/failed: {exc}")

if st.button("Describe ASG (read-only check)"):
    try:
        ctl = AutoScalingController(asg_name.strip() or "cloudpulse-demo-asg", region=aws_region,
                                    profile=scaling_profile, min_capacity=int(bounds_min),
                                    max_capacity=int(bounds_max))
        st.write(ctl.describe_asg())
    except Exception as exc:
        st.error(f"{exc}")

# ------------------------------------------------------------ policy comparison
st.subheader("Policy comparison — fixed vs reactive vs predictive (same chronological samples)")
p_run = st.button("Run policy comparison on loaded window", use_container_width=True)
if p_run:
    try:
        decisions, summary, opts = compare_on_dataframe(df, model_bundle=bundle)
        table = pd.DataFrame(summary).T[
            ["avg_recommended_capacity", "instance_hours", "estimated_cost_usd",
             "scale_up_events", "scale_down_events", "bounds_hits", "fallback_events",
             "median_samples_to_react"]]
        st.dataframe(table, use_container_width=True)
        st.caption(f"Forecast MAE {opts['forecast_mae']:.3f} vs persistence {opts['persistence_mae']:.3f} · "
                   f"cost = ESTIMATE at ${opts['rate_usd_per_instance_hour']}/instance-hour · "
                   "verify prices at aws.amazon.com/ec2/pricing/")
        DATA_DIR.mkdir(exist_ok=True)
        (DATA_DIR / "policy_summary.json").write_text(
            json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "options": opts, "summary": summary}, indent=2), encoding="utf-8")
    except Exception as exc:
        st.warning(f"Comparison unavailable: {exc}")

# ------------------------------------------------------------------- audit log
st.subheader("Action history (audit log)")
audit_rows = read_rows(DEFAULT_AUDIT_PATH, limit=100)
if audit_rows:
    st.dataframe(pd.DataFrame(audit_rows)[
        ["timestamp", "mode", "data_source", "observed_cpu", "predicted_cpu",
         "requested_capacity", "final_capacity", "result", "reason"]], use_container_width=True)
else:
    st.caption("No actions recorded yet. Every dry-run decision and every live attempt appears here.")

st.divider()
st.caption("⚠️ Cost disclaimer: AWS mode reads your real account; enabling live scaling changes real "
           "resources. Review the [Billing console](%s) after experiments and clean up test resources. "
           "This is a guarded educational prototype, not production automatic scaling." % BILLING_URL)
