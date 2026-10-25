"""Fixed / reactive / predictive capacity policies.

All three policies share one interface so they can be replayed over the
same chronological samples in evaluation/compare_policies.py:

    decision = policy(observed, predicted, *, current, bounds, **kwargs)
    decision -> {"policy", "observed", "predicted", "current",
                 "recommended", "action", "reason",
                 "bounds_hit", "fallback_triggered"}

  * fixed      — always keeps a constant capacity (baseline / over-provisioner).
  * reactive   — reacts to OBSERVED CPU only (crosses thresholds).
  * predictive — reacts to the FORECAST plus the same safety rules as the
                 allocator (bounds, high-CPU fallback, scale-down hysteresis).
"""
from __future__ import annotations

from typing import Callable

from controller.allocator import Allocator, CapacityBounds, Decision

REACTIVE_ALERT_CPU = 70.0   # scale up when observed CPU >= this
REACTIVE_LOW_CPU = 30.0     # scale down after sustained observed low CPU
REQUIRED_LOW_READINGS = 4   # with a 5-minute metric: ~20 minutes


def fixed_policy(observed, predicted, *, current, bounds: CapacityBounds, fixed_capacity: int = 1, **_) -> dict:
    recommended = bounds.clamp(fixed_capacity)
    return {
        "policy": "fixed",
        "observed": float(observed) if observed is not None else None,
        "predicted": predicted,
        "current": int(current),
        "recommended": recommended,
        "action": "hold",
        "reason": f"Fixed policy: always {fixed_capacity} instance(s), regardless of load.",
        "bounds_hit": recommended != int(fixed_capacity),
        "fallback_triggered": False,
    }


def reactive_policy(
    observed,
    predicted,
    *,
    current,
    bounds: CapacityBounds,
    consecutive_low: int = 0,
    alert_cpu: float = REACTIVE_ALERT_CPU,
    low_cpu: float = REACTIVE_LOW_CPU,
    required_low: int = REQUIRED_LOW_READINGS,
    **_,
) -> dict:
    if observed is None:
        raise ValueError("reactive_policy needs an observed CPU value.")
    observed = float(observed)
    if observed >= alert_cpu:
        recommended = bounds.max_capacity
        return {
            "policy": "reactive",
            "observed": observed,
            "predicted": predicted,
            "current": int(current),
            "recommended": recommended,
            "action": "scale_up",
            "reason": f"Observed CPU {observed:.1f}% >= alert threshold {alert_cpu:.0f}% — reacting now.",
            "bounds_hit": False,
            "fallback_triggered": False,
        }
    if observed <= low_cpu and consecutive_low >= required_low and current > bounds.min_capacity:
        recommended = bounds.min_capacity
        return {
            "policy": "reactive",
            "observed": observed,
            "predicted": predicted,
            "current": int(current),
            "recommended": recommended,
            "action": "scale_down",
            "reason": f"Observed CPU <= {low_cpu:.0f}% for {consecutive_low} consecutive readings — scaling down.",
            "bounds_hit": False,
            "fallback_triggered": False,
        }
    recommended = bounds.clamp(current)
    return {
        "policy": "reactive",
        "observed": observed,
        "predicted": predicted,
        "current": int(current),
        "recommended": recommended,
        "action": "hold",
        "reason": f"Observed CPU {observed:.1f}% within band — holding {recommended} instance(s).",
        "bounds_hit": False,
        "fallback_triggered": False,
    }


def predictive_policy(
    observed,
    predicted,
    *,
    current,
    bounds: CapacityBounds,
    consecutive_low: int = 0,
    last_action_at=None,
    now=None,
    allocator: Allocator | None = None,
    data_source: str = "evaluation",
    **_,
) -> dict:
    alloc = allocator or Allocator(bounds=bounds)
    decision: Decision = alloc.decide(
        observed,
        predicted,
        current,
        consecutive_low=consecutive_low,
        last_action_at=last_action_at,
        now=now,
        data_source=data_source,
    )
    return {
        "policy": "predictive",
        "observed": decision.observed_cpu,
        "predicted": decision.predicted_cpu,
        "current": decision.current_capacity,
        "recommended": decision.recommended_capacity,
        "action": decision.action,
        "reason": decision.reason,
        "bounds_hit": decision.bounds_hit,
        "fallback_triggered": decision.fallback_triggered,
    }


POLICIES: dict[str, Callable[..., dict]] = {
    "fixed": fixed_policy,
    "reactive": reactive_policy,
    "predictive": predictive_policy,
}
