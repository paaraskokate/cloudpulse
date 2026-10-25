"""Bounded capacity decision logic — pure Python, makes NO AWS calls.

Safety rules implemented here (matching the implementation guide):
  * Hard bounds: recommended capacity is always clamped to [min, max].
  * High-CPU fallback: if observed CPU >= fallback threshold, the
    recommendation is the configured maximum even when the model
    predicts low demand (protects against missed spikes).
  * Cooldown: scale-down is postponed while a previous action is recent.
  * Scale-down needs SUSTAINED low readings (consecutive_low_required).
  * Every decision can be written to the shared audit log.
Live cloud changes are handled separately by cloud.aws_scaling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from controller.audit import DEFAULT_AUDIT_PATH, append_row


@dataclass
class CapacityBounds:
    min_capacity: int = 1
    max_capacity: int = 2

    def __post_init__(self):
        self.min_capacity = int(self.min_capacity)
        self.max_capacity = int(self.max_capacity)
        if self.min_capacity < 1:
            raise ValueError("min_capacity must be >= 1.")
        if self.max_capacity < self.min_capacity:
            raise ValueError("max_capacity must be >= min_capacity.")

    def clamp(self, requested: float) -> int:
        return int(max(self.min_capacity, min(self.max_capacity, round(float(requested)))))


@dataclass
class Decision:
    timestamp: str
    observed_cpu: float
    predicted_cpu: float | None
    current_capacity: int
    recommended_capacity: int
    unbounded_target: int
    action: str  # "scale_up" | "hold" | "scale_down"
    reason: str
    fallback_triggered: bool = False
    bounds_hit: bool = False
    cooldown_active: bool = False
    data_source: str = "unknown"
    mode: str = "dry_run"

    def to_audit_row(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "data_source": self.data_source,
            "observed_cpu": round(float(self.observed_cpu), 2),
            "predicted_cpu": round(float(self.predicted_cpu), 2) if self.predicted_cpu is not None else None,
            "current_capacity": self.current_capacity,
            "requested_capacity": self.unbounded_target,
            "final_capacity": self.recommended_capacity,
            "action": self.action,
            "result": "logged",
            "reason": self.reason,
            "fallback_triggered": self.fallback_triggered,
            "bounds_hit": self.bounds_hit,
            "cooldown_active": self.cooldown_active,
        }


class Allocator:
    """Computes the recommended (bounded) desired capacity for one decision step."""

    def __init__(
        self,
        bounds: CapacityBounds | None = None,
        fallback_cpu: float = 85.0,
        scale_up_cpu: float = 70.0,
        scale_down_cpu: float = 30.0,
        consecutive_low_required: int = 4,
        cooldown_seconds: int = 300,
    ):
        self.bounds = bounds or CapacityBounds()
        self.fallback_cpu = float(fallback_cpu)
        self.scale_up_cpu = float(scale_up_cpu)
        self.scale_down_cpu = float(scale_down_cpu)
        self.consecutive_low_required = int(consecutive_low_required)
        self.cooldown_seconds = int(cooldown_seconds)

    def raw_target(self, predicted_cpu: float | None) -> int:
        """Unbounded desired-capacity target from the forecast (can exceed max).

        Each 25 points of predicted CPU above the scale-up threshold adds one
        capacity unit; e.g. predicted 100% with threshold 70% -> 3 units.
        The allocator clamps this later; tests assert the clamp happens.
        """
        if predicted_cpu is None:
            return self.bounds.min_capacity
        if predicted_cpu >= self.scale_up_cpu:
            return self.bounds.min_capacity + max(1, math.ceil((predicted_cpu - self.scale_up_cpu + 1) / 25.0))
        return self.bounds.min_capacity

    def consecutive_low_count(self, observations) -> int:
        """Trailing count of readings <= scale_down_cpu."""
        n = 0
        for v in reversed(list(observations or [])):
            if v is not None and float(v) <= self.scale_down_cpu:
                n += 1
            else:
                break
        return n

    def decide(
        self,
        observed_cpu: float,
        predicted_cpu: float | None,
        current_capacity: int,
        recent_observations=None,
        consecutive_low: int | None = None,
        last_action_at: datetime | None = None,
        now: datetime | None = None,
        data_source: str = "dashboard",
    ) -> Decision:
        now = now or datetime.now(timezone.utc)
        lows = (
            int(consecutive_low)
            if consecutive_low is not None
            else self.consecutive_low_count(recent_observations)
        )
        cooldown_active = bool(
            last_action_at
            and self.cooldown_seconds > 0
            and (now - last_action_at).total_seconds() < self.cooldown_seconds
        )
        unbounded = self.raw_target(predicted_cpu)
        fallback = float(observed_cpu) >= self.fallback_cpu

        if fallback:
            # Safety fallback overrides the prediction: observed CPU is already high.
            target = max(unbounded, self.bounds.max_capacity)
            action = "scale_up"
            reason = (
                f"Safety fallback: observed CPU {observed_cpu:.1f}% >= fallback threshold "
                f"{self.fallback_cpu:.0f}% (predicted was {predicted_cpu:.1f}%). "
                "Recommending maximum capacity."
            )
        elif unbounded > current_capacity:
            target = unbounded
            action = "scale_up"
            reason = (
                f"Forecast CPU {predicted_cpu:.1f}% >= scale-up threshold {self.scale_up_cpu:.0f}%. "
                f"Predictive recommendation."
            )
        elif (
            current_capacity > self.bounds.min_capacity
            and predicted_cpu is not None
            and predicted_cpu <= self.scale_down_cpu
            and observed_cpu <= self.scale_down_cpu
            and lows >= self.consecutive_low_required
        ):
            if cooldown_active:
                target = current_capacity
                action = "hold"
                reason = (
                    f"Sustained low CPU (predicted {predicted_cpu:.1f}%, observed {observed_cpu:.1f}%) "
                    f"for {lows} readings, but cooldown is active — scale-down postponed."
                )
            else:
                target = self.bounds.min_capacity
                action = "scale_down"
                reason = (
                    f"Sustained low CPU for {lows} consecutive readings "
                    f"(predicted {predicted_cpu:.1f}%, observed {observed_cpu:.1f}%). "
                    "Scaling down to minimum."
                )
        else:
            target = current_capacity
            action = "hold"
            reason = (
                f"Within safe band (predicted {predicted_cpu:.1f}%, observed {observed_cpu:.1f}%). "
                "Holding current capacity."
            )

        recommended = self.bounds.clamp(target)
        bounds_hit = recommended != target
        return Decision(
            timestamp=now.isoformat(),
            observed_cpu=float(observed_cpu),
            predicted_cpu=float(predicted_cpu) if predicted_cpu is not None else None,
            current_capacity=int(current_capacity),
            recommended_capacity=recommended,
            unbounded_target=int(target),
            action=action,
            reason=reason,
            fallback_triggered=fallback,
            bounds_hit=bounds_hit,
            cooldown_active=cooldown_active,
            data_source=data_source,
            mode="dry_run",
        )

    def log_decision(self, decision: Decision, path: str | Path | None = None):
        return append_row(decision.to_audit_row(), path or DEFAULT_AUDIT_PATH)
