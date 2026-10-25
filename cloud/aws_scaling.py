"""Guarded EC2 Auto Scaling integration — DISABLED and dry-run by default.

A live change requires ALL of:
  * environment variable CLOUDPULSE_ENABLE_LIVE_SCALING=true
  * a separate write-capable AWS profile (CLOUDPULSE_SCALING_PROFILE,
    default "cloudpulse-scaler") — never the read-only metrics profile
  * explicit confirmation (confirm_live=True) from the user
  * the target ASG must exist, belong to the expected region/account

Every attempt (dry-run, blocked, applied, error) is appended to the shared
audit log. No credentials or secrets are written to the log.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from botocore.exceptions import ClientError

from cloud.aws_metrics import build_session
from controller.audit import DEFAULT_AUDIT_PATH, append_row, read_rows

ENV_ENABLE = "CLOUDPULSE_ENABLE_LIVE_SCALING"


class ASGNotFoundError(RuntimeError):
    """Target ASG missing, wrong region, or account mismatch. NO change is made."""


class LiveScalingNotEnabledError(RuntimeError):
    """Live change refused because the mode/confirmation requirements are unmet."""


def live_scaling_enabled() -> bool:
    return os.getenv(ENV_ENABLE, "false").strip().lower() == "true"


class AutoScalingController:
    """Validates the ASG target, clamps capacity, enforces cooldown, writes audit rows."""

    def __init__(
        self,
        asg_name: str,
        region: str | None = None,
        profile: str | None = None,
        min_capacity: int = 1,
        max_capacity: int = 2,
        cooldown_seconds: int = 300,
        expected_account_id: str | None = None,
        audit_path: str | Path | None = None,
    ):
        self.asg_name = str(asg_name).strip()
        self.profile = profile or os.getenv("CLOUDPULSE_SCALING_PROFILE", "cloudpulse-scaler")
        self.region = (region or os.getenv("CLOUDPULSE_REGION") or "").strip() or None
        self.expected_account_id = (
            expected_account_id or os.getenv("CLOUDPULSE_EXPECTED_ACCOUNT_ID") or ""
        ).strip() or None
        self.min_capacity = int(min_capacity)
        self.max_capacity = int(max_capacity)
        self.cooldown_seconds = int(cooldown_seconds)
        self.audit_path = Path(audit_path) if audit_path else DEFAULT_AUDIT_PATH
        if not self.asg_name:
            raise ValueError("asg_name is required.")
        if self.min_capacity < 1 or self.max_capacity < self.min_capacity:
            raise ValueError("Bounds must satisfy 1 <= min_capacity <= max_capacity.")

    # ------------------------------------------------------------------ AWS
    def _session(self):
        return build_session(profile_name=self.profile, region_name=self.region)

    def describe_asg(self) -> dict:
        """Read-only check that the ASG exists in the expected region/account."""
        session = self._session()
        client = session.client("autoscaling")
        try:
            resp = client.describe_auto_scaling_groups(AutoScalingGroupNames=[self.asg_name])
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "Unknown")
            raise ASGNotFoundError(
                f"AWS refused to describe ASG '{self.asg_name}' ({code}). "
                "Check that the scaling profile/token is valid."
            ) from exc
        groups = resp.get("AutoScalingGroups", [])
        if not groups:
            raise ASGNotFoundError(
                f"Auto Scaling Group '{self.asg_name}' not found. "
                "Check the exact ASG name and the region before any change is attempted."
            )
        g = groups[0]
        azs = list(g.get("AvailabilityZones", []))
        region = session.region_name
        if azs and region and not any(az.startswith(region) for az in azs):
            raise ASGNotFoundError(
                f"ASG '{self.asg_name}' exists but its availability zones {azs} are outside region '{region}'."
            )
        caller_account = None
        if self.expected_account_id:
            caller_account = session.client("sts").get_caller_identity()["Account"]
            if caller_account != self.expected_account_id:
                raise ASGNotFoundError(
                    f"Account mismatch: caller account {caller_account} != expected "
                    f"{self.expected_account_id}. NO change was made."
                )
        return {
            "name": g.get("AutoScalingGroupName"),
            "region": region,
            "account": caller_account or "current",
            "min_size": g.get("MinSize"),
            "max_size": g.get("MaxSize"),
            "desired_capacity": g.get("DesiredCapacity"),
            "instances_healthy": len(
                [i for i in g.get("Instances", []) if i.get("HealthStatus") == "Healthy"]
            ),
            "instances_total": len(g.get("Instances", [])),
            "azs": azs,
        }

    # ------------------------------------------------------------- clamping
    def clamp(self, requested: float, asg_min: int | None = None, asg_max: int | None = None) -> int:
        """Clamp into [max(CP_min, asg_min), min(CP_max, asg_max)]."""
        lo = max(self.min_capacity, int(asg_min) if asg_min is not None else self.min_capacity)
        hi = min(self.max_capacity, int(asg_max) if asg_max is not None else self.max_capacity)
        if lo > hi:
            raise ValueError(
                f"Conflicting bounds: CloudPulse allows [{self.min_capacity}, {self.max_capacity}] "
                f"but the ASG itself allows [{asg_min}, {asg_max}]."
            )
        return int(max(lo, min(hi, round(float(requested)))))

    # ------------------------------------------------------- capacity change
    def _last_live_applied_at(self) -> datetime | None:
        for row in read_rows(self.audit_path, limit=100):
            if row.get("mode") == "live_scaling" and row.get("result") == "applied":
                ts = str(row.get("timestamp", ""))
                try:
                    parsed = datetime.fromisoformat(ts)
                except ValueError:
                    continue
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
        return None

    def apply_capacity(
        self,
        requested: int,
        *,
        reason: str,
        observed_cpu: float | None = None,
        predicted_cpu: float | None = None,
        data_source: str = "dashboard",
        live: bool = False,
        confirm_live: bool = False,
    ) -> dict:
        """Request one bounded capacity change. Dry-run by default.

        Order of operations guarantees the safety contract:
          allow-list -> validate ASG target -> clamp -> cooldown -> audit -> API.
        """
        requested = int(requested)
        now = datetime.now(timezone.utc)
        entry = {
            "timestamp": now.isoformat(),
            "mode": "live_scaling" if live else "dry_run",
            "asg": self.asg_name,
            "data_source": data_source,
            "observed_cpu": round(float(observed_cpu), 2) if observed_cpu is not None else None,
            "predicted_cpu": round(float(predicted_cpu), 2) if predicted_cpu is not None else None,
            "requested_capacity": requested,
            "reason": reason,
        }

        if live and not live_scaling_enabled():
            entry["result"] = "blocked_live_scaling_disabled"
            entry["detail"] = (
                f"Live scaling blocked: environment variable {ENV_ENABLE} is not 'true'. "
                "No AWS API call was made. Dry-run mode remains available."
            )
            append_row(entry, self.audit_path)
            raise LiveScalingNotEnabledError(entry["detail"])
        if live and not confirm_live:
            entry["result"] = "blocked_no_confirmation"
            entry["detail"] = "Live change refused: explicit confirmation (confirm_live=True) is required."
            append_row(entry, self.audit_path)
            raise LiveScalingNotEnabledError(entry["detail"])

        if live:
            info = self.describe_asg()  # raises ASGNotFoundError -> nothing changed
            entry["asg_state"] = info
            old_desired = info["desired_capacity"]
            clamped = self.clamp(requested, asg_min=info["min_size"], asg_max=info["max_size"])
        else:
            info = None
            old_desired = None
            clamped = self.clamp(requested)

        entry["old_desired_capacity"] = old_desired
        entry["final_capacity"] = clamped

        if not live:
            entry["result"] = "dry_run_recorded"
            entry["detail"] = "No AWS API call was made (dry run); no cloud resource changed."
        elif clamped == old_desired:
            entry["result"] = "no_change"
            entry["detail"] = f"Desired capacity is already {clamped}; nothing to do."
        else:
            last_applied = self._last_live_applied_at()
            if last_applied and (now - last_applied).total_seconds() < self.cooldown_seconds:
                entry["result"] = "cooldown_skipped"
                entry["detail"] = (
                    f"Cooldown of {self.cooldown_seconds}s is active since {last_applied.isoformat()}; "
                    "no change applied."
                )
            else:
                client = self._session().client("autoscaling")
                try:
                    client.set_desired_capacity(
                        AutoScalingGroupName=self.asg_name,
                        DesiredCapacity=clamped,
                        HonorCooldown=False,  # CloudPulse enforces its own cooldown
                    )
                    entry["result"] = "applied"
                    entry["detail"] = (
                        f"SetDesiredCapacity({self.asg_name}, {clamped}) accepted by AWS."
                    )
                except ClientError as exc:
                    code = exc.response.get("Error", {}).get("Code", "Unknown")
                    entry["result"] = "error"
                    entry["detail"] = (
                        f"SetDesiredCapacity failed ({code}). The scaling profile needs "
                        "autoscaling:DescribeAutoScalingGroups and autoscaling:SetDesiredCapacity."
                    )
                    append_row(entry, self.audit_path)
                    raise RuntimeError(entry["detail"]) from exc

        append_row(entry, self.audit_path)
        return entry
