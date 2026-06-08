"""
SafetyFilter — validates and clamps every action before it reaches any backend.

SafetyFilter is always active, in both sim and real. This is non-negotiable:
  - In sim it prevents learning policies from exploiting unphysical actions.
  - On real hardware it prevents joint limit violations and motor damage.

Access via description.get_safety_filter() — one instance per robot, cached.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action

if TYPE_CHECKING:
    from robodeploy.core.types import Observation
    from robodeploy.description.base import RobotDescription


ViolationKind = Literal["joint_position", "joint_velocity", "joint_acceleration", "ee_workspace", "force", "torque"]


@dataclass
class SafetyLimits:
    joint_position_min: np.ndarray | None = None
    joint_position_max: np.ndarray | None = None
    joint_velocity_max: np.ndarray | None = None
    joint_acceleration_max: np.ndarray | None = None
    workspace_box: tuple[np.ndarray, np.ndarray] | None = None
    ee_velocity_max: float | None = None
    force_max: float | None = None
    torque_max: float | None = None
    control_hz: float = 100.0


@dataclass
class FilterViolationRecord:
    kind: ViolationKind
    value: float
    limit: float
    timestamp: float = field(default_factory=time.time)
    joint_idx: int | None = None


def limits_from_description(description: RobotDescription) -> SafetyLimits:
    return SafetyLimits(
        joint_position_min=description.joint_position_limits[:, 0],
        joint_position_max=description.joint_position_limits[:, 1],
        joint_velocity_max=description.joint_velocity_limits,
        joint_acceleration_max=None,
        workspace_box=None,
        force_max=None,
        torque_max=float(np.max(description.joint_torque_limits)),
        control_hz=100.0,
    )


class SafetyFilter:
    """Clamps actions to a safe envelope. Raises SafetyError on hard violations."""

    def __init__(
        self,
        description: RobotDescription | None = None,
        *,
        limits: SafetyLimits | None = None,
        on_violation: Literal["clamp", "halt", "raise"] = "clamp",
        verbose: bool = False,
    ) -> None:
        if description is None and limits is None:
            raise ValueError("SafetyFilter requires description or limits.")
        if limits is None:
            assert description is not None
            limits = limits_from_description(description)
        self._limits = limits
        self._mode = on_violation
        self._verbose = verbose
        self._dof = int(limits.joint_position_min.shape[0]) if limits.joint_position_min is not None else 0
        if description is not None and self._dof == 0:
            self._dof = description.dof

        self._pos_min = np.asarray(limits.joint_position_min, dtype=np.float64)
        self._pos_max = np.asarray(limits.joint_position_max, dtype=np.float64)
        self._vel_max = np.asarray(limits.joint_velocity_max, dtype=np.float64)
        self._acc_max = (
            np.asarray(limits.joint_acceleration_max, dtype=np.float64)
            if limits.joint_acceleration_max is not None
            else None
        )
        self._workspace = limits.workspace_box
        self._ee_vel_max = limits.ee_velocity_max
        self._force_max = limits.force_max
        self._torque_max = limits.torque_max
        self._dt = 1.0 / max(float(limits.control_hz), 1.0)
        self._home_qpos = (
            np.asarray(description.home_qpos, dtype=np.float64)
            if description is not None
            else None
        )

        self._estop = False
        self._prev_pos: np.ndarray | None = None
        self._prev_vel: np.ndarray | None = None
        self._violations: list[FilterViolationRecord] = []

    def filter(
        self,
        action: Action,
        action_space: ActionSpace,
        *,
        obs: Observation | None = None,
        ignore_slew: bool = False,
    ) -> Action:
        self._violations.clear()
        if self._estop:
            return self._freeze_action()

        if obs is not None:
            self._check_force_torque(obs)

        if action_space == ActionSpace.JOINT_POS:
            return self._filter_joint_pos(action, ignore_slew=ignore_slew)
        if action_space == ActionSpace.JOINT_VEL:
            return self._filter_joint_vel(action)
        if action_space == ActionSpace.JOINT_TORQUE:
            return self._filter_joint_torque(action)
        if action_space in (ActionSpace.CARTESIAN_POSE, ActionSpace.DELTA_EE):
            return self._filter_cartesian(action, action_space)

        return action

    def violations(self) -> list[FilterViolationRecord]:
        return list(self._violations)

    def trigger_estop(self) -> None:
        self._estop = True

    def clear_estop(self) -> None:
        self._estop = False
        self._prev_pos = None
        self._prev_vel = None

    @property
    def estop_active(self) -> bool:
        return self._estop

    def _record(self, kind: ViolationKind, value: float, limit: float, *, joint_idx: int | None = None) -> None:
        rec = FilterViolationRecord(kind=kind, value=value, limit=limit, joint_idx=joint_idx)
        self._violations.append(rec)
        if self._verbose:
            print(f"[SafetyFilter] {kind}: {value:.4f} (limit {limit:.4f})")

    def _maybe_raise_workspace(self) -> None:
        if self._mode != "raise":
            return
        workspace_hits = [v for v in self._violations if v.kind == "ee_workspace"]
        if workspace_hits:
            from robodeploy.safety.violation import Hazard, SafetyError, Severity, ViolationRecord

            hit = workspace_hits[-1]
            raise SafetyError(
                ViolationRecord(
                    hazard=Hazard.EE_WORKSPACE_LIMIT,
                    severity=Severity.CRITICAL,
                    message=f"EE workspace violation (value={hit.value:.4f}, limit={hit.limit:.4f})",
                    value=hit.value,
                    limit=hit.limit,
                )
            )

    def _check_force_torque(self, obs: Observation) -> None:
        if self._force_max is not None and obs.ft_force is not None:
            force = float(np.linalg.norm(np.asarray(obs.ft_force, dtype=np.float64)))
            if force > self._force_max:
                self._record("force", force, self._force_max)
                if self._mode == "halt":
                    self.trigger_estop()
                elif self._mode == "raise":
                    from robodeploy.safety.violation import Hazard, SafetyError, Severity, ViolationRecord

                    raise SafetyError(
                        ViolationRecord(
                            hazard=Hazard.FORCE_LIMIT,
                            severity=Severity.CRITICAL,
                            message=f"|F|={force:.1f}N exceeds {self._force_max:.1f}N",
                            value=force,
                            limit=self._force_max,
                        )
                    )
        if self._torque_max is not None and obs.joint_torques is not None:
            torques = np.asarray(obs.joint_torques, dtype=np.float64)
            peak = float(np.max(np.abs(torques)))
            if peak > self._torque_max:
                self._record("torque", peak, self._torque_max)

    def _filter_joint_pos(self, action: Action, *, ignore_slew: bool = False) -> Action:
        if action.joint_positions is None:
            return action
        raw = np.asarray(action.joint_positions, dtype=np.float64)
        self._validate_shape(raw, "joint_positions")

        if self._prev_pos is not None and not ignore_slew:
            max_step = self._vel_max * self._dt
            slew = np.clip(raw - self._prev_pos, -max_step, max_step)
            raw = self._prev_pos + slew
            if np.any(np.abs(slew - (np.asarray(action.joint_positions, dtype=np.float64) - self._prev_pos)) > 1e-9):
                for idx in range(raw.shape[0]):
                    self._record("joint_velocity", float(abs(slew[idx])), float(max_step[idx]), joint_idx=idx)

        clamped = np.clip(raw, self._pos_min, self._pos_max)
        for idx in range(raw.shape[0]):
            if raw[idx] != clamped[idx]:
                self._record("joint_position", float(raw[idx]), float(clamped[idx]), joint_idx=idx)

        if self._prev_vel is not None and self._acc_max is not None and not ignore_slew:
            acc = (clamped - self._prev_pos) / self._dt - self._prev_vel
            acc_clamped = np.clip(acc, -self._acc_max, self._acc_max)
            if np.any(acc != acc_clamped):
                for idx in range(acc.shape[0]):
                    self._record("joint_acceleration", float(acc[idx]), float(self._acc_max[idx]), joint_idx=idx)
            clamped = self._prev_pos + (self._prev_vel + acc_clamped) * self._dt
            clamped = np.clip(clamped, self._pos_min, self._pos_max)

        self._prev_vel = (clamped - self._prev_pos) / self._dt if self._prev_pos is not None else np.zeros_like(clamped)
        self._prev_pos = clamped.copy()
        return Action(joint_positions=clamped.astype(np.float32), gripper=action.gripper, timestamp=action.timestamp)

    def _filter_joint_vel(self, action: Action) -> Action:
        if action.joint_velocities is None:
            return action
        raw = np.asarray(action.joint_velocities, dtype=np.float64)
        self._validate_shape(raw, "joint_velocities")
        clamped = np.clip(raw, -self._vel_max, self._vel_max)
        for idx in range(raw.shape[0]):
            if raw[idx] != clamped[idx]:
                self._record("joint_velocity", float(raw[idx]), float(clamped[idx]), joint_idx=idx)
        self._prev_pos = None
        return Action(joint_velocities=clamped.astype(np.float32), gripper=action.gripper, timestamp=action.timestamp)

    def _filter_joint_torque(self, action: Action) -> Action:
        if action.joint_torques is None:
            return action
        raw = np.asarray(action.joint_torques, dtype=np.float64)
        self._validate_shape(raw, "joint_torques")
        limit = self._torque_max if self._torque_max is not None else np.max(self._vel_max)
        clamped = np.clip(raw, -limit, limit)
        return Action(joint_torques=clamped.astype(np.float32), gripper=action.gripper, timestamp=action.timestamp)

    def _filter_cartesian(self, action: Action, action_space: ActionSpace) -> Action:
        self._validate_cartesian(action)
        if action.ee_position is None or self._workspace is None:
            return action
        low, high = self._workspace
        raw = np.asarray(action.ee_position, dtype=np.float64).reshape(3)
        clamped = np.clip(raw, low, high)
        if np.any(raw != clamped):
            self._record("ee_workspace", float(np.linalg.norm(raw - clamped)), float(np.linalg.norm(high - low)))
            self._maybe_raise_workspace()
        ee_vel = action.ee_velocity
        if ee_vel is not None and self._ee_vel_max is not None:
            speed = float(np.linalg.norm(np.asarray(ee_vel, dtype=np.float64)))
            if speed > self._ee_vel_max:
                scale = self._ee_vel_max / max(speed, 1e-9)
                ee_vel = (np.asarray(ee_vel, dtype=np.float64) * scale).astype(np.float32)
        return Action(
            joint_positions=action.joint_positions,
            joint_velocities=action.joint_velocities,
            joint_torques=action.joint_torques,
            ee_position=clamped.astype(np.float32),
            ee_orientation=action.ee_orientation,
            ee_velocity=ee_vel,
            gripper=action.gripper,
            timestamp=action.timestamp,
            action_space=action_space,
            is_delta_ee=action.is_delta_ee,
        )

    def _freeze_action(self) -> Action:
        if self._prev_pos is not None:
            pos = self._prev_pos
        elif self._home_qpos is not None:
            pos = self._home_qpos
        else:
            pos = np.zeros(self._dof, dtype=np.float64)
        return Action(joint_positions=pos.astype(np.float32), gripper=0.0)

    def _validate_shape(self, arr: np.ndarray, field: str) -> None:
        if arr.shape != (self._dof,):
            raise ValueError(f"Action.{field} has shape {arr.shape}, expected ({self._dof},).")

    def _validate_cartesian(self, action: Action) -> None:
        if action.ee_position is not None:
            pos = np.asarray(action.ee_position, dtype=np.float64)
            if pos.shape != (3,):
                raise ValueError(f"Action.ee_position has shape {pos.shape}, expected (3,).")
        if action.ee_orientation is not None:
            quat = np.asarray(action.ee_orientation, dtype=np.float64)
            if quat.shape != (4,):
                raise ValueError(f"Action.ee_orientation has shape {quat.shape}, expected (4,).")
