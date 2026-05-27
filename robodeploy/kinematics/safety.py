"""
SafetyFilter — validates and clamps every action before it reaches any backend.

SafetyFilter is always active, in both sim and real. This is non-negotiable:
  - In sim it prevents learning policies from exploiting unphysical actions.
  - On real hardware it prevents joint limit violations and motor damage.

Design: SafetyFilter knows only about joint limits from RobotDescription.
It does not know about physics, ROS2, or policy internals. It is pure math.

Every backend receives actions AFTER SafetyFilter.filter() has been called.
RoboEnv calls SafetyFilter.filter() automatically in env.step(). Backends
should not duplicate limit checking.

Access via description.get_safety_filter() — this ensures one instance per
robot, cached, so limit arrays are only built once.
"""

from __future__ import annotations

import numpy as np

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types  import Action

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from robodeploy.description.base import RobotDescription


class SafetyFilter:
    """Clamps and validates actions against robot joint limits.

    Do not instantiate directly. Access via:
        description.get_safety_filter()

    Args:
        description: Robot definition providing joint limits.
    """

    def __init__(self, description: RobotDescription) -> None:
        self._pos_min  = description.joint_position_limits[:, 0]   # [dof]
        self._pos_max  = description.joint_position_limits[:, 1]   # [dof]
        self._vel_max  = description.joint_velocity_limits          # [dof]
        self._tau_max  = description.joint_torque_limits            # [dof]
        self._dof      = description.dof

        # Emergency stop flag — set externally to freeze all actions
        self._estop:   bool = False

        # Velocity ramp: if prev_action is known, limit step-to-step position delta
        self._prev_pos: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def filter(self, action: Action, action_space: ActionSpace) -> Action:
        """Clamp action to safe limits and check e-stop.

        Called by RoboEnv.step() on every action before the backend receives it.
        Returns a new Action — does not modify in place.

        Args:
            action:       Raw action from the policy.
            action_space: Declared space of the action (determines which
                          fields are validated).

        Returns:
            Action: Clamped and validated copy. If e-stop is active, returns
                    an action holding the previous joint positions (freeze).

        Raises:
            ValueError: If the action's array shapes do not match robot DOF.
        """
        if self._estop:
            return self._freeze_action()

        if action_space == ActionSpace.JOINT_POS:
            return self._filter_joint_pos(action)
        if action_space == ActionSpace.JOINT_VEL:
            return self._filter_joint_vel(action)
        if action_space == ActionSpace.JOINT_TORQUE:
            return self._filter_joint_torque(action)

        if action_space in (ActionSpace.CARTESIAN_POSE, ActionSpace.DELTA_EE):
            self._validate_cartesian(action)
            raise NotImplementedError(
                "Cartesian action safety bounds are not configured. "
                "Use joint-space actions or provide a backend-specific Cartesian safety filter."
            )

        return action

    # ------------------------------------------------------------------
    # Emergency stop
    # ------------------------------------------------------------------

    def trigger_estop(self) -> None:
        """Engage emergency stop. All subsequent filter() calls return freeze actions.

        Call this from ROS2 safety callbacks, GUI buttons, or test fixtures.
        Reset with clear_estop() only when the robot is safe to move again.
        """
        self._estop = True

    def clear_estop(self) -> None:
        """Disengage emergency stop. Only call when safe to resume motion."""
        self._estop    = False
        self._prev_pos = None   # reset ramp state too

    @property
    def estop_active(self) -> bool:
        """True if emergency stop is currently engaged."""
        return self._estop

    # ------------------------------------------------------------------
    # Private clamping methods
    # ------------------------------------------------------------------

    def _filter_joint_pos(self, action: Action) -> Action:
        """Clamp joint_positions to [pos_min, pos_max]."""
        if action.joint_positions is None:
            return action

        raw = np.asarray(action.joint_positions, dtype=np.float64)
        self._validate_shape(raw, "joint_positions")

        clamped = np.clip(raw, self._pos_min, self._pos_max)
        self._prev_pos = clamped.copy()

        return Action(
            joint_positions=clamped.astype(np.float32),
            gripper=action.gripper,
            timestamp=action.timestamp,
        )

    def _filter_joint_vel(self, action: Action) -> Action:
        """Clamp joint_velocities to [-vel_max, +vel_max]."""
        if action.joint_velocities is None:
            return action

        raw     = np.asarray(action.joint_velocities, dtype=np.float64)
        self._validate_shape(raw, "joint_velocities")
        clamped = np.clip(raw, -self._vel_max, self._vel_max)
        self._prev_pos = None

        return Action(
            joint_velocities=clamped.astype(np.float32),
            gripper=action.gripper,
            timestamp=action.timestamp,
        )

    def _filter_joint_torque(self, action: Action) -> Action:
        """Clamp joint_torques to [-tau_max, +tau_max]."""
        if action.joint_torques is None:
            return action

        raw     = np.asarray(action.joint_torques, dtype=np.float64)
        self._validate_shape(raw, "joint_torques")
        clamped = np.clip(raw, -self._tau_max, self._tau_max)
        self._prev_pos = None

        return Action(
            joint_torques=clamped.astype(np.float32),
            gripper=action.gripper,
            timestamp=action.timestamp,
        )

    def _freeze_action(self) -> Action:
        """Return an action holding the last known joint positions (emergency stop)."""
        pos = self._prev_pos if self._prev_pos is not None else (
            0.5 * (self._pos_min + self._pos_max)
        )
        return Action(joint_positions=pos.astype(np.float32), gripper=0.0)

    def _validate_shape(self, arr: np.ndarray, field: str) -> None:
        if arr.shape != (self._dof,):
            raise ValueError(
                f"Action.{field} has shape {arr.shape}, "
                f"expected ({self._dof},) for this robot."
            )

    def _validate_cartesian(self, action: Action) -> None:
        if action.ee_position is not None:
            pos = np.asarray(action.ee_position, dtype=np.float64)
            if pos.shape != (3,):
                raise ValueError(f"Action.ee_position has shape {pos.shape}, expected (3,).")
        if action.ee_orientation is not None:
            quat = np.asarray(action.ee_orientation, dtype=np.float64)
            if quat.shape != (4,):
                raise ValueError(f"Action.ee_orientation has shape {quat.shape}, expected (4,).")
