"""
ActionAdapter — composable transforms applied to Actions before the backend.

Mirrors ObsPipeline exactly, but operates on Action objects instead of
Observations. It sits between the policy and the SafetyFilter:

    policy.get_action(obs)
        → ActionAdapter.process(action)   ← space/frame conversions
        → SafetyFilter.filter(action)     ← limit clamping, e-stop
        → backend.step(action)            ← hardware / physics

Why this matters:
  A policy trained to output Cartesian deltas (ActionSpace.DELTA_EE) cannot
  be used directly with a backend that expects joint positions. Instead of
  writing conversion code inside the policy or backend (coupling them),
  you insert an IKActionTransform into the adapter:

      adapter = ActionAdapter([
          DeltaEEToJointPosTransform(description.get_kinematics_solver()),
      ])

  Now the policy is decoupled from the backend's control mode. Swap the
  transform to support a different backend with no policy or backend changes.

Built-in action transforms:
  - IdentityActionTransform:         pass-through, default
  - DeltaEEToJointPosTransform:      Cartesian delta → joint positions via IK
  - ActionChunkTransform:            buffers a trajectory, yields one step at a time
  - ScaleActionTransform:            rescale action ranges (e.g. normalized [-1,1] → radians)

Design note on not using gym.spaces yet:
  Full gym.spaces.Dict support for heterogeneous multi-body robots (arm + base +
  gripper as separate spaces) is deferred. The current Action dataclass covers
  the common case. When a multi-body robot is added, the pattern is:
    - Extend Action with additional optional fields.
    - Add a SplitActionTransform that routes fields to different controllers.
  No redesign of ActionAdapter or IBackend is required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from robodeploy.core.types import Action


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class IActionTransform(ABC):
    """A single composable transform applied to an Action.

    Implement forward() to convert between action representations.
    Called in order by ActionAdapter.process() after policy inference
    and before SafetyFilter.
    """

    @abstractmethod
    def forward(self, action: Action) -> Action:
        """Transform action and return a new Action. Must not mutate in place.

        Args:
            action: Raw action from the policy (or previous transform).

        Returns:
            Transformed action.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state at the start of a new episode.

        Default is a no-op. Override for stateful transforms like
        ActionChunkTransform that buffer trajectory segments.
        """
        pass


# ---------------------------------------------------------------------------
# Built-in action transforms
# ---------------------------------------------------------------------------

class IdentityActionTransform(IActionTransform):
    """Pass-through. Default when no ActionAdapter is specified."""

    def forward(self, action: Action) -> Action:
        return action


class ScaleActionTransform(IActionTransform):
    """Rescale joint_positions from a normalized range to physical radians.

    Policies trained on demonstrations normalized to [-1, 1] need their
    outputs scaled back to the robot's actual joint range before use.

    Args:
        in_min:  Lower bound of the policy's output range (e.g. -1.0).
        in_max:  Upper bound of the policy's output range (e.g.  1.0).
        out_min: Lower joint limits [dof] in radians.
        out_max: Upper joint limits [dof] in radians.
    """

    def __init__(
        self,
        in_min:  float,
        in_max:  float,
        out_min: np.ndarray,
        out_max: np.ndarray,
    ) -> None:
        self._in_min  = in_min
        self._in_max  = in_max
        self._out_min = out_min
        self._out_max = out_max

    def forward(self, action: Action) -> Action:
        if action.joint_positions is None:
            return action
        raw   = np.asarray(action.joint_positions, dtype=np.float32)
        t     = (raw - self._in_min) / (self._in_max - self._in_min)  # → [0, 1]
        scaled = self._out_min + t * (self._out_max - self._out_min)
        return Action(
            joint_positions=scaled,
            gripper=action.gripper,
            timestamp=action.timestamp,
        )


class DeltaEEToJointPosTransform(IActionTransform):
    """Convert Cartesian end-effector delta → joint position targets via IK.

    Policies that output small Cartesian displacements (ActionSpace.DELTA_EE)
    can be used with joint-position backends by inserting this transform.

    Args:
        solver:  KinematicsSolver for the active robot description.
        dt:      Control period in seconds, used to integrate deltas into poses.
    """

    def __init__(self, solver, dt: float = 0.01) -> None:
        self._solver    = solver
        self._dt        = dt
        self._last_qpos: Optional[np.ndarray] = None

    def forward(self, action: Action) -> Action:
        if action.ee_position is None:
            return action   # not a Cartesian action, pass through

        delta_pos  = np.asarray(action.ee_position,    dtype=np.float64)
        q_init     = self._last_qpos

        # Integrate delta into absolute target
        if q_init is not None:
            current_pos, current_quat = self._solver.fk(q_init)
            target_pos  = current_pos + delta_pos * self._dt
            target_quat = current_quat   # orientation held unless ee_orientation provided
        else:
            target_pos  = delta_pos
            target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        if action.ee_orientation is not None:
            target_quat = np.asarray(action.ee_orientation, dtype=np.float64)

        joint_targets = self._solver.ik(target_pos, target_quat, q_init=q_init)
        self._last_qpos = joint_targets.copy()

        return Action(
            joint_positions=joint_targets.astype(np.float32),
            gripper=action.gripper,
            timestamp=action.timestamp,
        )

    def warm_start(self, q_init: np.ndarray) -> None:
        """Seed the IK solver with the robot's current joint state.

        Must be called after env.reset() and before the first get_action()
        call. Without this, the first IK query has no prior joint state,
        which means the solver cannot avoid elbow flips on the first step.

        RoboEnv.reset() calls this automatically if an ActionAdapter
        containing a DeltaEEToJointPosTransform is provided and the backend
        exposes its initial joint positions.

        Args:
            q_init: Current joint positions [dof] in radians.
        """
        self._last_qpos = np.asarray(q_init, dtype=np.float64).copy()

    def reset(self) -> None:
        self._last_qpos = None


class ActionChunkTransform(IActionTransform):
    """Buffer a trajectory chunk from the policy, yield one step per call.

    Used with models that output action sequences (ACT, Diffusion Policy).
    The policy is called infrequently (every chunk_size steps) and outputs
    a full trajectory. This transform serves one action per control step.

    Args:
        chunk_size: Number of steps per policy call. Policy must return
                    Action with joint_positions of shape [chunk_size, dof].
    """

    def __init__(self, chunk_size: int = 20) -> None:
        self._chunk_size = chunk_size
        self._buffer: list[np.ndarray] = []
        self._gripper_buffer: list[Optional[float]] = []

    def forward(self, action: Action) -> Action:
        """If action contains a full chunk, buffer it. Return one step."""
        if action.joint_positions is not None:
            pos = np.asarray(action.joint_positions)
            if pos.ndim == 2 and pos.shape[0] > 1:
                # Full chunk received — load into buffer
                self._buffer = [pos[i] for i in range(pos.shape[0])]
                if action.gripper is not None and hasattr(action.gripper, '__len__'):
                    self._gripper_buffer = list(action.gripper)
                else:
                    self._gripper_buffer = [action.gripper] * len(self._buffer)

        # Pop one step from buffer
        if self._buffer:
            step_pos     = self._buffer.pop(0)
            step_gripper = self._gripper_buffer.pop(0) if self._gripper_buffer else action.gripper
            return Action(
                joint_positions=step_pos.astype(np.float32),
                gripper=step_gripper,
                timestamp=action.timestamp,
            )

        return action   # buffer empty — pass through as-is

    def reset(self) -> None:
        self._buffer.clear()
        self._gripper_buffer.clear()

    @property
    def steps_remaining(self) -> int:
        """Steps remaining in the current chunk buffer."""
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ActionAdapter:
    """Applies an ordered list of IActionTransforms to each action.

    Mirrors ObsPipeline in structure. Insert between policy inference
    and SafetyFilter.

    Args:
        transforms: Ordered list of action transforms. Applied left-to-right.
                    Defaults to [IdentityActionTransform()] — a no-op.

    Example:
        adapter = ActionAdapter([
            ScaleActionTransform(in_min=-1, in_max=1, out_min=limits[:,0], out_max=limits[:,1]),
        ])

        # In control loop:
        raw_action   = policy.get_action(obs)
        adapted      = adapter.process(raw_action)
        safe_action  = safety_filter.filter(adapted, ...)
        backend.step(safe_action)
    """

    def __init__(self, transforms: list[IActionTransform] | None = None) -> None:
        self.transforms: list[IActionTransform] = transforms or [IdentityActionTransform()]

    def process(self, action: Action) -> Action:
        """Apply all transforms in order.

        Args:
            action: Raw action from the policy.

        Returns:
            Transformed action ready for SafetyFilter.
        """
        for transform in self.transforms:
            action = transform.forward(action)
        return action

    def reset(self) -> None:
        """Reset all stateful transforms. Called by RoboEnv at episode start."""
        for transform in self.transforms:
            transform.reset()

    def warm_start(self, q_init: np.ndarray) -> None:
        """Seed stateful transforms (e.g. DeltaEEToJointPosTransform) with current q."""
        for transform in self.transforms:
            warm = getattr(transform, "warm_start", None)
            if callable(warm):
                warm(q_init)

    def __repr__(self) -> str:
        names = [type(t).__name__ for t in self.transforms]
        return f"ActionAdapter([{', '.join(names)}])"
