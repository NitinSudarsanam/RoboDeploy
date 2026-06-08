"""TeleopPolicy — maps ITeleopDevice commands to RoboDeploy Action objects."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from robodeploy.core.registry import register_policy
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.base import PolicyBase
from robodeploy.teleop.base import ITeleopDevice, TeleopCommand, TeleopSafetyError
from robodeploy.teleop.ik_bridge import IKSolver, build_ik

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]


def _to_numpy(arr) -> np.ndarray:  # noqa: ANN001
    if arr is None:
        raise ValueError("expected array")
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _rpy_delta_to_quat(delta_rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in delta_rpy.reshape(3)]
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


@register_policy("teleop")
class TeleopPolicy(PolicyBase):
    """Adapts an ITeleopDevice to PolicyBase with optional cartesian IK."""

    def __init__(
        self,
        *,
        device: ITeleopDevice,
        action_space: ActionSpace = ActionSpace.JOINT_POS,
        ik_solver: IKSolver | None = None,
        default_action: Literal["hold", "zero"] = "hold",
        max_step_position_m: float = 0.05,
        max_step_orientation_rad: float = 0.1,
        config: dict | None = None,
    ) -> None:
        policy_cfg = {"action_hz": 50.0}
        if config:
            policy_cfg.update(dict(config))
        super().__init__(action_space=action_space, config=policy_cfg)
        self._device = device
        self._ik = ik_solver
        self._default_action = str(default_action)
        self._max_step_position_m = float(max_step_position_m)
        self._max_step_orientation_rad = float(max_step_orientation_rad)
        self._last_action: Action | None = None
        self._last_command: TeleopCommand | None = None
        self._backend = None
        self._description = None
        self._gripper_state = 0.0

    @property
    def device(self) -> ITeleopDevice:
        return self._device

    def bind_runtime(self, backend, description=None) -> None:
        self._backend = backend
        self._description = description
        if self._ik is None:
            self._ik = build_ik(backend, description)

    def _reset_impl(self, *, seed: int | None = None) -> None:
        del seed
        self._last_action = None
        self._last_command = None

    def _hold(self, obs: Observation) -> Action:
        q = _to_numpy(obs.joint_positions)
        return Action(joint_positions=jnp.asarray(q, dtype=jnp.float32), gripper=self._gripper_state)

    def _zero(self, obs: Observation) -> Action:
        q = _to_numpy(obs.joint_positions)
        zeros = np.zeros_like(q, dtype=np.float32)
        return Action(joint_positions=jnp.asarray(zeros, dtype=jnp.float32), gripper=self._gripper_state)

    def _default(self, obs: Observation) -> Action:
        if self._default_action == "zero":
            return self._zero(obs)
        return self._hold(obs)

    def _clamp_delta(self, cmd: TeleopCommand) -> TeleopCommand:
        if cmd.delta_position is not None:
            delta = np.asarray(cmd.delta_position, dtype=np.float32).reshape(3)
            norm = float(np.linalg.norm(delta))
            if norm > self._max_step_position_m:
                delta = delta * (self._max_step_position_m / norm)
            cmd.delta_position = delta
        if cmd.delta_orientation_rpy is not None:
            delta = np.asarray(cmd.delta_orientation_rpy, dtype=np.float32).reshape(3)
            norm = float(np.linalg.norm(delta))
            if norm > self._max_step_orientation_rad:
                delta = delta * (self._max_step_orientation_rad / norm)
            cmd.delta_orientation_rpy = delta
        return cmd

    def _integrate_ee(self, obs: Observation, cmd: TeleopCommand) -> tuple[np.ndarray, np.ndarray]:
        pos = _to_numpy(obs.ee_position).reshape(3)
        quat = _to_numpy(obs.ee_orientation).reshape(4)
        if cmd.delta_position is not None:
            pos = pos + np.asarray(cmd.delta_position, dtype=np.float32).reshape(3)
        if cmd.delta_orientation_rpy is not None:
            dq = _rpy_delta_to_quat(cmd.delta_orientation_rpy)
            quat = _quat_multiply(dq, quat)
        return pos.astype(np.float32), quat.astype(np.float32)

    def get_action(self, obs: Observation) -> Action:
        cmd = self._device.poll()
        self._last_command = cmd
        if cmd is None:
            return self._last_action or self._default(obs)
        if cmd.e_stop:
            raise TeleopSafetyError("operator e-stop")

        cmd = self._clamp_delta(cmd)
        if cmd.gripper_command is not None:
            self._gripper_state = float(cmd.gripper_command)

        q = _to_numpy(obs.joint_positions)
        action: Action

        if cmd.delta_joint_positions is not None:
            target_q = q + np.asarray(cmd.delta_joint_positions, dtype=np.float32).reshape(-1)
            action = Action(
                joint_positions=jnp.asarray(target_q, dtype=jnp.float32),
                gripper=self._gripper_state,
            )
        elif cmd.delta_position is not None or cmd.delta_orientation_rpy is not None:
            target_pos, target_quat = self._integrate_ee(obs, cmd)
            if self._ik is not None:
                solved = self._ik.solve(q, target_pos, target_quat=target_quat)
                action = Action(
                    joint_positions=jnp.asarray(solved, dtype=jnp.float32),
                    gripper=self._gripper_state,
                )
            else:
                action = self._default(obs)
                if cmd.gripper_command is not None:
                    action = Action(
                        joint_positions=action.joint_positions,
                        gripper=self._gripper_state,
                    )
        elif cmd.gripper_command is not None:
            action = Action(
                joint_positions=jnp.asarray(q, dtype=jnp.float32),
                gripper=self._gripper_state,
            )
        else:
            return self._last_action or self._default(obs)

        self._last_action = action
        return action
