"""LearnedActionAdapter — bridge model tensor outputs to backend Action objects."""

from __future__ import annotations

from typing import Any

import numpy as np

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]


class LearnedActionAdapter:
    """Convert raw model outputs into Actions for the declared target space."""

    def __init__(
        self,
        *,
        source_space: ActionSpace,
        target_space: ActionSpace,
        source_dim: int,
        target_dim: int,
        ik_solver: Any | None = None,
        normalization: dict[str, Any] | None = None,
        gripper_index_source: int | None = None,
        gripper_index_target: int | None = None,
        arm_dof: int | None = None,
        dt: float = 0.01,
    ) -> None:
        self._source_space = source_space
        self._target_space = target_space
        self._source_dim = int(source_dim)
        self._target_dim = int(target_dim)
        self._ik_solver = ik_solver
        self._normalization = dict(normalization or {})
        self._gripper_index_source = gripper_index_source
        self._gripper_index_target = gripper_index_target
        self._arm_dof = arm_dof
        self._dt = float(dt)
        self._last_q: np.ndarray | None = None

    def __call__(self, model_output: np.ndarray, obs: Observation) -> Action:
        raw = np.asarray(model_output, dtype=np.float64).reshape(-1)
        if raw.size < self._source_dim:
            raise ValueError(
                f"Model output has dim {raw.size}, expected at least {self._source_dim}."
            )
        vector = self._unnormalize(raw[: self._source_dim])
        gripper = self._extract_gripper(raw)
        action = self._to_source_action(vector, gripper, obs)
        converted = self._convert_space(action, obs)
        self._validate_target(converted)
        return converted

    def warm_start(self, q_init: np.ndarray) -> None:
        self._last_q = np.asarray(q_init, dtype=np.float64).copy()

    def reset(self) -> None:
        self._last_q = None

    def _unnormalize(self, vector: np.ndarray) -> np.ndarray:
        norm = self._normalization
        if not norm:
            return vector
        low = np.asarray(norm.get("low", -1.0), dtype=np.float64)
        high = np.asarray(norm.get("high", 1.0), dtype=np.float64)
        out_min = np.asarray(norm.get("out_min", low), dtype=np.float64)
        out_max = np.asarray(norm.get("out_max", high), dtype=np.float64)
        if low.size == 1:
            low = np.full(vector.shape, float(low.reshape(-1)[0]))
        if high.size == 1:
            high = np.full(vector.shape, float(high.reshape(-1)[0]))
        if out_min.size == 1:
            out_min = np.full(vector.shape, float(out_min.reshape(-1)[0]))
        if out_max.size == 1:
            out_max = np.full(vector.shape, float(out_max.reshape(-1)[0]))
        t = (vector - low) / np.maximum(high - low, 1e-8)
        return out_min + t * (out_max - out_min)

    def _extract_gripper(self, raw: np.ndarray) -> float | None:
        idx = self._gripper_index_source
        if idx is None:
            if raw.size > self._source_dim:
                return float(np.clip(raw[self._source_dim], 0.0, 1.0))
            return None
        if idx < 0 or idx >= raw.size:
            return None
        return float(np.clip(raw[idx], 0.0, 1.0))

    def _to_source_action(self, vector: np.ndarray, gripper: float | None, obs: Observation) -> Action:
        space = self._source_space
        if space == ActionSpace.JOINT_POS:
            return Action(
                joint_positions=jnp.asarray(vector, dtype=jnp.float32),
                gripper=gripper,
                action_space=space,
            )
        if space == ActionSpace.JOINT_VEL:
            return Action(
                joint_velocities=jnp.asarray(vector, dtype=jnp.float32),
                gripper=gripper,
                action_space=space,
            )
        if space in (ActionSpace.DELTA_EE, ActionSpace.CARTESIAN_POSE):
            return Action(
                ee_position=jnp.asarray(vector[:3], dtype=jnp.float32),
                ee_orientation=(
                    jnp.asarray(vector[3:7], dtype=jnp.float32)
                    if vector.size >= 7
                    else None
                ),
                gripper=gripper,
                action_space=space,
                is_delta_ee=space == ActionSpace.DELTA_EE,
            )
        return Action(joint_positions=jnp.asarray(vector, dtype=jnp.float32), gripper=gripper, action_space=space)

    def _convert_space(self, action: Action, obs: Observation) -> Action:
        if self._source_space == self._target_space:
            return action
        if self._source_space == ActionSpace.DELTA_EE and self._target_space == ActionSpace.JOINT_POS:
            return self._delta_ee_to_joint_pos(action, obs)
        if self._source_space == ActionSpace.CARTESIAN_POSE and self._target_space == ActionSpace.JOINT_POS:
            return self._cartesian_to_joint_pos(action, obs)
        if self._source_space == ActionSpace.JOINT_POS and self._target_space == ActionSpace.JOINT_POS:
            return self._resize_joint_pos(action, obs)
        raise ValueError(
            f"No adapter path from {self._source_space.name} to {self._target_space.name}."
        )

    def _delta_ee_to_joint_pos(self, action: Action, obs: Observation) -> Action:
        if self._ik_solver is None:
            raise ValueError("IK solver required for DELTA_EE → JOINT_POS adaptation.")
        q_init = self._last_q
        if q_init is None and obs.joint_positions is not None:
            q_init = np.asarray(obs.joint_positions, dtype=np.float64)
        delta = np.asarray(action.ee_position, dtype=np.float64)
        if q_init is not None:
            current_pos, current_quat = self._ik_solver.fk(q_init)
            target_pos = current_pos + delta * self._dt
            target_quat = current_quat
        else:
            target_pos = delta
            target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if action.ee_orientation is not None:
            target_quat = np.asarray(action.ee_orientation, dtype=np.float64)
        joint_targets = self._ik_solver.ik(target_pos, target_quat, q_init=q_init)
        self._last_q = joint_targets.copy()
        return Action(
            joint_positions=jnp.asarray(joint_targets, dtype=jnp.float32),
            gripper=action.gripper,
            action_space=ActionSpace.JOINT_POS,
        )

    def _cartesian_to_joint_pos(self, action: Action, obs: Observation) -> Action:
        if self._ik_solver is None:
            raise ValueError("IK solver required for CARTESIAN_POSE → JOINT_POS adaptation.")
        q_init = self._last_q
        if q_init is None and obs.joint_positions is not None:
            q_init = np.asarray(obs.joint_positions, dtype=np.float64)
        target_pos = np.asarray(action.ee_position, dtype=np.float64)
        target_quat = (
            np.asarray(action.ee_orientation, dtype=np.float64)
            if action.ee_orientation is not None
            else np.array([1.0, 0.0, 0.0, 0.0])
        )
        joint_targets = self._ik_solver.ik(target_pos, target_quat, q_init=q_init)
        self._last_q = joint_targets.copy()
        return Action(
            joint_positions=jnp.asarray(joint_targets, dtype=jnp.float32),
            gripper=action.gripper,
            action_space=ActionSpace.JOINT_POS,
        )

    def _resize_joint_pos(self, action: Action, obs: Observation) -> Action:
        arm_dof = self._arm_dof or self._target_dim
        joints = np.asarray(action.joint_positions, dtype=np.float64)
        if joints.size == arm_dof and self._target_dim == arm_dof:
            return Action(
                joint_positions=jnp.asarray(joints, dtype=jnp.float32),
                gripper=action.gripper,
                action_space=ActionSpace.JOINT_POS,
            )
        if joints.size < self._target_dim and obs.joint_positions is not None:
            base = np.asarray(obs.joint_positions, dtype=np.float64)
            merged = base.copy()
            merged[: joints.size] = joints
            return Action(
                joint_positions=jnp.asarray(merged[: self._target_dim], dtype=jnp.float32),
                gripper=action.gripper,
                action_space=ActionSpace.JOINT_POS,
            )
        return Action(
            joint_positions=jnp.asarray(joints[: self._target_dim], dtype=jnp.float32),
            gripper=action.gripper,
            action_space=ActionSpace.JOINT_POS,
        )

    def _validate_target(self, action: Action) -> None:
        if self._target_space == ActionSpace.JOINT_POS and action.joint_positions is not None:
            size = int(np.asarray(action.joint_positions).size)
            if size != self._target_dim:
                raise ValueError(f"Adapted joint_positions dim {size} != target_dim {self._target_dim}.")
