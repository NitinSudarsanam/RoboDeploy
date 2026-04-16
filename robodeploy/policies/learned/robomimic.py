"""RobomimicPolicy — learned policy wrapper (structure-only migration).

This ports the previous standalone wrapper into the architecture's PolicyBase
interface and registers it under the string name ``robomimic``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from robodeploy.core.registry import register_policy
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.base import PolicyBase


@register_policy("robomimic")
class RobomimicPolicy(PolicyBase):
    """Wrapper around a robomimic checkpoint policy."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        obs_key: str = "state",
        action_smooth: float = 0.2,
        use_cuda: bool = True,
        arm_dof: int = 7,
    ) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS)
        self._checkpoint_path = Path(checkpoint_path)
        self._obs_key = obs_key
        self._action_smooth = float(np.clip(action_smooth, 0.0, 1.0))
        self._use_cuda = use_cuda
        self._arm_dof = arm_dof

        self._policy = None
        self._prev_ctrl: Optional[np.ndarray] = None

        self._load_policy()

    def _reset_impl(self) -> None:
        # robomimic policies use start_episode() as their episode boundary hook
        if self._policy is not None:
            self._policy.start_episode()
        self._prev_ctrl = None

    def _load_policy(self) -> None:
        try:
            import robomimic.utils.file_utils as FileUtils
            import robomimic.utils.torch_utils as TorchUtils
        except ImportError as exc:
            raise ImportError(
                "robomimic is not installed. Install it with:\n"
                "    pip install robomimic\n"
                "See https://robomimic.github.io for details."
            ) from exc

        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"Robomimic checkpoint not found: {self._checkpoint_path}"
            )

        device = TorchUtils.get_torch_device(try_to_use_cuda=self._use_cuda)
        self._policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=str(self._checkpoint_path),
            device=device,
            verbose=True,
        )

    def _build_obs_dict(self, obs: Observation) -> dict[str, np.ndarray]:
        joint_pos = np.asarray(obs.joint_positions, dtype=np.float32)[: self._arm_dof]
        joint_vel = np.asarray(obs.joint_velocities, dtype=np.float32)[: self._arm_dof]

        if obs.gripper_state is not None:
            gripper = np.array([obs.gripper_state], dtype=np.float32)
        else:
            gripper = np.zeros(1, dtype=np.float32)

        state = np.concatenate([joint_pos, joint_vel, gripper])
        return {self._obs_key: state}

    def _smooth_action(self, raw_action: np.ndarray) -> np.ndarray:
        if self._prev_ctrl is None or self._action_smooth <= 0.0:
            return raw_action
        alpha = self._action_smooth
        return (1.0 - alpha) * self._prev_ctrl + alpha * raw_action

    def get_action(self, obs: Observation) -> Action:
        if self._policy is None:
            raise RuntimeError("RobomimicPolicy not loaded.")

        obs_dict = self._build_obs_dict(obs)
        raw = np.asarray(self._policy(ob=obs_dict), dtype=np.float64).reshape(-1)

        smoothed = self._smooth_action(raw)
        self._prev_ctrl = smoothed.copy()

        arm_action = smoothed[: self._arm_dof]
        gripper_val = (
            float(np.clip(smoothed[self._arm_dof], 0.0, 1.0))
            if smoothed.size > self._arm_dof
            else None
        )

        try:
            import jax.numpy as jnp
        except ImportError:
            import numpy as jnp  # type: ignore[assignment]

        return Action(
            joint_positions=jnp.asarray(arm_action, dtype=jnp.float32),
            gripper=gripper_val,
        )

