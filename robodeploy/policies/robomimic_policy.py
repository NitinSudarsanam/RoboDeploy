"""Robomimic checkpoint policy wrapper for RoboDeploy.

Converts a RoboDeploy ``Observation`` into the flat state vector expected by
robomimic policies, runs inference, and converts the output back to a
RoboDeploy ``Action``.

State vector layout (matches move_panda_robomimic.py):
    [joint_positions(7), joint_velocities(7), gripper(1)] → 15-dim

Usage::

    from robodeploy.policies import RobomimicPolicy

    policy = RobomimicPolicy(checkpoint_path="path/to/policy.pth")
    policy.start_episode()

    action = policy.get_action(obs)   # obs: Observation
    await engine.apply_action(action)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from robodeploy.core.types import Action, Observation


class RobomimicPolicy:
    """Wrapper around a robomimic checkpoint policy.

    Args:
        checkpoint_path: Path to the ``.pth`` robomimic checkpoint file.
        obs_key:         Observation dict key the policy expects for the
                         low-dimensional state input (default ``"state"``).
        action_smooth:   Exponential smoothing factor in [0, 1] applied to
                         the raw policy output before it is executed.
                         0.0 = no smoothing, 1.0 = always use new action.
        use_cuda:        Whether to run inference on GPU (default True).
        arm_dof:         Number of arm DOF to include in the state vector
                         (default 7 for Franka Panda).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        obs_key: str = "state",
        action_smooth: float = 0.2,
        use_cuda: bool = True,
        arm_dof: int = 7,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._obs_key = obs_key
        self._action_smooth = float(np.clip(action_smooth, 0.0, 1.0))
        self._use_cuda = use_cuda
        self._arm_dof = arm_dof

        self._policy = None
        self._prev_ctrl: Optional[np.ndarray] = None

        self._load_policy()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

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

    def start_episode(self) -> None:
        """Reset internal policy state at the start of a new episode."""
        if self._policy is None:
            raise RuntimeError("Policy not loaded. Call __init__ first.")
        self._policy.start_episode()
        self._prev_ctrl = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _build_obs_dict(self, obs: Observation) -> dict[str, np.ndarray]:
        """Convert a RoboDeploy Observation into the flat dict robomimic expects."""
        joint_pos = np.asarray(obs.joint_positions, dtype=np.float32)[: self._arm_dof]
        joint_vel = np.asarray(obs.joint_velocities, dtype=np.float32)[: self._arm_dof]

        # Gripper: scalar normalized [0=open, 1=closed] → single-element array
        if obs.gripper_state is not None:
            gripper = np.array([obs.gripper_state], dtype=np.float32)
        else:
            gripper = np.zeros(1, dtype=np.float32)

        state = np.concatenate([joint_pos, joint_vel, gripper])
        return {self._obs_key: state}

    def _smooth_action(self, raw_action: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing between previous and current action."""
        if self._prev_ctrl is None or self._action_smooth <= 0.0:
            return raw_action

        alpha = self._action_smooth
        smoothed = (1.0 - alpha) * self._prev_ctrl + alpha * raw_action
        return smoothed

    def get_action(self, obs: Observation) -> Action:
        """Run policy inference and return a RoboDeploy Action.

        Args:
            obs: Current robot observation.

        Returns:
            Action with ``joint_positions`` (7-DOF) and ``gripper`` fields.
        """
        obs_dict = self._build_obs_dict(obs)
        raw = np.asarray(self._policy(ob=obs_dict), dtype=np.float64).reshape(-1)

        smoothed = self._smooth_action(raw)
        self._prev_ctrl = smoothed.copy()

        # Last element of the action vector is the gripper command
        arm_action = smoothed[: self._arm_dof]
        gripper_val = float(np.clip(smoothed[self._arm_dof], 0.0, 1.0)) if smoothed.size > self._arm_dof else None

        try:
            import jax.numpy as jnp
        except ImportError:
            import numpy as jnp  # type: ignore[assignment]
        return Action(
            joint_positions=jnp.asarray(arm_action, dtype=jnp.float32),
            gripper=gripper_val,
        )
