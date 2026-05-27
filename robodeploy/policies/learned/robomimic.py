"""RobomimicPolicy — learned policy wrapper with injectable inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from robodeploy.core.registry import register_policy
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.base import PolicyBase

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

PredictFn = Callable[[dict[str, np.ndarray]], np.ndarray]


@register_policy("robomimic")
class RobomimicPolicy(PolicyBase):
    """Wrapper around a robomimic checkpoint or an injected predict callable."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config: dict | None = None,
        *,
        obs_key: str = "state",
        action_smooth: float = 0.2,
        use_cuda: bool = True,
        arm_dof: int = 7,
        predict_fn: PredictFn | None = None,
    ) -> None:
        cfg = dict(config or {})
        if checkpoint_path is not None:
            cfg.setdefault("checkpoint_path", checkpoint_path)
        super().__init__(action_space=ActionSpace.JOINT_POS, config=cfg)
        self._obs_key = str(self.config.get("obs_key", obs_key))
        self._action_smooth = float(
            np.clip(float(self.config.get("action_smooth", action_smooth)), 0.0, 1.0)
        )
        self._use_cuda = bool(self.config.get("use_cuda", use_cuda))
        self._arm_dof = int(self.config.get("arm_dof", arm_dof))
        self._predict_fn: PredictFn | None = predict_fn or self.config.get("predict_fn")
        self._policy: Any = None
        self._prev_ctrl: Optional[np.ndarray] = None
        if self._predict_fn is None:
            self._load_policy()

    def _reset_impl(self) -> None:
        if self._policy is not None and hasattr(self._policy, "start_episode"):
            self._policy.start_episode()
        self._prev_ctrl = None

    def _load_policy(self) -> None:
        checkpoint = Path(str(self.config.get("checkpoint_path", "")))
        try:
            import robomimic.utils.file_utils as FileUtils
            import robomimic.utils.torch_utils as TorchUtils
        except ImportError as exc:
            raise ImportError(
                "robomimic is not installed. Install it with:\n"
                "    pip install robomimic\n"
                "Or pass config={'predict_fn': callable} for lightweight tests."
            ) from exc

        if not checkpoint.exists():
            raise FileNotFoundError(f"Robomimic checkpoint not found: {checkpoint}")

        device = TorchUtils.get_torch_device(try_to_use_cuda=self._use_cuda)
        self._policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=str(checkpoint),
            device=device,
            verbose=bool(self.config.get("verbose", True)),
        )

    def _build_obs_dict(self, obs: Observation) -> dict[str, np.ndarray]:
        joint_pos = np.asarray(obs.joint_positions, dtype=np.float32)[: self._arm_dof]
        joint_vel = np.asarray(obs.joint_velocities, dtype=np.float32)[: self._arm_dof]
        if obs.gripper_state is not None:
            gripper = np.array([obs.gripper_state], dtype=np.float32)
        else:
            gripper = np.zeros(1, dtype=np.float32)
        if obs.rgb is not None:
            rgb = np.asarray(obs.rgb, dtype=np.float32)
            if rgb.ndim == 3:
                rgb = rgb.reshape(-1)
            state = np.concatenate([joint_pos, joint_vel, gripper, rgb[:32]])
        else:
            state = np.concatenate([joint_pos, joint_vel, gripper])
        return {self._obs_key: state}

    def _smooth_action(self, raw_action: np.ndarray) -> np.ndarray:
        if self._prev_ctrl is None or self._action_smooth <= 0.0:
            return raw_action
        alpha = self._action_smooth
        return (1.0 - alpha) * self._prev_ctrl + alpha * raw_action

    def _predict_raw(self, obs: Observation) -> np.ndarray:
        obs_dict = self._build_obs_dict(obs)
        if self._predict_fn is not None:
            return np.asarray(self._predict_fn(obs_dict), dtype=np.float64).reshape(-1)
        if self._policy is None:
            raise RuntimeError("RobomimicPolicy not loaded.")
        return np.asarray(self._policy(ob=obs_dict), dtype=np.float64).reshape(-1)

    def get_action(self, obs: Observation) -> Action:
        smoothed = self._smooth_action(self._predict_raw(obs))
        self._prev_ctrl = smoothed.copy()
        arm_action = smoothed[: self._arm_dof]
        gripper_val = (
            float(np.clip(smoothed[self._arm_dof], 0.0, 1.0))
            if smoothed.size > self._arm_dof
            else None
        )
        return Action(
            joint_positions=jnp.asarray(arm_action, dtype=jnp.float32),
            gripper=gripper_val,
        )
