"""User-style components for a Kuka sinusoid demo.

These classes intentionally live outside the `robodeploy/` package to prove
that end-users can define and register their own robots, tasks, and policies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from robodeploy.core.registry import register_policy, register_robot, register_task
from robodeploy.core.spaces import ActionSpace, AssetFormat
from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
from robodeploy.description.base import RobotDescription
from robodeploy.policies.base import PolicyBase
from robodeploy.tasks.base import TaskBase


@register_robot("user_kuka")
class UserKukaDescription(RobotDescription):
    """User-defined robot description providing URDF + MJCF assets."""

    dof = 7
    display_name = "UserKuka"
    ee_link_name = "robot0/ee_link"
    joint_names = [f"robot0/joint{i}" for i in range(1, 8)]

    joint_position_limits = np.array([[-3.14, 3.14]] * dof, dtype=np.float64)
    joint_velocity_limits = np.array([2.0] * dof, dtype=np.float64)
    joint_torque_limits = np.array([50.0] * dof, dtype=np.float64)
    home_qpos = np.array([0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0], dtype=np.float64)

    def asset_path(self, fmt: AssetFormat, variant: str = "default") -> Path:
        del variant
        repo_root = Path(__file__).resolve().parents[2]
        if fmt == AssetFormat.MJCF:
            path = repo_root / "robodeploy" / "description" / "kuka" / "assets" / "mjcf" / "kuka.xml"
            if not path.exists():
                raise FileNotFoundError(f"Expected MJCF at {path}")
            return path
        if fmt == AssetFormat.URDF:
            path = repo_root / "examples" / "user_kuka_sinusoid" / "assets" / "user_kuka.urdf"
            if not path.exists():
                raise FileNotFoundError(f"Expected URDF at {path}")
            return path
        raise FileNotFoundError(f"UserKukaDescription demo provides URDF+MJCF only (requested {fmt}).")


@register_task("user_kuka_sinusoid")
class UserKukaSinusoidTask(TaskBase):
    """User-defined task: just run for N steps."""

    def __init__(self, max_steps: int = 1000) -> None:
        super().__init__(config={"max_steps": int(max_steps)})

    def obs_spec(self) -> ObsSpec:
        return ObsSpec(rgb=False, depth=False)

    def scene_spec(self) -> SceneSpec:
        return SceneSpec()

    def language_instruction(self) -> str:
        return "Move the Kuka arm with sinusoidal joint motion."

    def reset_fn(self, backend) -> None:
        return

    def reward_fn(self, obs: Observation, action: Action) -> float:
        return 0.0

    def success_fn(self, obs: Observation) -> bool:
        return False


@register_policy("user_sinusoid")
class UserSinusoidPolicy(PolicyBase):
    """User-defined scripted sinusoidal joint-position policy."""

    def __init__(
        self,
        dof: int = 7,
        amplitude: float = 0.25,
        frequency_hz: float = 0.2,
        phase: float = 0.0,
        home_qpos: Optional[list[float]] = None,
        joint_mask: Optional[list[int]] = None,
    ) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS, config={
            "action_hz": 50.0,
        })
        self._dof = int(dof)
        self._amp = float(amplitude)
        self._freq = float(frequency_hz)
        self._phase = float(phase)
        self._t = 0.0
        self._dt = 1.0 / 50.0
        self._home = np.array(home_qpos if home_qpos is not None else [0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0], dtype=np.float64)
        if self._home.shape[0] != self._dof:
            raise ValueError("home_qpos length must match dof.")
        self._mask = np.array(joint_mask if joint_mask is not None else [1] * self._dof, dtype=np.float64)

    def _reset_impl(self) -> None:
        self._t = 0.0

    def get_action(self, obs: Observation) -> Action:
        del obs
        self._t += self._dt
        w = 2.0 * np.pi * self._freq
        delta = self._amp * np.sin(w * self._t + self._phase) * self._mask
        q = (self._home + delta).astype(np.float32)
        return Action(joint_positions=q)

