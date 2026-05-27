"""User-style components: canonical URDF description.

This example demonstrates:
- user defines a robot from URDF (canonical input)
- MuJoCo backend requires MJCF (so it fails without override)
- user can override MJCF path explicitly via backend config
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robodeploy.core.registry import register_robot, register_task, register_policy
from robodeploy.description.base import URDFRobotDescription
from robodeploy.core.types import Action, Observation, ObsSpec, SceneSpec
from robodeploy.tasks.base import TaskBase
from robodeploy.policies.base import PolicyBase
from robodeploy.core.spaces import ActionSpace


@register_robot("user_urdf_robot")
class UserURDFKuka(URDFRobotDescription):
    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        urdf_path = repo_root / "examples" / "user_urdf_asset_override" / "assets" / "user_kuka.urdf"
        super().__init__(
            urdf_path,
            ee_link_name="robot0/ee_link",
            display_name="UserURDFKuka",
            home_qpos=np.array([0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0], dtype=np.float64),
        )


@register_task("user_dummy_task")
class UserDummyTask(TaskBase):
    def obs_spec(self) -> ObsSpec:
        return ObsSpec(rgb=False, depth=False)

    def scene_spec(self) -> SceneSpec:
        return SceneSpec()

    def language_instruction(self) -> str:
        return "Hold / move joints."

    def reset_fn(self, backend) -> None:
        return

    def reward_fn(self, obs: Observation, action: Action) -> float:
        return 0.0

    def success_fn(self, obs: Observation) -> bool:
        return False


@register_policy("user_hold_policy")
class UserHoldPolicy(PolicyBase):
    def __init__(self) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS)
        self._home_qpos = np.array([0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0], dtype=np.float64)

    def get_action(self, obs: Observation) -> Action:
        return Action(joint_positions=self._home_qpos.copy(), action_space=ActionSpace.JOINT_POS)

