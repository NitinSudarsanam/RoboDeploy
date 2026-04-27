"""PourTask placeholder matching architecture layout."""

from __future__ import annotations

import math

from robodeploy.core.registry import register_task
from robodeploy.core.types import Action, ObsSpec, Observation, PropConfig, SceneSpec
from robodeploy.tasks.base import TaskBase


@register_task("pour")
class PourTask(TaskBase):
    def obs_spec(self) -> ObsSpec:
        return ObsSpec(rgb=False, depth=False)

    def scene_spec(self) -> SceneSpec:
        return SceneSpec(
            props=[
                PropConfig(name="cup_source", asset_path="", position=(0.5, -0.15, 0.02), is_fixed=False),
                PropConfig(name="cup_target", asset_path="", position=(0.55, 0.15, 0.02), is_fixed=True),
            ]
        )

    def language_instruction(self) -> str:
        return "Pour from source into target."

    def reset_fn(self, backend) -> None:
        return

    def reward_fn(self, obs: Observation, action: Action) -> float:
        del action
        target = (0.55, 0.15, 0.02)
        dx = float(obs.ee_position[0]) - target[0]
        dy = float(obs.ee_position[1]) - target[1]
        dz = float(obs.ee_position[2]) - target[2]
        return -math.sqrt(dx * dx + dy * dy + dz * dz)

    def success_fn(self, obs: Observation) -> bool:
        target = (0.55, 0.15, 0.02)
        dx = float(obs.ee_position[0]) - target[0]
        dy = float(obs.ee_position[1]) - target[1]
        dz = float(obs.ee_position[2]) - target[2]
        return (dx * dx + dy * dy + dz * dz) < (0.05 * 0.05)

