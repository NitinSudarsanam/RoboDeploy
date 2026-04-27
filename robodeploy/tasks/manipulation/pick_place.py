"""PickPlaceTask — architecture-compliant manipulation task stub."""

from __future__ import annotations

import math

from robodeploy.core.registry import register_task
from robodeploy.core.types import Action, ObsSpec, Observation, PropConfig, SceneSpec
from robodeploy.tasks.base import TaskBase


@register_task("pick_place")
class PickPlaceTask(TaskBase):
    """Pick-and-place task (minimal)."""

    def obs_spec(self) -> ObsSpec:
        # Minimal default: no vision until concrete sensor stack is added.
        return ObsSpec(rgb=False, depth=False)

    def scene_spec(self) -> SceneSpec:
        # Minimal scene: a "source" prop and a "target" prop marker.
        # Backends may ignore asset_path for now; RViz can still show markers.
        return SceneSpec(
            props=[
                PropConfig(name="source", asset_path="", position=(0.5, 0.0, 0.02), is_fixed=False),
                PropConfig(name="target", asset_path="", position=(0.6, 0.2, 0.02), is_fixed=True),
            ],
            table_height=0.0,
            lighting="default",
        )

    def language_instruction(self) -> str:
        return "Pick the object and place it at the target."

    def reset_fn(self, backend) -> None:
        # DomainRandomizer integration (future): backend.teleport_object() calls go here.
        return

    def reward_fn(self, obs: Observation, action: Action) -> float:
        del action
        target = (0.6, 0.2, 0.02)
        dx = float(obs.ee_position[0]) - target[0]
        dy = float(obs.ee_position[1]) - target[1]
        dz = float(obs.ee_position[2]) - target[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        return -dist

    def success_fn(self, obs: Observation) -> bool:
        target = (0.6, 0.2, 0.02)
        dx = float(obs.ee_position[0]) - target[0]
        dy = float(obs.ee_position[1]) - target[1]
        dz = float(obs.ee_position[2]) - target[2]
        return (dx * dx + dy * dy + dz * dz) < (0.05 * 0.05)

