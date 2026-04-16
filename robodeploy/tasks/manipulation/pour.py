"""PourTask placeholder matching architecture layout."""

from __future__ import annotations

from robodeploy.core.registry import register_task
from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
from robodeploy.tasks.base import TaskBase


@register_task("pour")
class PourTask(TaskBase):
    def obs_spec(self) -> ObsSpec:
        return ObsSpec(rgb=False, depth=False)

    def scene_spec(self) -> SceneSpec:
        return SceneSpec()

    def language_instruction(self) -> str:
        return "Pour from source into target."

    def reset_fn(self, backend) -> None:
        return

    def reward_fn(self, obs: Observation, action: Action) -> float:
        return 0.0

    def success_fn(self, obs: Observation) -> bool:
        return False

