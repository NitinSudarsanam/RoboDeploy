"""Minimal demo task for plugin entry-point smoke tests."""

from __future__ import annotations

from robodeploy.core.registry import register_task
from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
from robodeploy.tasks.base import TaskBase


@register_task("demo_task")
class DemoTask(TaskBase):
    def obs_spec(self) -> ObsSpec:
        return ObsSpec()

    def scene_spec(self) -> SceneSpec:
        return SceneSpec()

    def language_instruction(self) -> str:
        return "Demo plugin task."

    def reward_fn(self, obs: Observation, action: Action) -> float:
        del obs, action
        return 0.0

    def success_fn(self, obs: Observation) -> bool:
        del obs
        return False
