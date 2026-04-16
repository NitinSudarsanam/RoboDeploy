"""PickPlaceTask — architecture-compliant manipulation task stub."""

from __future__ import annotations

from robodeploy.core.registry import register_task
from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
from robodeploy.tasks.base import TaskBase


@register_task("pick_place")
class PickPlaceTask(TaskBase):
    """Pick-and-place task (stub)."""

    def obs_spec(self) -> ObsSpec:
        # Minimal default: no vision until concrete sensor stack is added.
        return ObsSpec(rgb=False, depth=False)

    def scene_spec(self) -> SceneSpec:
        # Scene content is backend-specific to load; keep empty in stub.
        return SceneSpec(objects=[], table_height=0.0, lighting="default")

    def language_instruction(self) -> str:
        return "Pick the object and place it at the target."

    def reset_fn(self, backend) -> None:
        # DomainRandomizer integration (future): backend.teleport_object() calls go here.
        return

    def reward_fn(self, obs: Observation, action: Action) -> float:
        # Placeholder shaping — returns 0 until task is defined.
        return 0.0

    def success_fn(self, obs: Observation) -> bool:
        return False

