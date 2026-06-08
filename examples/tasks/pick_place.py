"""PickPlaceTask — example manipulation task (not part of robodeploy core)."""

from __future__ import annotations

from robodeploy.core.registry import register_task
from robodeploy.core.types import SceneSpec
from robodeploy.scene_builder import SceneBuilder
from robodeploy.tasks.templates.pick_place import PickPlaceTemplate


@register_task("pick_place")
class PickPlaceTask(PickPlaceTemplate):
    def scene_spec(self) -> SceneSpec:
        return (SceneBuilder()
            .add_box(self.source_name, size=(0.025, 0.025, 0.025), pos=(0.55, 0.0, 0.38), mass=0.05, rgba=(1.0, 0.0, 0.0, 1.0))
            .add_target(self.target_name, pos=(0.60, 0.20, 0.38))
            .build_spec())
