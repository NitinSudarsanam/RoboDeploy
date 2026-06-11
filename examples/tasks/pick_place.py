"""PickPlaceTask — reference manipulation task for examples and benchmarks."""

from __future__ import annotations

from robodeploy.core.registry import register_task
from robodeploy.core.types import SceneSpec
from robodeploy.tasks.templates.pick_place import PickPlaceTemplate

from examples.scenes.pick_table import build_pick_place_scene


@register_task("pick_place")
class PickPlaceTask(PickPlaceTemplate):
    def scene_spec(self) -> SceneSpec:
        return build_pick_place_scene()
