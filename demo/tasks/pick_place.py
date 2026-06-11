"""Pick-and-place task for the Kuka demo."""

from __future__ import annotations

from robodeploy.core.registry import register_task
from robodeploy.core.types import SceneSpec
from robodeploy.tasks.templates.pick_place import PickPlaceTemplate

from demo.scenes.pick_table import build_pick_place_scene


@register_task("demo_pick_place")
class DemoPickPlaceTask(PickPlaceTemplate):
    def scene_spec(self) -> SceneSpec:
        return build_pick_place_scene()
