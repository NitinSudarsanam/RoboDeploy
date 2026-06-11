"""PegTask — reference manipulation task."""

from __future__ import annotations

from robodeploy.core.registry import register_task
from robodeploy.core.types import SceneSpec
from robodeploy.scene_builder import SceneBuilder
from robodeploy.tasks.templates.insertion import InsertionTemplate


@register_task("peg_insertion")
class PegTask(InsertionTemplate):
    def scene_spec(self) -> SceneSpec:
        return (
            SceneBuilder()
            .add_cylinder(self.peg_name, radius=0.012, height=0.06, pos=(0.5, 0.0, 0.06), mass=0.05, rgba=(0.9, 0.7, 0.2, 1.0))
            .add_box(self.hole_name, size=(0.05, 0.05, 0.006), pos=(0.6, 0.0, 0.006), fixed=True, rgba=(0.1, 0.1, 0.1, 1.0))
            .build_spec()
        )
