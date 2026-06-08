"""PourTask — example manipulation task (not part of robodeploy core)."""

from __future__ import annotations

from robodeploy.core.registry import register_task
from robodeploy.core.types import SceneSpec
from robodeploy.scene_builder import SceneBuilder
from robodeploy.tasks.templates.pour import PourTemplate


@register_task("pour")
class PourTask(PourTemplate):
    def scene_spec(self) -> SceneSpec:
        return (
            SceneBuilder()
            .add_cylinder(self.cup_name, radius=0.035, height=0.04, pos=(0.5, -0.15, 0.04), mass=0.08, rgba=(0.2, 0.3, 1.0, 1.0))
            .add_cylinder(self.target_zone_name, radius=0.045, height=0.04, pos=(0.55, 0.15, 0.04), fixed=True, rgba=(0.1, 0.8, 0.3, 0.8))
            .build_spec()
        )
