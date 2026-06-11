"""ShowcaseSceneTask — demo scene with every procedural geom kind."""

from __future__ import annotations

from robodeploy.core.registry import register_task
from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
from robodeploy.scene_builder import SceneBuilder
from robodeploy.tasks.base import TaskBase


@register_task("showcase_scene")
class ShowcaseSceneTask(TaskBase):
    """Inspectable scene exercising box, cylinder, sphere, and capsule props."""

    def obs_spec(self) -> ObsSpec:
        if self.config.get("require_rgb"):
            return ObsSpec(rgb=True, depth=False, objects=True)
        if self.config.get("require_objects"):
            return ObsSpec(rgb=False, depth=False, objects=True)
        return ObsSpec(rgb=False, depth=False)

    def scene_spec(self) -> SceneSpec:
        return (
            SceneBuilder()
            .add_box(
                "showcase_box",
                size=(0.03, 0.03, 0.03),
                pos=(0.50, -0.15, 0.38),
                mass=0.05,
                rgba=(1.0, 0.1, 0.1, 1.0),
            )
            .add_cylinder(
                "showcase_cylinder",
                radius=0.025,
                height=0.04,
                pos=(0.55, 0.15, 0.38),
                fixed=True,
                rgba=(0.1, 0.3, 1.0, 1.0),
            )
            .add_sphere(
                "showcase_sphere",
                radius=0.03,
                pos=(0.62, -0.05, 0.40),
                mass=0.04,
                rgba=(1.0, 0.9, 0.1, 1.0),
            )
            .add_capsule(
                "showcase_capsule",
                radius=0.02,
                length=0.05,
                pos=(0.68, 0.10, 0.42),
                fixed=True,
                rgba=(1.0, 0.5, 0.0, 1.0),
            )
            .set_table_height(0.0)
            .set_lighting("minimal")
            .set_cameras("overview")
            .set_terrain("flat", size=(4.0, 4.0))
            .build_spec()
        )

    def language_instruction(self) -> str:
        return "Inspect the showcase objects."

    def reset_fn(self, backend) -> None:
        self._bind_backend(backend)
        self._apply_domain_randomization(backend)

    def reward_fn(self, obs: Observation, action: Action) -> float:
        del obs, action
        return 0.0

    def done_fn(self, obs: Observation) -> bool:
        del obs
        return False

    def success_fn(self, obs: Observation) -> bool:
        del obs
        return False
