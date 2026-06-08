"""ShowcaseSceneTask — demo scene with every procedural geom kind."""

from __future__ import annotations

from robodeploy.core.registry import register_task
from robodeploy.core.types import (
    Action,
    CameraSpec,
    GeomSpec,
    LightSpec,
    MaterialSpec,
    ObsSpec,
    Observation,
    PropConfig,
    SceneSpec,
    TerrainSpec,
    WorldSpec,
)
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
        return SceneSpec(
            props=[
                PropConfig(
                    name="showcase_box",
                    position=(0.50, -0.15, 0.38),
                    is_fixed=False,
                    mass=0.05,
                    geom=GeomSpec(kind="box", size=(0.03, 0.03, 0.03)),
                    material=MaterialSpec(rgba=(1.0, 0.1, 0.1, 1.0)),
                ),
                PropConfig(
                    name="showcase_cylinder",
                    position=(0.55, 0.15, 0.38),
                    is_fixed=True,
                    geom=GeomSpec(kind="cylinder", size=(0.025, 0.04)),
                    material=MaterialSpec(rgba=(0.1, 0.3, 1.0, 1.0)),
                ),
                PropConfig(
                    name="showcase_sphere",
                    position=(0.62, -0.05, 0.40),
                    is_fixed=False,
                    mass=0.04,
                    geom=GeomSpec(kind="sphere", size=(0.03,)),
                    material=MaterialSpec(rgba=(1.0, 0.9, 0.1, 1.0)),
                ),
                PropConfig(
                    name="showcase_capsule",
                    position=(0.68, 0.10, 0.42),
                    is_fixed=True,
                    geom=GeomSpec(kind="capsule", size=(0.02, 0.05)),
                    material=MaterialSpec(rgba=(1.0, 0.5, 0.0, 1.0)),
                ),
            ],
            table_height=0.0,
            lighting="default",
            world=WorldSpec(
                lights=[
                    LightSpec(
                        position=(0.5, -0.5, 1.5),
                        direction=(0.0, 0.3, -1.0),
                        diffuse=(0.85, 0.85, 0.85),
                        kind="directional",
                    )
                ],
                cameras=[
                    CameraSpec(
                        name="scene_overview",
                        position=(0.0, -1.2, 0.9),
                        orientation=(1.0, 0.0, 0.0, 0.0),
                        fov_deg=70.0,
                        resolution=(320, 240),
                    )
                ],
                terrain=TerrainSpec(kind="flat", size=(4.0, 4.0)),
            ),
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
