"""PickPlaceTask — architecture-compliant manipulation task stub."""

from __future__ import annotations

import math

from robodeploy.core.registry import register_task
from robodeploy.core.types import Action, GeomSpec, MaterialSpec, ObsSpec, Observation, PropConfig, SceneSpec
from robodeploy.tasks.base import TaskBase


@register_task("pick_place")
class PickPlaceTask(TaskBase):
    """Pick-and-place task (minimal)."""

    def obs_spec(self) -> ObsSpec:
        # Minimal default: no vision until concrete sensor stack is added.
        return ObsSpec(rgb=False, depth=False)

    def scene_spec(self) -> SceneSpec:
        # Minimal scene: a "source" prop and a "target" prop marker.
        # Backends may ignore asset_path for now; RViz can still show markers.
        return SceneSpec(
            props=[
                PropConfig(
                    name="source",
                    position=(0.5, 0.0, 0.025),
                    is_fixed=False,
                    mass=0.05,
                    geom=GeomSpec(kind="box", size=(0.025, 0.025, 0.025)),
                    material=MaterialSpec(rgba=(1.0, 0.0, 0.0, 1.0)),
                ),
                PropConfig(
                    name="target",
                    position=(0.6, 0.2, 0.003),
                    is_fixed=True,
                    geom=GeomSpec(kind="box", size=(0.04, 0.04, 0.003)),
                    material=MaterialSpec(rgba=(0.0, 0.8, 0.0, 0.7)),
                ),
            ],
            table_height=0.0,
            lighting="default",
        )

    def language_instruction(self) -> str:
        return "Pick the object and place it at the target."

    def reset_fn(self, backend) -> None:
        self._bind_backend(backend)
        self._apply_domain_randomization(backend)

    def reward_fn(self, obs: Observation, action: Action) -> float:
        del action
        source_goal = self._placement_goal()
        source_pose = self.prop_pose("source")
        if source_pose is None:
            return -self._distance3(tuple(float(v) for v in obs.ee_position), source_goal)
        source_pos, _ = source_pose
        ee_dist = self._distance3(tuple(float(v) for v in obs.ee_position), source_pos)
        source_dist = self._distance3(source_pos, source_goal)
        lift_bonus = max(0.0, float(source_pos[2]) - float(self.scene_prop("source").position[2]))
        return -(0.35 * ee_dist + source_dist) + min(lift_bonus, 0.1)

    def success_fn(self, obs: Observation) -> bool:
        source_pose = self.prop_pose("source")
        if source_pose is None:
            return self._distance3(tuple(float(v) for v in obs.ee_position), self._placement_goal()) < 0.05
        source_pos, _ = source_pose
        return self._distance3(source_pos, self._placement_goal()) < 0.04

    def _placement_goal(self) -> tuple[float, float, float]:
        target = self.scene_prop("target")
        source = self.scene_prop("source")
        if target is None:
            return (0.6, 0.2, 0.02)
        z = float(target.position[2])
        if source is not None and source.geom is not None and source.geom.kind == "box" and len(source.geom.size) >= 3:
            z += float(source.geom.size[2])
        return (float(target.position[0]), float(target.position[1]), z)

    @staticmethod
    def _distance3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        dz = float(a[2]) - float(b[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

