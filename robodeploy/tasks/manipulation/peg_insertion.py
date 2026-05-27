"""PegTask placeholder matching architecture layout."""

from __future__ import annotations

import math

from robodeploy.core.registry import register_task
from robodeploy.core.types import Action, GeomSpec, MaterialSpec, ObsSpec, Observation, PropConfig, SceneSpec
from robodeploy.tasks.base import TaskBase


@register_task("peg_insertion")
class PegTask(TaskBase):
    def obs_spec(self) -> ObsSpec:
        return ObsSpec(rgb=False, depth=False)

    def scene_spec(self) -> SceneSpec:
        return SceneSpec(
            props=[
                PropConfig(
                    name="peg",
                    position=(0.5, 0.0, 0.06),
                    is_fixed=False,
                    mass=0.05,
                    geom=GeomSpec(kind="cylinder", size=(0.012, 0.06)),
                    material=MaterialSpec(rgba=(0.9, 0.7, 0.2, 1.0)),
                ),
                PropConfig(
                    name="hole",
                    position=(0.6, 0.0, 0.006),
                    is_fixed=True,
                    geom=GeomSpec(kind="box", size=(0.05, 0.05, 0.006)),
                    material=MaterialSpec(rgba=(0.1, 0.1, 0.1, 1.0)),
                ),
            ]
        )

    def language_instruction(self) -> str:
        return "Insert peg into hole."

    def reset_fn(self, backend) -> None:
        self._bind_backend(backend)
        return

    def reward_fn(self, obs: Observation, action: Action) -> float:
        del action
        peg_pose = self.prop_pose("peg")
        if peg_pose is None:
            return -self._distance3(tuple(float(v) for v in obs.ee_position), self._insert_goal())
        peg_pos, _ = peg_pose
        ee_dist = self._distance3(tuple(float(v) for v in obs.ee_position), peg_pos)
        peg_dist = self._distance3(peg_pos, self._insert_goal())
        insertion_bonus = max(0.0, float(self.scene_prop("peg").position[2]) - float(peg_pos[2]))
        return -(0.3 * ee_dist + peg_dist) + min(insertion_bonus, 0.08)

    def success_fn(self, obs: Observation) -> bool:
        peg_pose = self.prop_pose("peg")
        if peg_pose is None:
            return self._distance3(tuple(float(v) for v in obs.ee_position), self._insert_goal()) < 0.03
        peg_pos, _ = peg_pose
        return self._distance3(peg_pos, self._insert_goal()) < 0.025

    def _insert_goal(self) -> tuple[float, float, float]:
        hole = self.scene_prop("hole")
        peg = self.scene_prop("peg")
        if hole is None:
            return (0.6, 0.0, 0.03)
        z = float(hole.position[2]) + 0.02
        if peg is not None and peg.geom is not None and peg.geom.kind == "cylinder" and len(peg.geom.size) > 1:
            z = float(hole.position[2]) + float(peg.geom.size[1]) * 0.5
        return (float(hole.position[0]), float(hole.position[1]), z)

    @staticmethod
    def _distance3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        dz = float(a[2]) - float(b[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

