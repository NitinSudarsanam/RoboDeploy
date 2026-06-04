"""PourTask — example manipulation task (not part of robodeploy core)."""

from __future__ import annotations

import math

from robodeploy.core.registry import register_task
from robodeploy.core.types import Action, GeomSpec, MaterialSpec, ObsSpec, Observation, PropConfig, SceneSpec
from robodeploy.tasks.base import TaskBase


@register_task("pour")
class PourTask(TaskBase):
    def obs_spec(self) -> ObsSpec:
        return ObsSpec(rgb=False, depth=False)

    def scene_spec(self) -> SceneSpec:
        return SceneSpec(
            props=[
                PropConfig(
                    name="cup_source",
                    position=(0.5, -0.15, 0.04),
                    is_fixed=False,
                    mass=0.08,
                    geom=GeomSpec(kind="cylinder", size=(0.035, 0.04)),
                    material=MaterialSpec(rgba=(0.2, 0.3, 1.0, 1.0)),
                ),
                PropConfig(
                    name="cup_target",
                    position=(0.55, 0.15, 0.04),
                    is_fixed=True,
                    geom=GeomSpec(kind="cylinder", size=(0.045, 0.04)),
                    material=MaterialSpec(rgba=(0.1, 0.8, 0.3, 0.8)),
                ),
            ]
        )

    def language_instruction(self) -> str:
        return "Pour from source into target."

    def reset_fn(self, backend) -> None:
        self._bind_backend(backend)

    def reward_fn(self, obs: Observation, action: Action) -> float:
        del action
        source_pose = self.prop_pose("cup_source")
        if source_pose is None:
            return -self._distance3(tuple(float(v) for v in obs.ee_position), self._pour_goal())
        source_pos, source_quat = source_pose
        target_dist = self._distance3(source_pos, self._pour_goal())
        ee_dist = self._distance3(tuple(float(v) for v in obs.ee_position), source_pos)
        tip_bonus = min(self._tilt_angle_rad(source_quat), 1.2)
        return -(0.3 * ee_dist + target_dist) + 0.1 * tip_bonus

    def success_fn(self, obs: Observation) -> bool:
        source_pose = self.prop_pose("cup_source")
        if source_pose is None:
            return self._distance3(tuple(float(v) for v in obs.ee_position), self._pour_goal()) < 0.05
        source_pos, source_quat = source_pose
        return (
            self._distance3(source_pos, self._pour_goal()) < 0.06
            and self._tilt_angle_rad(source_quat) > 0.5
        )

    def _pour_goal(self) -> tuple[float, float, float]:
        target = self.scene_prop("cup_target")
        if target is None:
            return (0.55, 0.15, 0.12)
        return (
            float(target.position[0]),
            float(target.position[1]),
            float(target.position[2]) + 0.08,
        )

    @staticmethod
    def _distance3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        dz = float(a[2]) - float(b[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _tilt_angle_rad(quat: tuple[float, float, float, float]) -> float:
        w = max(-1.0, min(1.0, abs(float(quat[0]))))
        return 2.0 * math.acos(w)
