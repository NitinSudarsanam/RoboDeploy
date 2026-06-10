"""Pick-and-place task template."""

from __future__ import annotations

from abc import abstractmethod

from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
from robodeploy.tasks.base import TaskBase
from robodeploy.tasks.reward_builder import RewardBuilder


class PickPlaceTemplate(TaskBase):
    """Shared pick-place reward/success; subclass sets names + scene_spec()."""

    source_name: str = "source"
    target_name: str = "target"
    reward_weights: dict[str, float] = {
        "ee_to_source": 0.35,
        "source_to_target": 1.0,
        "lift_bonus": 0.1,
    }
    success_threshold: float = 0.04

    def obs_spec(self) -> ObsSpec:
        spec = ObsSpec(rgb=False, depth=False)
        if self.config.get("require_objects"):
            spec = ObsSpec(rgb=False, depth=False, objects=True)
        if self.config.get("grasp_success_force_min"):
            spec = ObsSpec(
                rgb=spec.rgb,
                depth=spec.depth,
                objects=spec.objects,
                ft_sensor=True,
            )
        return spec

    @abstractmethod
    def scene_spec(self) -> SceneSpec:
        ...

    def language_instruction(self) -> str:
        return f"Pick {self.source_name} and place it at {self.target_name}."

    def reset_fn(self, backend) -> None:
        self._bind_backend(backend)
        self._apply_domain_randomization(backend)

    def reward_fn(self, obs: Observation, action: Action) -> float:
        weights = {**self.reward_weights, **self.config.get("reward_weights", {})}
        builder = RewardBuilder().with_pose_resolver(self._pose_for_reward)
        builder.distance(
            "ee",
            self.source_name,
            scale=weights.get("ee_to_source", 0.35),
            name="ee_to_source",
        )
        builder.distance_to_point(
            self.source_name,
            self._placement_goal(),
            scale=weights.get("source_to_target", 1.0),
            name="source_to_target",
        )
        source = self.scene_prop(self.source_name)
        initial_z = float(source.position[2]) if source else 0.38
        builder.bonus_lift(
            self.source_name,
            initial_z=initial_z,
            max_bonus=weights.get("lift_bonus", 0.1),
        )
        return builder.build()(obs, action)

    def failure_fn(self, obs: Observation) -> bool:
        """Fail when FT force exceeds ``failure_force_max_N`` (collision guard)."""
        limit = self.config.get("failure_force_max_N")
        if limit is None:
            return False
        force = obs.ft_force
        if force is None and getattr(obs, "ft_forces", None):
            forces = obs.ft_forces
            if forces:
                force = next(iter(forces.values()))
        if force is None:
            return False
        mag = float(sum(float(v) ** 2 for v in force) ** 0.5)
        return mag > float(limit)

    def success_fn(self, obs: Observation) -> bool:
        if not self.grasp_confirmed(obs):
            return False
        goal = self._placement_goal()
        source_pose = self.object_pose(self.source_name, obs)
        threshold = float(self.config.get("success_threshold", self.success_threshold))
        if source_pose is None:
            ee = tuple(float(v) for v in obs.ee_position)
            return self._distance3(ee, goal) < 0.05
        pos, _ = source_pose
        return self._distance3(tuple(float(v) for v in pos), goal) < threshold

    def _pose_for_reward(self, key: str, obs: Observation):
        if key == "ee":
            return tuple(float(v) for v in obs.ee_position)
        if key == "placement_goal":
            return self._placement_goal()
        pose = self.object_pose(key, obs)
        if pose is None:
            prop = self.scene_prop(key)
            if prop is not None:
                return tuple(prop.position)
            return None
        pos, _ = pose
        return tuple(float(v) for v in pos)

    def _placement_goal(self) -> tuple[float, float, float]:
        target = self.scene_prop(self.target_name)
        source = self.scene_prop(self.source_name)
        if target is None:
            return (0.6, 0.2, 0.41)
        z = float(target.position[2])
        if source is not None and source.geom is not None and source.geom.kind == "box":
            if len(source.geom.size) >= 3:
                z += float(source.geom.size[2])
        return (float(target.position[0]), float(target.position[1]), z)

    @staticmethod
    def _distance3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        import math

        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        dz = float(a[2]) - float(b[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)
