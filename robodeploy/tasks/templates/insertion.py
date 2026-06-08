"""Peg insertion task template."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
from robodeploy.tasks.base import TaskBase
from robodeploy.tasks.choreography import TaskChoreography
from robodeploy.tasks.reward_builder import RewardBuilder


class InsertionTemplate(TaskBase):
    peg_name: str = "peg"
    hole_name: str = "hole"
    reward_weights: dict[str, float] = {"ee_to_peg": 0.35, "peg_to_hole": 1.0}
    choreography_path: str | None = None

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config)
        self._choreography = self._load_choreography()

    def obs_spec(self) -> ObsSpec:
        if self.config.get("require_objects"):
            return ObsSpec(rgb=False, depth=False, objects=True)
        return ObsSpec(rgb=False, depth=False)

    @abstractmethod
    def scene_spec(self) -> SceneSpec:
        ...

    def language_instruction(self) -> str:
        return f"Insert {self.peg_name} into {self.hole_name}."

    def reset_fn(self, backend) -> None:
        self._bind_backend(backend)
        if self._choreography is not None:
            self._choreography.reset()

    def reward_fn(self, obs: Observation, action: Action) -> float:
        weights = {**self.reward_weights, **self.config.get("reward_weights", {})}
        goal = self._hole_goal()
        builder = RewardBuilder().with_pose_resolver(self._pose_for_reward)
        builder.distance("ee", self.peg_name, scale=weights.get("ee_to_peg", 0.35), name="ee_to_peg")
        builder.distance_to_point(self.peg_name, goal, scale=weights.get("peg_to_hole", 1.0), name="peg_to_hole")
        reward = builder.build()(obs, action)
        if self._choreography is not None:
            reward *= self._choreography.phase_reward_scale()
        return reward

    def success_fn(self, obs: Observation) -> bool:
        if self._choreography is not None:
            self._choreography.advance(obs, pose_resolver=self._pose_for_reward)
            return self._choreography.complete
        pose = self.object_pose(self.peg_name, obs)
        if pose is None:
            return False
        pos, _ = pose
        goal = self._hole_goal()
        return self._distance3(tuple(float(v) for v in pos), goal) < 0.03

    def _load_choreography(self) -> TaskChoreography | None:
        path = self.config.get("choreography_path") or self.choreography_path
        data = self.config.get("choreography")
        if data:
            return TaskChoreography.from_dict(data)
        if path:
            return TaskChoreography.from_yaml(Path(path))
        return None

    def _hole_goal(self) -> tuple[float, float, float]:
        hole = self.scene_prop(self.hole_name)
        if hole is None:
            return (0.55, 0.0, 0.38)
        return tuple(float(v) for v in hole.position)

    def _insert_goal(self) -> tuple[float, float, float]:
        """Alias for tests and legacy callers."""
        return self._hole_goal()

    def _pose_for_reward(self, key: str, obs: Observation):
        if key == "ee":
            return tuple(float(v) for v in obs.ee_position)
        pose = self.object_pose(key, obs)
        if pose is None:
            prop = self.scene_prop(key)
            return tuple(prop.position) if prop else None
        pos, _ = pose
        return tuple(float(v) for v in pos)

    @staticmethod
    def _distance3(a, b) -> float:  # noqa: ANN001
        import math

        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
