"""Stacking task template."""

from __future__ import annotations

from abc import abstractmethod

from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
from robodeploy.tasks.base import TaskBase
from robodeploy.tasks.reward_builder import RewardBuilder


class StackingTemplate(TaskBase):
    cube_names: list[str] = ["cube_a", "cube_b"]
    base_name: str = "base"
    reward_weights: dict[str, float] = {"stack": 1.0}

    def obs_spec(self) -> ObsSpec:
        if self.config.get("require_objects"):
            return ObsSpec(rgb=False, depth=False, objects=True)
        return ObsSpec(rgb=False, depth=False)

    @abstractmethod
    def scene_spec(self) -> SceneSpec:
        ...

    def language_instruction(self) -> str:
        return f"Stack cubes on {self.base_name}."

    def reset_fn(self, backend) -> None:
        self._bind_backend(backend)

    def reward_fn(self, obs: Observation, action: Action) -> float:
        weights = {**self.reward_weights, **self.config.get("reward_weights", {})}
        builder = RewardBuilder().with_pose_resolver(self._pose_for_reward)
        if len(self.cube_names) >= 2:
            builder.distance(
                self.cube_names[0],
                self.cube_names[1],
                scale=weights.get("stack", 1.0),
                name="stack",
            )
        return builder.build()(obs, action)

    def success_fn(self, obs: Observation) -> bool:
        if len(self.cube_names) < 2:
            return False
        top_pose = self.object_pose(self.cube_names[0], obs)
        base_pose = self.object_pose(self.cube_names[1], obs)
        if top_pose is None or base_pose is None:
            return False
        top_pos, _ = top_pose
        base_pos, _ = base_pose
        xy = sum((float(top_pos[i]) - float(base_pos[i])) ** 2 for i in range(2)) ** 0.5
        z_delta = float(top_pos[2]) - float(base_pos[2])
        return xy < 0.03 and 0.02 < z_delta < 0.08

    def _pose_for_reward(self, key: str, obs: Observation):
        pose = self.object_pose(key, obs)
        if pose is None:
            prop = self.scene_prop(key)
            return tuple(prop.position) if prop else None
        pos, _ = pose
        return tuple(float(v) for v in pos)
