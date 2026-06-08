"""Tier-1 reach_target benchmark task (dummy + sim presets)."""

from __future__ import annotations

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.registry import register_policy, register_task
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
from robodeploy.policies.base import PolicyBase
from robodeploy.tasks.base import TaskBase


@register_task("benchmark_reach_target")
class BenchmarkReachTargetTask(TaskBase):
    """Reach a fixed target pose; succeeds within joint-space tolerance."""

    def obs_spec(self) -> ObsSpec:
        return ObsSpec()

    def scene_spec(self) -> SceneSpec:
        return SceneSpec()

    def language_instruction(self) -> str:
        return "Reach the target position."

    def max_steps(self) -> int:
        return int(self.config.get("max_steps", 300))

    def reset_fn(self, backend) -> None:
        del backend

    def reward_fn(self, obs: Observation, action: Action) -> float:
        del action
        target = float(self.config.get("target_q", 0.4))
        err = abs(float(obs.joint_positions[0]) - target)
        return -err

    def success_fn(self, obs: Observation) -> bool:
        target = float(self.config.get("target_q", 0.4))
        tol = float(self.config.get("success_tol", 0.05))
        return abs(float(obs.joint_positions[0]) - target) <= tol

    def failure_fn(self, obs: Observation) -> bool:
        del obs
        return False


@register_policy("benchmark_reach_scripted")
class BenchmarkReachScriptedPolicy(PolicyBase):
    """Deterministic ramp toward the reach target on joint 0."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS, config=config)
        self._step = 0

    def _reset_impl(self, *, seed: int | None = None) -> None:
        del seed
        self._step = 0

    def get_action(self, obs: Observation) -> Action:
        del obs
        self._step += 1
        target = float(self.config.get("target_q", 0.4))
        rate = float(self.config.get("ramp_rate", 0.08))
        value = min(target, rate * self._step)
        return Action(joint_positions=jnp.asarray([value, 0.0], dtype=jnp.float32))
