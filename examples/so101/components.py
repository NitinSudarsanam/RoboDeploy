"""SO-101 demo task/policy (robot description lives in ``robodeploy.description.so101``)."""

from __future__ import annotations

import numpy as np

from robodeploy.core.registry import register_policy, register_task
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
from robodeploy.description.so101 import SO101Description
from robodeploy.policies.base import PolicyBase
from robodeploy.tasks.base import TaskBase


@register_task("so101_demo")
class SO101DemoTask(TaskBase):
    """Run for N steps (any backend)."""

    def __init__(self, max_steps: int = 2000) -> None:
        super().__init__(config={"max_steps": int(max_steps)})

    def obs_spec(self) -> ObsSpec:
        return ObsSpec(rgb=False, depth=False)

    def scene_spec(self) -> SceneSpec:
        return SceneSpec()

    def language_instruction(self) -> str:
        return "Exercise SO-101 joints with a slow sinusoid."

    def reset_fn(self, backend) -> None:
        return

    def reward_fn(self, obs: Observation, action: Action) -> float:
        return 0.0

    def success_fn(self, obs: Observation) -> bool:
        return False


@register_policy("so101_sinusoid")
class SO101SinusoidPolicy(PolicyBase):
    """Small-motion sinusoid around ``SO101Description.home_qpos``."""

    def __init__(
        self,
        amplitude: float = 0.15,
        frequency_hz: float = 0.15,
    ) -> None:
        super().__init__(
            action_space=ActionSpace.JOINT_POS,
            config={"action_hz": 50.0},
        )
        self._home = SO101Description().home_qpos.astype(np.float64)
        self._amp = float(amplitude)
        self._freq = float(frequency_hz)
        self._t = 0.0
        self._dt = 1.0 / 50.0
        self._mask = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)

    def _reset_impl(self) -> None:
        self._t = 0.0

    def get_action(self, obs: Observation) -> Action:
        del obs
        self._t += self._dt
        w = 2.0 * np.pi * self._freq
        delta = self._amp * np.sin(w * self._t) * self._mask
        q = (self._home + delta).astype(np.float32)
        return Action(joint_positions=q)
