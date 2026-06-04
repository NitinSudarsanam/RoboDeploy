"""Joint-space waypoint tracking policy for scripted demos."""

from __future__ import annotations

import numpy as np

from robodeploy.core.registry import register_policy
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.base import PolicyBase


@register_policy("example_joint_track")
class JointTrackPolicy(PolicyBase):
    """Move toward a target joint configuration with bounded per-step deltas."""

    def __init__(
        self,
        *,
        target_qpos: list[float] | None = None,
        home_qpos: list[float] | None = None,
        max_delta: float = 0.08,
    ) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS, config={"action_hz": 50.0})
        home = np.array(
            home_qpos if home_qpos is not None else [0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0],
            dtype=np.float32,
        )
        target = np.array(target_qpos if target_qpos is not None else home, dtype=np.float32)
        if home.shape != target.shape:
            raise ValueError("home_qpos and target_qpos must have the same length.")
        self._home = home
        self._target = target
        self._max_delta = float(max_delta)

    def _reset_impl(self) -> None:
        pass

    def get_action(self, obs: Observation) -> Action:
        q = np.asarray(obs.joint_positions, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._target.shape[0]:
            q = self._home.copy()
        err = self._target - q
        step = np.clip(err, -self._max_delta, self._max_delta)
        return Action(joint_positions=(q + step).astype(np.float32))
