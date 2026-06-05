"""Sensor-driven reach policy — object poses from Observation.objects only."""

from __future__ import annotations

import numpy as np

from robodeploy.core.registry import register_policy
from robodeploy.core.types import Observation

from examples.policies.reach_pick_place import ReachPickPlacePolicy, _Phase


@register_policy("example_sensor_reach_pick")
class SensorReachPickPlacePolicy(ReachPickPlacePolicy):
    """Reach pick-place using prop poses published by sensors into ``obs.objects``."""

    def __init__(self, *args, carry_mode: str = "follow", **kwargs) -> None:
        super().__init__(*args, carry_mode=carry_mode, **kwargs)

    def _update_targets_from_obs(self, obs: Observation) -> None:
        objects = getattr(obs, "objects", None) or {}
        if "source" not in objects or "target" not in objects:
            return
        src_pos, _ = objects["source"]
        tgt_pos, _ = objects["target"]
        self._set_ee_targets(
            np.array(src_pos, dtype=np.float32),
            np.array(tgt_pos, dtype=np.float32),
        )

    def attach_mujoco(self, backend, description=None) -> None:
        """Bind IK only; do not read prop poses directly from the backend."""
        desc = description or self._description
        if desc is None:
            return
        self._backend = backend
        from examples.policies.mujoco_ik import attach_mujoco_ik

        attach_mujoco_ik(self, backend, desc)

    def get_action(self, obs: Observation):
        self._update_targets_from_obs(obs)
        return super().get_action(obs)

    def _reset_impl(self) -> None:
        super()._reset_impl()
        self._phase = _Phase.SETTLE_HOME
