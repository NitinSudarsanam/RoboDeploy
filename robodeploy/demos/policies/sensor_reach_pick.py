"""Sensor-driven reach policy — object poses from Observation.objects only."""

from __future__ import annotations

from robodeploy.core.registry import register_policy

from robodeploy.demos.policies.reach_pick_place import ReachPickPlacePolicy


@register_policy("example_sensor_reach_pick")
class SensorReachPickPlacePolicy(ReachPickPlacePolicy):
    """Reach pick-place using prop poses published by sensors into ``obs.objects``.

    Waypoints refresh from ``obs.objects`` each step via the base
    ``_update_targets_from_obs``; this subclass only changes the default carry
    mode and keeps runtime binding to IK attachment (no backend prop reads).
    """

    def __init__(self, *args, carry_mode: str = "follow", **kwargs) -> None:
        config = dict(kwargs.pop("config", None) or {})
        config.setdefault("carry_mode", carry_mode)
        super().__init__(*args, config=config, **kwargs)

    def bind_runtime(self, backend, description=None) -> None:
        """Bind IK only; do not read prop poses directly from the backend."""
        desc = description or self._description
        if desc is None:
            return
        self._backend = backend
        from robodeploy.kinematics.policy_ik import attach_policy_ik

        attach_policy_ik(self, backend, desc)

    def attach_mujoco(self, backend, description=None) -> None:
        self.bind_runtime(backend, description)
