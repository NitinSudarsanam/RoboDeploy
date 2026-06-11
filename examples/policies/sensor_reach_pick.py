"""Sensor-driven reach policy — object poses from Observation.objects only."""

from __future__ import annotations

from robodeploy.core.registry import register_policy

from examples.policies.reach_pick_place import ReachPickPlacePolicy


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
        """Bind IK + backend carry adapters; never read prop poses from the backend."""
        super().bind_runtime(backend, description)

    def attach_mujoco(self, backend, description=None) -> None:
        self.bind_runtime(backend, description)
