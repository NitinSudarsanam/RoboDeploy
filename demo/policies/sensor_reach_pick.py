"""Sensor-driven reach policy for the demo — targets from ``obs.objects`` only."""

from __future__ import annotations

from robodeploy.core.registry import register_policy

from demo.policies.reach_pick_place import DemoReachPickPolicy


@register_policy("demo_sensor_reach_pick")
class DemoSensorReachPickPolicy(DemoReachPickPolicy):
    """Reach pick-place using prop poses from sensors in ``obs.objects``."""

    def __init__(self, *args, carry_mode: str = "follow", **kwargs) -> None:
        config = dict(kwargs.pop("config", None) or {})
        config.setdefault("carry_mode", carry_mode)
        super().__init__(*args, config=config, **kwargs)

    def bind_runtime(self, backend, description=None) -> None:
        super().bind_runtime(backend, description)

    def attach_mujoco(self, backend, description=None) -> None:
        self.bind_runtime(backend, description)
