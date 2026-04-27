from __future__ import annotations

from .base import ControllerConfig, register_controller


@register_controller("joint_velocity")
def _stub_joint_velocity(cfg: ControllerConfig, backend_config: dict):
    raise NotImplementedError("joint_velocity controller adapter not implemented yet.")

