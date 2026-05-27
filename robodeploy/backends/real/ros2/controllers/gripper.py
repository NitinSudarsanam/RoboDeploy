from __future__ import annotations

from .base import ControllerConfig, register_controller


@register_controller("gripper_stub")
def _stub_gripper(cfg: ControllerConfig, backend_config: dict):
    raise NotImplementedError("gripper controller adapter not implemented yet.")

