from __future__ import annotations

from .base import ControllerConfig, register_controller


@register_controller("joint_effort_stub")
def _stub_joint_effort(cfg: ControllerConfig, backend_config: dict):
    raise NotImplementedError("joint_effort controller adapter not implemented yet.")

