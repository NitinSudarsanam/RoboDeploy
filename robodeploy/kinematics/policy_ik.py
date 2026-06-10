"""Attach position IK solvers to policies (MuJoCo first, Pinocchio fallback)."""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)


def _try_mujoco_ik(policy, backend, description) -> bool:
    try:
        from robodeploy.kinematics.mujoco_ik import attach_mujoco_ik

        return attach_mujoco_ik(policy, backend, description) is not None
    except Exception:
        return False


def _try_pin_ik(policy, description) -> bool:
    try:
        from robodeploy.kinematics.pin_ik import attach_pin_ik

        return attach_pin_ik(policy, description) is not None
    except Exception:
        return False


def attach_policy_ik(policy, backend, description) -> None:
    """Bind MuJoCo IK when available; otherwise try Pinocchio URDF IK."""
    if backend is not None and hasattr(backend, "_model"):
        if _try_mujoco_ik(policy, backend, description):
            return
    if _try_pin_ik(policy, description):
        return
    _logger.warning(
        "No IK solver attached for %s; policy will use delta-home fallback.",
        type(policy).__name__,
    )
