"""Deprecated: import from ``robodeploy.kinematics.pin_ik`` instead."""

from __future__ import annotations

import warnings

from robodeploy.kinematics.pin_ik import PinIkSolver, attach_pin_ik

warnings.warn(
    "examples.policies.pin_ik is deprecated; use robodeploy.kinematics.pin_ik",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PinIkSolver", "attach_pin_ik"]
