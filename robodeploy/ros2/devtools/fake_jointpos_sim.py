"""Fake ROS2 joint-position simulator (robot-agnostic).

This is a stable import path used by examples and internal backends.
Implementation currently reuses the ROS2 backend dev helper.
"""

from __future__ import annotations

from robodeploy.backends.real.ros2.dev.fake_joint_sim import (  # noqa: F401
    FakeJointPosSim,
    FakeJointPosSimConfig,
)

__all__ = ["FakeJointPosSim", "FakeJointPosSimConfig"]

