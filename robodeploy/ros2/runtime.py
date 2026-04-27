"""Stable ROS2 runtime surface.

This is a thin wrapper over the existing ROS2 runtime implementation used by
the ROS2 backend. Keeping this in `robodeploy.ros2` makes it accessible to
robot-agnostic utilities and devtools without reaching into backend packages.
"""

from __future__ import annotations

from robodeploy.backends.real.ros2.runtime import Ros2Runtime

__all__ = ["Ros2Runtime"]

