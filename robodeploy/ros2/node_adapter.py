"""Ros2NodeAdapter base class (stable import path).

Controllers/sensors/devtools should import from `robodeploy.ros2` rather than
reaching into backend packages.
"""

from __future__ import annotations

from robodeploy.backends.real.ros2.adapters_base import Ros2NodeAdapter

__all__ = ["Ros2NodeAdapter"]
