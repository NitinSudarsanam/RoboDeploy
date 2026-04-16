"""
ROS2Backend — real-hardware backend stub.

Defines the architecture-compliant ROS 2 backend and registration hook.
Detailed real-hardware I/O is intentionally deferred; this is the structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robodeploy.backends.base import BackendBase
from robodeploy.core.registry import register_backend
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation

from ._driver import FrankaROS2Driver

if TYPE_CHECKING:
    from robodeploy.core.interfaces.sensor import ISensor
    from robodeploy.core.interfaces.task import ITask
    from robodeploy.description.base import RobotDescription


@register_backend("ros2")
class ROS2Backend(BackendBase):
    """ROS 2 hardware backend (stub; Franka-focused for now)."""

    is_real = True
    control_hz = 100.0
    supported_action_spaces = [ActionSpace.JOINT_POS]

    def _load(
        self,
        description: RobotDescription,
        task: ITask,
        sensors: list[ISensor],
    ) -> None:
        # Minimal structural wiring: create driver but do not implement full I/O yet.
        self._driver = FrankaROS2Driver(config=self.config)
        raise NotImplementedError(
            "ROS2Backend implementation deferred. "
            "Wire FrankaROS2Driver into _get_obs_impl/_step_impl."
        )

    def _reset_impl(self) -> Observation:
        raise NotImplementedError

    def _step_impl(self, action: Action) -> Observation:
        raise NotImplementedError

    def _get_obs_impl(self) -> Observation:
        raise NotImplementedError

    def _close_impl(self) -> None:
        if hasattr(self, "_driver"):
            try:
                self._driver.stop()
            except Exception:
                pass

