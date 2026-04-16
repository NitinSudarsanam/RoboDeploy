"""
MuJoCoBackend — simulation backend stub.

This file defines the *architecture-compliant* MuJoCo backend class and
registration hook. Detailed MuJoCo/MJX implementation is intentionally
deferred; this is the structural migration only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robodeploy.backends.base import BackendBase
from robodeploy.core.registry import register_backend
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation

if TYPE_CHECKING:
    from robodeploy.core.interfaces.sensor import ISensor
    from robodeploy.core.interfaces.task import ITask
    from robodeploy.description.base import RobotDescription


@register_backend("mujoco")
class MuJoCoBackend(BackendBase):
    """MuJoCo simulation backend (stub)."""

    is_real = False
    control_hz = 100.0
    supported_action_spaces = [ActionSpace.JOINT_POS, ActionSpace.JOINT_TORQUE]

    def _load(
        self,
        description: RobotDescription,
        task: ITask,
        sensors: list[ISensor],
    ) -> None:
        raise NotImplementedError(
            "MuJoCoBackend implementation deferred. "
            "Port MJX/MuJoCo code here from the legacy engine."
        )

    def _reset_impl(self) -> Observation:
        raise NotImplementedError

    def _step_impl(self, action: Action) -> Observation:
        raise NotImplementedError

    def _get_obs_impl(self) -> Observation:
        raise NotImplementedError

    def _close_impl(self) -> None:
        # If a viewer or GPU contexts were opened, close them here.
        pass

