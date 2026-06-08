"""Device-agnostic teleop command and input device interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class TeleopSafetyError(RuntimeError):
    """Raised when the operator triggers an e-stop or other critical halt."""


@dataclass
class TeleopCommand:
    """Device-agnostic operator input for one control tick."""

    delta_position: np.ndarray | None = None
    delta_orientation_rpy: np.ndarray | None = None
    delta_joint_positions: np.ndarray | None = None
    gripper_command: float | None = None
    button_pressed: dict[str, bool] = field(default_factory=dict)
    record_toggle: bool = False
    reset_episode: bool = False
    e_stop: bool = False

    def has_motion(self) -> bool:
        return (
            self.delta_position is not None
            or self.delta_orientation_rpy is not None
            or self.delta_joint_positions is not None
            or self.gripper_command is not None
        )


class ITeleopDevice(ABC):
    """Pollable operator input source (keyboard, gamepad, ROS topic, etc.)."""

    @abstractmethod
    def start(self) -> None:
        """Acquire device resources (listeners, ROS subscriptions, etc.)."""

    @abstractmethod
    def poll(self) -> TeleopCommand | None:
        """Return the latest command snapshot, or None when idle."""

    @abstractmethod
    def stop(self) -> None:
        """Release device resources."""

    @property
    def is_alive(self) -> bool:
        return True
