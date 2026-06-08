"""Factory for teleop device implementations."""

from __future__ import annotations

from robodeploy.teleop.base import ITeleopDevice
from robodeploy.teleop.gamepad import GamepadTeleop
from robodeploy.teleop.keyboard import KeyboardTeleop
from robodeploy.teleop.mujoco_mouse import MuJoCoMouseIKTeleop
from robodeploy.teleop.ros2_bridge import Ros2JoyTeleop, Ros2TwistTeleop
from robodeploy.teleop.spacemouse import SpaceMouseTeleop

_DEVICE_ALIASES = {
    "keyboard": KeyboardTeleop,
    "kb": KeyboardTeleop,
    "spacemouse": SpaceMouseTeleop,
    "space_mouse": SpaceMouseTeleop,
    "3dconnexion": SpaceMouseTeleop,
    "gamepad": GamepadTeleop,
    "joystick": GamepadTeleop,
    "xbox": GamepadTeleop,
    "ps4": GamepadTeleop,
    "mujoco_mouse": MuJoCoMouseIKTeleop,
    "mouse_ik": MuJoCoMouseIKTeleop,
    "ros2_twist": Ros2TwistTeleop,
    "ros_twist": Ros2TwistTeleop,
    "cmd_vel": Ros2TwistTeleop,
    "ros2_joy": Ros2JoyTeleop,
    "ros_joy": Ros2JoyTeleop,
    "joy": Ros2JoyTeleop,
}


def make_teleop_device(name: str, **kwargs) -> ITeleopDevice:
    """Construct a registered teleop device by short name."""
    key = str(name).strip().lower()
    cls = _DEVICE_ALIASES.get(key)
    if cls is None:
        supported = ", ".join(sorted({k for k in _DEVICE_ALIASES if "_" not in k or k.endswith("_mouse")}))
        raise ValueError(f"Unknown teleop device {name!r}. Supported: {supported}.")
    return cls(**kwargs)
