"""ROS2 controller-family adapters.

Design goal: avoid robot-specific Python.

We adapt *controller families* (joint_position, joint_trajectory, ...) rather than
individual robot models. Robot-specific defaults live in data-only presets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation


@dataclass(frozen=True)
class ControllerConfig:
    """Per-robot ROS2 wiring for a specific controller family."""

    robot_id: str
    namespace: str = ""  # e.g. "/robot0"

    # ROS topic names relative to namespace (unless absolute paths are used by caller).
    joint_states_topic: str = "joint_states"
    cmd_topic: str = "joint_position_commands"

    # Frames for TF lookup.
    base_frame: str = "base_link"
    ee_frame: str = "ee_link"

    # Optional canonical joint ordering; if omitted, adopt first JointState ordering.
    joint_names: Optional[list[str]] = None

    # Timing / reliability.
    joint_state_timeout_s: float = 1.0

    # Optional outgoing command pacing.
    command_hz: float = 0.0


@runtime_checkable
class IControllerAdapter(Protocol):
    """Controller-family adapter surface used by ROS2Backend."""

    controller_type: str
    supported_action_spaces: list[ActionSpace]

    def start(self) -> None: ...
    def stop(self) -> None: ...

    @property
    def robot_id(self) -> str: ...

    @property
    def base_frame(self) -> str: ...

    @property
    def ee_frame(self) -> str: ...

    @property
    def joint_names(self) -> list[str]: ...

    def get_obs(self) -> Observation: ...
    def send_action(self, action: Action) -> None: ...

    # Optional: block until a newer JointState is received (best-effort).
    def send_action_and_wait(self, action: Action) -> None: ...

    # Optional: best-effort diagnostics dict.
    def get_diagnostics(self) -> dict: ...


_CONTROLLERS: dict[str, Callable[[ControllerConfig, dict], IControllerAdapter]] = {}


def register_controller(controller_type: str):
    def decorator(factory: Callable[[ControllerConfig, dict], IControllerAdapter]):
        if controller_type in _CONTROLLERS:
            raise KeyError(f"ROS2 controller '{controller_type}' already registered.")
        _CONTROLLERS[controller_type] = factory
        return factory

    return decorator


def make_controller(controller_type: str, cfg: ControllerConfig, backend_config_dict: dict) -> IControllerAdapter:
    if controller_type not in _CONTROLLERS:
        raise KeyError(f"ROS2 controller '{controller_type}' not found. Registered: {list(_CONTROLLERS)}")
    return _CONTROLLERS[controller_type](cfg, backend_config_dict)

