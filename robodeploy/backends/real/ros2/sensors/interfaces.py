"""ROS2 sensor-stream interfaces.

These are used by ROS2Backend to ingest extra sensor streams (camera, wrench, etc.)
without tying ROS2Backend to concrete message types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from robodeploy.core.types import SensorData


@dataclass(frozen=True)
class Ros2SensorConfig:
    """Configuration for a ROS2 sensor stream (namespaced by ROS2Backend)."""

    robot_id: str
    name: str  # user-friendly name (e.g. "front_cam")
    namespace: str = ""  # e.g. "/robot0"

    # Implementation-specific topic mapping.
    topics: dict[str, str] = None  # e.g. {"rgb": "camera/color/image_raw", ...}

    # Optional generic knobs.
    qos_depth: int = 10
    frame_id: Optional[str] = None
    target_frame: Optional[str] = None
    encoding: Optional[str] = None

    def __post_init__(self) -> None:
        if self.topics is None:
            object.__setattr__(self, "topics", {})


@runtime_checkable
class IRos2Sensor(Protocol):
    """ROS2 sensor stream adapter."""

    sensor_type: str

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def read(self) -> SensorData: ...

    # Optional diagnostics payload.
    def get_diagnostics(self) -> dict[str, Any]: ...

