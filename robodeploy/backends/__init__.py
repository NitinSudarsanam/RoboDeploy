"""Execution backends for simulation and real hardware."""

from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
from robodeploy.backends.simulator import backend_for_simulator, merge_simulator_config


def __getattr__(name: str):
    if name in {"ROS2Backend", "ROS2RealBackend", "ROS2RvizBackend"}:
        try:
            from robodeploy.backends.real.ros2.backend import ROS2RealBackend, ROS2RvizBackend
        except ImportError as exc:  # pragma: no cover - depends on ROS2 install
            raise ImportError(
                "ROS2 backends require ROS2 Python packages such as rclpy. "
                "Source your ROS2 environment or install the ROS2 dependencies before importing this backend."
            ) from exc
        if name == "ROS2RvizBackend":
            return ROS2RvizBackend
        return ROS2RealBackend
    if name == "ROS2GazeboBackend":
        try:
            from robodeploy.backends.sim.gazebo import ROS2GazeboBackend
        except ImportError as exc:  # pragma: no cover - depends on ROS2 install
            raise ImportError(
                "ROS2GazeboBackend requires ROS2 and Gazebo bridge dependencies. "
                "Install/source ROS2 and ros_gz_bridge before importing this backend."
            ) from exc
        return ROS2GazeboBackend
    raise AttributeError(name)


__all__ = [
    "MuJoCoBackend",
    "ROS2Backend",
    "ROS2RealBackend",
    "ROS2RvizBackend",
    "ROS2GazeboBackend",
    "backend_for_simulator",
    "merge_simulator_config",
]
