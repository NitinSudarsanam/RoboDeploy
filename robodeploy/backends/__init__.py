"""Execution backends for simulation and real hardware."""

from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend

try:
    from robodeploy.backends.real.ros2.backend import ROS2RealBackend as ROS2Backend  # type: ignore
except ImportError:  # pragma: no cover
    ROS2Backend = None  # type: ignore[assignment]

__all__ = ["MuJoCoBackend", "ROS2Backend"]
