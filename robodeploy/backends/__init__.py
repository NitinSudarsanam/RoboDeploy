"""Execution backends for simulation and real hardware.

Imports are lazy: MujocoEngine requires JAX; FrankaRealBackend only requires
rclpy (ros2_env).  Neither is pulled in until explicitly accessed.
"""

__all__ = ["MujocoEngine", "FrankaRealBackend"]


def __getattr__(name: str):
    if name == "MujocoEngine":
        from .sim import MujocoEngine
        return MujocoEngine
    if name == "FrankaRealBackend":
        from .real import FrankaRealBackend
        return FrankaRealBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
