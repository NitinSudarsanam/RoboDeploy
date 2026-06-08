"""Real-hardware backends."""

__all__ = ["ROS2Backend"]


def __getattr__(name: str):
    if name == "ROS2Backend":
        from robodeploy.backends.real.ros2.backend import ROS2RealBackend as ROS2Backend

        return ROS2Backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
