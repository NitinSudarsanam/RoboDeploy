"""Process-wide ROS2 runtime utilities (rclpy init + shared executor).

ROS 2 requires that `rclpy.init()` is called at most once per process (per context).
RoboDeploy composes multiple ROS-enabled components (controllers, sensors, RViz),
so they must share a single init+executor lifecycle.
"""

from __future__ import annotations

import threading
from typing import Optional


class Ros2Runtime:
    _lock = threading.Lock()
    _started = False
    _rclpy = None
    _executor = None
    _spin_thread: Optional[threading.Thread] = None
    _node_count = 0
    # When True, RoboDeploy-created rclpy nodes set ``use_sim_time`` (Gazebo + bridged /clock).
    use_sim_time: bool = False

    @classmethod
    def ensure_started(cls) -> None:
        with cls._lock:
            if cls._started:
                return

            try:
                import rclpy
                from rclpy.executors import MultiThreadedExecutor
            except ImportError as exc:
                raise ImportError(
                    "ROS 2 packages not found. Ensure you are in a ROS 2 Jazzy Python environment "
                    "with rclpy installed and your ROS setup sourced."
                ) from exc

            # Idempotent init: ok() is False before init, True after.
            if not rclpy.ok():
                rclpy.init()

            cls._rclpy = rclpy
            cls._executor = MultiThreadedExecutor()
            cls._spin_thread = threading.Thread(
                target=cls._executor.spin,
                daemon=True,
                name="robodeploy_ros2_executor",
            )
            cls._spin_thread.start()
            cls._started = True

    @classmethod
    def add_node(cls, node) -> None:
        cls.ensure_started()
        with cls._lock:
            if cls._executor is None:
                raise RuntimeError("Ros2Runtime executor not available.")
            cls._executor.add_node(node)
            cls._node_count += 1

    @classmethod
    def remove_node(cls, node) -> None:
        with cls._lock:
            try:
                if cls._executor is not None:
                    cls._executor.remove_node(node)
            except Exception:
                pass
            cls._node_count = max(0, int(cls._node_count) - 1)

    @classmethod
    def ros_graph_has_clock(cls, timeout_s: float = 0.5) -> bool:
        """Return True when ``/clock`` has at least one publisher (sim-time graph)."""
        cls.ensure_started()
        try:
            import rclpy.node
        except ImportError:
            return False

        probe = rclpy.node.Node("robodeploy_clock_probe")
        cls.add_node(probe)
        try:
            if timeout_s > 0:
                import time

                time.sleep(float(timeout_s))
            return int(probe.count_publishers("/clock")) > 0
        except Exception:
            return False
        finally:
            try:
                cls.remove_node(probe)
                probe.destroy_node()
            except Exception:
                pass

    @classmethod
    def shutdown(cls) -> None:
        with cls._lock:
            if not cls._started:
                return
            rclpy = cls._rclpy
            executor = cls._executor
            cls._executor = None
            cls._rclpy = None
            cls._started = False

        try:
            if executor is not None:
                executor.shutdown()
        except Exception:
            pass
        try:
            if rclpy is not None and rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

