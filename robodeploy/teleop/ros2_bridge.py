"""ROS2 Twist / Joy topic bridges for external teleop nodes."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from robodeploy.teleop.base import ITeleopDevice, TeleopCommand

_DEFAULT_JOY_AXES = {
    "x": 1,
    "y": 0,
    "z_up": 5,
    "z_down": 4,
    "yaw": 3,
    "pitch": 4,
}

_DEFAULT_JOY_BUTTONS = {
    "reset_episode": 0,
    "e_stop": 1,
    "record_toggle": 2,
}


class Ros2TwistTeleop(ITeleopDevice):
    """Subscribe to geometry_msgs/Twist (e.g. /cmd_vel) for EE deltas."""

    def __init__(
        self,
        *,
        topic: str = "/cmd_vel",
        scale_position: float = 0.01,
        scale_orientation: float = 0.05,
        node: Any | None = None,
    ) -> None:
        self._topic = str(topic)
        self._scale_position = float(scale_position)
        self._scale_orientation = float(scale_orientation)
        self._node = node
        self._alive = False
        self._lock = threading.Lock()
        self._linear = np.zeros(3, dtype=np.float32)
        self._angular = np.zeros(3, dtype=np.float32)

    def start(self) -> None:
        if self._alive:
            return
        self._alive = True
        if self._node is not None:
            return
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from rclpy.node import Node
        except ImportError as exc:
            raise ImportError(
                "Ros2TwistTeleop requires rclpy and geometry_msgs. "
                "Install ROS2 Python bindings for your distro."
            ) from exc

        teleop = self

        class _TwistNode(Node):
            def __init__(self) -> None:
                super().__init__("robodeploy_twist_teleop")
                self.create_subscription(Twist, teleop._topic, self._on_twist, 10)

            def _on_twist(self, msg: Twist) -> None:
                teleop.inject_twist(
                    linear=[msg.linear.x, msg.linear.y, msg.linear.z],
                    angular=[msg.angular.x, msg.angular.y, msg.angular.z],
                )

        if not rclpy.ok():
            rclpy.init()
        self._node = _TwistNode()

    def stop(self) -> None:
        self._alive = False
        node = self._node
        self._node = None
        if node is not None and hasattr(node, "destroy_node"):
            try:
                import rclpy

                node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    @property
    def is_alive(self) -> bool:
        return self._alive

    def inject_twist(
        self,
        *,
        linear: np.ndarray | list[float],
        angular: np.ndarray | list[float] | None = None,
    ) -> None:
        """Test helper: push a Twist message without ROS."""
        with self._lock:
            self._linear = np.asarray(linear, dtype=np.float32).reshape(3)
            if angular is not None:
                self._angular = np.asarray(angular, dtype=np.float32).reshape(3)

    def poll(self) -> TeleopCommand | None:
        with self._lock:
            linear = self._linear.copy()
            angular = self._angular.copy()
            self._linear.fill(0.0)
            self._angular.fill(0.0)

        delta_pos = linear * self._scale_position
        delta_rpy = angular * self._scale_orientation
        has_pos = bool(np.any(np.abs(delta_pos) > 1e-8))
        has_rpy = bool(np.any(np.abs(delta_rpy) > 1e-8))
        if not has_pos and not has_rpy:
            return None
        return TeleopCommand(
            delta_position=delta_pos.astype(np.float32) if has_pos else None,
            delta_orientation_rpy=delta_rpy.astype(np.float32) if has_rpy else None,
        )


class Ros2JoyTeleop(ITeleopDevice):
    """Subscribe to sensor_msgs/Joy for analog gamepad-style input."""

    def __init__(
        self,
        *,
        topic: str = "/joy",
        axes_map: dict[str, int] | None = None,
        button_map: dict[str, int] | None = None,
        deadzone: float = 0.1,
        scale_position: float = 0.005,
        scale_orientation: float = 0.05,
        node: Any | None = None,
    ) -> None:
        self._topic = str(topic)
        self._axes_map = dict(_DEFAULT_JOY_AXES)
        if axes_map:
            self._axes_map.update(axes_map)
        self._button_map = dict(_DEFAULT_JOY_BUTTONS)
        if button_map:
            self._button_map.update(button_map)
        self._deadzone = float(deadzone)
        self._scale_position = float(scale_position)
        self._scale_orientation = float(scale_orientation)
        self._node = node
        self._alive = False
        self._lock = threading.Lock()
        self._axes: list[float] = []
        self._buttons: list[int] = []
        self._prev_buttons: dict[int, bool] = {}
        self._edge: dict[str, bool] = {
            "record_toggle": False,
            "reset_episode": False,
            "e_stop": False,
        }

    def start(self) -> None:
        if self._alive:
            return
        self._alive = True
        if self._node is not None:
            return
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Joy
        except ImportError as exc:
            raise ImportError(
                "Ros2JoyTeleop requires rclpy and sensor_msgs. "
                "Install ROS2 Python bindings for your distro."
            ) from exc

        teleop = self

        class _JoyNode(Node):
            def __init__(self) -> None:
                super().__init__("robodeploy_joy_teleop")
                self.create_subscription(Joy, teleop._topic, self._on_joy, 10)

            def _on_joy(self, msg: Joy) -> None:
                teleop.inject_joy(axes=list(msg.axes), buttons=list(msg.buttons))

        if not rclpy.ok():
            rclpy.init()
        self._node = _JoyNode()

    def stop(self) -> None:
        self._alive = False
        node = self._node
        self._node = None
        if node is not None and hasattr(node, "destroy_node"):
            try:
                import rclpy

                node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    @property
    def is_alive(self) -> bool:
        return self._alive

    def inject_joy(self, *, axes: list[float], buttons: list[int] | None = None) -> None:
        """Test helper: push a Joy message without ROS."""
        with self._lock:
            self._axes = [float(v) for v in axes]
            if buttons is not None:
                self._buttons = [int(v) for v in buttons]

    def _axis(self, name: str) -> float:
        with self._lock:
            axes = list(self._axes)
        index = self._axes_map.get(name)
        if index is None or index >= len(axes):
            return 0.0
        value = float(axes[index])
        if abs(value) < self._deadzone:
            return 0.0
        sign = 1.0 if value >= 0.0 else -1.0
        return sign * (abs(value) - self._deadzone) / max(1e-6, 1.0 - self._deadzone)

    def _edge_button(self, name: str) -> bool:
        with self._lock:
            buttons = list(self._buttons)
        index = self._button_map.get(name)
        if index is None or index >= len(buttons):
            return False
        pressed = bool(buttons[index])
        prev = self._prev_buttons.get(index, False)
        self._prev_buttons[index] = pressed
        return pressed and not prev

    def poll(self) -> TeleopCommand | None:
        x = self._axis("x")
        y = self._axis("y")
        z = self._axis("z_up") - self._axis("z_down")
        yaw = self._axis("yaw")
        pitch = self._axis("pitch")

        delta_pos = np.array([x, y, z], dtype=np.float32) * self._scale_position
        delta_rpy = np.array([0.0, pitch, yaw], dtype=np.float32) * self._scale_orientation

        if self._edge_button("reset_episode"):
            self._edge["reset_episode"] = True
        if self._edge_button("e_stop"):
            self._edge["e_stop"] = True
        if self._edge_button("record_toggle"):
            self._edge["record_toggle"] = True
        edge = dict(self._edge)
        for flag in self._edge:
            self._edge[flag] = False

        has_pos = bool(np.any(np.abs(delta_pos) > 1e-8))
        has_rpy = bool(np.any(np.abs(delta_rpy) > 1e-8))
        has_hotkey = any(edge.values())
        if not has_pos and not has_rpy and not has_hotkey:
            return None

        return TeleopCommand(
            delta_position=delta_pos if has_pos else None,
            delta_orientation_rpy=delta_rpy if has_rpy else None,
            record_toggle=bool(edge.get("record_toggle")),
            reset_episode=bool(edge.get("reset_episode")),
            e_stop=bool(edge.get("e_stop")),
        )
