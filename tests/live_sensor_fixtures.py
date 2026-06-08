"""Shared ROS2 publishers for live sensor CI tests."""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class LiveRos2SensorPublishers:
    """Minimal wrist camera + FT + joint_states + TF publishers for ros2_rviz tests."""

    def __init__(self, *, robot_id: str = "robot0") -> None:
        import rclpy
        from geometry_msgs.msg import TransformStamped, WrenchStamped
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from std_msgs.msg import Header
        from tf2_ros import StaticTransformBroadcaster

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._node = Node("robodeploy_live_sensor_test")
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

        self._JointState = JointState
        self._Image = Image
        self._CameraInfo = CameraInfo
        self._WrenchStamped = WrenchStamped
        self._Header = Header

        js_topic = f"/{robot_id}/joint_states"
        self._js_pub = self._node.create_publisher(JointState, js_topic, 10)
        self._img_pub = self._node.create_publisher(Image, "/wrist_camera/image_raw", 10)
        self._info_pub = self._node.create_publisher(CameraInfo, "/wrist_camera/camera_info", 10)
        self._wrench_pub = self._node.create_publisher(WrenchStamped, "/wrist_ft/wrench", 10)

        self._tf_broadcaster = StaticTransformBroadcaster(self._node)
        tf = TransformStamped()
        tf.header.frame_id = "world"
        tf.child_frame_id = "wrist_camera"
        tf.transform.translation.x = 0.4
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.6
        tf.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(tf)

        self._joint_names = [f"joint{i}" for i in range(1, 8)]
        self._timer = self._node.create_timer(0.05, self._publish_all)

    def _publish_all(self) -> None:
        now = self._node.get_clock().now().to_msg()
        js = self._JointState()
        js.header.stamp = now
        js.header.frame_id = ""
        js.name = list(self._joint_names)
        js.position = [0.0] * len(self._joint_names)
        js.velocity = [0.0] * len(self._joint_names)
        js.effort = [0.0] * len(self._joint_names)
        self._js_pub.publish(js)

        img = self._Image()
        img.header.stamp = now
        img.header.frame_id = "wrist_camera"
        img.height = 48
        img.width = 64
        img.encoding = "rgb8"
        img.step = 64 * 3
        img.data = list(np.zeros((48, 64, 3), dtype=np.uint8).reshape(-1))
        self._img_pub.publish(img)

        info = self._CameraInfo()
        info.header.stamp = now
        info.header.frame_id = "wrist_camera"
        info.height = 48
        info.width = 64
        info.k = [64.0, 0.0, 32.0, 0.0, 48.0, 24.0, 0.0, 0.0, 1.0]
        self._info_pub.publish(info)

        wrench = self._WrenchStamped()
        wrench.header.stamp = now
        wrench.header.frame_id = "wrist_ft"
        wrench.wrench.force.x = 1.0
        wrench.wrench.force.y = 0.5
        wrench.wrench.force.z = -0.2
        wrench.wrench.torque.x = 0.1
        wrench.wrench.torque.y = 0.0
        wrench.wrench.torque.z = 0.0
        self._wrench_pub.publish(wrench)

    def spin_until(self, predicate, *, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        return False

    def close(self) -> None:
        self._timer.cancel()
        self._executor.shutdown()
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


def rclpy_available() -> bool:
    try:
        import rclpy  # noqa: F401

        return True
    except Exception:
        return False


def gazebo_binary_available() -> bool:
    import shutil

    return shutil.which("gz") is not None
