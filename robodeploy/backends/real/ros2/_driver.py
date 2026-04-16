"""ROS 2 Jazzy hardware driver for the Franka Emika Panda arm.

Migrated from `robodeploy/robots/franka/real/ros2_driver.py` as part of the
backend architecture migration. This module is treated as a private helper
for `ROS2Backend` in `backends/real/ros2/backend.py`.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

# Franka arm joint names published by franka_ros2
_PANDA_ARM_JOINTS: list[str] = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
_PANDA_FINGER_JOINTS: list[str] = ["panda_finger_joint1", "panda_finger_joint2"]

_BASE_FRAME = "panda_link0"
_EE_FRAME = "panda_hand"

# Finger stroke in metres (each finger travels 0–0.04 m; combined = 0–0.08 m)
_FINGER_MAX_M: float = 0.04


class FrankaROS2Driver:
    """Low-level ROS 2 bridge for the Franka Panda arm.

    Runs an rclpy node in a daemon thread so the caller can use Python without
    blocking the ROS executor.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._config = config or {}
        self._base_frame: str = self._config.get("base_frame", _BASE_FRAME)
        self._ee_frame: str = self._config.get("ee_frame", _EE_FRAME)

        # Protected state (written by ROS callbacks, read by getters)
        self._lock = threading.Lock()
        self._joint_positions = np.zeros(7, dtype=np.float64)
        self._joint_velocities = np.zeros(7, dtype=np.float64)
        self._joint_torques = np.zeros(7, dtype=np.float64)
        self._gripper_opening_m = float(_FINGER_MAX_M)  # default: fully open
        self._has_joint_state = False

        self._node = None
        self._spin_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, timeout_s: float = 10.0) -> None:
        """Initialise rclpy, subscribe, and wait for the first joint-state message."""
        try:
            import rclpy
            import rclpy.node
            import tf2_ros
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Float64MultiArray
        except ImportError as exc:
            raise ImportError(
                "ROS 2 packages not found. Activate ros2_env:\n"
                "    conda activate ros2_env\n"
                "then ensure franka_ros2 is sourced."
            ) from exc

        rclpy.init()
        self._node = rclpy.node.Node("robodeploy_franka_driver")

        self._cmd_pub = self._node.create_publisher(
            Float64MultiArray,
            "/joint_group_impedance_controller/commands",
            10,
        )

        self._node.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            10,
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)

        self._spin_thread = threading.Thread(
            target=rclpy.spin,
            args=(self._node,),
            daemon=True,
            name="robodeploy_franka_ros2_spin",
        )
        self._spin_thread.start()

        # Block until first joint-state message or timeout
        deadline = time.monotonic() + timeout_s
        while not self._has_joint_state and time.monotonic() < deadline:
            time.sleep(0.01)

        if not self._has_joint_state:
            self.stop()
            raise RuntimeError(
                f"Timed out after {timeout_s}s waiting for /joint_states. "
                "Is franka_ros2 running and sourced?"
            )

    def stop(self) -> None:
        """Destroy the ROS 2 node and shut down rclpy."""
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        try:
            import rclpy

            rclpy.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ROS 2 callbacks (run in spin thread)
    # ------------------------------------------------------------------

    def _on_joint_state(self, msg) -> None:
        """Parse /joint_states and store arm joint + finger state."""
        name_to_idx = {n: i for i, n in enumerate(msg.name)}

        positions = np.zeros(7, dtype=np.float64)
        velocities = np.zeros(7, dtype=np.float64)
        torques = np.zeros(7, dtype=np.float64)

        for out_idx, joint_name in enumerate(_PANDA_ARM_JOINTS):
            src = name_to_idx.get(joint_name)
            if src is None:
                continue
            if msg.position:
                positions[out_idx] = msg.position[src]
            if msg.velocity:
                velocities[out_idx] = msg.velocity[src]
            if msg.effort:
                torques[out_idx] = msg.effort[src]

        # Finger opening: average of the two finger joints (each 0–0.04 m)
        finger_vals: list[float] = []
        for joint_name in _PANDA_FINGER_JOINTS:
            src = name_to_idx.get(joint_name)
            if src is not None and msg.position:
                finger_vals.append(float(msg.position[src]))
        gripper_m = float(np.mean(finger_vals)) if finger_vals else _FINGER_MAX_M

        with self._lock:
            self._joint_positions[:] = positions
            self._joint_velocities[:] = velocities
            self._joint_torques[:] = torques
            self._gripper_opening_m = gripper_m
            self._has_joint_state = True

    # ------------------------------------------------------------------
    # Public getters
    # ------------------------------------------------------------------

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (positions_rad, velocities_rad_s, torques_Nm) as copies."""
        with self._lock:
            return (
                self._joint_positions.copy(),
                self._joint_velocities.copy(),
                self._joint_torques.copy(),
            )

    def get_gripper_state(self) -> float:
        """Return normalized gripper openness: 0.0 = fully open, 1.0 = fully closed."""
        with self._lock:
            opening_m = self._gripper_opening_m
        return float(1.0 - opening_m / _FINGER_MAX_M)

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (position_m [3], quaternion [4] w,x,y,z) from TF.

        Falls back to zeros if the TF lookup fails (e.g. on startup).
        """
        try:
            import rclpy.time

            tf_stamped = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._ee_frame,
                rclpy.time.Time(),
            )
            t = tf_stamped.transform.translation
            r = tf_stamped.transform.rotation
            position = np.array([t.x, t.y, t.z], dtype=np.float64)
            quaternion = np.array([r.w, r.x, r.y, r.z], dtype=np.float64)
            return position, quaternion
        except Exception:
            return np.zeros(3, dtype=np.float64), np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

    # ------------------------------------------------------------------
    # Command sender
    # ------------------------------------------------------------------

    def send_joint_positions(self, positions_rad: np.ndarray) -> None:
        """Publish a 7-DOF joint position command to the impedance controller."""
        from std_msgs.msg import Float64MultiArray

        msg = Float64MultiArray()
        msg.data = positions_rad[:7].astype(np.float64).tolist()
        self._cmd_pub.publish(msg)

    def send_gripper_command(self, normalized_close: float) -> None:
        """Send a gripper command via the finger position controller."""
        try:
            from std_msgs.msg import Float64MultiArray

            finger_m = float(_FINGER_MAX_M * (1.0 - normalized_close))
            msg = Float64MultiArray()
            msg.data = [finger_m, finger_m]
            self._cmd_pub.publish(msg)
        except Exception:
            pass

