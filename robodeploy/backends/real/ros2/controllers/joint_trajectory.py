"""ROS2 joint-trajectory controller adapter (generic).

Preferred controller family for ros2_control in most setups. Initial impl is
minimal: publish JointTrajectory with a single point per command.
"""

from __future__ import annotations

import numpy as np

from robodeploy.core.spaces import ActionSpace

from .base import ControllerConfig, register_controller
from .joint_position import JointPositionControllerAdapter


class JointTrajectoryControllerAdapter(JointPositionControllerAdapter):
    """Implements `joint_trajectory` commands; shares JointState + TF logic."""

    controller_type = "joint_trajectory"
    supported_action_spaces = [ActionSpace.JOINT_POS]

    def __init__(self, cfg: ControllerConfig, backend_config=None) -> None:
        super().__init__(cfg, backend_config)
        self.node_name = f"robodeploy_jointtraj_{self._robot_id}"

    def _on_node_ready(self, node) -> None:
        try:
            import tf2_ros
            from sensor_msgs.msg import JointState
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        except Exception as exc:
            raise ImportError(
                "trajectory_msgs / tf2_ros not available. Install ROS 2 message packages."
            ) from exc

        self._JointTrajectory = JointTrajectory
        self._JointTrajectoryPoint = JointTrajectoryPoint
        self._cmd_msg_type = JointTrajectory

        self._cmd_pub = node.create_publisher(
            JointTrajectory,
            self._resolved_cmd_topic,
            10,
        )

        node.create_subscription(
            JointState,
            self._resolved_joint_states_topic,
            self._on_joint_state,
            10,
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

        self._wait_for_joint_state(timeout_s=10.0)

    def _publish_joint_positions(self, positions_rad: np.ndarray) -> None:
        msg = self._JointTrajectory()
        msg.joint_names = self.joint_names
        pt = self._JointTrajectoryPoint()
        pt.positions = [float(x) for x in np.asarray(positions_rad, dtype=np.float64).reshape(-1).tolist()]
        try:
            from builtin_interfaces.msg import Duration

            d = Duration()
            d.sec = 0
            d.nanosec = int(200_000_000)  # 0.2s
            pt.time_from_start = d
        except Exception:
            pass
        msg.points = [pt]
        self._cmd_pub.publish(msg)


@register_controller("joint_trajectory")
def create_joint_trajectory_adapter(cfg: ControllerConfig, backend_config: dict):
    return JointTrajectoryControllerAdapter(cfg, backend_config)
