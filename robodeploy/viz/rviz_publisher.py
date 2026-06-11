"""RViz publishing helpers (optional).

This module is ROS-only and must not be imported by core layers.

It lives outside the backend packages so both sim and real backends can compose
RViz publishing without cross-layer imports.
"""

from __future__ import annotations

import threading
import time
import warnings
from typing import Optional

from robodeploy.core.types import Observation, SceneSpec

from robodeploy.backends.real.ros2.runtime import Ros2Runtime


class RvizPublisher:
    def __init__(
        self,
        *,
        fixed_frame: str,
        publish_hz: float = 10.0,
        namespace: str = "/robodeploy",
        base_frame: str = "base_link",
    ) -> None:
        self._fixed_frame = fixed_frame
        self._publish_hz = float(publish_hz)
        self._ns = namespace.rstrip("/")
        self._base_frame = base_frame
        self._node = None
        self._spin_thread: Optional[threading.Thread] = None
        self._running = False

        self._scene_topic = f"{self._ns}/scene/markers"
        self._task_topic = f"{self._ns}/tasks/markers"

        # State for traces
        self._trace_by_robot: dict[str, list[tuple[float, float, float]]] = {}

    def start(self) -> None:
        try:
            import rclpy.node
            from geometry_msgs.msg import PoseStamped
            from visualization_msgs.msg import MarkerArray
        except ImportError as exc:
            raise ImportError("RViz publishing requires ROS2 Python packages (rclpy, visualization_msgs).") from exc

        Ros2Runtime.ensure_started()
        self._node = rclpy.node.Node("robodeploy_rviz_publisher")
        Ros2Runtime.add_node(self._node)
        self._MarkerArray = MarkerArray
        self._PoseStamped = PoseStamped

        # RViz default fixed frame is often "world"; connect it to the robot root so markers + RobotModel align.
        if self._fixed_frame == "world":
            try:
                from geometry_msgs.msg import TransformStamped
                from tf2_ros import StaticTransformBroadcaster

                br = StaticTransformBroadcaster(self._node)
                t = TransformStamped()
                t.header.stamp = self._node.get_clock().now().to_msg()
                t.header.frame_id = "world"
                t.child_frame_id = self._base_frame
                t.transform.translation.x = 0.0
                t.transform.translation.y = 0.0
                t.transform.translation.z = 0.0
                t.transform.rotation.w = 1.0
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                br.sendTransform(t)
            except Exception as exc:
                warnings.warn(f"Failed to publish static RViz transform: {exc}", RuntimeWarning, stacklevel=2)

        # Latch the scene topic: it is published once at backend init, and RViz
        # typically attaches afterwards. Transient-local delivers the last
        # MarkerArray to late subscribers (display must also request it).
        from rclpy.qos import DurabilityPolicy, QoSProfile

        scene_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._scene_pub = self._node.create_publisher(MarkerArray, self._scene_topic, scene_qos)
        self._task_pub = self._node.create_publisher(MarkerArray, self._task_topic, 10)
        self._last_scene = None
        self._last_scene_publish_s = 0.0
        self._ee_pubs: dict[str, object] = {}
        self._trace_pubs: dict[str, object] = {}
        self._running = True

    def close(self) -> None:
        self._running = False
        if self._node is not None:
            try:
                Ros2Runtime.remove_node(self._node)
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None

    def reset(self) -> None:
        self._trace_by_robot.clear()

    def publish_scene(self, scene: SceneSpec) -> None:
        if self._node is None:
            return
        self._last_scene = scene
        self._last_scene_publish_s = time.monotonic()
        from visualization_msgs.msg import Marker
        from visualization_msgs.msg import MarkerArray

        ma = MarkerArray()
        stamp = self._node.get_clock().now().to_msg()

        world = scene.to_world() if hasattr(scene, "to_world") else scene

        idx = 0
        for prop in getattr(world, "props", []) or []:
            m = Marker()
            m.header.frame_id = self._fixed_frame
            m.header.stamp = stamp
            m.ns = "scene"
            m.id = idx
            idx += 1
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = prop.position
            qw, qx, qy, qz = prop.orientation
            m.pose.orientation.w = qw
            m.pose.orientation.x = qx
            m.pose.orientation.y = qy
            m.pose.orientation.z = qz
            geom = getattr(prop, "geom", None)
            size = tuple(getattr(geom, "size", ()) or ())
            if getattr(geom, "kind", "") == "sphere":
                m.type = Marker.SPHERE
            elif getattr(geom, "kind", "") == "cylinder":
                m.type = Marker.CYLINDER
            if len(size) >= 3:
                m.scale.x = float(size[0]) * 2.0
                m.scale.y = float(size[1]) * 2.0
                m.scale.z = float(size[2]) * 2.0
            elif len(size) >= 1:
                m.scale.x = m.scale.y = m.scale.z = float(size[0]) * 2.0
            else:
                m.scale.x = m.scale.y = m.scale.z = 0.05
            rgba = tuple(getattr(getattr(prop, "material", None), "rgba", (0.2, 0.8, 0.2, 0.8)))
            m.color.r = float(rgba[0])
            m.color.g = float(rgba[1])
            m.color.b = float(rgba[2])
            m.color.a = float(rgba[3])
            ma.markers.append(m)

        # Back-compat: objects
        for obj in getattr(scene, "objects", []) or []:
            m = Marker()
            m.header.frame_id = self._fixed_frame
            m.header.stamp = stamp
            m.ns = "scene_obj"
            m.id = idx
            idx += 1
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = obj.position
            qw, qx, qy, qz = obj.orientation
            m.pose.orientation.w = qw
            m.pose.orientation.x = qx
            m.pose.orientation.y = qy
            m.pose.orientation.z = qz
            m.scale.x = 0.05
            m.scale.y = 0.05
            m.scale.z = 0.05
            m.color.r = 0.2
            m.color.g = 0.2
            m.color.b = 0.8
            m.color.a = 0.8
            ma.markers.append(m)

        self._scene_pub.publish(ma)

    def publish_robot_state(self, robot_id: str, obs: Observation) -> None:
        if self._node is None:
            return

        # Per-robot EE pose topic: /robodeploy/<robot_id>/ee_pose
        topic = f"{self._ns}/{robot_id}/ee_pose"
        if robot_id not in self._ee_pubs:
            self._ee_pubs[robot_id] = self._node.create_publisher(self._PoseStamped, topic, 10)
        ee_pub = self._ee_pubs[robot_id]

        pose = self._PoseStamped()
        pose.header.frame_id = self._fixed_frame
        pose.header.stamp = self._node.get_clock().now().to_msg()
        pose.pose.position.x = float(obs.ee_position[0])
        pose.pose.position.y = float(obs.ee_position[1])
        pose.pose.position.z = float(obs.ee_position[2])
        pose.pose.orientation.w = float(obs.ee_orientation[0])
        pose.pose.orientation.x = float(obs.ee_orientation[1])
        pose.pose.orientation.y = float(obs.ee_orientation[2])
        pose.pose.orientation.z = float(obs.ee_orientation[3])
        ee_pub.publish(pose)

        # Track trace history
        self._trace_by_robot.setdefault(robot_id, []).append(
            (float(obs.ee_position[0]), float(obs.ee_position[1]), float(obs.ee_position[2]))
        )
        if len(self._trace_by_robot[robot_id]) > 2000:
            self._trace_by_robot[robot_id] = self._trace_by_robot[robot_id][-2000:]

        # Publish line-strip trace as Marker
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker

        trace_topic = f"{self._ns}/{robot_id}/trace"
        if robot_id not in self._trace_pubs:
            self._trace_pubs[robot_id] = self._node.create_publisher(Marker, trace_topic, 10)
        trace_pub = self._trace_pubs[robot_id]

        pts = self._trace_by_robot[robot_id]
        m = Marker()
        m.header.frame_id = self._fixed_frame
        m.header.stamp = self._node.get_clock().now().to_msg()
        m.ns = f"trace_{robot_id}"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.01
        m.color.r = 0.1
        m.color.g = 0.8
        m.color.b = 0.9
        m.color.a = 0.9
        for x, y, z in pts:
            p = Point()
            p.x, p.y, p.z = x, y, z
            m.points.append(p)
        trace_pub.publish(m)

    def publish_task_viz(self, payload: dict) -> None:
        """Publish task-goal markers from env-provided payload."""
        if self._node is None:
            return
        # Fallback for subscribers whose QoS still misses the latched scene
        # message: re-publish the cached scene at ~1 Hz off the step loop.
        if self._last_scene is not None and time.monotonic() - self._last_scene_publish_s >= 1.0:
            self.publish_scene(self._last_scene)
        from visualization_msgs.msg import Marker
        from visualization_msgs.msg import MarkerArray

        ma = MarkerArray()
        stamp = self._node.get_clock().now().to_msg()

        idx = 0
        tasks = (payload or {}).get("tasks", {}) if isinstance(payload, dict) else {}
        for task_id, task_payload in tasks.items():
            # Back-compat: flat list under task_id
            if isinstance(task_payload, list):
                items_by_robot = {"": task_payload}
            else:
                per_robot = task_payload.get("per_robot", {}) if isinstance(task_payload, dict) else {}
                items_by_robot = per_robot if isinstance(per_robot, dict) else {}

            for robot_id, items in items_by_robot.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    kind = item.get("kind")
                    m = Marker()
                    m.header.frame_id = self._fixed_frame
                    m.header.stamp = stamp
                    m.ns = f"task_{task_id}_{robot_id}" if robot_id else f"task_{task_id}"
                    m.id = idx
                    idx += 1
                    m.action = Marker.ADD

                    # Position / orientation
                    pos = item.get("position", [0, 0, 0])
                    quat = item.get("orientation", [1, 0, 0, 0])
                    m.pose.position.x = float(pos[0])
                    m.pose.position.y = float(pos[1])
                    m.pose.position.z = float(pos[2])
                    m.pose.orientation.w = float(quat[0])
                    m.pose.orientation.x = float(quat[1])
                    m.pose.orientation.y = float(quat[2])
                    m.pose.orientation.z = float(quat[3])

                    # Default sizes and colors
                    m.scale.x = float(item.get("scale_x", 0.2))
                    m.scale.y = float(item.get("scale_y", 0.02))
                    m.scale.z = float(item.get("scale_z", 0.02))
                    m.color.r = float(item.get("color_r", 0.95))
                    m.color.g = float(item.get("color_g", 0.6))
                    m.color.b = float(item.get("color_b", 0.1))
                    m.color.a = float(item.get("color_a", 0.9))

                    if kind == "pose":
                        m.type = Marker.ARROW
                    elif kind == "point":
                        m.type = Marker.SPHERE
                    elif kind == "bbox":
                        m.type = Marker.CUBE
                    else:
                        continue

                    ma.markers.append(m)

        self._task_pub.publish(ma)

