"""Optional ROS2/RViz publishing bridge for MuJoCoBackend.

This module must only be imported when RViz is enabled, so MuJoCoBackend remains
usable without ROS installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from robodeploy.core.types import Observation, SceneSpec


@dataclass(frozen=True)
class MujocoRos2BridgeConfig:
    fixed_frame: str = "world"
    publish_hz: float = 10.0
    namespace: str = "/robodeploy"
    base_frame: str = "base_link"


class MujocoRos2Bridge:
    def __init__(self, cfg: MujocoRos2BridgeConfig) -> None:
        from robodeploy.viz.rviz_publisher import RvizPublisher

        self._rviz = RvizPublisher(
            fixed_frame=cfg.fixed_frame,
            publish_hz=cfg.publish_hz,
            namespace=cfg.namespace,
            base_frame=cfg.base_frame,
        )

    def start(self) -> None:
        self._rviz.start()

    def close(self) -> None:
        self._rviz.close()

    def reset(self) -> None:
        self._rviz.reset()

    def publish_scene(self, scene: SceneSpec) -> None:
        self._rviz.publish_scene(scene)

    def publish_robot_state(self, robot_id: str, obs: Observation) -> None:
        self._rviz.publish_robot_state(robot_id, obs)

    def publish_task_viz(self, payload: Optional[dict]) -> None:
        self._rviz.publish_task_viz(payload)

