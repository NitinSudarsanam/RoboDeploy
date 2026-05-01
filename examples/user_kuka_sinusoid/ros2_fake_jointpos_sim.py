"""Thin wrapper around RoboDeploy's ROS2 devtool fake sim.

This module intentionally contains *no* rclpy boilerplate. It exists so users can
run a local ROS2 graph for RViz demos on machines without Gazebo.
"""

from __future__ import annotations

from robodeploy.ros2.devtools.fake_jointpos_sim import FakeJointPosSim, FakeJointPosSimConfig

__all__ = ["FakeJointPosSim", "FakeJointPosSimConfig"]

