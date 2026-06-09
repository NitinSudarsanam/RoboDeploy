"""Regression tests for ROS topic namespace resolution."""

from __future__ import annotations

import unittest

from robodeploy.backends.real.ros2.controllers.base import resolve_ros_topic


class ResolveRosTopicTests(unittest.TestCase):
    def test_absolute_topic_not_double_prefixed(self):
        self.assertEqual(resolve_ros_topic("/robot0", "/joint_states"), "/joint_states")

    def test_relative_topic_joins_namespace(self):
        self.assertEqual(resolve_ros_topic("/robot0", "joint_states"), "/robot0/joint_states")

    def test_empty_namespace_prepends_slash(self):
        self.assertEqual(resolve_ros_topic("", "joint_states"), "/joint_states")

    def test_jtc_absolute_cmd_topic(self):
        topic = resolve_ros_topic("/robot0", "/joint_trajectory_controller/joint_trajectory")
        self.assertEqual(topic, "/joint_trajectory_controller/joint_trajectory")


if __name__ == "__main__":
    unittest.main()
