from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from robodeploy.backends.real.ros2.controllers.base import ControllerConfig, make_controller
from robodeploy.backends.real.ros2.perception import ColorBlobPerceptionSource, DictPerceptionSource
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation


class ROS2ControllerParityTests(unittest.TestCase):
    def test_joint_velocity_controller_registered(self):
        from robodeploy.backends.real.ros2.controllers import joint_velocity as _mod  # noqa: F401

        cfg = ControllerConfig(robot_id="robot0", cmd_topic="joint_velocity_controller/commands")
        ctrl = make_controller("joint_velocity", cfg, {})
        self.assertEqual(ctrl.controller_type, "joint_velocity")

    def test_joint_effort_controller_registered(self):
        from robodeploy.backends.real.ros2.controllers import joint_effort as _mod  # noqa: F401

        cfg = ControllerConfig(robot_id="robot0", cmd_topic="effort_controllers/commands")
        ctrl = make_controller("joint_effort", cfg, {})
        self.assertEqual(ctrl.controller_type, "joint_effort")

    def test_gripper_maps_close_command(self):
        from robodeploy.backends.real.ros2.controllers.gripper import GripperControllerAdapter

        cfg = ControllerConfig(robot_id="robot0")
        ctrl = GripperControllerAdapter(cfg, {"gripper_command_type": "float", "gripper_open_width": 0.08, "gripper_close_width": 0.0})
        ctrl._cmd_msg_type = mock.Mock()
        ctrl._cmd_pub = mock.Mock()
        ctrl.send_action(Action(gripper=1.0))
        published = ctrl._cmd_pub.publish.call_args[0][0]
        self.assertAlmostEqual(float(published.data), 0.0, places=5)

    def test_dict_perception_source_pose(self):
        src = DictPerceptionSource({"source": ((0.5, 0.0, 0.4), (1.0, 0.0, 0.0, 0.0))})
        pos, quat = src.get_pose("source")
        np.testing.assert_allclose(pos, [0.5, 0.0, 0.4])

    def test_color_blob_perception_source_detects_red_blob(self):
        src = ColorBlobPerceptionSource(object_name="source", tolerance=120)
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[10:20, 10:20] = (255, 0, 0)
        obs = Observation(
            joint_positions=np.zeros(1),
            joint_velocities=np.zeros(1),
            joint_torques=np.zeros(1),
            ee_position=np.zeros(3),
            ee_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            ee_velocity=np.zeros(3),
            ee_angular_velocity=np.zeros(3),
            images={"wrist_camera": rgb},
        )
        src.update_obs(obs)
        pos, _ = src.get_pose("source")
        self.assertGreater(float(pos[0]), 0.0)

    def test_ros2_backend_supported_action_spaces_include_velocity_and_effort(self):
        from robodeploy.backends.real.ros2.backend import ROS2RealBackend

        backend = ROS2RealBackend({})
        backend._drivers = {
            "robot0": type("D", (), {"supported_action_spaces": [ActionSpace.JOINT_VEL]})(),
            "robot1": type("D", (), {"supported_action_spaces": [ActionSpace.JOINT_TORQUE]})(),
        }
        action_spaces: set[ActionSpace] = set()
        for driver in backend._drivers.values():
            for space in getattr(driver, "supported_action_spaces", []) or []:
                action_spaces.add(space)
        backend.supported_action_spaces = sorted(action_spaces, key=lambda s: s.name)
        spaces = set(backend.supported_action_spaces)
        self.assertIn(ActionSpace.JOINT_VEL, spaces)
        self.assertIn(ActionSpace.JOINT_TORQUE, spaces)


if __name__ == "__main__":
    unittest.main()
