from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robodeploy.core.sensor_rig import SensorRig, _apply_backend_sensor_defaults
from robodeploy.core.types import SensorMount
from robodeploy.backends.sim.gazebo.urdf_sensors import _resolve_mount


class SensorMountTests(unittest.TestCase):
    def test_rig_overhead_camera_mount_from_config(self):
        rig = SensorRig.robot_mounted(
            "rig",
            ee_link="robot0/ee_link",
            overhead_rgbd={
                "mount": {
                    "parent_link": "world",
                    "position": [0.0, -1.0, 1.2],
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                }
            },
        )
        spec = rig.specs[0]
        self.assertEqual(spec.kind, "overhead_camera")
        mount = spec.mount
        self.assertIsNotNone(mount)
        assert mount is not None
        if isinstance(mount, dict):
            self.assertEqual(mount["parent_link"], "world")
            self.assertEqual(mount["position"][2], 1.2)
        else:
            self.assertEqual(mount.parent_link, "world")
            self.assertEqual(mount.position[2], 1.2)

    def test_wrist_ft_defaults_parent_to_ee_link(self):
        rig = SensorRig.robot_mounted("rig", ee_link="robot0/ee_link", wrist_ft={})
        spec = next(s for s in rig.specs if s.kind == "wrist_ft")
        self.assertEqual(spec.mount.parent_link, "robot0/ee_link")

    def test_resolve_mount_reads_config_dict_when_sensor_mount_empty(self):
        sensor = mock.Mock()
        sensor.mount = SensorMount()
        sensor.config = {
            "mount": {
                "parent_link": "ee_link",
                "position": [0.1, 0.0, 0.05],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            }
        }
        mount = _resolve_mount(sensor)
        self.assertEqual(mount.parent_link, "ee_link")
        self.assertAlmostEqual(mount.position[0], 0.1)

    def test_ros2_topic_defaults_for_overhead_camera(self):
        cfg = _apply_backend_sensor_defaults(
            "overhead_camera",
            {"name": "overhead_camera"},
            backend_name="gazebo",
        )
        self.assertEqual(cfg["namespace"], "/overhead_camera")
        self.assertEqual(cfg["rgb"], "image_raw")


if __name__ == "__main__":
    unittest.main()
