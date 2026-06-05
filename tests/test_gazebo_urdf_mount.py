from __future__ import annotations

import unittest

from robodeploy.backends.real.ros2.sensors.camera_rgbd import Ros2RgbdCameraISensor
from robodeploy.backends.sim.gazebo.urdf_sensors import inject_sensors_into_urdf

MINIMAL_URDF = """<?xml version="1.0"?>
<robot name="arm">
  <link name="ee_link"/>
</robot>
"""


class GazeboUrdfMountTests(unittest.TestCase):
    def test_inject_reads_mount_from_config_when_sensor_mount_empty(self):
        camera = Ros2RgbdCameraISensor(
            config={
                "name": "wrist_camera",
                "mount": {
                    "parent_link": "ee_link",
                    "position": (0.0, 0.0, 0.05),
                    "orientation": (1.0, 0.0, 0.0, 0.0),
                },
            }
        )
        patched = inject_sensors_into_urdf(MINIMAL_URDF, [camera])
        self.assertIn('joint name="wrist_camera_joint"', patched)
        self.assertIn('sensor name="wrist_camera" type="camera"', patched)


if __name__ == "__main__":
    unittest.main()
