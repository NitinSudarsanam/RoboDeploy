from __future__ import annotations

import unittest

from robodeploy.backends.real.ros2.sim_launchers.ros_gz_bridge import image_bridge_rules, wrench_bridge_rules
from robodeploy.backends.sim.gazebo.urdf_sensors import inject_sensors_into_urdf
from robodeploy.core.types import SensorMount
from robodeploy.sensors.camera.sim.mujoco_camera import MuJoCoCameraRenderer
from robodeploy.sensors.ft_sensor.sim.mujoco_ft import MuJoCoFTSensor


MINIMAL_URDF = """<?xml version="1.0"?>
<robot name="arm">
  <link name="ee_link"/>
</robot>
"""


class GazeboUrdfSensorTests(unittest.TestCase):
    def test_inject_camera_and_ft_links(self):
        camera = MuJoCoCameraRenderer(
            "wrist_camera",
            config={"width": 64, "height": 48},
            mount=SensorMount(parent_link="ee_link", position=(0.0, 0.0, 0.05)),
        )
        ft = MuJoCoFTSensor(
            "wrist_ft",
            mount=SensorMount(parent_link="ee_link"),
        )
        patched = inject_sensors_into_urdf(MINIMAL_URDF, [camera, ft])
        self.assertIn('joint name="wrist_camera_joint"', patched)
        self.assertIn('sensor name="wrist_camera" type="camera"', patched)
        self.assertIn('sensor name="wrist_ft" type="force_torque"', patched)

    def test_bridge_rules_for_image_and_wrench(self):
        rules = image_bridge_rules("/robot0/wrist_camera/image_raw")
        self.assertTrue(any("sensor_msgs/msg/Image" in r for r in rules))
        wrules = wrench_bridge_rules("/robot0/wrist_ft/wrench")
        self.assertTrue(any("WrenchStamped" in r for r in wrules))


if __name__ == "__main__":
    unittest.main()
