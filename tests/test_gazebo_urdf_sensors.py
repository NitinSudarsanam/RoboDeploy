from __future__ import annotations

import unittest

from robodeploy.backends.real.ros2.sim_launchers.ros_gz_bridge import (
    camera_info_bridge_rules,
    image_bridge_rules,
    imu_bridge_rules,
    wrench_bridge_rules,
)
from robodeploy.backends.sim.gazebo.urdf_sensors import inject_sensors_into_urdf, patch_urdf_controller_yaml
from robodeploy.core.types import SensorMount
from robodeploy.description.kuka.description import KukaDescription
from robodeploy.core.spaces import AssetFormat
from robodeploy.sensors.camera.sim.mujoco_camera import MuJoCoCameraRenderer
from robodeploy.sensors.ft_sensor.sim.mujoco_ft import MuJoCoFTSensor
from robodeploy.sensors.imu.sim.mujoco_imu import MuJoCoIMUSensor


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
        self.assertIn("<topic>/wrist_camera/image_raw</topic>", patched)
        self.assertIn("<topic>/wrist_ft/wrench</topic>", patched)

    def test_inject_imu_link(self):
        imu = MuJoCoIMUSensor(
            "wrist_imu",
            mount=SensorMount(parent_link="ee_link"),
        )
        patched = inject_sensors_into_urdf(MINIMAL_URDF, [imu])
        self.assertIn('joint name="wrist_imu_joint"', patched)
        self.assertIn('sensor name="wrist_imu" type="imu"', patched)
        self.assertIn("<topic>/wrist_imu/imu</topic>", patched)

    def test_inject_rgbd_depth_topic(self):
        camera = MuJoCoCameraRenderer(
            "wrist_camera",
            config={"width": 64, "height": 48, "depth": True},
            mount=SensorMount(parent_link="ee_link"),
        )
        patched = inject_sensors_into_urdf(MINIMAL_URDF, [camera])
        self.assertIn('type="rgbd_camera"', patched)
        self.assertIn("<depth_topic>/wrist_camera/depth/image_raw</depth_topic>", patched)

    def test_kuka_urdf_has_ros2_control(self):
        path = KukaDescription().asset_path(AssetFormat.URDF)
        text = path.read_text(encoding="utf-8")
        self.assertIn("<ros2_control", text)
        self.assertIn("gz_ros2_control", text)
        patched = patch_urdf_controller_yaml(text, path)
        self.assertNotIn("__CONTROLLER_YAML__", patched)
        ctrl = path.parent / "kuka_controllers.yaml"
        self.assertTrue(ctrl.exists())
        self.assertIn("joint_trajectory_controller", ctrl.read_text(encoding="utf-8"))

    def test_kuka_urdf_has_arm_collision_geometry(self):
        path = KukaDescription().asset_path(AssetFormat.URDF)
        text = path.read_text(encoding="utf-8")
        for link in ("base_link", "link1", "link7", "ee_link"):
            self.assertIn(f'<link name="{link}">', text)
        self.assertGreaterEqual(text.count("<collision>"), 8)

    def test_bridge_rules_for_image_wrench_imu(self):
        rules = image_bridge_rules("/robot0/wrist_camera/image_raw")
        self.assertTrue(any("sensor_msgs/msg/Image" in r for r in rules))
        wrules = wrench_bridge_rules("/robot0/wrist_ft/wrench")
        self.assertTrue(any("WrenchStamped" in r for r in wrules))
        irules = camera_info_bridge_rules("/wrist_camera/camera_info")
        self.assertTrue(any("CameraInfo" in r for r in irules))
        imu_rules = imu_bridge_rules("/wrist_imu/imu")
        self.assertTrue(any("sensor_msgs/msg/Imu" in r for r in imu_rules))


if __name__ == "__main__":
    unittest.main()
