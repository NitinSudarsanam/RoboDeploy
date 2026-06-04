from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robodeploy import RoboEnv
from robodeploy.core.registry import list_registered, register_policy, resolve_sensor_class, unregister_policy


class RegistryHonestyTests(unittest.TestCase):
    def tearDown(self) -> None:
        unregister_policy("unit_registry_policy")

    def test_same_class_reregistration_is_noop(self):
        class UnitPolicy:
            pass

        register_policy("unit_registry_policy")(UnitPolicy)
        register_policy("unit_registry_policy")(UnitPolicy)
        self.assertIn("unit_registry_policy", list_registered()["policies"])

    def test_different_class_collision_still_raises(self):
        class UnitPolicyA:
            pass

        class UnitPolicyB:
            pass

        register_policy("unit_registry_policy")(UnitPolicyA)
        with self.assertRaises(KeyError):
            register_policy("unit_registry_policy")(UnitPolicyB)

    def test_wrist_camera_pair_resolves_sim_and_real(self):
        import robodeploy.builtins  # noqa: F401
        from robodeploy.backends.real.ros2.sensors.camera_rgbd import Ros2RgbdCameraISensor
        from robodeploy.sensors.camera.sim.isaacsim_camera import IsaacSimCameraRenderer
        from robodeploy.sensors.camera.real.realsense import RealSenseCamera
        from robodeploy.sensors.camera.sim.mujoco_camera import MuJoCoCameraRenderer

        self.assertIs(resolve_sensor_class("wrist_camera", is_real=False), MuJoCoCameraRenderer)
        self.assertIs(resolve_sensor_class("wrist_camera", is_real=True), RealSenseCamera)
        self.assertIs(
            resolve_sensor_class("wrist_camera", is_real=False, backend_name="isaacsim"),
            IsaacSimCameraRenderer,
        )
        self.assertIs(
            resolve_sensor_class("wrist_camera", is_real=False, backend_name="ros2_rviz"),
            Ros2RgbdCameraISensor,
        )
        self.assertIs(
            resolve_sensor_class("wrist_camera", is_real=False, backend_name="gazebo"),
            Ros2RgbdCameraISensor,
        )
        self.assertIs(
            resolve_sensor_class("wrist_camera", is_real=True, backend_name="ros2"),
            Ros2RgbdCameraISensor,
        )

    def test_make_uses_backend_aware_sensor_resolution(self):
        import robodeploy.builtins  # noqa: F401
        from robodeploy.core.registry import use

        use("examples.tasks")
        from robodeploy.backends.real.ros2.sensors.camera_rgbd import Ros2RgbdCameraISensor
        from robodeploy.sensors.camera.sim.isaacsim_camera import IsaacSimCameraRenderer
        from robodeploy.sensors.camera.sim.mujoco_camera import MuJoCoCameraRenderer
        from robodeploy.sensors.ft_sensor.sim.isaacsim_ft import IsaacSimFTSensor

        mujoco_env = RoboEnv.make(
            robot="franka",
            backend="mujoco",
            task="pick_place",
            policy="vla_stub",
            sensors=["wrist_camera"],
        )
        self.assertIsInstance(mujoco_env.robots[0].sensors[0], MuJoCoCameraRenderer)

        ros_env = RoboEnv.make(
            robot="franka",
            backend="ros2_rviz",
            task="pick_place",
            policy="vla_stub",
            sensors=["wrist_camera"],
            sensor_kwargs={
                "wrist_camera": {
                    "rgb": "camera/color/image_raw",
                    "depth": "camera/depth/image_raw",
                    "info": "camera/color/camera_info",
                }
            },
        )
        self.assertIsInstance(ros_env.robots[0].sensors[0], Ros2RgbdCameraISensor)

        isaac_env = RoboEnv.make(
            robot="franka",
            backend="isaacsim",
            task="pick_place",
            policy="vla_stub",
            sensors=["wrist_camera"],
        )
        self.assertIsInstance(isaac_env.robots[0].sensors[0], IsaacSimCameraRenderer)

        isaac_ft_env = RoboEnv.make(
            robot="franka",
            backend="isaacsim",
            task="pick_place",
            policy="vla_stub",
            sensors=["wrist_ft"],
        )
        self.assertIsInstance(isaac_ft_env.robots[0].sensors[0], IsaacSimFTSensor)


if __name__ == "__main__":
    unittest.main()
