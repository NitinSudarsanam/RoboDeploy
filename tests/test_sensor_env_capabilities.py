from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.backends.base import BackendBase
from robodeploy.backends.sim.gazebo.backend import ROS2GazeboBackend
from robodeploy.backends.sim.isaacsim.backend import IsaacSimBackend
from robodeploy.backends.sim.mujoco.scene_builder import MjcfSceneBuilder
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.spaces import ActionSpace, AssetFormat
from robodeploy.core.types import (
    Action,
    CameraSpec,
    GeomSpec,
    LightSpec,
    ObsSpec,
    Observation,
    PropConfig,
    SceneSpec,
    SensorData,
    SensorMount,
    TerrainSpec,
    WorldSpec,
)
from robodeploy.description.base import RobotDescription
from robodeploy.env import RoboEnv
from robodeploy.policies.base import PolicyBase
from robodeploy.sensors.base import SensorBase
from robodeploy.tasks.base import TaskBase


def make_obs(value: float = 0.0) -> Observation:
    return Observation(
        joint_positions=jnp.asarray([value, value], dtype=jnp.float32),
        joint_velocities=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_torques=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        ee_position=jnp.asarray([value, 0.0, 0.0], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_angular_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        timestamp=value,
        timestamp_hw=value,
        timestamp_recv=value,
    )


class DummyRobot(RobotDescription):
    dof = 2
    display_name = "dummy"
    ee_link_name = "ee_link"
    joint_names = ["joint1", "joint2"]
    joint_position_limits = jnp.asarray([[-3.14, 3.14], [-3.14, 3.14]], dtype=jnp.float32)
    joint_velocity_limits = jnp.asarray([2.0, 2.0], dtype=jnp.float32)
    joint_torque_limits = jnp.asarray([10.0, 10.0], dtype=jnp.float32)
    home_qpos = jnp.asarray([0.0, 0.0], dtype=jnp.float32)

    def asset_path(self, fmt, variant: str = "default"):
        del fmt, variant
        return ""


class DummyGazeboRobot(DummyRobot):
    def asset_path(self, fmt, variant: str = "default"):
        del variant
        if fmt == AssetFormat.URDF:
            return Path(__file__)
        return super().asset_path(fmt, variant="default")


class DummyPolicy(PolicyBase):
    def __init__(self) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS)

    def _reset_impl(self) -> None:
        return

    def get_action(self, obs: Observation) -> Action:
        del obs
        return Action(joint_positions=jnp.asarray([0.0, 0.0], dtype=jnp.float32))


class DummyTask(TaskBase):
    def obs_spec(self) -> ObsSpec:
        return ObsSpec()

    def scene_spec(self) -> SceneSpec:
        return SceneSpec(
            world=WorldSpec(
                props=[PropConfig(name="cube", geom=GeomSpec(kind="box", size=(0.02, 0.02, 0.02)))],
            )
        )

    def language_instruction(self) -> str:
        return "test"

    def reset_fn(self, backend) -> None:
        del backend

    def reward_fn(self, obs: Observation, action: Action) -> float:
        del obs, action
        return 0.0

    def success_fn(self, obs: Observation) -> bool:
        del obs
        return False


class CollisionTask(DummyTask):
    def scene_spec(self) -> SceneSpec:
        return SceneSpec(
            world=WorldSpec(
                props=[PropConfig(name="cube", geom=GeomSpec(kind="sphere", size=(0.01,)))],
            )
        )


class DummyBackend(BackendBase):
    is_real = False
    control_hz = 20.0
    supported_action_spaces = [ActionSpace.JOINT_POS]

    def initialize_multi(self, robots, scene, shared_sensors) -> None:
        robot = robots[0]
        self.initialize(robot.description, scene, [*robot.sensors, *shared_sensors])

    def reset_multi(self, robot_ids=None) -> list[Observation]:
        del robot_ids
        return [self.reset()]

    def step_multi(self, actions: list[Action]) -> list[Observation]:
        return [self.step(actions[0])]

    def get_obs_multi(self) -> list[Observation]:
        return [self.get_obs()]

    def _load(self, description, scene, sensors) -> None:
        del description, scene, sensors

    def _reset_impl(self) -> Observation:
        return self._merge_sensor_data(make_obs(), self._sensors)

    def _step_impl(self, action: Action) -> Observation:
        del action
        return self._merge_sensor_data(make_obs(), self._sensors)

    def _get_obs_impl(self) -> Observation:
        return self._merge_sensor_data(make_obs(), self._sensors)

    def _close_impl(self) -> None:
        return


class DummySensor(SensorBase):
    def __init__(self) -> None:
        super().__init__(name="cam", is_real=False)
        self.events: list[str] = []

    def _init_impl(self, backend) -> None:
        self.events.append(f"init:{type(backend).__name__}")

    def _read_impl(self) -> SensorData:
        self.events.append("read")
        return SensorData(rgb=np.zeros((2, 3, 3), dtype=np.uint8), timestamp_hw=1.0, timestamp_recv=2.0)

    def _close_impl(self) -> None:
        self.events.append("close")


class DummyMountedCamera(SensorBase):
    def __init__(self) -> None:
        super().__init__(
            name="wrist",
            is_real=False,
            config={"camera_name": "wrist", "width": 32, "height": 24},
            mount=SensorMount(parent_link="ee_link", position=(0.01, 0.0, 0.0)),
        )

    def _init_impl(self, backend) -> None:
        del backend

    def _read_impl(self) -> SensorData:
        return SensorData()

    def _close_impl(self) -> None:
        return


class DummyMountedFT(SensorBase):
    def __init__(self) -> None:
        super().__init__(
            name="wrist_ft",
            is_real=False,
            config={"site": "wrist_ft_site"},
            mount=SensorMount(parent_link="ee_link"),
        )

    def _init_impl(self, backend) -> None:
        del backend

    def _read_impl(self) -> SensorData:
        return SensorData()

    def _close_impl(self) -> None:
        return


def make_robot(task=None) -> Robot:
    return Robot(
        robot_id="robot0",
        description=DummyRobot(),
        tasks={"task0": RobotTask(task=task or DummyTask(), policies={"p": DummyPolicy()})},
    )


class SensorEnvCapabilityTests(unittest.TestCase):
    def test_scene_builder_emits_world_props_lights_cameras_and_actuators(self):
        builder = MjcfSceneBuilder("<mujoco><worldbody><body name='ee_link'/></worldbody></mujoco>")
        world = WorldSpec(
            props=[PropConfig(name="cube", geom=GeomSpec(kind="box", size=(0.02, 0.02, 0.02)))],
            lights=[LightSpec(position=(0, 0, 2), direction=(0, 0, -1))],
            cameras=[CameraSpec(name="overhead_camera", position=(0, -1, 1), orientation=(1, 0, 0, 0))],
            terrain=TerrainSpec(kind="flat", size=(3.0, 3.0)),
        )
        builder.ensure_world_defaults(add_camera=False)
        builder.attach_actuators(["joint1"])
        builder.attach_world(world)
        xml = builder.emit()
        self.assertIn('name="cube"', xml)
        self.assertIn('type="box"', xml)
        self.assertIn('name="overhead_camera"', xml)
        self.assertIn('joint="joint1"', xml)

    def test_scene_builder_emits_mounted_camera_and_ft_sensors(self):
        builder = MjcfSceneBuilder("<mujoco><worldbody><body name='ee_link'/></worldbody></mujoco>")
        builder.attach_sensors([DummyMountedCamera(), DummyMountedFT()])
        xml = builder.emit()
        self.assertIn('name="wrist"', xml)
        self.assertIn('name="wrist_ft_site"', xml)
        self.assertIn('name="wrist_ft_force"', xml)
        self.assertIn('name="wrist_ft_torque"', xml)

    def test_env_initializes_sensors_before_warmup_and_merges_named_images(self):
        sensor = DummySensor()
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy()})},
            sensors=[sensor],
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        obs, _ = env.reset()
        self.assertEqual(sensor.events[0], "init:DummyBackend")
        self.assertIn("cam", obs.images)
        self.assertIs(obs.rgb, obs.images["cam"])
        env.close()

    def test_explicit_action_list_length_mismatch_raises(self):
        env = RoboEnv(backend=DummyBackend(), robots=[make_robot()])
        env.reset()
        with self.assertRaises(ValueError):
            env.step([
                Action(joint_positions=jnp.asarray([0.0, 0.0], dtype=jnp.float32)),
                Action(joint_positions=jnp.asarray([1.0, 1.0], dtype=jnp.float32)),
            ])
        env.close()

    def test_scene_prop_collision_raises(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={
                "task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy()}),
                "task1": RobotTask(task=CollisionTask(), policies={"p": DummyPolicy()}, mode="concurrent"),
            },
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        with self.assertRaises(ValueError):
            env.reset()

    def test_gazebo_backend_derives_namespaced_bridge_rules_from_declared_sensors(self):
        from robodeploy.backends.real.ros2.sensors.camera_rgbd import Ros2RgbdCameraISensor

        robot = Robot(
            robot_id="robot0",
            description=DummyGazeboRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy()})},
            sensors=[
                Ros2RgbdCameraISensor(
                    config={
                        "name": "wrist_camera",
                        "rgb": "camera/color/image_raw",
                        "depth": "camera/depth/image_raw",
                        "info": "camera/color/camera_info",
                    }
                )
            ],
        )
        shared = [
            Ros2RgbdCameraISensor(
                config={
                    "name": "overhead_camera",
                    "namespace": "/scene",
                    "rgb": "overhead/rgb",
                    "depth": "overhead/depth",
                }
            )
        ]
        backend = ROS2GazeboBackend(config={"sim": {"kind": "gazebo", "world": "demo_world.sdf"}})

        with (
            mock.patch("robodeploy.backends.real.ros2.sim_launchers.gazebo.GazeboLauncher") as launcher_cls,
            mock.patch("robodeploy.backends.real.ros2.backend.ROS2RealBackend.initialize_multi", return_value=None),
        ):
            backend.initialize_multi([robot], SceneSpec(), shared)

        launch_cfg = launcher_cls.call_args.args[0]
        self.assertEqual(launch_cfg.robot_name, "robot0")
        self.assertEqual(str(launch_cfg.robot_urdf), str(Path(__file__)))
        self.assertIn("/robot0/camera/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image", launch_cfg.bridge_rules)
        self.assertIn("/robot0/camera/depth/image_raw@sensor_msgs/msg/Image[gz.msgs.Image", launch_cfg.bridge_rules)
        self.assertIn("/scene/overhead/rgb@sensor_msgs/msg/Image[gz.msgs.Image", launch_cfg.bridge_rules)
        self.assertIn("/scene/overhead/depth@sensor_msgs/msg/Image[gz.msgs.Image", launch_cfg.bridge_rules)
        self.assertIn("/robot0/camera/color/image_raw", launch_cfg.wait_for_topics)
        self.assertIn("/scene/overhead/rgb", launch_cfg.wait_for_topics)

    def test_gazebo_backend_generates_world_from_scene_when_missing(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyGazeboRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy()})},
        )
        scene = SceneSpec(
            world=WorldSpec(
                props=[
                    PropConfig(
                        name="cube",
                        geom=GeomSpec(kind="box", size=(0.02, 0.03, 0.04)),
                        position=(0.1, 0.2, 0.3),
                    )
                ],
                lights=[LightSpec(position=(0.0, 0.0, 2.0), direction=(0.0, 0.0, -1.0))],
                cameras=[CameraSpec(name="overhead_camera", position=(0.0, -1.0, 1.0), orientation=(1, 0, 0, 0))],
            )
        )
        backend = ROS2GazeboBackend(config={"sim": {"kind": "gazebo"}})

        with (
            mock.patch("robodeploy.backends.real.ros2.sim_launchers.gazebo.GazeboLauncher") as launcher_cls,
            mock.patch("robodeploy.backends.real.ros2.backend.ROS2RealBackend.initialize_multi", return_value=None),
        ):
            backend.initialize_multi([robot], scene, [])

        launch_cfg = launcher_cls.call_args.args[0]
        world_path = Path(launch_cfg.world)
        self.addCleanup(lambda: world_path.unlink(missing_ok=True))
        self.assertTrue(world_path.exists())
        xml = world_path.read_text(encoding="utf-8")
        self.assertIn('model name="cube"', xml)
        self.assertIn('sensor name="overhead_camera"', xml)
        self.assertIn("<gravity>0.0 0.0 -9.81</gravity>", xml)
        self.assertIn('light name="light_0"', xml)

    def test_isaac_backend_applies_prop_and_camera_orientation_to_stage(self):
        class _FakePrim:
            def __init__(self, stage, path: str, type_name: str):
                self.stage = stage
                self.path = path
                self.type_name = type_name
                self.references: list[str] = []

            def GetReferences(self):
                return self

            def AddReference(self, ref: str) -> None:
                self.references.append(ref)

            def IsValid(self) -> bool:
                return True

        class _FakeAttr:
            def __init__(self) -> None:
                self.value = None

            def Set(self, value) -> None:  # noqa: ANN001
                self.value = value

        class _FakeStage:
            def __init__(self) -> None:
                self.prims: dict[str, _FakePrim] = {}
                self.transforms: dict[str, dict[str, object]] = {}
                self.attrs: dict[tuple[str, str], _FakeAttr] = {}

            def DefinePrim(self, path: str, type_name: str):
                prim = _FakePrim(self, path, type_name)
                self.prims[path] = prim
                self.transforms.setdefault(path, {"translate": None, "rotate": None, "scale": None})
                return prim

            def GetPrimAtPath(self, path: str):
                return self.prims.get(path)

        class _FakeGeomWrapper:
            def __init__(self, stage: _FakeStage, path: str, type_name: str):
                self.stage = stage
                self.path = path
                self.prim = stage.DefinePrim(path, type_name)

            def GetRadiusAttr(self):
                return self.stage.attrs.setdefault((self.path, "radius"), _FakeAttr())

            def GetHeightAttr(self):
                return self.stage.attrs.setdefault((self.path, "height"), _FakeAttr())

            def GetVerticalApertureAttr(self):
                return self.stage.attrs.setdefault((self.path, "vertical_aperture"), _FakeAttr())

        class _FakeXformCommonAPI:
            def __init__(self, obj) -> None:  # noqa: ANN001
                target = getattr(obj, "prim", obj)
                self._path = getattr(obj, "path", target.path)
                self._stage = getattr(obj, "stage", target.stage)

            def SetTranslate(self, value) -> None:  # noqa: ANN001
                self._stage.transforms[self._path]["translate"] = tuple(value)

            def SetRotate(self, value) -> None:  # noqa: ANN001
                self._stage.transforms[self._path]["rotate"] = tuple(value)

            def SetScale(self, value) -> None:  # noqa: ANN001
                self._stage.transforms[self._path]["scale"] = tuple(value)

        stage = _FakeStage()
        omni_mod = types.ModuleType("omni")
        omni_usd_mod = types.ModuleType("omni.usd")
        omni_usd_mod.get_context = lambda: types.SimpleNamespace(get_stage=lambda: stage)
        omni_mod.usd = omni_usd_mod

        usd_geom = types.SimpleNamespace(
            Xform=types.SimpleNamespace(Define=lambda s, p: _FakeGeomWrapper(s, p, "Xform")),
            Cube=types.SimpleNamespace(Define=lambda s, p: _FakeGeomWrapper(s, p, "Cube")),
            Sphere=types.SimpleNamespace(Define=lambda s, p: _FakeGeomWrapper(s, p, "Sphere")),
            Cylinder=types.SimpleNamespace(Define=lambda s, p: _FakeGeomWrapper(s, p, "Cylinder")),
            Camera=types.SimpleNamespace(Define=lambda s, p: _FakeGeomWrapper(s, p, "Camera")),
            XformCommonAPI=_FakeXformCommonAPI,
            Xformable=lambda prim: types.SimpleNamespace(path=prim.path, stage=prim.stage),
        )
        usd_lux = types.SimpleNamespace(
            DistantLight=types.SimpleNamespace(Define=lambda s, p: _FakeGeomWrapper(s, p, "DistantLight")),
            SphereLight=types.SimpleNamespace(Define=lambda s, p: _FakeGeomWrapper(s, p, "SphereLight")),
        )
        gf = types.SimpleNamespace(Vec3d=lambda *args: tuple(args))
        pxr_mod = types.ModuleType("pxr")
        pxr_mod.Gf = gf
        pxr_mod.UsdGeom = usd_geom
        pxr_mod.UsdLux = usd_lux

        backend = IsaacSimBackend(config={})
        backend._warnings = []
        backend._scene_prop_paths = {}

        world = WorldSpec(
            props=[
                PropConfig(
                    name="cube",
                    geom=GeomSpec(kind="box", size=(0.02, 0.03, 0.04)),
                    position=(0.1, 0.2, 0.3),
                    orientation=(0.70710678, 0.0, 0.0, 0.70710678),
                )
            ],
            cameras=[
                CameraSpec(
                    name="overhead_camera",
                    position=(0.0, -1.0, 1.0),
                    orientation=(0.70710678, 0.70710678, 0.0, 0.0),
                )
            ],
        )

        with mock.patch.dict(
            sys.modules,
            {
                "omni": omni_mod,
                "omni.usd": omni_usd_mod,
                "pxr": pxr_mod,
            },
        ):
            backend._load_world_spec_into_stage(world)
            self.assertTrue(backend._write_usd_prop_pose("cube", (1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0)))

        self.assertEqual(backend._scene_prop_paths["cube"], "/World/RoboDeployProps/cube")
        self.assertEqual(stage.transforms["/World/RoboDeployProps/cube"]["translate"], (1.0, 2.0, 3.0))
        self.assertIsNotNone(stage.transforms["/World/RoboDeployProps/cube"]["rotate"])
        self.assertEqual(stage.transforms["/World/RoboDeployProps/cube"]["scale"], (0.04, 0.06, 0.08))
        self.assertEqual(stage.transforms["/World/overhead_camera"]["translate"], (0.0, -1.0, 1.0))
        self.assertIsNotNone(stage.transforms["/World/overhead_camera"]["rotate"])
        self.assertFalse(backend._warnings)

    def test_isaac_camera_sensor_reads_mocked_frames(self):
        from robodeploy.sensors.camera.sim.isaacsim_camera import IsaacSimCameraRenderer

        class _FakePrim:
            def __init__(self, stage, path: str, type_name: str):
                self.stage = stage
                self.path = path
                self.type_name = type_name

            def IsValid(self) -> bool:
                return True

        class _FakeStage:
            def __init__(self) -> None:
                self.prims: dict[str, _FakePrim] = {}
                self.transforms: dict[str, dict[str, object]] = {}

            def DefinePrim(self, path: str, type_name: str):
                prim = _FakePrim(self, path, type_name)
                self.prims[path] = prim
                self.transforms.setdefault(path, {"translate": None, "rotate": None})
                return prim

            def GetPrimAtPath(self, path: str):
                return self.prims.get(path)

        class _FakeGeomWrapper:
            def __init__(self, stage: _FakeStage, path: str, type_name: str):
                self.stage = stage
                self.path = path
                self.prim = stage.DefinePrim(path, type_name)

        class _FakeXformCommonAPI:
            def __init__(self, obj) -> None:  # noqa: ANN001
                target = getattr(obj, "prim", obj)
                self._path = getattr(obj, "path", target.path)
                self._stage = getattr(obj, "stage", target.stage)

            def SetTranslate(self, value) -> None:  # noqa: ANN001
                self._stage.transforms[self._path]["translate"] = tuple(value)

            def SetRotate(self, value) -> None:  # noqa: ANN001
                self._stage.transforms[self._path]["rotate"] = tuple(value)

        class _FakeCamera:
            def __init__(self, *args, **kwargs):  # noqa: ANN002,ANN003
                self.args = args
                self.kwargs = kwargs
                self.initialized = False

            def initialize(self) -> None:
                self.initialized = True

            def get_rgba(self):
                return np.ones((2, 3, 4), dtype=np.float32)

            def get_depth(self):
                return np.full((2, 3), 0.5, dtype=np.float32)

        stage = _FakeStage()
        omni_mod = types.ModuleType("omni")
        omni_usd_mod = types.ModuleType("omni.usd")
        omni_usd_mod.get_context = lambda: types.SimpleNamespace(get_stage=lambda: stage)
        omni_mod.usd = omni_usd_mod
        pxr_mod = types.ModuleType("pxr")
        pxr_mod.Gf = types.SimpleNamespace(Vec3d=lambda *args: tuple(args))
        pxr_mod.UsdGeom = types.SimpleNamespace(
            Camera=types.SimpleNamespace(Define=lambda s, p: _FakeGeomWrapper(s, p, "Camera")),
            XformCommonAPI=_FakeXformCommonAPI,
        )
        isaacsim_mod = types.ModuleType("isaacsim")
        isaacsim_sensors_mod = types.ModuleType("isaacsim.sensors")
        isaacsim_camera_mod = types.ModuleType("isaacsim.sensors.camera")
        isaacsim_camera_mod.Camera = _FakeCamera
        isaacsim_mod.sensors = isaacsim_sensors_mod
        isaacsim_sensors_mod.camera = isaacsim_camera_mod

        backend = types.SimpleNamespace(
            _world=object(),
            _simulation_app=object(),
            _robot_prim_path="/World/robot0",
            control_hz=20.0,
            _sim_time=1.25,
        )
        sensor = IsaacSimCameraRenderer(
            config={"name": "wrist_camera", "depth": True},
            mount=SensorMount(parent_link="ee_link", position=(0.1, 0.2, 0.3)),
        )

        with mock.patch.dict(
            sys.modules,
            {
                "omni": omni_mod,
                "omni.usd": omni_usd_mod,
                "pxr": pxr_mod,
                "isaacsim": isaacsim_mod,
                "isaacsim.sensors": isaacsim_sensors_mod,
                "isaacsim.sensors.camera": isaacsim_camera_mod,
            },
        ):
            sensor.initialize(backend)
            reading = sensor.read()

        self.assertEqual(sensor._prim_path, "/World/robot0/ee_link/wrist_camera")
        self.assertTrue(sensor._camera.initialized)
        self.assertEqual(stage.transforms["/World/robot0/ee_link/wrist_camera"]["translate"], (0.1, 0.2, 0.3))
        self.assertEqual(reading.rgb.shape, (2, 3, 3))
        self.assertEqual(reading.rgb.dtype, np.uint8)
        self.assertEqual(reading.depth.dtype, np.float32)
        self.assertEqual(reading.timestamp, 1.25)
        self.assertEqual(reading.timestamp_source, "sim")
        self.assertIsNotNone(reading.intrinsics)
        self.assertIn("fx", reading.intrinsics)
        self.assertIsNotNone(reading.extrinsics)
        self.assertIn("position", reading.extrinsics)

    def test_isaac_ft_sensor_reads_measured_joint_forces(self):
        from robodeploy.sensors.ft_sensor.sim.isaacsim_ft import IsaacSimFTSensor

        backend = types.SimpleNamespace(
            _robot=types.SimpleNamespace(
                get_measured_joint_forces=lambda: np.asarray(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    ],
                    dtype=np.float32,
                )
            ),
            _description=types.SimpleNamespace(joint_names=["joint1", "joint2"]),
            _sim_time=2.5,
        )
        sensor = IsaacSimFTSensor(config={"joint_name": "joint2"})
        sensor.initialize(backend)
        reading = sensor.read()

        np.testing.assert_allclose(reading.ft_force, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(reading.ft_torque, np.asarray([4.0, 5.0, 6.0], dtype=np.float32))
        self.assertIn("wrist_ft", reading.ft_forces)
        self.assertIn("wrist_ft", reading.ft_torques)
        np.testing.assert_allclose(reading.ft_forces["wrist_ft"], reading.ft_force)
        self.assertEqual(reading.timestamp, 2.5)
        self.assertEqual(reading.timestamp_source, "sim")


if __name__ == "__main__":
    unittest.main()
