"""
ROS2RealBackend — ROS 2 transport backend (ROS 2 Jazzy).

This backend is transport-only: it communicates with an existing ROS 2 graph
(real hardware or an externally launched simulator).

Sim orchestration (Gazebo, ros_gz_bridge, controller spawning) must live in a
separate simulated backend so `is_real` remains honest.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from robodeploy.backends.errors import BackendTimeoutError
from robodeploy.safety.violation import Hazard, SafetyError
from robodeploy.backends.base import BackendBase
from robodeploy.core.registry import register_backend
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation, SceneSpec

from .dev import FakeJointPosSim, FakeJointPosSimConfig
from .interfaces import ControllerConfig, Ros2BackendConfig, make_controller
from .presets import PRESETS
from .sensors.interfaces import Ros2SensorConfig
from .sensors.registry import make_ros2_sensor
from .runtime import Ros2Runtime

# Ensure built-in controller adapters are registered (side-effect imports).
from .controllers import gripper as _builtin_gripper  # noqa: F401
from .controllers import joint_effort as _builtin_joint_effort  # noqa: F401
from .controllers import joint_position as _builtin_joint_position  # noqa: F401
from .controllers import joint_trajectory as _builtin_joint_trajectory  # noqa: F401
from .controllers import joint_velocity as _builtin_joint_velocity  # noqa: F401
from .controllers import so101_feetech as _builtin_so101_feetech  # noqa: F401
from .sensors import camera_rgbd as _builtin_rgbd_sensor  # noqa: F401
from .sensors import wrench as _builtin_wrench_sensor  # noqa: F401

if TYPE_CHECKING:
    from robodeploy.core.interfaces.sensor import ISensor
    from robodeploy.core.robot import Robot
    from robodeploy.description.base import RobotDescription


@register_backend("ros2")
class ROS2RealBackend(BackendBase):
    """ROS 2 transport backend (multi-robot capable)."""

    is_real = True
    sensor_backend_name = "ros2"
    control_hz = 100.0
    supported_action_spaces = [ActionSpace.JOINT_POS]

    def _load(
        self,
        description: RobotDescription,
        scene: SceneSpec,
        sensors: list[ISensor],
    ) -> None:
        # Single-agent path is no longer used by RoboEnv (env always calls
        # initialize_multi). Keep this raising so misuse fails loudly.
        del description, scene, sensors
        raise RuntimeError(
            "ROS2RealBackend.initialize() is not supported. "
            "Use RoboEnv(robots=[...]) which calls initialize_multi() directly."
        )

    def _reset_impl(self) -> Observation:
        return self.reset_multi()[0]

    def _step_impl(self, action: Action) -> Observation:
        return self.step_multi([action])[0]

    def _get_obs_impl(self) -> Observation:
        return self.get_obs_multi()[0]

    def _close_impl(self) -> None:
        if hasattr(self, "_fake_sims"):
            for sim in self._fake_sims:
                try:
                    sim.stop()
                except Exception:
                    pass
            self._fake_sims = []
        if hasattr(self, "_drivers"):
            for d in self._drivers.values():
                try:
                    d.stop()
                except Exception:
                    pass
        if hasattr(self, "_sensors_by_robot"):
            for sensors in self._sensors_by_robot.values():
                for s in sensors:
                    try:
                        s.stop()
                    except Exception:
                        pass
        if hasattr(self, "_rviz") and self._rviz is not None:
            try:
                self._rviz.close()
            except Exception:
                pass
        if hasattr(self, "_sim_launcher") and self._sim_launcher is not None:
            try:
                self._sim_launcher.stop()
            except Exception:
                pass
        if hasattr(self, "_rsp_launchers"):
            for rsp in self._rsp_launchers:
                try:
                    rsp.stop()
                except Exception:
                    pass
            self._rsp_launchers = []
        try:
            Ros2Runtime.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Multi-robot implementation
    # ------------------------------------------------------------------

    def initialize_multi(
        self,
        robots: list["Robot"],
        scene: SceneSpec,
        shared_sensors: list["ISensor"],
    ) -> None:
        if not robots:
            raise ValueError("ROS2RealBackend.initialize_multi requires at least one robot.")

        cfg = self._parse_backend_config(self.config)
        self._backend_cfg = cfg
        self._scene = scene
        self._scene_prop_names = [prop.name for prop in scene.to_world().props]
        self._sim_launcher = None
        self._drivers = {}
        self._sensors_by_robot: dict[str, list[object]] = {}
        self._robot_sensors_by_robot: dict[str, list["ISensor"]] = {r.robot_id: list(r.sensors) for r in robots}
        self._shared_sensors = list(shared_sensors)
        self._sensors = [sensor for sensors in self._robot_sensors_by_robot.values() for sensor in sensors] + self._shared_sensors
        self._latest_obs: dict[str, Observation] = {}
        self._latest_viz_payload: Optional[dict] = None
        self._diagnostics: dict = {"backend": "ros2", "robots": {}, "warnings": []}
        self._rsp_launchers: list = []
        self._fake_sims: list[FakeJointPosSim] = []
        self._perception_source = self._resolve_perception_source()
        self._recovery_managers: dict[str, object] = {}

        fake_sim_cfg_raw = self.config.get("dev_fake_sim")
        if fake_sim_cfg_raw is not None:
            fake_sim_specs = fake_sim_cfg_raw if isinstance(fake_sim_cfg_raw, list) else [fake_sim_cfg_raw]
            for spec in fake_sim_specs:
                fake_cfg = self._coerce_fake_sim_config(spec, robots)
                sim = FakeJointPosSim(fake_cfg)
                sim.start()
                self._fake_sims.append(sim)

        # RViz RobotModel needs /robot_description + /tf from robot_state_publisher.
        # MuJoCo may use different joint naming (e.g. robot0/joint1); ROS URDF should use valid names
        # and per-robot config should set robot0.joint_names to match /joint_states.
        if cfg.rviz_enabled and cfg.rviz_launch_robot_state_publisher and robots:
            from robodeploy.core.spaces import AssetFormat

            from .sim_launchers.robot_state_publisher import RobotStatePublisherLauncher

            try:
                urdf_path = robots[0].description.asset_path(AssetFormat.URDF)
                urdf_text = Path(urdf_path).read_text(encoding="utf-8")
                # Remap /joint_states -> /<robot_id>/joint_states for multi-robot parity.
                rsp = RobotStatePublisherLauncher(
                    urdf_text,
                    namespace=str(robots[0].robot_id),
                    joint_states_topic="joint_states",
                )
                rsp.start()
                self._rsp_launchers.append(rsp)
            except Exception as exc:
                self._diagnostics.setdefault("warnings", []).append(
                    f"robot_state_publisher not started: {exc}"
                )

        for robot in robots:
            robot_id = robot.robot_id
            ns = f"/{robot_id}"
            # Merge: defaults < preset < per-robot explicit keys.
            preset_name = self.config.get(f"{robot_id}.preset", getattr(robot.description, "ros2_preset_name", None))
            preset = PRESETS.get(str(preset_name), {}) if preset_name else {}

            def _get(key: str, default):
                return self.config.get(f"{robot_id}.{key}", preset.get(key, default))

            joint_names = _get("joint_names", None)
            if not isinstance(joint_names, list):
                joint_names = getattr(robot.description, "joint_names", None)
            ee_frame = getattr(robot.description, "ee_link_name", "ee_link")
            base_frame = self.config.get("base_frame", "base_link")

            # Controller selection: default to joint_position until presets land.
            controller_type = str(
                self.config.get(
                    f"{robot_id}.controller",
                    preset.get("controller_type", cfg.controller_by_robot_id.get(robot_id, "joint_position")),
                )
            )
            hz = float(cfg.command_hz_by_robot_id.get(robot_id, cfg.command_hz) or 0.0)

            mv = _get("max_joint_velocity", None)
            if isinstance(mv, list):
                mv_t = tuple(float(x) for x in mv)
            else:
                mv_t = None

            home_q = getattr(robot.description, "home_qpos", None)
            home_q_over = _get("home_qpos", None)
            if isinstance(home_q_over, list):
                home_q = home_q_over
            home_t = (
                tuple(float(x) for x in np.asarray(home_q, dtype=np.float64).reshape(-1))
                if home_q is not None
                else None
            )
            jv = getattr(robot.description, "joint_velocity_limits", None)
            jv_t = (
                tuple(float(x) for x in np.asarray(jv, dtype=np.float64).reshape(-1))
                if jv is not None
                else None
            )
            port_raw = _get("port", None)
            cal_path = _get("calibration_path", None)

            ctl_cfg = ControllerConfig(
                robot_id=robot_id,
                namespace=ns,
                base_frame=str(_get("base_frame", base_frame)),
                ee_frame=str(_get("ee_frame", ee_frame)),
                joint_states_topic=str(_get("joint_states_topic", "joint_states")),
                cmd_topic=str(_get("joint_cmd_topic", _get("joint_pos_cmd_topic", "joint_position_commands"))),
                joint_names=list(joint_names) if joint_names else None,
                joint_state_timeout_s=float(_get("joint_state_timeout_s", 1.0)),
                command_hz=hz,
                max_joint_velocity=mv_t,
                port=str(port_raw) if port_raw else None,
                baud=int(_get("baud", 1_000_000)),
                state_hz=float(_get("state_hz", 0.0) or 0.0),
                calibration_path=str(cal_path) if cal_path else None,
                reset_ramp_s=float(_get("reset_ramp_s", 3.0)),
                watchdog_timeout_s=float(_get("watchdog_timeout_s", 0.5)),
                temperature_max_c=float(_get("temperature_max_c", 70.0)),
                temperature_poll_s=float(_get("temperature_poll_s", 0.5)),
                enable_console_estop=bool(_get("enable_console_estop", True)),
                publish_state=bool(_get("publish_state", True)),
                publish_command_echo=bool(_get("publish_command_echo", True)),
                allow_uncalibrated=bool(_get("allow_uncalibrated", False)),
                apply_motor_limits=bool(_get("apply_motor_limits", bool(self.config.get("apply_motor_limits", True)))),
                home_qpos=home_t,
                joint_velocity_limits=jv_t,
            )

            controller = make_controller(controller_type, ctl_cfg, dict(self.config))
            controller.start()
            self._drivers[robot_id] = controller
            self._latest_obs[robot_id] = controller.get_obs()
            ctl_diag = getattr(controller, "get_diagnostics", None)
            if callable(ctl_diag):
                self._diagnostics["robots"][robot_id] = ctl_diag()

            # Optional per-robot sensor streams from config:
            #   robot0.sensors = [{"type": "rgbd", "name": "front", "rgb": "...", "depth": "...", "info": "..."}]
            sensors_cfg = _get("sensors", None)
            sensors: list[object] = []
            if isinstance(sensors_cfg, list):
                for item in sensors_cfg:
                    if not isinstance(item, dict):
                        continue
                    sensor_type = str(item.get("type", "") or "")
                    if not sensor_type:
                        continue
                    name = str(item.get("name", sensor_type) or sensor_type or "sensor")
                    topics = {
                        "rgb": item.get("rgb"),
                        "depth": item.get("depth"),
                        "info": item.get("info"),
                    }
                    topics = {k: v for k, v in topics.items() if isinstance(v, str) and v}
                    s_cfg = Ros2SensorConfig(robot_id=robot_id, name=name, namespace=ns, topics=topics)
                    s = make_ros2_sensor(sensor_type, s_cfg, dict(self.config))
                    s.start()
                    sensors.append(s)
            self._sensors_by_robot[robot_id] = sensors

        # Optional RViz sidecar
        self._rviz = None
        if cfg.rviz_enabled:
            from .rviz import RvizPublisher

            # Use first driver's node if available; drivers own ROS nodes/executors.
            # For simplicity, we create a separate node inside RvizPublisher.
            self._rviz = RvizPublisher(
                fixed_frame=cfg.rviz_fixed_frame,
                publish_hz=cfg.rviz_publish_hz,
                namespace="/robodeploy",
                base_frame=robots[0].description.ros_base_frame_id(),
            )
            self._rviz.start()
            self._rviz.publish_scene(scene)

        action_spaces: set[ActionSpace] = set()
        for driver in self._drivers.values():
            for space in getattr(driver, "supported_action_spaces", []) or []:
                action_spaces.add(space)
        if not action_spaces:
            action_spaces.add(ActionSpace.JOINT_POS)
        self.supported_action_spaces = sorted(action_spaces, key=lambda s: s.name)

        self._initialized = True

    def reset_multi(self, robot_ids: list[str] | None = None) -> list[Observation]:
        self._require_initialized("reset_multi")
        ids = robot_ids or list(self._drivers.keys())

        # Real reset is “synchronize”; optional homing can be added per driver later.
        out: list[Observation] = []
        if self._rviz is not None:
            self._rviz.reset()
        for rid in ids:
            obs = self._drivers[rid].get_obs()
            obs = self._merge_sensor_data(obs, [
                *self._robot_sensors_by_robot.get(rid, []),
                *self._sensors_by_robot.get(rid, []),
                *self._shared_sensors,
            ])
            self._latest_obs[rid] = obs
            out.append(obs)
            self._update_perception_source(obs)
            self._capture_driver_diagnostics(rid)
            if self._rviz is not None:
                self._rviz.publish_robot_state(rid, obs)
        return out

    def step_multi(self, actions: list[Action]) -> list[Observation]:
        self._require_initialized("step_multi")
        if len(actions) != len(self._drivers):
            raise ValueError(
                f"ROS2RealBackend.step_multi expected {len(self._drivers)} actions (one per robot), got {len(actions)}."
            )

        robot_ids = list(self._drivers.keys())
        for rid, act in zip(robot_ids, actions):
            driver = self._drivers[rid]
            send_wait = getattr(driver, "send_action_and_wait", None)
            if callable(send_wait):
                send_wait(act)
            else:
                driver.send_action(act)

        # Wait up to each driver's timeout for new state; fallback to latest.
        out: list[Observation] = []
        for rid in robot_ids:
            obs = self._drivers[rid].get_obs()
            obs = self._merge_sensor_data(obs, [
                *self._robot_sensors_by_robot.get(rid, []),
                *self._sensors_by_robot.get(rid, []),
                *self._shared_sensors,
            ])
            self._latest_obs[rid] = obs
            out.append(obs)
            self._update_perception_source(obs)
            self._capture_driver_diagnostics(rid)
            if self._rviz is not None:
                self._rviz.publish_robot_state(rid, obs)
        if self._rviz is not None and self._latest_viz_payload is not None:
            self._rviz.publish_task_viz(self._latest_viz_payload)
        return out

    def get_obs_multi(self) -> list[Observation]:
        self._require_initialized("get_obs_multi")
        return [self._latest_obs[rid] for rid in self._drivers.keys()]

    def get_prop_names(self) -> list[str]:
        return list(getattr(self, "_scene_prop_names", []))

    def get_prop_pose(self, name: str):
        perception = getattr(self, "_perception_source", None)
        if perception is None:
            raise NotImplementedError(
                "ROS2RealBackend cannot infer pose for prop without a perception source. "
                "Inject via config: perception_source = 'dict' | object with get_pose()."
            )
        return perception.get_pose(name)

    def _resolve_perception_source(self):
        raw = self.config.get("perception_source")
        if raw is None:
            return None
        if hasattr(raw, "get_pose"):
            return raw
        if isinstance(raw, dict):
            kind = str(raw.get("kind", "dict"))
            if kind == "dict":
                from .perception import DictPerceptionSource

                return DictPerceptionSource(raw.get("poses", {}))
            if kind == "tf":
                from .perception import TFPerceptionSource

                return TFPerceptionSource(
                    tf_buffer=raw.get("tf_buffer"),
                    frame_by_prop=dict(raw.get("frame_by_prop", {}) or {}),
                    target_frame=str(raw.get("target_frame", "world")),
                )
            if kind == "color_blob":
                from .perception import ColorBlobPerceptionSource

                return ColorBlobPerceptionSource(
                    camera=str(raw.get("camera", "wrist_camera")),
                    object_name=str(raw.get("object_name", "source")),
                    target_rgb=tuple(raw.get("target_rgb", (255, 0, 0))),
                    tolerance=int(raw.get("tolerance", 90)),
                    default_z=float(raw.get("default_z", 0.38)),
                    world_origin=tuple(raw.get("world_origin", (0.55, 0.0, 0.38))),
                    camera_to_world_scale=tuple(raw.get("camera_to_world_scale", (0.15, 0.15, 1.0))),
                )
        return None

    def _update_perception_source(self, obs: Observation) -> None:
        perception = getattr(self, "_perception_source", None)
        updater = getattr(perception, "update_obs", None)
        if callable(updater):
            updater(obs)

    def set_prop_pose(self, name: str, position, orientation) -> None:  # noqa: ANN001
        del position, orientation
        raise NotImplementedError(f"ROS2RealBackend cannot teleport physical prop '{name}'.")

    def teleport_object(self, name: str, position: tuple[float, float, float]) -> None:
        del position
        raise NotImplementedError(f"ROS2RealBackend cannot teleport physical prop '{name}'.")

    # ------------------------------------------------------------------
    # Optional hook for RoboEnv to provide task-goal visualization payload
    # ------------------------------------------------------------------

    def set_viz_payload(self, payload: Optional[dict]) -> None:
        self._latest_viz_payload = payload

    def get_diagnostics(self) -> dict:
        result = dict(self._diagnostics)
        result["control_hz"] = self.control_hz
        result["robot_count"] = len(getattr(self, "_drivers", {}))
        result["sensors"] = self._collect_sensor_diagnostics()
        result.update(self._sensor_diagnostics())
        return result

    def _collect_sensor_diagnostics(self) -> dict:
        sensor_diag: dict[str, dict] = {}
        for rid, sensors in getattr(self, "_sensors_by_robot", {}).items():
            for sensor in sensors:
                getter = getattr(sensor, "get_diagnostics", None)
                name = str(getattr(sensor, "name", type(sensor).__name__))
                if callable(getter):
                    sensor_diag[f"{rid}/{name}"] = getter()
        for rid, sensors in getattr(self, "_robot_sensors_by_robot", {}).items():
            for sensor in sensors:
                getter = getattr(sensor, "get_diagnostics", None)
                name = str(getattr(sensor, "name", type(sensor).__name__))
                if callable(getter):
                    sensor_diag[f"{rid}/{name}"] = getter()
        for sensor in getattr(self, "_shared_sensors", []):
            getter = getattr(sensor, "get_diagnostics", None)
            name = str(getattr(sensor, "name", type(sensor).__name__))
            if callable(getter):
                sensor_diag[f"shared/{name}"] = getter()
        return sensor_diag

    def _capture_driver_diagnostics(self, robot_id: str) -> None:
        driver = self._drivers[robot_id]
        driver_diag = getattr(driver, "get_diagnostics", None)
        if not callable(driver_diag):
            return
        diag = driver_diag()
        self._diagnostics["robots"][robot_id] = diag
        timeout_s = float(diag.get("joint_state_timeout_s", 0.0) or 0.0)
        state_age_s = float(diag.get("last_joint_state_age_s", 0.0) or 0.0)
        ack_timeouts = int(diag.get("pending_ack_timeouts", 0) or 0)
        if ack_timeouts > 0:
            warning = (
                f"Robot '{robot_id}' has {ack_timeouts} command(s) without joint-state ack "
                f"within {float(diag.get('ack_timeout_s', 0.0) or 0.0):.3f}s."
            )
            warnings = self._diagnostics.setdefault("warnings", [])
            if warning not in warnings:
                warnings.append(warning)
            self._diagnostics.setdefault("safety", {})["last_hazard"] = Hazard.COMMAND_REJECTED.name
        if timeout_s > 0.0 and state_age_s > timeout_s:
            manager = self._recovery_manager_for(robot_id, timeout_s=timeout_s, driver=driver)
            try:
                manager.on_state_stale(state_age_s)
            except SafetyError as exc:
                self._diagnostics.setdefault("safety", {})["last_hazard"] = exc.violation.hazard.name
                raise
            warning = str(
                BackendTimeoutError(
                    robot_id,
                    timeout_s,
                    f"Latest ROS2 state age is {state_age_s:.3f}s; recovery in progress.",
                )
            )
            warnings = self._diagnostics.setdefault("warnings", [])
            if warning not in warnings:
                warnings.append(warning)

    def _recovery_manager_for(self, robot_id: str, *, timeout_s: float, driver) -> object:
        from .recovery import ROS2RecoveryManager

        existing = self._recovery_managers.get(robot_id)
        if existing is not None:
            return existing

        def _reconnect() -> bool:
            getter = getattr(driver, "get_obs", None)
            if not callable(getter):
                return False
            try:
                getter()
                fresh = driver.get_diagnostics()
                age = float(fresh.get("last_joint_state_age_s", 1e9) or 1e9)
                return age <= timeout_s
            except Exception:
                return False

        manager = ROS2RecoveryManager(
            reconnect_fn=_reconnect,
            state_timeout_s=timeout_s,
            max_retries=int(self.config.get("recovery_max_retries", 5)),
            initial_backoff_s=float(self.config.get("recovery_initial_backoff_s", 0.0)),
            max_backoff_s=float(self.config.get("recovery_max_backoff_s", 1.0)),
        )
        self._recovery_managers[robot_id] = manager
        return manager

    @staticmethod
    def _coerce_fake_sim_config(spec, robots) -> FakeJointPosSimConfig:
        """Build a FakeJointPosSimConfig from a dict / dataclass / single ns string."""
        if isinstance(spec, FakeJointPosSimConfig):
            return spec
        if isinstance(spec, str):
            return FakeJointPosSimConfig(robot_ns=spec)
        if not isinstance(spec, dict):
            raise TypeError(
                "backend_kwargs['dev_fake_sim'] must be a dict, FakeJointPosSimConfig, "
                f"or list of those — got {type(spec).__name__}."
            )

        kwargs = dict(spec)
        if "joint_names" not in kwargs:
            for robot in robots:
                if str(robot.robot_id) == kwargs.get("robot_ns", "/robot0").lstrip("/"):
                    jn = getattr(robot.description, "joint_names", None)
                    if jn:
                        kwargs["joint_names"] = tuple(jn)
                    break
            else:
                jn = getattr(robots[0].description, "joint_names", None) if robots else None
                if jn:
                    kwargs.setdefault("joint_names", tuple(jn))
        else:
            kwargs["joint_names"] = tuple(kwargs["joint_names"])
        return FakeJointPosSimConfig(**kwargs)

    @staticmethod
    def _parse_backend_config(config: dict) -> Ros2BackendConfig:
        rviz_cfg = (config.get("rviz") or {}) if isinstance(config.get("rviz"), dict) else {}
        return Ros2BackendConfig(
            spin_hz=float(config.get("spin_hz", 200.0)),
            rviz_enabled=bool(rviz_cfg.get("enabled", False)),
            rviz_fixed_frame=str(rviz_cfg.get("fixed_frame", "world")),
            rviz_publish_hz=float(rviz_cfg.get("publish_hz", 10.0)),
            rviz_launch_robot_state_publisher=bool(rviz_cfg.get("launch_robot_state_publisher", True)),
            controller_by_robot_id=dict(config.get("controller_by_robot_id", {}) or {}),
            command_hz=float(config.get("command_hz", 0.0) or 0.0),
            command_hz_by_robot_id=dict(config.get("command_hz_by_robot_id", {}) or {}),
        )


@register_backend("ros2_rviz")
class ROS2RvizBackend(ROS2RealBackend):
    """ROS2/RViz transport backed by fake or external simulated joint state."""

    is_real = False
    sensor_backend_name = "ros2_rviz"

