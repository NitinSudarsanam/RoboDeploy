"""MuJoCoBackend — minimal runnable MuJoCo simulation backend.

This backend is intentionally small: it exists so users can run real scripts
through the `RoboEnv` contract with a MuJoCo world, swap in other backends,
and keep tasks/policies backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from robodeploy.backends.base import BackendBase
from robodeploy.core.registry import register_backend
from robodeploy.core.spaces import ActionSpace, AssetFormat
from robodeploy.core.types import Action, Observation, Pose3D, SceneSpec

from .multi_robot_builder import MultiRobotMjcfBuilder
from .scene_builder import MjcfSceneBuilder

if TYPE_CHECKING:
    from robodeploy.core.interfaces.sensor import ISensor
    from robodeploy.core.robot import Robot
    from robodeploy.description.base import RobotDescription


@dataclass
class _RobotRuntimeState:
    robot_id: str
    joint_names: list[str]
    ee_link_name: str
    home_qpos: list[float] = field(default_factory=list)
    joint_ids: list[int] = field(default_factory=list)
    qpos_addr: list[int] = field(default_factory=list)
    dof_addr: list[int] = field(default_factory=list)
    actuator_ids: list[int] = field(default_factory=list)
    ee_body_id: int = -1


@register_backend("mujoco")
class MuJoCoBackend(BackendBase):
    """MuJoCo simulation backend (minimal)."""

    is_real = False
    sensor_backend_name = "mujoco"
    control_hz = 100.0
    supported_action_spaces = [ActionSpace.JOINT_POS, ActionSpace.JOINT_TORQUE]

    def initialize_multi(self, robots: list["Robot"], scene: SceneSpec, shared_sensors) -> None:  # type: ignore[override]
        if len(robots) == 1:
            robot = robots[0]
            self._robot_id = str(robot.robot_id or "robot0")
            self._multi_mode = False
            super().initialize(robot.description, scene, [*robot.sensors, *shared_sensors])
            return
        self._multi_mode = True
        self._description = robots[0].description
        self._scene = scene
        sensors: list[ISensor] = []
        seen: set[int] = set()
        for robot in robots:
            for sensor in robot.sensors:
                if id(sensor) not in seen:
                    sensors.append(sensor)
                    seen.add(id(sensor))
        for sensor in shared_sensors or []:
            if id(sensor) not in seen:
                sensors.append(sensor)
                seen.add(id(sensor))
        self._sensors = sensors
        self._asset_selections.clear()
        self._load_multi(robots, scene, sensors)
        self._initialized = True

    def reset_multi(self, robot_ids: list[str] | None = None) -> list[Observation]:
        del robot_ids
        if not getattr(self, "_multi_mode", False):
            return [self.reset()]
        self._require_initialized("reset_multi")
        mujoco = self._mujoco
        mujoco.mj_resetData(self._model, self._data)
        self._grasp_prop = None
        for eq_id in getattr(self, "_grasp_eq_ids", {}).values():
            self._data.eq_active[eq_id] = 0
        self._set_home_qpos_multi()
        mujoco.mj_forward(self._model, self._data)
        obs_list = [self._obs_for_robot(rid) for rid in self._robot_order]
        self._episode_count += 1
        self._step_count = 0
        return obs_list

    def step_multi(self, actions: list[Action]) -> list[Observation]:
        if not getattr(self, "_multi_mode", False):
            if len(actions) != 1:
                raise ValueError(f"MuJoCoBackend.step_multi expected 1 action, got {len(actions)}.")
            return [self.step(actions[0])]
        self._require_initialized("step_multi")
        if len(actions) != len(self._robot_order):
            raise ValueError(
                f"MuJoCoBackend.step_multi expected {len(self._robot_order)} actions, got {len(actions)}."
            )
        mujoco = self._mujoco
        for rid, action in zip(self._robot_order, actions):
            state = self._robot_states[rid]
            if action.joint_positions is None:
                continue
            q = action.joint_positions
            for i, aid in enumerate(state.actuator_ids):
                self._data.ctrl[aid] = float(q[i])
        dt = float(self._model.opt.timestep)
        steps = max(1, int(round((1.0 / float(self.control_hz)) / dt)))
        for _ in range(steps):
            mujoco.mj_step(self._model, self._data)
            self._sync_grasped_prop()
        if self._viewer is not None:
            try:
                self._viewer.sync()
            except Exception:
                pass
        obs_list = [self._obs_for_robot(rid) for rid in self._robot_order]
        self._step_count += 1
        return obs_list

    def get_obs_multi(self) -> list[Observation]:
        if not getattr(self, "_multi_mode", False):
            return [self.get_obs()]
        self._require_initialized("get_obs_multi")
        return [self._obs_for_robot(rid) for rid in self._robot_order]

    def _load(
        self,
        description: RobotDescription,
        scene: SceneSpec,
        sensors: list[ISensor],
    ) -> None:
        try:
            import mujoco
        except Exception as exc:
            raise ImportError(
                "MuJoCoBackend requires the `mujoco` Python package.\n"
                "Install with:\n"
                "  pip install mujoco\n"
                f"Original error: {exc}"
            ) from exc

        self._mujoco = mujoco
        mjcf_path = None
        try:
            mjcf_path = self._resolve_asset_path(self._robot_id, description, AssetFormat.MJCF, variant="sim")
        except FileNotFoundError:
            mjcf_path = None

        world = scene.to_world()
        has_sensor_camera = any("camera" in type(sensor).__name__.lower() or "camera" in str(getattr(sensor, "name", "")).lower() for sensor in sensors)
        if mjcf_path is not None:
            builder = MjcfSceneBuilder(Path(mjcf_path).read_text(encoding="utf-8"), config=self.config)
            builder.ensure_compiler_meshdir(str(Path(mjcf_path).resolve().parent))
            builder.ensure_world_defaults(add_camera=not bool(world.cameras or has_sensor_camera))
            builder.attach_world(world)
            if bool(self.config.get("enable_grasp_welds", False)):
                builder.attach_grasp_welds(description.ee_link_name, world.props)
            builder.attach_sensors(sensors)
            self._model = mujoco.MjModel.from_xml_string(builder.emit())
        else:
            # Auto-path: load the URDF directly using MuJoCo's compiler.
            # This keeps robot definitions URDF-first while letting MuJoCo run without a hand-written MJCF.
            # Users can still supply an MJCF via asset_path/asset_overrides for full feature support.
            try:
                urdf_path = self._resolve_asset_path(self._robot_id, description, AssetFormat.URDF, variant="sim")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "MuJoCoBackend needs either an MJCF or a URDF asset.\n"
                    "Provide one of:\n"
                    "- RobotDescription.asset_path(AssetFormat.MJCF, variant='sim')\n"
                    "- RobotDescription.asset_path(AssetFormat.URDF, variant='sim')\n"
                    "- backend config override: config={'asset_overrides': {'robot0': {'mjcf': '/path/to/model.xml'}}}\n"
                ) from exc

            # Step 1: compile URDF once.
            urdf_model = mujoco.MjModel.from_xml_path(str(urdf_path))

            # Step 2: convert to MJCF text (MuJoCo's last compiled XML), then inject position actuators.
            # URDF import does not create actuators, but RoboDeploy's JOINT_POS control path expects them.
            xml_text = self._compile_mjcf_with_position_actuators(
                mujoco,
                urdf_model,
                joint_names=list(description.joint_names),
                meshdir=str(Path(urdf_path).resolve().parent),
                scene=scene,
                ee_link=description.ee_link_name,
            )

            # Step 3: recompile the augmented MJCF into the actual runtime model.
            self._model = mujoco.MjModel.from_xml_string(xml_text)

            # Best-effort cache: if MuJoCo exposes the last-compiled MJCF, write it out so users can iterate.
            # This is optional and failure is non-fatal.
            if bool(self.config.get("cache_compiled_mjcf", True)):
                try:
                    cache_dir = Path(self.config.get("compiled_cache_dir", Path.home() / ".robodeploy" / "mujoco")).expanduser()
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    out_path = cache_dir / f"{self._robot_id}_compiled.xml"
                    out_path.write_text(xml_text, encoding="utf-8")
                    self.config.setdefault("compiled_mjcf_path", str(out_path))
                except Exception:
                    pass
        self._data = mujoco.MjData(self._model)

        self._prop_body_ids: dict[str, int] = {}
        self._prop_qpos_addr: dict[str, int] = {}
        for prop in world.props:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, prop.name)
            if bid >= 0:
                self._prop_body_ids[prop.name] = int(bid)
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, f"{prop.name}_freejoint")
            if jid >= 0:
                self._prop_qpos_addr[prop.name] = int(self._model.jnt_qposadr[jid])

        self._joint_ids: list[int] = []
        self._qpos_addr: list[int] = []
        self._dof_addr: list[int] = []
        self._actuator_ids: list[int] = []

        for jname in description.joint_names:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise KeyError(f"MuJoCo joint not found in model: '{jname}'")
            self._joint_ids.append(jid)
            self._qpos_addr.append(int(self._model.jnt_qposadr[jid]))
            self._dof_addr.append(int(self._model.jnt_dofadr[jid]))

            # Prefer an actuator with matching joint name if present
            aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
            if aid < 0:
                if not bool(self.config.get("allow_actuator_name_fallback", False)):
                    raise KeyError(
                        f"MuJoCo actuator not found for joint '{jname}'. "
                        "Either name actuators the same as joints, or enable "
                        "`allow_actuator_name_fallback=True` to use the bundled "
                        f"'{self._robot_id}/act{{i}}' convention."
                    )
                # fallback: common naming used in bundled MJCF (`act1..act7`)
                idx = len(self._actuator_ids) + 1
                aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self._robot_id}/act{idx}")
            if aid < 0:
                raise KeyError(f"MuJoCo actuator not found for joint '{jname}'")
            self._actuator_ids.append(aid)

        self._ee_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, description.ee_link_name)
        self._grasp_prop: str | None = None
        self._grasp_mode: str = "follow"
        self._grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.03)
        self._grasp_eq_ids: dict[str, int] = {}
        if bool(self.config.get("enable_grasp_welds", False)):
            for prop_name in self._prop_body_ids:
                eq_name = f"grasp_{prop_name}"
                eid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
                if eid >= 0:
                    self._grasp_eq_ids[prop_name] = int(eid)
        if self._ee_body_id < 0:
            raise KeyError(f"MuJoCo body not found for ee_link_name '{description.ee_link_name}'")

        self._viewer: Optional[object] = None
        self._enable_viewer = bool(self.config.get("enable_viewer", False))
        if self._enable_viewer:
            try:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
            except Exception as exc:
                self._viewer = None
                raise RuntimeError(f"Failed to launch MuJoCo viewer: {exc}") from exc

        # Start at home
        self._set_home_qpos()
        mujoco.mj_forward(self._model, self._data)

        # Optional RViz bridge (keeps ROS imports out unless enabled).
        self._rviz_bridge = None
        self._latest_viz_payload: Optional[dict] = None
        rviz_cfg = (self.config.get("rviz") or {}) if isinstance(self.config.get("rviz"), dict) else {}
        if bool(rviz_cfg.get("enabled", False)):
            from .ros2_bridge import MujocoRos2Bridge, MujocoRos2BridgeConfig

            self._rviz_bridge = MujocoRos2Bridge(MujocoRos2BridgeConfig(
                fixed_frame=str(rviz_cfg.get("fixed_frame", "world")),
                publish_hz=float(rviz_cfg.get("publish_hz", 10.0)),
                namespace="/robodeploy",
                base_frame=description.ros_base_frame_id(),
            ))
            self._rviz_bridge.start()
            try:
                self._rviz_bridge.publish_scene(self._scene)
            except Exception:
                pass

    def _load_multi(self, robots: list["Robot"], scene: SceneSpec, sensors: list[ISensor]) -> None:
        try:
            import mujoco
        except Exception as exc:
            raise ImportError(
                "MuJoCoBackend requires the `mujoco` Python package.\n"
                "Install with:\n"
                "  pip install mujoco\n"
                f"Original error: {exc}"
            ) from exc

        self._mujoco = mujoco
        builder = MultiRobotMjcfBuilder(scene, config=self.config)
        for robot in robots:
            rid = str(robot.robot_id)
            desc = robot.description
            try:
                mjcf_path = self._resolve_asset_path(rid, desc, AssetFormat.MJCF, variant="sim")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"MuJoCo multi-robot mode requires MJCF assets for robot '{rid}'."
                ) from exc
            mjcf_path = Path(mjcf_path)
            builder.add_robot(
                rid,
                desc,
                mjcf_path.read_text(encoding="utf-8"),
                base_pose=robot.base_pose,
                meshdir=str(mjcf_path.resolve().parent),
            )

        xml_text = builder.finalize(sensors)
        self._model = mujoco.MjModel.from_xml_string(xml_text)
        self._data = mujoco.MjData(self._model)

        world = scene.to_world()
        self._prop_body_ids: dict[str, int] = {}
        self._prop_qpos_addr: dict[str, int] = {}
        for prop in world.props:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, prop.name)
            if bid >= 0:
                self._prop_body_ids[prop.name] = int(bid)
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, f"{prop.name}_freejoint")
            if jid >= 0:
                self._prop_qpos_addr[prop.name] = int(self._model.jnt_qposadr[jid])

        self._robot_order = [str(robot.robot_id) for robot in robots]
        self._robot_states: dict[str, _RobotRuntimeState] = {}
        for rid, sl in builder.robot_slices.items():
            state = _RobotRuntimeState(
                robot_id=rid,
                joint_names=list(sl.joint_names),
                ee_link_name=sl.ee_link_name,
                home_qpos=list(sl.home_qpos),
            )
            for jname in state.joint_names:
                jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    raise KeyError(f"MuJoCo joint not found in multi-robot model: '{jname}'")
                state.joint_ids.append(jid)
                state.qpos_addr.append(int(self._model.jnt_qposadr[jid]))
                state.dof_addr.append(int(self._model.jnt_dofadr[jid]))
                aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
                if aid < 0 and bool(self.config.get("allow_actuator_name_fallback", False)):
                    idx = len(state.actuator_ids) + 1
                    aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{rid}/act{idx}")
                if aid < 0:
                    raise KeyError(f"MuJoCo actuator not found for joint '{jname}'")
                state.actuator_ids.append(aid)
            state.ee_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, state.ee_link_name)
            if state.ee_body_id < 0:
                raise KeyError(f"MuJoCo body not found for ee_link '{state.ee_link_name}'")
            self._robot_states[rid] = state

        self._robot_id = self._robot_order[0]
        self._ee_body_id = self._robot_states[self._robot_id].ee_body_id
        self._grasp_prop: str | None = None
        self._grasp_mode: str = "follow"
        self._grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.03)
        self._grasp_eq_ids: dict[str, int] = {}
        self._viewer: Optional[object] = None
        self._enable_viewer = bool(self.config.get("enable_viewer", False))
        if self._enable_viewer:
            try:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
            except Exception as exc:
                self._viewer = None
                raise RuntimeError(f"Failed to launch MuJoCo viewer: {exc}") from exc
        self._set_home_qpos_multi()
        mujoco.mj_forward(self._model, self._data)
        self._rviz_bridge = None
        self._latest_viz_payload = None

    def _set_home_qpos_multi(self) -> None:
        for state in self._robot_states.values():
            home = state.home_qpos
            if not home:
                continue
            for i, addr in enumerate(state.qpos_addr):
                self._data.qpos[addr] = float(home[i])
            for i, aid in enumerate(state.actuator_ids):
                if i < len(home):
                    self._data.ctrl[aid] = float(home[i])

    def _obs_for_robot(self, robot_id: str) -> Observation:
        state = self._robot_states[robot_id]
        obs = self._build_obs_for_state(state)
        return self._merge_sensor_data(obs, self._sensors)

    def _build_obs_for_state(self, state: _RobotRuntimeState) -> Observation:
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        dof = len(state.dof_addr)
        qpos = jnp.asarray([self._data.qpos[a] for a in state.qpos_addr], dtype=jnp.float32)
        qvel = jnp.asarray([self._data.qvel[a] for a in state.dof_addr], dtype=jnp.float32)
        qfrc = jnp.asarray([self._data.qfrc_actuator[a] for a in state.dof_addr], dtype=jnp.float32)
        ee_pos = jnp.asarray(self._data.xpos[state.ee_body_id].copy(), dtype=jnp.float32)
        ee_quat = jnp.asarray(self._data.xquat[state.ee_body_id].copy(), dtype=jnp.float32)
        return Observation(
            joint_positions=qpos,
            joint_velocities=qvel,
            joint_torques=qfrc if qfrc.shape[0] == dof else jnp.zeros(dof, dtype=jnp.float32),
            ee_position=ee_pos,
            ee_orientation=ee_quat,
            ee_velocity=jnp.zeros(3, dtype=jnp.float32),
            ee_angular_velocity=jnp.zeros(3, dtype=jnp.float32),
            timestamp=float(self._data.time),
            timestamp_hw=float(self._data.time),
            timestamp_recv=float(self._data.time),
        )

    def _compile_mjcf_with_position_actuators(  # noqa: ANN001
        self,
        mujoco,
        model,
        *,
        joint_names: list[str],
        meshdir: str | None = None,
        scene: SceneSpec | None = None,
        ee_link: str | None = None,
    ) -> str:
        """Return MJCF XML with `<actuator><position joint=.../></actuator>` for each joint.

        MuJoCo can compile URDF directly, but the resulting model typically has zero actuators.
        RoboDeploy's MuJoCo backend expects actuators for JOINT_POS control, so we inject them.
        """
        builder = MjcfSceneBuilder.from_compiled_model(mujoco, model, config=self.config)
        builder.ensure_compiler_meshdir(meshdir)
        builder.stabilize_urdf_import()
        world = scene.to_world() if scene is not None else SceneSpec().to_world()
        builder.ensure_world_defaults(add_camera=not bool(world.cameras or getattr(self, "_sensors", [])))
        builder.attach_actuators(joint_names)
        builder.attach_world(world)
        if ee_link and bool(self.config.get("enable_grasp_welds", False)):
            builder.attach_grasp_welds(ee_link, world.props)
        builder.attach_sensors(list(getattr(self, "_sensors", [])))
        return builder.emit()

    def _reset_impl(self) -> Observation:
        mujoco = self._mujoco
        mujoco.mj_resetData(self._model, self._data)
        self._grasp_prop = None
        for eq_id in self._grasp_eq_ids.values():
            self._data.eq_active[eq_id] = 0
        self._set_home_qpos()
        mujoco.mj_forward(self._model, self._data)
        obs = self._merge_sensor_data(self._build_obs(), self._sensors)
        if self._rviz_bridge is not None:
            self._rviz_bridge.reset()
            self._rviz_bridge.publish_robot_state(self._robot_id, obs)
            if self._latest_viz_payload is not None:
                self._rviz_bridge.publish_task_viz(self._latest_viz_payload)
        return obs

    def _step_impl(self, action: Action) -> Observation:
        mujoco = self._mujoco

        if action.joint_positions is not None:
            q = action.joint_positions
            for i, aid in enumerate(self._actuator_ids):
                self._data.ctrl[aid] = float(q[i])

        # Step enough substeps to match control_hz based on model timestep
        dt = float(self._model.opt.timestep)
        steps = max(1, int(round((1.0 / float(self.control_hz)) / dt)))
        for _ in range(steps):
            mujoco.mj_step(self._model, self._data)
            self._sync_grasped_prop()

        if self._viewer is not None:
            try:
                self._viewer.sync()
            except Exception:
                pass

        obs = self._merge_sensor_data(self._build_obs(), self._sensors)
        if self._rviz_bridge is not None:
            self._rviz_bridge.publish_robot_state(self._robot_id, obs)
            if self._latest_viz_payload is not None:
                self._rviz_bridge.publish_task_viz(self._latest_viz_payload)
        return obs

    def _get_obs_impl(self) -> Observation:
        return self._merge_sensor_data(self._build_obs(), self._sensors)

    def _close_impl(self) -> None:
        if getattr(self, "_rviz_bridge", None) is not None:
            try:
                self._rviz_bridge.close()
            except Exception:
                pass
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    # Optional hook for RoboEnv to provide task-goal visualization payload.
    def set_viz_payload(self, payload: Optional[dict]) -> None:
        self._latest_viz_payload = payload

    def render(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.sync()
            except Exception:
                return

    def has_prop_contact(self, prop_name: str, *, other_body: str | None = None) -> bool:
        del other_body
        """Return True when MuJoCo reports contact between a prop and the EE body."""
        prop_bid = self._prop_body_ids.get(prop_name)
        if prop_bid is None or self._ee_body_id < 0:
            return False
        for i in range(int(self._data.ncon)):
            con = self._data.contact[i]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            b1 = int(self._model.geom_bodyid[g1])
            b2 = int(self._model.geom_bodyid[g2])
            if prop_bid in (b1, b2) and self._ee_body_id in (b1, b2):
                return True
        return False

    def prop_near_ee(self, prop_name: str, *, threshold: float = 0.06) -> bool:
        """Distance-based grasp proxy when explicit contacts are absent in sim."""
        if prop_name not in self._prop_body_ids or self._ee_body_id < 0:
            return False
        prop_pos = self._data.xpos[self._prop_body_ids[prop_name]]
        ee_pos = self._data.xpos[self._ee_body_id]
        dx = float(prop_pos[0] - ee_pos[0])
        dy = float(prop_pos[1] - ee_pos[1])
        dz = float(prop_pos[2] - ee_pos[2])
        return (dx * dx + dy * dy + dz * dz) ** 0.5 <= float(threshold)

    def attach_grasp_welds(self, prop_names: list[str]) -> None:
        """Mark props as weld-grasp candidates (welds emitted at scene build)."""
        self._grasp_weld_props = [str(name) for name in prop_names]

    def set_grasp_prop(
        self,
        prop_name: str | None,
        *,
        offset: tuple[float, float, float] | None = None,
        mode: str | None = None,
    ) -> None:
        """Grasp helper: ``follow`` (kinematic EE tracking) or ``weld`` (physics equality)."""
        mujoco = self._mujoco
        if self._grasp_prop and self._grasp_prop in self._grasp_eq_ids:
            self._data.eq_active[self._grasp_eq_ids[self._grasp_prop]] = 0
        self._grasp_prop = str(prop_name) if prop_name else None
        if offset is not None:
            self._grasp_offset = (float(offset[0]), float(offset[1]), float(offset[2]))
        if mode is not None:
            self._grasp_mode = str(mode).lower()
        if not self._grasp_prop:
            mujoco.mj_forward(self._model, self._data)
            return
        if (
            self._grasp_mode == "weld"
            and self._grasp_prop in self._grasp_eq_ids
        ):
            self._snap_prop_to_ee(self._grasp_prop)
            self._data.eq_active[self._grasp_eq_ids[self._grasp_prop]] = 1
            mujoco.mj_forward(self._model, self._data)

    def _snap_prop_to_ee(self, prop_name: str) -> None:
        ee_pos = self._data.xpos[self._ee_body_id]
        pos = (
            float(ee_pos[0]) + self._grasp_offset[0],
            float(ee_pos[1]) + self._grasp_offset[1],
            float(ee_pos[2]) + self._grasp_offset[2],
        )
        _, quat = self.get_prop_pose(prop_name)
        self.set_prop_pose(prop_name, pos, quat)

    def _sync_grasped_prop(self) -> None:
        if self._grasp_mode == "weld" or not self._grasp_prop or self._grasp_prop not in self._prop_body_ids:
            return
        ee_pos = self._data.xpos[self._ee_body_id]
        pos = (
            float(ee_pos[0]) + self._grasp_offset[0],
            float(ee_pos[1]) + self._grasp_offset[1],
            float(ee_pos[2]) + self._grasp_offset[2],
        )
        _, quat = self.get_prop_pose(self._grasp_prop)
        self.set_prop_pose(self._grasp_prop, pos, quat)

    def get_prop_names(self) -> list[str]:
        return sorted(self._prop_body_ids)

    def get_prop_pose(self, name: str):
        if name not in self._prop_body_ids:
            raise KeyError(f"Unknown MuJoCo prop '{name}'.")
        bid = self._prop_body_ids[name]
        return (
            tuple(float(v) for v in self._data.xpos[bid].copy()),
            tuple(float(v) for v in self._data.xquat[bid].copy()),
        )

    def set_prop_pose(self, name: str, position, orientation) -> None:  # noqa: ANN001
        if name not in self._prop_body_ids:
            raise KeyError(f"Unknown MuJoCo prop '{name}'.")
        pos = [float(v) for v in position]
        quat = [float(v) for v in orientation]
        if name in self._prop_qpos_addr:
            addr = self._prop_qpos_addr[name]
            self._data.qpos[addr : addr + 3] = pos
            self._data.qpos[addr + 3 : addr + 7] = quat
        else:
            bid = self._prop_body_ids[name]
            self._model.body_pos[bid] = pos
            self._model.body_quat[bid] = quat
        self._mujoco.mj_forward(self._model, self._data)

    def teleport_object(self, name: str, position: tuple[float, float, float]) -> None:
        _, quat = self.get_prop_pose(name)
        self.set_prop_pose(name, position, quat)

    def set_prop_mass(self, name: str, mass: float) -> None:
        if name not in self._prop_body_ids:
            raise KeyError(f"Unknown MuJoCo prop '{name}'.")
        self._model.body_mass[self._prop_body_ids[name]] = float(mass)

    def set_physics_params(self, **kwargs) -> None:
        if "gravity" in kwargs:
            self._model.opt.gravity[:] = [float(v) for v in kwargs["gravity"]]
        if "friction" in kwargs:
            scale = float(kwargs["friction"])
            self._model.geom_friction[:, 0] *= scale

    def _set_home_qpos(self) -> None:
        # Set joint qpos for named joints
        home = getattr(self._description, "home_qpos", None)
        if home is None:
            return
        for i, addr in enumerate(self._qpos_addr):
            self._data.qpos[addr] = float(home[i])

    def get_sim_state(self) -> dict:
        import numpy as np

        return {
            "qpos": np.asarray(self._data.qpos, dtype=np.float64).tolist(),
            "qvel": np.asarray(self._data.qvel, dtype=np.float64).tolist(),
            "ctrl": np.asarray(self._data.ctrl, dtype=np.float64).tolist(),
            "time": float(self._data.time),
        }

    def set_sim_state(self, state: dict) -> None:
        self._data.qpos[:] = state["qpos"]
        self._data.qvel[:] = state["qvel"]
        if "ctrl" in state:
            self._data.ctrl[:] = state["ctrl"]
        if "time" in state:
            self._data.time = float(state["time"])
        self._mujoco.mj_forward(self._model, self._data)

    def _build_obs(self) -> Observation:
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        dof = len(self._dof_addr)
        qpos = jnp.asarray([self._data.qpos[a] for a in self._qpos_addr], dtype=jnp.float32)
        qvel = jnp.asarray([self._data.qvel[a] for a in self._dof_addr], dtype=jnp.float32)
        # actuator forces are indexed by dof; use dof addresses for arm joints
        qfrc = jnp.asarray([self._data.qfrc_actuator[a] for a in self._dof_addr], dtype=jnp.float32)

        ee_pos = jnp.asarray(self._data.xpos[self._ee_body_id].copy(), dtype=jnp.float32)
        ee_quat = jnp.asarray(self._data.xquat[self._ee_body_id].copy(), dtype=jnp.float32)
        ee_vel = jnp.zeros(3, dtype=jnp.float32)
        ee_avel = jnp.zeros(3, dtype=jnp.float32)

        return Observation(
            joint_positions=qpos,
            joint_velocities=qvel,
            joint_torques=qfrc if qfrc.shape[0] == dof else jnp.zeros(dof, dtype=jnp.float32),
            ee_position=ee_pos,
            ee_orientation=ee_quat,
            ee_velocity=ee_vel,
            ee_angular_velocity=ee_avel,
            timestamp=float(self._data.time),
            timestamp_hw=float(self._data.time),
            timestamp_recv=float(self._data.time),
        )

