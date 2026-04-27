"""MuJoCoBackend — minimal runnable MuJoCo simulation backend.

This backend is intentionally small: it exists so users can run real scripts
through the `RoboEnv` contract with a MuJoCo world, swap in other backends,
and keep tasks/policies backend-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from robodeploy.backends.base import BackendBase
from robodeploy.core.registry import register_backend
from robodeploy.core.spaces import ActionSpace, AssetFormat
from robodeploy.core.types import Action, Observation, SceneSpec

if TYPE_CHECKING:
    from robodeploy.core.interfaces.sensor import ISensor
    from robodeploy.description.base import RobotDescription


@register_backend("mujoco")
class MuJoCoBackend(BackendBase):
    """MuJoCo simulation backend (minimal)."""

    is_real = False
    control_hz = 100.0
    supported_action_spaces = [ActionSpace.JOINT_POS, ActionSpace.JOINT_TORQUE]

    def initialize_multi(self, robots, scene: SceneSpec, shared_sensors) -> None:  # type: ignore[override]
        if len(robots) != 1:
            raise NotImplementedError("MuJoCoBackend currently supports a single robot per backend instance.")
        if shared_sensors:
            raise NotImplementedError("MuJoCoBackend does not support shared sensors yet.")
        robot = robots[0]
        self._robot_id = str(robot.robot_id or "robot0")
        super().initialize(robot.description, scene, robot.sensors)

    def _load(
        self,
        description: RobotDescription,
        scene: SceneSpec,
        sensors: list[ISensor],
    ) -> None:
        del scene
        del sensors
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
        try:
            mjcf_path = self._resolve_asset_path(self._robot_id, description, AssetFormat.MJCF, variant="sim")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "MuJoCoBackend needs an MJCF asset.\n"
                "Provide one of:\n"
                "- RobotDescription.asset_path(AssetFormat.MJCF, variant='sim')\n"
                "- backend config override: config={'asset_overrides': {'robot0': {'mjcf': '/path/to/model.xml'}}}\n"
                "URDF-only descriptions are supported as canonical input, but MuJoCo requires MJCF or a conversion step."
            ) from exc
        self._model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self._data = mujoco.MjData(self._model)

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
            ))
            self._rviz_bridge.start()
            try:
                self._rviz_bridge.publish_scene(self._scene)
            except Exception:
                pass

    def _reset_impl(self) -> Observation:
        mujoco = self._mujoco
        mujoco.mj_resetData(self._model, self._data)
        self._set_home_qpos()
        mujoco.mj_forward(self._model, self._data)
        obs = self._build_obs()
        if self._rviz_bridge is not None:
            self._rviz_bridge.publish_robot_state("robot0", obs)
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

        if self._viewer is not None:
            try:
                self._viewer.sync()
            except Exception:
                pass

        obs = self._build_obs()
        if self._rviz_bridge is not None:
            self._rviz_bridge.publish_robot_state("robot0", obs)
            if self._latest_viz_payload is not None:
                self._rviz_bridge.publish_task_viz(self._latest_viz_payload)
        return obs

    def _get_obs_impl(self) -> Observation:
        return self._build_obs()

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

    def _set_home_qpos(self) -> None:
        # Set joint qpos for named joints
        home = getattr(self._description, "home_qpos", None)
        if home is None:
            return
        for i, addr in enumerate(self._qpos_addr):
            self._data.qpos[addr] = float(home[i])

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

