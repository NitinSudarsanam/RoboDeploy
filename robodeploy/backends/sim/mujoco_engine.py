"""Robot-agnostic MuJoCo MJX simulation engine."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from robodeploy.core.bridge import BaseRobot
from robodeploy.core.task import BaseTask
from robodeploy.core.types import Action, Observation


class MujocoEngine(BaseRobot):
    """MJX-powered simulator that loads robots from robodeploy/robots/<name>/sim/."""

    def __init__(self, robots: Optional[list[str]] = None, config: Optional[dict] = None):
        super().__init__(config)
        self._repo_root = Path(__file__).resolve().parents[3]
        self._available_robots = self._discover_available_robots()

        self.robots = robots or ["franka"]
        if not self.robots:
            raise ValueError("At least one robot name must be provided.")

        self.robot_name = self.robots[0]
        if self.robot_name not in self._available_robots:
            raise ValueError(
                f"Unknown robot '{self.robot_name}'. Available: {self._available_robots}"
            )

        self.timestep = float(self.config.get("timestep", 0.002))
        self.control_hz = float(self.config.get("control_hz", 100.0))
        self.steps_per_control = max(1, int((1.0 / self.control_hz) / self.timestep))

        self._model_path: Path
        self._dof = 0
        self._ee_body = -1
        self._gripper_actuator_indices: tuple[int, ...] = ()

        self._load_robot_model(self.robot_name)

    def _discover_available_robots(self) -> list[str]:
        robots_dir = self._repo_root / "robodeploy" / "robots"
        names: list[str] = []
        if not robots_dir.exists():
            return names

        for robot_dir in sorted(robots_dir.iterdir()):
            if not robot_dir.is_dir():
                continue
            sim_dir = robot_dir / "sim"
            if sim_dir.exists() and any(p.suffix == ".xml" for p in sim_dir.glob("*.xml")):
                names.append(robot_dir.name)
        return names

    def _resolve_model_path(self, robot_name: str) -> Path:
        sim_dir = self._repo_root / "robodeploy" / "robots" / robot_name / "sim"
        xml_candidates = sorted(sim_dir.glob("*.xml"))
        if not xml_candidates:
            raise FileNotFoundError(
                f"Could not find MJCF for robot '{robot_name}' under {sim_dir}"
            )
        return xml_candidates[0]

    def _find_gripper_actuator_indices(self) -> tuple[int, ...]:
        gripper_indices: list[int] = []
        for idx in range(self._model.nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx)
            if not name:
                continue
            lowered = name.lower()
            if "grip" in lowered or "finger" in lowered:
                gripper_indices.append(idx)
        return tuple(gripper_indices)

    def _resolve_ee_body_id(self) -> int:
        candidate_names = [
            str(self.config.get("ee_body", "")),
            "robot0/ee_link",
            "robot0/tool0",
            "robot0/flange",
            "ee_link",
            "tool0",
        ]

        for body_name in candidate_names:
            if not body_name:
                continue
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                return body_id

        raise ValueError(
            "End-effector body not found. Set config['ee_body'] to a valid MuJoCo body name."
        )

    def _load_robot_model(self, robot_name: str) -> None:
        self._model_path = self._resolve_model_path(robot_name)
        self._model = mujoco.MjModel.from_xml_path(str(self._model_path))
        self._model.opt.timestep = self.timestep
        self._data = mujoco.MjData(self._model)

        self._mjx_model = mjx.put_model(self._model)
        self._mjx_data = mjx.put_data(self._model, self._data)
        self._mjx_step = jax.jit(mjx.step)

        configured_dof = self.config.get("dof")
        if configured_dof is None:
            configured_dof = self._model.nu
        self._dof = int(configured_dof)

        self._ee_body = self._resolve_ee_body_id()
        self._gripper_actuator_indices = self._find_gripper_actuator_indices()
        self.robot_name = robot_name

    async def initialize(self) -> None:
        await super().initialize()
        mujoco.mj_resetData(self._model, self._data)
        self._mjx_data = mjx.put_data(self._model, self._data)

    async def get_obs(self, tasks: Optional[list[BaseTask]] = None) -> Observation:
        qpos = self._mjx_data.qpos
        qvel = self._mjx_data.qvel
        qfrc = self._mjx_data.qfrc_actuator

        ee_pos = self._mjx_data.xpos[self._ee_body]
        ee_quat = self._mjx_data.xquat[self._ee_body]

        # Basic engine keeps velocity terms lightweight and JAX-native.
        ee_vel = jnp.zeros((3,), dtype=jnp.float32)
        ee_ang_vel = jnp.zeros((3,), dtype=jnp.float32)

        rgb = None
        depth = None
        if tasks and any(t.get_observation_spec().get("rgb", False) for t in tasks):
            rgb = jnp.zeros((128, 128, 3), dtype=jnp.uint8)
        if tasks and any(t.get_observation_spec().get("depth", False) for t in tasks):
            depth = jnp.zeros((128, 128), dtype=jnp.float32)

        return Observation(
            joint_positions=jnp.asarray(qpos[: self._dof], dtype=jnp.float32),
            joint_velocities=jnp.asarray(qvel[: self._dof], dtype=jnp.float32),
            joint_torques=jnp.asarray(qfrc[: self._dof], dtype=jnp.float32),
            ee_position=jnp.asarray(ee_pos, dtype=jnp.float32),
            ee_orientation=jnp.asarray(ee_quat, dtype=jnp.float32),
            ee_velocity=ee_vel,
            ee_angular_velocity=ee_ang_vel,
            rgb=rgb,
            depth=depth,
            timestamp=float(self._mjx_data.time),
        )

    async def apply_action(self, action: Action) -> None:
        ctrl = self._mjx_data.ctrl

        if action.joint_positions is not None:
            targets = jnp.asarray(action.joint_positions, dtype=ctrl.dtype)
            ctrl = ctrl.at[: self._dof].set(targets[: self._dof])

        if action.gripper is not None and self._gripper_actuator_indices:
            # 0.0=open, 1.0=closed mapped to finger joint targets [0, 0.04].
            closed = jnp.asarray(0.0, dtype=ctrl.dtype)
            opened = jnp.asarray(0.04, dtype=ctrl.dtype)
            gval = jnp.asarray(action.gripper, dtype=ctrl.dtype)
            finger_target = opened * (1.0 - gval) + closed * gval

            for idx in self._gripper_actuator_indices:
                ctrl = ctrl.at[idx].set(finger_target)

        if action.joint_torques is not None:
            torques = jnp.asarray(action.joint_torques, dtype=self._mjx_data.qfrc_applied.dtype)
            qfrc = self._mjx_data.qfrc_applied.at[: self._dof].set(torques[: self._dof])
            self._mjx_data = self._mjx_data.replace(qfrc_applied=qfrc)

        self._mjx_data = self._mjx_data.replace(ctrl=ctrl)

        for _ in range(self.steps_per_control):
            self._mjx_data = self._mjx_step(self._mjx_model, self._mjx_data)

    async def reset(self) -> Observation:
        mujoco.mj_resetData(self._model, self._data)
        self._mjx_data = mjx.put_data(self._model, self._data)

        default_qpos = self.config.get("default_qpos")
        if default_qpos is None:
            default_qpos = [0.0] * self._dof

        home = jnp.asarray(
            default_qpos,
            dtype=self._mjx_data.qpos.dtype,
        )
        qpos = self._mjx_data.qpos.at[: self._dof].set(home[: self._dof])
        self._mjx_data = self._mjx_data.replace(qpos=qpos)

        return await self.get_obs()

    def set_physics(self, gravity: list[float]) -> None:
        self._model.opt.gravity[:] = gravity
        self._mjx_model = mjx.put_model(self._model)

    def teleport_object(self, body_name: str, position: list[float]) -> None:
        joint_name = f"{body_name}_joint"
        j_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if j_id < 0:
            raise ValueError(f"Free joint '{joint_name}' was not found.")

        qpos_adr = self._model.jnt_qposadr[j_id]
        qpos = self._mjx_data.qpos.at[qpos_adr : qpos_adr + 3].set(jnp.asarray(position, dtype=self._mjx_data.qpos.dtype))
        self._mjx_data = self._mjx_data.replace(qpos=qpos)

    def switch_robot(self, robot_name: str, should_reset: bool = True) -> None:
        if robot_name not in self._available_robots:
            raise ValueError(
                f"Unknown robot '{robot_name}'. Available: {self._available_robots}"
            )

        self._load_robot_model(robot_name)
        if should_reset:
            mujoco.mj_resetData(self._model, self._data)
            self._mjx_data = mjx.put_data(self._model, self._data)

    def get_available_robots(self) -> list[str]:
        return list(self._available_robots)

    def get_info(self) -> dict:
        return {
            "robot": self.robot_name,
            "available_robots": self._available_robots,
            "model_path": str(self._model_path),
            "control_hz": self.control_hz,
            "timestep": self.timestep,
            "dof": self._dof,
            "gripper_actuators": list(self._gripper_actuator_indices),
        }
