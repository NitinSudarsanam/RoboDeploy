"""Robot-agnostic MuJoCo MJX simulation engine."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from robodeploy.core.bridge import BaseRobot
from robodeploy.core.task import BaseTask
from robodeploy.core.types import Action, Observation

# JAX and MJX are imported lazily at first instantiation so that this module
# can be imported even in environments where JAX / mujoco-mjx are not installed
# (e.g. ros2_env used only for real-hardware deployment).
try:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx as _mjx
    _JAX_MJX_AVAILABLE = True
except ImportError:
    _JAX_MJX_AVAILABLE = False
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _mjx = None  # type: ignore[assignment]


def _require_jax_mjx() -> None:
    if not _JAX_MJX_AVAILABLE:
        raise ImportError(
            "MujocoEngine requires JAX and mujoco-mjx.\n"
            "Install them with:\n"
            "    pip install jax\n"
            "    pip install mujoco-mjx\n"
            "or use FrankaRealBackend for hardware-only deployments."
        )


class MujocoEngine(BaseRobot):
    """MJX-powered simulator that loads robots from robodeploy/robots/<name>/sim/.

    Args:
        robots: List of robot names to load (only the first is used currently).
        config: Optional configuration overrides. Recognized keys:
            - timestep (float): Physics timestep in seconds (default 0.002).
            - control_hz (float): Control loop frequency in Hz (default 100.0).
            - dof (int): Number of controlled DOF (defaults to model.nu).
            - ee_body (str): Name of the end-effector body in the MJCF.
            - default_qpos (list[float]): Home joint positions for reset.
        enable_viewer (bool): If True, open a passive MuJoCo viewer window.
    """

    def __init__(
        self,
        robots: Optional[list[str]] = None,
        config: Optional[dict] = None,
        enable_viewer: bool = False,
    ):
        _require_jax_mjx()
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

        self._enable_viewer = enable_viewer
        self._viewer = None

        self._model_path: Path
        self._dof = 0
        self._ee_body = -1
        self._gripper_actuator_indices: tuple[int, ...] = ()

        self._load_robot_model(self.robot_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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

    def _load_robot_config(self, robot_name: str) -> dict:
        """Load and merge robots/<name>/config.yaml into self.config (config wins)."""
        import yaml
        config_path = (
            self._repo_root / "robodeploy" / "robots" / robot_name / "config.yaml"
        )
        yaml_cfg: dict = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    yaml_cfg = yaml.safe_load(f) or {}
            except Exception:
                pass
        # explicit config dict overrides yaml values
        merged = {**yaml_cfg, **self.config}
        return merged

    def _find_arm_joint_qpos_indices(self) -> tuple[int, ...]:
        """Return qpos addresses for arm (non-gripper) actuated joints, in actuator order."""
        indices: list[int] = []
        for act_idx in range(self._model.nu):
            if act_idx in self._gripper_actuator_indices:
                continue
            joint_id = int(self._model.actuator_trnid[act_idx, 0])
            qpos_adr = int(self._model.jnt_qposadr[joint_id])
            indices.append(qpos_adr)
        return tuple(indices)

    def _load_robot_model(self, robot_name: str) -> None:
        self._model_path = self._resolve_model_path(robot_name)
        self._model = mujoco.MjModel.from_xml_path(str(self._model_path))
        self._model.opt.timestep = self.timestep
        self._data = mujoco.MjData(self._model)

        self._mjx_model = _mjx.put_model(self._model)
        self._mjx_data = _mjx.put_data(self._model, self._data)
        self._mjx_step = jax.jit(_mjx.step)

        self._ee_body = self._resolve_ee_body_id()
        self._gripper_actuator_indices = self._find_gripper_actuator_indices()

        # Merge config.yaml so callers don't need to specify robot-level params
        merged_cfg = self._load_robot_config(robot_name)

        self._arm_qpos_indices = self._find_arm_joint_qpos_indices()
        self._dof = len(self._arm_qpos_indices)

        self._default_qpos: list[float] = merged_cfg.get(
            "default_qpos", [0.0] * self._dof
        )
        self.robot_name = robot_name

    def _sync_mjx_to_cpu(self) -> None:
        """Copy current MJX state back into self._data for viewer/FK use."""
        self._data.qpos[:] = np.asarray(self._mjx_data.qpos)
        self._data.qvel[:] = np.asarray(self._mjx_data.qvel)
        self._data.ctrl[:] = np.asarray(self._mjx_data.ctrl)
        self._data.time = float(self._mjx_data.time)
        mujoco.mj_forward(self._model, self._data)

    def _compute_gripper_state(self) -> Optional[float]:
        """Return normalized gripper openness [0=open, 1=closed] from finger joints."""
        if not self._gripper_actuator_indices:
            return None
        # Read first finger ctrl value and normalize to [0, 1]
        idx = self._gripper_actuator_indices[0]
        lo, hi = float(self._model.actuator_ctrlrange[idx, 0]), float(self._model.actuator_ctrlrange[idx, 1])
        if math.isclose(hi, lo):
            return 0.0
        ctrl_val = float(self._mjx_data.ctrl[idx])
        # hi = fully open (0.04 m), lo = fully closed (0.0 m) for Franka fingers
        # Normalize: 0.0=open (hi), 1.0=closed (lo)
        return float(1.0 - (ctrl_val - lo) / (hi - lo))

    # ------------------------------------------------------------------
    # BaseRobot interface
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        await super().initialize()
        mujoco.mj_resetData(self._model, self._data)
        self._mjx_data = _mjx.put_data(self._model, self._data)

        if self._enable_viewer:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

    async def get_obs(self, tasks: Optional[list[BaseTask]] = None) -> Observation:
        qpos = self._mjx_data.qpos
        qvel = self._mjx_data.qvel
        qfrc = self._mjx_data.qfrc_actuator

        ee_pos = self._mjx_data.xpos[self._ee_body]
        ee_quat = self._mjx_data.xquat[self._ee_body]

        ee_vel = jnp.zeros((3,), dtype=jnp.float32)
        ee_ang_vel = jnp.zeros((3,), dtype=jnp.float32)

        rgb = None
        depth = None
        if tasks and any(t.get_observation_spec().get("rgb", False) for t in tasks):
            rgb = jnp.zeros((128, 128, 3), dtype=jnp.uint8)
        if tasks and any(t.get_observation_spec().get("depth", False) for t in tasks):
            depth = jnp.zeros((128, 128), dtype=jnp.float32)

        arm_qpos = jnp.stack([qpos[i] for i in self._arm_qpos_indices])
        arm_qvel = jnp.stack([qvel[i] for i in self._arm_qpos_indices])
        arm_qfrc = jnp.stack([qfrc[i] for i in self._arm_qpos_indices])

        return Observation(
            joint_positions=jnp.asarray(arm_qpos, dtype=jnp.float32),
            joint_velocities=jnp.asarray(arm_qvel, dtype=jnp.float32),
            joint_torques=jnp.asarray(arm_qfrc, dtype=jnp.float32),
            ee_position=jnp.asarray(ee_pos, dtype=jnp.float32),
            ee_orientation=jnp.asarray(ee_quat, dtype=jnp.float32),
            ee_velocity=ee_vel,
            ee_angular_velocity=ee_ang_vel,
            rgb=rgb,
            depth=depth,
            gripper_state=self._compute_gripper_state(),
            timestamp=float(self._mjx_data.time),
        )

    async def apply_action(self, action: Action) -> None:
        ctrl = self._mjx_data.ctrl

        if action.joint_positions is not None:
            targets = jnp.asarray(action.joint_positions, dtype=ctrl.dtype)
            n = min(targets.shape[0], ctrl.shape[0])
            ctrl = ctrl.at[:n].set(targets[:n])

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
            n = min(torques.shape[0], self._mjx_data.qfrc_applied.shape[0])
            qfrc = self._mjx_data.qfrc_applied.at[:n].set(torques[:n])
            self._mjx_data = self._mjx_data.replace(qfrc_applied=qfrc)

        self._mjx_data = self._mjx_data.replace(ctrl=ctrl)

        for _ in range(self.steps_per_control):
            self._mjx_data = self._mjx_step(self._mjx_model, self._mjx_data)

        if self._viewer is not None and self._viewer.is_running():
            self._sync_mjx_to_cpu()
            self._viewer.sync()

    async def reset(self) -> Observation:
        mujoco.mj_resetData(self._model, self._data)
        self._mjx_data = _mjx.put_data(self._model, self._data)

        home = self._default_qpos[: self._dof]
        qpos = self._mjx_data.qpos
        for arm_idx, qpos_addr in enumerate(self._arm_qpos_indices):
            val = jnp.asarray(home[arm_idx] if arm_idx < len(home) else 0.0,
                              dtype=qpos.dtype)
            qpos = qpos.at[qpos_addr].set(val)
        self._mjx_data = self._mjx_data.replace(qpos=qpos)

        return await self.get_obs()

    async def shutdown(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        await super().shutdown()

    # ------------------------------------------------------------------
    # Utility / inspection
    # ------------------------------------------------------------------

    def set_physics(self, gravity: list[float]) -> None:
        self._model.opt.gravity[:] = gravity
        self._mjx_model = _mjx.put_model(self._model)

    def teleport_object(self, body_name: str, position: list[float]) -> None:
        joint_name = f"{body_name}_joint"
        j_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if j_id < 0:
            raise ValueError(f"Free joint '{joint_name}' was not found.")

        qpos_adr = self._model.jnt_qposadr[j_id]
        qpos = self._mjx_data.qpos.at[qpos_adr : qpos_adr + 3].set(
            jnp.asarray(position, dtype=self._mjx_data.qpos.dtype)
        )
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
            "viewer_enabled": self._enable_viewer,
        }
