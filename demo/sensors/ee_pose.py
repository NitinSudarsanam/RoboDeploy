"""EE pose sensor — FK from joint encoders (not backend body_xpos oracle)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from robodeploy.core.registry import register_sensor, register_sensor_pair
from robodeploy.core.types import SensorData
from robodeploy.sensors.base import SensorBase


def _mujoco_fk_from_q(backend: Any, q: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """FK from encoder qpos via MuJoCo mj_forward (no Pinocchio required)."""
    model = getattr(backend, "_model", None)
    data = getattr(backend, "_data", None)
    addrs = getattr(backend, "_qpos_addr", None)
    ee_id = getattr(backend, "_ee_body_id", None)
    if model is None or data is None or not addrs or ee_id is None:
        return None
    try:
        import mujoco
    except ImportError:
        return None
    saved = [float(data.qpos[a]) for a in addrs]
    try:
        for i, addr in enumerate(addrs):
            data.qpos[addr] = float(q[i])
        mujoco.mj_forward(model, data)
        pos = np.asarray(data.xpos[int(ee_id)], dtype=np.float32).copy()
        quat = np.asarray(data.xquat[int(ee_id)], dtype=np.float32).copy()
        return pos, quat
    finally:
        for i, addr in enumerate(addrs):
            data.qpos[addr] = saved[i]
        mujoco.mj_forward(model, data)


def _ee_from_ros_transport(backend: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """EE pose from ROS2 driver obs (TF + joint_states via robot_state_publisher)."""
    latest = getattr(backend, "_latest_obs", None)
    if isinstance(latest, dict) and latest:
        obs = next(iter(latest.values()))
        ee_pos = getattr(obs, "ee_position", None)
        ee_quat = getattr(obs, "ee_orientation", None)
        if ee_pos is not None:
            pos = np.asarray(ee_pos, dtype=np.float32).reshape(3)
            if np.isfinite(pos).all():
                if ee_quat is not None:
                    quat = np.asarray(ee_quat, dtype=np.float32).reshape(4)
                    if not np.isfinite(quat).all():
                        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                else:
                    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                return pos, quat
    drivers = getattr(backend, "_drivers", None)
    if not isinstance(drivers, dict) or not drivers:
        return None
    driver = next(iter(drivers.values()))
    tf_buffer = getattr(driver, "_tf_buffer", None)
    base_frame = getattr(driver, "base_frame", None) or getattr(driver, "_base_frame", None)
    ee_frame = getattr(driver, "ee_frame", None) or getattr(driver, "_ee_frame", None)
    if tf_buffer is None or not base_frame or not ee_frame:
        return None
    try:
        import rclpy.time
        from rclpy.duration import Duration

        tf_stamped = tf_buffer.lookup_transform(
            str(base_frame),
            str(ee_frame),
            rclpy.time.Time(),
            timeout=Duration(seconds=0.25),
        )
        tr = tf_stamped.transform.translation
        rot = tf_stamped.transform.rotation
        pos = np.array([tr.x, tr.y, tr.z], dtype=np.float32)
        quat = np.array([rot.w, rot.x, rot.y, rot.z], dtype=np.float32)
        if np.isfinite(pos).all() and np.isfinite(quat).all():
            return pos, quat
    except Exception:
        return None
    return None


def _prefer_world_fk(backend: Any) -> bool:
    """True when EE should be expressed in world frame (matches scene prop poses)."""
    cfg = getattr(backend, "config", None) or {}
    if bool(cfg.get("prefer_fk_ee_pose", False)):
        return True
    for rid_key, frame in cfg.items():
        if str(rid_key).endswith(".base_frame") and str(frame).strip().lower() == "world":
            return True
    for driver in (getattr(backend, "_drivers", None) or {}).values():
        if str(getattr(driver, "base_frame", "")).strip().lower() == "world":
            return True
    return False


def _pin_fk_from_q(fk: Any, q: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        pos = fk.fk_position(q)
        _, quat = fk._solver.fk(q)
        quat_arr = np.asarray(quat, dtype=np.float32).reshape(4)
        return pos, quat_arr
    except Exception:
        return None


def _joint_positions_from_backend(backend: Any) -> np.ndarray | None:
    """Read encoder positions from the active backend (proprioception path)."""
    latest = getattr(backend, "_latest_obs", None)
    if isinstance(latest, dict) and latest:
        obs = next(iter(latest.values()))
        q = getattr(obs, "joint_positions", None)
        if q is not None:
            return np.asarray(q, dtype=np.float64).reshape(-1)
    data = getattr(backend, "_data", None)
    addrs = getattr(backend, "_qpos_addr", None)
    if data is not None and addrs:
        return np.array([data.qpos[a] for a in addrs], dtype=np.float64)
    return None


@register_sensor("ee_pose_sim")
class EePoseSensor(SensorBase):
    """Publish ``obs.ee_pose`` from FK(joint_positions) + URDF (mimics robot_state_publisher)."""

    def __init__(self, name: str | dict | None = None, *, config: dict | None = None) -> None:
        if isinstance(name, dict) and config is None:
            cfg = dict(name)
            sensor_name = str(cfg.get("name", "ee_pose"))
        else:
            cfg = dict(config or {})
            sensor_name = str(name or cfg.get("name", "ee_pose"))
        super().__init__(name=sensor_name, is_real=False, config=cfg)
        self._backend = None
        self._fk = None
        self._prefer_mujoco_fk = False
        self._pos_noise = float(cfg.get("position_noise_std", 0.0))

    def _init_impl(self, backend) -> None:
        self._backend = backend
        self._prefer_mujoco_fk = (
            getattr(backend, "_model", None) is not None and getattr(backend, "_data", None) is not None
        )
        if self._prefer_mujoco_fk:
            return
        desc = getattr(backend, "_description", None)
        if desc is None:
            robots = getattr(backend, "_robots", None)
            if robots:
                desc = robots[0].description
        if desc is None:
            return
        try:
            from robodeploy.kinematics.pin_ik import PinIkSolver

            self._fk = PinIkSolver(desc.get_kinematics_solver())
        except Exception:
            self._fk = None

    def _read_impl(self) -> SensorData:
        assert self._backend is not None
        q = _joint_positions_from_backend(self._backend)
        ts = time.monotonic()
        data = getattr(self._backend, "_data", None)
        if data is not None and hasattr(data, "time"):
            ts = float(data.time)
        if q is None:
            return SensorData(
                timestamp=ts,
                timestamp_hw=ts,
                timestamp_recv=time.monotonic(),
                timestamp_source="sim",
                status="stale",
            )
        pos: np.ndarray | None = None
        quat_arr: np.ndarray | None = None
        prefer_world = _prefer_world_fk(self._backend)
        if self._prefer_mujoco_fk:
            mj_fk = _mujoco_fk_from_q(self._backend, q)
            if mj_fk is not None:
                pos, quat_arr = mj_fk
        if pos is None and prefer_world and self._fk is not None:
            pin_fk = _pin_fk_from_q(self._fk, q)
            if pin_fk is not None:
                pos, quat_arr = pin_fk
        if pos is None and not self._prefer_mujoco_fk and not prefer_world:
            ros_ee = _ee_from_ros_transport(self._backend)
            if ros_ee is not None:
                pos, quat_arr = ros_ee
        if pos is None and self._fk is not None:
            pin_fk = _pin_fk_from_q(self._fk, q)
            if pin_fk is not None:
                pos, quat_arr = pin_fk
        if pos is None:
            mj_fk = _mujoco_fk_from_q(self._backend, q)
            if mj_fk is not None:
                pos, quat_arr = mj_fk
        if pos is None:
            return SensorData(
                timestamp=ts,
                timestamp_hw=ts,
                timestamp_recv=time.monotonic(),
                timestamp_source="sim",
                status="stale",
            )
        if self._pos_noise > 0.0:
            rng = np.random.default_rng(int(self.config.get("noise_seed", 0)))
            pos = pos + rng.normal(0.0, self._pos_noise, size=3).astype(np.float32)
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]
        return SensorData(
            ee_pose=jnp.asarray(pos, dtype=jnp.float32),
            ee_pose_orientation=jnp.asarray(quat_arr, dtype=jnp.float32),
            timestamp=ts,
            timestamp_hw=ts,
            timestamp_recv=time.monotonic(),
            timestamp_source="sim",
            status="ok",
        )

    def warmup(self, n_frames: int = 3) -> None:
        """Prime cache with a valid FK sample (avoids first-read stale on sim reset)."""
        attempts = max(1, int(n_frames))
        for _ in range(attempts):
            try:
                data = self._read_impl()
            except Exception:
                continue
            if getattr(data, "ee_pose", None) is not None and str(getattr(data, "status", "ok")) == "ok":
                self._last_data = data
                return

    def _close_impl(self) -> None:
        self._backend = None
        self._fk = None
        self._prefer_mujoco_fk = False


@register_sensor_pair(
    "ee_pose",
    sim=EePoseSensor,
    by_backend={"mujoco": EePoseSensor, "ros2_rviz": EePoseSensor, "gazebo": EePoseSensor},
)
class EePosePair:
    pass
