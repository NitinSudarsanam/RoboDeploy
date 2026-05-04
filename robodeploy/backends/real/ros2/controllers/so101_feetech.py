"""SO-101 follower: Feetech STS3215 bus via HuggingFace ``lerobot`` + ROS2 state/command topics."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import numpy as np

from robodeploy.backends.real.common import Commander, StateCache
from robodeploy.backends.real.ros2.safety import EStop, JointLimitGuard, SafetyError, TemperatureGuard, Watchdog
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.description.so101.calibration import SO101Calibration
from robodeploy.ros2 import Ros2NodeAdapter

from ._clamp import slew_limit_command
from .base import ControllerConfig, register_controller


def _import_feetech_stack() -> tuple[type, Any, Any]:
    """Return ``(FeetechMotorsBus, Motor, MotorNormMode)`` for current ``lerobot`` layout."""
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        return FeetechMotorsBus, Motor, MotorNormMode
    except ImportError as e_new:
        try:
            from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus  # type: ignore[no-redef]

            Motor = None  # type: ignore[assignment]
            MotorNormMode = None  # type: ignore[assignment]
            return FeetechMotorsBus, Motor, MotorNormMode
        except ImportError as e_old:
            raise ImportError(
                "SO-101 real adapter requires `lerobot` with Feetech support. Install with:\n"
                '  pip install "lerobot[feetech]"'
            ) from e_new


def _build_motors_dict(Motor: Any, MotorNormMode: Any) -> dict[str, Any]:
    if Motor is None or MotorNormMode is None:
        raise ImportError(
            "This lerobot version is too old for the SO-101 adapter (missing Motor / MotorNormMode). "
            "Upgrade lerobot to >= 0.1 with `src/lerobot/motors/` layout."
        )
    motors: dict[str, Any] = {}
    norm = MotorNormMode.DEGREES
    for i in range(1, 7):
        motors[str(i)] = Motor(id=i, model="sts3215", norm_mode=norm)
    return motors


class SO101FeetechControllerAdapter(Ros2NodeAdapter):
    controller_type = "so101_feetech"
    supported_action_spaces = [ActionSpace.JOINT_POS]

    def __init__(self, cfg: ControllerConfig, backend_config: Optional[dict] = None) -> None:
        super().__init__()
        self._cfg = cfg
        self._backend_config = backend_config or {}

        self._robot_id = cfg.robot_id
        self._ns = (cfg.namespace or "").rstrip("/")
        self._cmd_hz = float(cfg.command_hz or 0.0)
        self._state_hz = float(cfg.state_hz or 0.0) or (self._cmd_hz if self._cmd_hz > 0 else 50.0)

        self._max_vel = (
            np.asarray(cfg.max_joint_velocity, dtype=np.float64).reshape(-1)
            if cfg.max_joint_velocity is not None
            else None
        )
        self._jv_guard = (
            np.asarray(cfg.joint_velocity_limits, dtype=np.float64).reshape(-1)
            if cfg.joint_velocity_limits is not None
            else np.full(6, 10.0, dtype=np.float64)
        )

        self._home = (
            np.asarray(cfg.home_qpos, dtype=np.float64).reshape(-1)
            if cfg.home_qpos is not None
            else np.zeros(6, dtype=np.float64)
        )
        if self._home.shape[0] != 6:
            raise ValueError("home_qpos must have length 6 for SO101FeetechControllerAdapter.")

        self._lock = threading.Lock()
        self._q = np.zeros(6, dtype=np.float64)
        self._qd = np.zeros(6, dtype=np.float64)
        self._tau = np.zeros(6, dtype=np.float64)

        self._obs_cache = StateCache()
        self._commander = Commander(
            self._apply_hardware_goal,
            min_period_s=(1.0 / self._cmd_hz) if self._cmd_hz > 0 else 0.0,
        )

        self.node_name = f"robodeploy_so101_{self._robot_id}"

        self._bus: Any = None
        self._calib: SO101Calibration | None = None
        self._motor_names = [str(i) for i in range(1, 7)]

        self._js_pub = None
        self._echo_pub = None
        self._cmd_msg_type = None

        self._stop_state = threading.Event()
        self._state_thread: threading.Thread | None = None

        self._estop = EStop(enable_console=bool(cfg.enable_console_estop))
        self._watchdog: Watchdog | None = None
        self._watchdog_armed: bool = False
        self._limit_guard: JointLimitGuard | None = None
        self._temp_guard: TemperatureGuard | None = None

        self._last_write_wall_s = 0.0
        self._diag_trip = ""
        self._hard_stop_msg: str | None = None

    @property
    def robot_id(self) -> str:
        return self._robot_id

    @property
    def base_frame(self) -> str:
        return self._cfg.base_frame

    @property
    def ee_frame(self) -> str:
        return self._cfg.ee_frame

    @property
    def joint_names(self) -> list[str]:
        return list(self._cfg.joint_names or self._motor_names)

    def _hard_stop(self, msg: str) -> None:
        """Record fault and disable torque (safe to call from background threads)."""
        with self._lock:
            if self._hard_stop_msg is None:
                self._hard_stop_msg = str(msg)
        self._diag_trip = str(msg)
        try:
            if self._bus is not None and getattr(self._bus, "is_connected", False):
                self._bus.disable_torque()
        except Exception:
            pass

    def _ensure_ok(self) -> None:
        if self._hard_stop_msg:
            raise SafetyError(self._hard_stop_msg)

    def _on_watchdog_timeout(self) -> None:
        self._hard_stop("Watchdog: no command within timeout.")

    def _bootstrap_bus(self) -> None:
        if not self._cfg.port:
            raise RuntimeError(
                "SO-101 real adapter requires a serial port. Set e.g. "
                "config_overrides={'robot0.port': '/dev/ttyACM0'} or env ROBODEPLOY_SO101_PORT."
            )
        FeetechMotorsBus, Motor, MotorNormMode = _import_feetech_stack()
        motors = _build_motors_dict(Motor, MotorNormMode)
        self._bus = FeetechMotorsBus(str(self._cfg.port), motors)
        self._bus.connect(handshake=True)
        if hasattr(self._bus, "set_baudrate"):
            try:
                self._bus.set_baudrate(int(self._cfg.baud))
            except Exception:
                pass

        _, self._calib = SO101Calibration.locate(
            explicit_path=self._cfg.calibration_path,
            allow_template=bool(self._cfg.allow_uncalibrated),
        )
        lowers = np.array([j.soft_min_rad for j in self._calib.joints], dtype=np.float64)
        uppers = np.array([j.soft_max_rad for j in self._calib.joints], dtype=np.float64)
        self._limit_guard = JointLimitGuard(lowers, uppers, self._jv_guard)

        self._bus.disable_torque()
        ticks = self._bus.sync_read("Present_Position", self._motor_names, normalize=False)
        q0 = self._calib.to_radians({k: float(v) for k, v in ticks.items()})
        with self._lock:
            self._q[:] = q0

        self._bus.enable_torque()
        self._ramp_to_home(q0)

    def _ramp_to_home(self, q_start: np.ndarray) -> None:
        assert self._bus is not None and self._calib is not None
        ramp_s = max(float(self._cfg.reset_ramp_s), 0.05)
        rate = 50.0
        n = max(int(ramp_s * rate), 1)
        q_home = self._home.copy()
        dt_cmd = 1.0 / rate
        for k in range(1, n + 1):
            self._estop.check()
            alpha = k / float(n)
            q_tgt = (1.0 - alpha) * q_start + alpha * q_home
            q_tgt = slew_limit_command(
                q_tgt,
                self._q if k > 1 else q_start,
                max_joint_velocity=self._max_vel,
                command_hz=1.0 / dt_cmd if dt_cmd > 0 else 0.0,
            )
            for i, jc in enumerate(self._calib.joints):
                q_tgt[i] = float(np.clip(q_tgt[i], jc.soft_min_rad, jc.soft_max_rad))
            if self._limit_guard is not None:
                try:
                    self._limit_guard.check(q_tgt, dt=dt_cmd)
                except SafetyError as exc:
                    self._hard_stop(str(exc))
                    raise
            self._bus.sync_write("Goal_Position", self._calib.to_ticks(q_tgt), normalize=False)
            with self._lock:
                self._q[:] = q_tgt
            time.sleep(dt_cmd)

    def _apply_hardware_goal(self, positions_rad: np.ndarray) -> None:
        assert self._bus is not None and self._calib is not None and self._limit_guard is not None
        self._estop.check()
        self._ensure_ok()
        q_des = np.asarray(positions_rad, dtype=np.float64).reshape(-1)
        with self._lock:
            q_cur = self._q.copy()
        q_des = slew_limit_command(
            q_des,
            q_cur,
            max_joint_velocity=self._max_vel,
            command_hz=self._cmd_hz,
        )
        for i, jc in enumerate(self._calib.joints):
            q_des[i] = float(np.clip(q_des[i], jc.soft_min_rad, jc.soft_max_rad))
        dt = 1.0 / self._cmd_hz if self._cmd_hz > 0 else None
        try:
            if self._limit_guard is not None:
                self._limit_guard.check(q_des, dt=dt)
        except SafetyError as exc:
            self._hard_stop(str(exc))
            raise

        t0 = time.perf_counter()
        self._bus.sync_write("Goal_Position", self._calib.to_ticks(q_des), normalize=False)
        self._last_write_wall_s = time.perf_counter() - t0

        if self._echo_pub is not None and self._cmd_msg_type is not None:
            msg = self._cmd_msg_type()
            msg.data = [float(x) for x in q_des.tolist()]
            self._echo_pub.publish(msg)

        if not self._watchdog_armed and self._watchdog is not None:
            self._watchdog.arm()
            self._watchdog_armed = True
        if self._watchdog is not None:
            self._watchdog.feed()

    def start(self) -> None:
        if self._node is not None:
            return
        self._estop.start()
        try:
            self._bootstrap_bus()
            self._watchdog = Watchdog(float(self._cfg.watchdog_timeout_s), self._on_watchdog_timeout)
            super().start()
            self._stop_state.clear()
            self._state_thread = threading.Thread(target=self._state_loop, name=f"so101_state_{self._robot_id}", daemon=True)
            self._state_thread.start()

            def _read_temps() -> dict[str, float]:
                out: dict[str, float] = {}
                if self._bus is None or not getattr(self._bus, "is_connected", False):
                    return out
                for name in self._motor_names:
                    try:
                        out[name] = float(self._bus.read("Present_Temperature", name, normalize=False))
                    except Exception:
                        continue
                return out

            self._temp_guard = TemperatureGuard(
                _read_temps,
                max_c=float(self._cfg.temperature_max_c),
                period_s=float(self._cfg.temperature_poll_s),
                on_violation=lambda reason: self._hard_stop(str(reason)),
            )
            self._temp_guard.start()
        except BaseException:
            try:
                if self._watchdog is not None:
                    self._watchdog.disarm()
            except Exception:
                pass
            self._watchdog = None
            try:
                self._estop.stop()
            except Exception:
                pass
            if self._bus is not None and getattr(self._bus, "is_connected", False):
                try:
                    self._bus.disable_torque()
                except Exception:
                    pass
                try:
                    self._bus.disconnect(disable_torque=False)
                except Exception:
                    pass
            self._bus = None
            raise

    def stop(self) -> None:
        self._stop_state.set()
        if self._state_thread is not None and self._state_thread.is_alive():
            self._state_thread.join(timeout=2.5)
        self._state_thread = None
        try:
            if self._temp_guard is not None:
                self._temp_guard.stop()
        except Exception:
            pass
        self._temp_guard = None
        try:
            self._estop.stop()
        except Exception:
            pass
        try:
            if self._watchdog is not None:
                self._watchdog.disarm()
        except Exception:
            pass
        self._watchdog = None
        self._watchdog_armed = False

        if self._bus is not None and getattr(self._bus, "is_connected", False):
            try:
                self._bus.disable_torque()
            except Exception:
                pass
            try:
                self._bus.disconnect(disable_torque=False)
            except Exception:
                pass
        self._bus = None

        super().stop()

    def _state_loop(self) -> None:
        assert self._bus is not None and self._calib is not None
        period = 1.0 / self._state_hz if self._state_hz > 0 else 0.02
        q_prev = self._q.copy()
        while not self._stop_state.wait(timeout=period):
            try:
                self._estop.check()
                self._ensure_ok()
                ticks = self._bus.sync_read("Present_Position", self._motor_names, normalize=False)
                q = self._calib.to_radians({k: float(v) for k, v in ticks.items()})
                qd = (q - q_prev) / period
                q_prev = q.copy()
                with self._lock:
                    self._q[:] = q
                    self._qd[:] = qd
                if self._js_pub is not None and bool(self._cfg.publish_state):
                    import rclpy.clock

                    from sensor_msgs.msg import JointState

                    msg = JointState()
                    msg.header.stamp = self._node.get_clock().now().to_msg()
                    msg.name = self.joint_names
                    msg.position = [float(x) for x in q.tolist()]
                    msg.velocity = [float(x) for x in qd.tolist()]
                    self._js_pub.publish(msg)
            except SafetyError:
                break
            except Exception:
                if self._hard_stop_msg:
                    break
                continue

    def _on_node_ready(self, node) -> None:
        try:
            import rclpy.time
            import tf2_ros
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Float64MultiArray
        except ImportError as exc:
            raise ImportError(
                "ROS 2 packages not found. Source your ROS 2 environment with rclpy / tf2_ros / sensor_msgs."
            ) from exc

        self._cmd_msg_type = Float64MultiArray
        if bool(self._cfg.publish_state):
            self._js_pub = node.create_publisher(
                JointState,
                f"{self._ns}/{self._cfg.joint_states_topic}" if self._ns else f"/{self._cfg.joint_states_topic}",
                10,
            )
        if bool(self._cfg.publish_command_echo):
            self._echo_pub = node.create_publisher(
                Float64MultiArray,
                f"{self._ns}/{self._cfg.cmd_topic}" if self._ns else f"/{self._cfg.cmd_topic}",
                10,
            )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

    def _on_node_stopping(self, node) -> None:
        del node
        self._js_pub = None
        self._echo_pub = None

    def _get_ee_pose_from_tf(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            import rclpy.time

            tf_stamped = self._tf_buffer.lookup_transform(self._base_frame(), self._ee_frame(), rclpy.time.Time())
            tr = tf_stamped.transform.translation
            rot = tf_stamped.transform.rotation
            pos = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
            quat = np.array([rot.w, rot.x, rot.y, rot.z], dtype=np.float64)
            return pos, quat
        except Exception:
            return np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _base_frame(self) -> str:
        return str(self._cfg.base_frame)

    def _ee_frame(self) -> str:
        return str(self._cfg.ee_frame)

    def send_action(self, action: Action) -> None:
        if action.joint_positions is None:
            return
        self._estop.check()
        self._ensure_ok()
        q_des = np.asarray(action.joint_positions, dtype=np.float64).reshape(-1)
        self._commander.send(q_des)

    def send_action_and_wait(self, action: Action) -> None:
        self.send_action(action)
        if self._cmd_hz > 0:
            time.sleep(1.0 / self._cmd_hz)

    def get_obs(self) -> Observation:
        self._estop.check()
        self._ensure_ok()
        if self._watchdog is not None and self._watchdog_armed:
            self._watchdog.feed()
        with self._lock:
            q = self._q.copy()
            qd = self._qd.copy()
            tau = self._tau.copy()
        ee_pos, ee_quat = self._get_ee_pose_from_tf()
        obs = Observation(
            joint_positions=q.astype(np.float32),
            joint_velocities=qd.astype(np.float32),
            joint_torques=tau.astype(np.float32),
            ee_position=ee_pos.astype(np.float32),
            ee_orientation=ee_quat.astype(np.float32),
            ee_velocity=np.zeros((3,), dtype=np.float32),
            ee_angular_velocity=np.zeros((3,), dtype=np.float32),
        )
        self._obs_cache.write(obs)
        return obs

    def get_diagnostics(self) -> dict:
        temps: dict[str, float] = {}
        if self._bus is not None and getattr(self._bus, "is_connected", False):
            for name in self._motor_names:
                try:
                    temps[name] = float(self._bus.read("Present_Temperature", name, normalize=False))
                except Exception:
                    continue
        return {
            "robot_id": self._robot_id,
            "controller_type": self.controller_type,
            "command_count": self._commander.record.count,
            "last_command_wall_s": self._commander.record.sent_wall_s,
            "last_write_latency_s": float(self._last_write_wall_s),
            "estop_tripped": bool(self._estop.tripped),
            "watchdog_armed": bool(self._watchdog_armed),
            "trip": self._diag_trip,
            "hard_stop_msg": self._hard_stop_msg,
            "present_temperature_c": temps,
        }


@register_controller("so101_feetech")
def create_so101_feetech_adapter(cfg: ControllerConfig, backend_config: dict):
    return SO101FeetechControllerAdapter(cfg, backend_config)
