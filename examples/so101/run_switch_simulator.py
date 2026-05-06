"""SO-101 demo — change **one** setting to swap backend / simulator.

Edit ``BACKEND`` below (``"mujoco"`` | ``"isaacsim"`` | ``"ros2_rviz"`` | ``"gazebo"`` | ``"real_world"``), then:

    python -m examples.so101.run_switch_simulator

Optional: ``--profile default|smooth|fast|demo`` or ``ROBODEPLOY_PROFILE`` — merged with the
robot description's ``default_behavior_profile()`` and translated per backend so motion
and control rate stay consistent across simulators.

Robot model comes from the bundled URDF in ``robodeploy.description.so101``.

- **ros2_rviz**: RViz + ROS2 transport; use ``LOCAL_ROS_GRAPH = True`` or ``--fake-sim`` for embedded fake joint sim.
- **real_world**: SO-101 uses the built-in ``so101_feetech`` controller (lerobot + USB). Requires ``--port`` or
  ``ROBODEPLOY_SO101_PORT``, and a calibration file (see ``python -m examples.so101.calibrate_so101``). Other robots
  still use generic ``joint_position`` + your ROS graph.
- **mujoco**: loads URDF directly and auto-injects position actuators; you can still override with MJCF via ``asset_overrides``.
- **gazebo**: pass ``config_overrides['sim']`` with at least ``world``, or implement ``gazebo_sim_launch_config`` on a custom description subclass.
- **isaacsim**: requires Isaac Sim / omni stack; loads URDF by default.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import cast
import json

# ---------------------------------------------------------------------------
# Only edit this block (and optionally LOCAL_ROS_GRAPH for ros2_rviz).
# ---------------------------------------------------------------------------
BACKEND: SimulatorName = "ros2_rviz"
LOCAL_ROS_GRAPH = True  # True for ros2_rviz: embedded joint-position devtool
# ---------------------------------------------------------------------------


def _parse_backend_override(default_backend: str) -> str:
    if "--backend" in sys.argv:
        i = sys.argv.index("--backend")
        if i + 1 < len(sys.argv):
            return str(sys.argv[i + 1])
    return str(os.environ.get("ROBODEPLOY_BACKEND", default_backend))


def _parse_port() -> str | None:
    if "--port" in sys.argv:
        i = sys.argv.index("--port")
        if i + 1 < len(sys.argv):
            return str(sys.argv[i + 1]).strip()
    v = os.environ.get("ROBODEPLOY_SO101_PORT", "").strip()
    return v or None


def _parse_calibration_path() -> str | None:
    if "--calibration" in sys.argv:
        i = sys.argv.index("--calibration")
        if i + 1 < len(sys.argv):
            return str(sys.argv[i + 1]).strip()
    v = os.environ.get("ROBODEPLOY_SO101_CALIBRATION", "").strip()
    return v or None


def _torque_off_now(port: str) -> None:
    """Best-effort: disable torque and disconnect (LeRobot bus)."""
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError as exc:
        print(exc)
        print('\nInstall:  pip install "lerobot[feetech]"')
        raise SystemExit(1) from exc

    motors = {str(i): Motor(id=i, model="sts3215", norm_mode=MotorNormMode.DEGREES) for i in range(1, 7)}
    bus = FeetechMotorsBus(str(port), motors)
    bus.connect(handshake=True)
    bus.disable_torque()
    bus.disconnect(disable_torque=False)


def _read_positions_now(port: str, *, calibration_path: str | None) -> list[float]:
    """Best-effort: read current joint positions (radians) without commanding motion.

    Uses the same calibration resolution as the real backend:
    - If the calibration JSON is LeRobot-style, register it on the bus and read normalized degrees.
    - Otherwise, read raw ticks and convert via ``SO101Calibration``.
    """
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import MotorCalibration
    except ImportError as exc:
        print(exc)
        print('\nInstall:  pip install "lerobot[feetech]"')
        raise SystemExit(1) from exc

    motors = {str(i): Motor(id=i, model="sts3215", norm_mode=MotorNormMode.DEGREES) for i in range(1, 7)}
    bus = FeetechMotorsBus(str(port), motors)
    bus.connect(handshake=True)
    names = [str(i) for i in range(1, 7)]

    # Resolve calibration file (prefer explicit, else env/default via SO101Calibration.locate)
    try:
        cal_path, _ = SO101Calibration.locate(explicit_path=calibration_path, allow_template=True)
        raw = Path(cal_path).read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        data = None

    import numpy as np

    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()) and any(
        {"id", "range_min", "range_max"}.issubset(set(v.keys())) for v in data.values() if isinstance(v, dict)
    ):
        cal_dict: dict[str, MotorCalibration] = {}
        for v in data.values():
            if not isinstance(v, dict):
                continue
            try:
                mid = int(v["id"])
                cal_dict[str(mid)] = MotorCalibration(
                    id=mid,
                    drive_mode=int(v.get("drive_mode", 0)),
                    homing_offset=int(v.get("homing_offset", 0)),
                    range_min=int(v["range_min"]),
                    range_max=int(v["range_max"]),
                )
            except Exception:
                continue
        if len(cal_dict) == 6:
            bus.calibration = cal_dict
            pos_deg = bus.sync_read("Present_Position", names, normalize=True)
            bus.disconnect(disable_torque=False)
            q = np.deg2rad(np.array([float(pos_deg[str(i)]) for i in range(1, 7)], dtype=np.float64))
            return [float(x) for x in q.tolist()]

    # Fallback: raw tick -> radians using neutral calibration
    ticks = bus.sync_read("Present_Position", names, normalize=False)
    bus.disconnect(disable_torque=False)
    cal = SO101Calibration.load(SO101Calibration.locate(explicit_path=calibration_path, allow_template=True)[0])
    q = cal.to_radians({k: float(v) for k, v in ticks.items()})
    return [float(x) for x in q.tolist()]


def _parse_profile_override() -> str | None:
    if "--profile" in sys.argv:
        i = sys.argv.index("--profile")
        if i + 1 < len(sys.argv):
            return str(sys.argv[i + 1])
    v = os.environ.get("ROBODEPLOY_PROFILE")
    return str(v) if v else None


def _parse_home_qpos() -> list[float] | None:
    if "--home-qpos" in sys.argv:
        i = sys.argv.index("--home-qpos")
        vals = []
        for j in range(i + 1, min(i + 7, len(sys.argv))):
            vals.append(float(sys.argv[j]))
        if len(vals) != 6:
            raise ValueError("--home-qpos expects 6 floats (joint1..joint6)")
        return vals
    v = os.environ.get("ROBODEPLOY_SO101_HOME_QPOS", "").strip()
    if v:
        parts = v.replace(",", " ").split()
        if len(parts) != 6:
            raise ValueError("ROBODEPLOY_SO101_HOME_QPOS must contain 6 floats")
        return [float(x) for x in parts]
    return None


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from robodeploy.backends.simulator import SimulatorName, backend_for_simulator  # noqa: E402
from robodeploy.behavior import BehaviorProfile, PresetName  # noqa: E402
from robodeploy.description.so101.calibration import MissingCalibrationError, SO101Calibration  # noqa: E402
from robodeploy.core.robot import Robot, RobotTask  # noqa: E402
from robodeploy.description.so101 import SO101Description  # noqa: E402
from robodeploy.env import RoboEnv  # noqa: E402

from examples.so101.components import SO101DemoTask, SO101SinusoidPolicy  # noqa: E402


def main() -> None:
    global BACKEND  # noqa: PLW0603
    BACKEND = _parse_backend_override(str(BACKEND))  # type: ignore[assignment]

    local_graph = LOCAL_ROS_GRAPH or ("--fake-sim" in sys.argv) or (BACKEND == "ros2_rviz")
    if BACKEND == "ros2_rviz" and local_graph:
        time.sleep(0.2)

    desc = SO101Description()
    pr = _parse_profile_override()
    behavior = BehaviorProfile(preset=cast(PresetName, pr)) if pr else None
    resolved = desc.default_behavior_profile().merged_with(behavior).resolved()

    home_override = _parse_home_qpos()
    if home_override is not None:
        import numpy as np

        desc.home_qpos = np.asarray(home_override, dtype=np.float64)

    task = SO101DemoTask(max_steps=2000)
    home_only = "--home" in sys.argv
    policy = SO101SinusoidPolicy(
        amplitude=0.0 if home_only else 0.12,
        frequency_hz=0.0 if home_only else 0.12,
        action_hz=float(resolved.control_hz),
        home_qpos=desc.home_qpos,
    )
    robot = Robot(
        robot_id="robot0",
        description=desc,
        sensors=[],
        tasks={
            "demo": RobotTask(
                task=task,
                policies={"main": policy},
                mode="sequential",
            )
        },
    )

    config_overrides: dict | None = None
    if str(BACKEND) == "real_world":
        port = _parse_port()
        if not port:
            print(
                "real_world requires a serial port for SO-101.\n"
                "  Pass --port /dev/ttyACM0   or set   ROBODEPLOY_SO101_PORT\n"
                "Calibrate first:\n"
                "  python -m examples.so101.calibrate_so101 --port ... --out ~/.robodeploy/so101_calibration.json"
            )
            return
        if "--torque-off" in sys.argv:
            _torque_off_now(port)
            print(f"Torque disabled on SO-101 (port={port}). You can move the arm by hand now.")
            return
        if "--read-pos" in sys.argv:
            q = _read_positions_now(port, calibration_path=_parse_calibration_path())
            print("Current joint_positions (radians, joint1..joint6):")
            print("  " + " ".join(f"{x:.6f}" for x in q))
            print("\nAs --home-qpos:")
            print("  --home-qpos " + " ".join(f"{x:.6f}" for x in q))
            print("\nAs env:")
            print('  ROBODEPLOY_SO101_HOME_QPOS="' + " ".join(f"{x:.6f}" for x in q) + '"')
            return
        cal_path = _parse_calibration_path()
        allow_uncal = "--allow-uncalibrated" in sys.argv
        try:
            SO101Calibration.locate(explicit_path=cal_path, allow_template=allow_uncal)
        except MissingCalibrationError as exc:
            print(exc)
            return
        config_overrides = {"robot0.port": port}
        if cal_path:
            config_overrides["robot0.calibration_path"] = cal_path
        if allow_uncal:
            config_overrides["robot0.allow_uncalibrated"] = True
        if "--apply-motor-limits" in sys.argv:
            # Persistent write to servo registers. Use only when your calibration is correct.
            config_overrides["apply_motor_limits"] = True

    try:
        backend = backend_for_simulator(
            BACKEND,
            robots=[robot],
            local_ros_graph=local_graph,
            behavior=behavior,
            config_overrides=config_overrides,
        )
    except ValueError as exc:
        if BACKEND == "gazebo":
            print(exc)
            print(
                "\nGazebo: set BACKEND or pass sim world via backend config, e.g.\n"
                "  backend_for_simulator('gazebo', robots=[robot], config_overrides={'sim': {'world': '...'}})\n"
                "or subclass SO101Description with gazebo_sim_launch_config()."
            )
            return
        raise

    env = RoboEnv(backend=backend, robots=[robot])

    try:
        obs, info = env.reset()
    except ImportError as exc:
        if BACKEND == "mujoco":
            print(exc)
            print("\nInstall MuJoCo:  pip install mujoco")
            return
        if BACKEND == "isaacsim":
            print(exc)
            print("\nIsaac Sim backend requires the Isaac Sim / omni.isaac stack.")
            return
        raise
    except FileNotFoundError as exc:
        if BACKEND == "mujoco":
            print(exc)
            print("\nSO-101 URDF-only: add MJCF or use asset_overrides for MuJoCo.")
            return
        raise

    print("BACKEND =", BACKEND, "| reset:", info)

    if "--print-home" in sys.argv:
        if obs.joint_positions is not None:
            q = [float(x) for x in obs.joint_positions.tolist()]
            print("\nCurrent joint_positions (use as home):")
            print("  --home-qpos", " ".join(f"{x:.6f}" for x in q))
            print("Or env:")
            print('  ROBODEPLOY_SO101_HOME_QPOS="' + " ".join(f"{x:.6f}" for x in q) + '"')
        env.close()
        return

    step_sleep = 1.0 / max(float(getattr(backend, "control_hz", resolved.control_hz)), 1e-6)
    max_steps = 250 if home_only else 2000

    for i in range(max_steps):
        obs, reward, done, info = env.step()
        if i % 100 == 0 and obs.joint_positions is not None and len(obs.joint_positions) > 0:
            print("step", i, "q0", float(obs.joint_positions[0]))
        if done:
            break
        time.sleep(step_sleep)

    env.close()


if __name__ == "__main__":
    main()
