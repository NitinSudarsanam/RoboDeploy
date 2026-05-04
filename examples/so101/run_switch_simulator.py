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


def _parse_profile_override() -> str | None:
    if "--profile" in sys.argv:
        i = sys.argv.index("--profile")
        if i + 1 < len(sys.argv):
            return str(sys.argv[i + 1])
    v = os.environ.get("ROBODEPLOY_PROFILE")
    return str(v) if v else None


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

    task = SO101DemoTask(max_steps=2000)
    policy = SO101SinusoidPolicy(
        amplitude=0.12,
        frequency_hz=0.12,
        action_hz=float(resolved.control_hz),
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

    step_sleep = 1.0 / max(float(getattr(backend, "control_hz", resolved.control_hz)), 1e-6)
    max_steps = 2000

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
