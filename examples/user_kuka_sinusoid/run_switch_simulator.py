"""Kuka sinusoid demo — change **one** setting to swap backend / simulator.

Edit ``BACKEND`` below (``"mujoco"`` | ``"isaacsim"`` | ``"ros2_rviz"`` | ``"gazebo"`` | ``"real_world"``), then:

    python -m examples.user_kuka_sinusoid.run_switch_simulator

ROS2+RViz without an external robot graph: ``BACKEND = "ros2_rviz"`` and
``LOCAL_ROS_GRAPH = True`` (or pass ``--fake-sim`` on the command line).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from robodeploy.backends.simulator import SimulatorName, backend_for_simulator  # noqa: E402
from robodeploy.core.robot import Robot, RobotTask  # noqa: E402
from robodeploy.env import RoboEnv  # noqa: E402

from examples.user_kuka_sinusoid.components import (  # noqa: E402
    UserKukaDescription,
    UserKukaSinusoidTask,
    UserSinusoidPolicy,
)

# ---------------------------------------------------------------------------
# Only edit this block (and optionally LOCAL_ROS_GRAPH for ros2_rviz).
# ---------------------------------------------------------------------------
BACKEND: SimulatorName = "ros2_rviz"
LOCAL_ROS_GRAPH = True  # True for ros2_rviz: embedded joint-position devtool
# ---------------------------------------------------------------------------


def main() -> None:
    local_graph = LOCAL_ROS_GRAPH or ("--fake-sim" in sys.argv)
    if BACKEND == "ros2_rviz" and local_graph:
        time.sleep(0.2)

    desc = UserKukaDescription()
    task = UserKukaSinusoidTask(max_steps=2000)
    policy = UserSinusoidPolicy(amplitude=0.35, frequency_hz=0.2)
    robot = Robot(
        robot_id="robot0",
        description=desc,
        sensors=[],
        tasks={
            "sinusoid": RobotTask(
                task=task,
                policies={"main": policy},
                mode="sequential",
            )
        },
    )

    backend = backend_for_simulator(
        BACKEND,
        robots=[robot],
        local_ros_graph=local_graph,
    )
    env = RoboEnv(backend=backend, robots=[robot])

    try:
        obs, info = env.reset()
    except ImportError as exc:
        if BACKEND == "mujoco":
            print(exc)
            print("\nInstall MuJoCo:  pip install mujoco")
            return
        raise

    print("BACKEND =", BACKEND, "| reset:", info)

    step_sleep = 0.01 if BACKEND == "mujoco" else 0.02
    max_steps = 1000 if BACKEND == "mujoco" else 2000

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
