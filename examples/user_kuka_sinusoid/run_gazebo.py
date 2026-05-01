"""Run the user-defined Kuka sinusoid demo on Gazebo via ROS2GazeboBackend.

RoboDeploy starts Gazebo (best-effort) when configured on the description;
ROS2 transport + RViz follow the same ``backend_for_simulator`` path as other simulators.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from robodeploy.backends.simulator import backend_for_simulator
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.env import RoboEnv

from examples.user_kuka_sinusoid.components import (  # noqa: E402
    UserKukaDescription,
    UserKukaSinusoidTask,
    UserSinusoidPolicy,
)


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()


def main() -> None:
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

    backend = backend_for_simulator("gazebo", robots=[robot])
    env = RoboEnv(backend=backend, robots=[robot])

    obs, info = env.reset()
    print("reset:", info)
    for i in range(2000):
        obs, reward, done, info = env.step()
        if i % 100 == 0 and obs.joint_positions is not None and len(obs.joint_positions) > 0:
            print("step", i, "q0", float(obs.joint_positions[0]))
        if done:
            break
        time.sleep(0.02)
    env.close()


if __name__ == "__main__":
    main()
