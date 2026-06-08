"""Run the user-defined Kuka sinusoid demo on ROS 2 + RViz.

Prereq: a ROS 2 graph (real robot or simulator) that provides:
  - /robot0/joint_states (sensor_msgs/JointState) with joint names matching the URDF (joint1..joint7 for user_kuka.urdf)
  - /robot0/<joint_pos_cmd_topic> (std_msgs/Float64MultiArray) for joint targets
  - TF from robot_state_publisher for RViz RobotModel (started automatically when rviz.enabled=true)

Optional: ``--fake-sim`` uses RoboDeploy's embedded joint-position devtool (no rclpy here).
"""

from __future__ import annotations

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()
import sys
import time
from pathlib import Path


from robodeploy.backends.simulator import backend_for_simulator  # noqa: E402
from robodeploy.core.robot import Robot, RobotTask  # noqa: E402
from robodeploy.env import RoboEnv  # noqa: E402

from examples.user_kuka_sinusoid.components import (  # noqa: E402
    UserKukaDescription,
    UserKukaSinusoidTask,
    UserSinusoidPolicy,
)


def main() -> None:
    local_ros_graph = "--fake-sim" in sys.argv
    if local_ros_graph:
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
        "ros2_rviz",
        robots=[robot],
        local_ros_graph=local_ros_graph,
    )
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
