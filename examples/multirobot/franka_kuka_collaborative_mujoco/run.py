"""Franka + Kuka collaborative hand-off in one MuJoCo world (sequential coordination)."""

from __future__ import annotations

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np  # noqa: E402

from robodeploy.builtins import import_builtins  # noqa: E402
from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend  # noqa: E402
from robodeploy.core.registry import use  # noqa: E402
from robodeploy.core.robot import Robot, RobotTask  # noqa: E402
from robodeploy.core.types import Pose3D  # noqa: E402
from robodeploy.description.kuka.description import KukaDescription  # noqa: E402
from robodeploy.env import RoboEnv  # noqa: E402
from examples.franka_pick_place_mujoco.components import ExampleFrankaMujocoDescription  # noqa: E402
from examples.policies.joint_track import JointTrackPolicy  # noqa: E402
from examples.tasks.pick_place import PickPlaceTask  # noqa: E402


def main() -> None:
    import_builtins()
    use("examples.tasks")
    use("examples.policies")

    task = PickPlaceTask()
    franka_home = [float(x) for x in ExampleFrankaMujocoDescription().home_qpos]
    kuka_home = [float(x) for x in KukaDescription().home_qpos]
    franka_reach = (np.array(franka_home) + np.array([0.12, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])).tolist()
    kuka_hold = (np.array(kuka_home) + np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0])).tolist()

    robots = [
        Robot(
            robot_id="franka_placer",
            description=ExampleFrankaMujocoDescription(),
            base_pose=Pose3D(position=(-0.5, 0.15, 0.4)),
            tasks={
                "pick": RobotTask(
                    task=task,
                    policies={"track": JointTrackPolicy(home_qpos=franka_home, target_qpos=franka_reach)},
                    mode="sequential",
                )
            },
        ),
        Robot(
            robot_id="kuka_tray",
            description=KukaDescription(),
            base_pose=Pose3D(position=(0.55, -0.15, 0.4)),
            tasks={
                "pick": RobotTask(
                    task=task,
                    policies={"track": JointTrackPolicy(home_qpos=kuka_home, target_qpos=kuka_hold)},
                    mode="sequential",
                )
            },
        ),
    ]

    env = RoboEnv(
        backend=MuJoCoBackend(config={"allow_actuator_name_fallback": True, "enable_viewer": False}),
        robots=robots,
        max_episode_steps=100,
    )
    try:
        obs, info = env.reset()
        print("reset ok", info.episode_id, "robots", len(env.robots))
        for step in range(70):
            obs, reward, done, info = env.step()
            if step % 20 == 0:
                multi = env.get_processed_obs_by_robot()
                for rid, obs_r in multi.items():
                    print(f"  {rid} q0={float(obs_r.joint_positions[0]):.3f}")
            if done:
                break
        print("franka_kuka_collaborative_mujoco finished")
    finally:
        env.close()


if __name__ == "__main__":
    main()
