"""Two SO-101 arms on ROS2 — multi-robot namespaces (fake sim for local dev)."""

from __future__ import annotations

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from robodeploy.builtins import import_builtins  # noqa: E402
from robodeploy.backends.real.ros2.backend import ROS2RealBackend  # noqa: E402
from robodeploy.core.robot import Robot, RobotTask  # noqa: E402
from robodeploy.core.types import Pose3D  # noqa: E402
from robodeploy.description.so101.description import SO101Description  # noqa: E402
from robodeploy.env import RoboEnv  # noqa: E402
from examples.policies.joint_track import JointTrackPolicy  # noqa: E402
from examples.tasks.pick_place import PickPlaceTask  # noqa: E402


def main() -> None:
    import_builtins()

    task = PickPlaceTask()
    desc_a = SO101Description()
    desc_b = SO101Description()
    home_a = [float(x) for x in desc_a.home_qpos]
    home_b = [float(x) for x in desc_b.home_qpos]
    target_a = [v + 0.05 for v in home_a]
    target_b = [v - 0.05 for v in home_b]

    robots = [
        Robot(
            robot_id="so101_left",
            description=desc_a,
            base_pose=Pose3D(position=(-0.3, 0.0, 0.0)),
            tasks={
                "hold": RobotTask(
                    task=task,
                    policies={"track": JointTrackPolicy(home_qpos=home_a, target_qpos=target_a)},
                )
            },
        ),
        Robot(
            robot_id="so101_right",
            description=desc_b,
            base_pose=Pose3D(position=(0.3, 0.0, 0.0)),
            tasks={
                "hold": RobotTask(
                    task=task,
                    policies={"track": JointTrackPolicy(home_qpos=home_b, target_qpos=target_b)},
                )
            },
        ),
    ]

    backend = ROS2RealBackend(
        config={
            "dev_fake_sim": [
                {"robot_ns": "/so101_left", "joint_names": tuple(desc_a.joint_names)},
                {"robot_ns": "/so101_right", "joint_names": tuple(desc_b.joint_names)},
            ],
            "so101_left.controller": "joint_position",
            "so101_right.controller": "joint_position",
            "so101_left.joint_names": list(desc_a.joint_names),
            "so101_right.joint_names": list(desc_b.joint_names),
            "rviz": {"enabled": False},
        }
    )

    env = RoboEnv(backend=backend, robots=robots, max_episode_steps=40)
    try:
        obs, info = env.reset()
        print("reset ok", info.episode_id)
        for step in range(25):
            obs, reward, done, info = env.step()
            if step % 10 == 0:
                for rid, obs_r in env.get_processed_obs_by_robot().items():
                    q = obs_r.joint_positions
                    print(f"  {rid} q0={float(q[0]):.3f}" if q is not None else f"  {rid} (no q)")
            if done:
                break
        print("two_so101_real finished (fake sim). For hardware: set controller so101_feetech + --port.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
