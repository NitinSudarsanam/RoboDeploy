"""Three-arm cooperative assembly — shared-policy joint targets in MuJoCo."""

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
    franka_desc = ExampleFrankaMujocoDescription()
    kuka_desc = KukaDescription()
    franka_home = [float(x) for x in franka_desc.home_qpos]
    kuka_home = [float(x) for x in kuka_desc.home_qpos]

    shared_track = JointTrackPolicy(
        home_qpos=franka_home,
        target_qpos=(np.array(franka_home) + np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])).tolist(),
    )

    robots = [
        Robot(
            robot_id="arm_center",
            description=franka_desc,
            base_pose=Pose3D(position=(0.0, 0.0, 0.4)),
            tasks={"assemble": RobotTask(task=task, policies={"track": shared_track})},
        ),
        Robot(
            robot_id="arm_left",
            description=franka_desc,
            base_pose=Pose3D(position=(-0.7, 0.0, 0.4)),
            tasks={
                "assemble": RobotTask(
                    task=task,
                    policies={
                        "track": JointTrackPolicy(
                            home_qpos=franka_home,
                            target_qpos=(np.array(franka_home) + np.array([0.08, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])).tolist(),
                        )
                    },
                )
            },
        ),
        Robot(
            robot_id="arm_right",
            description=kuka_desc,
            base_pose=Pose3D(position=(0.7, 0.0, 0.4)),
            tasks={
                "assemble": RobotTask(
                    task=task,
                    policies={
                        "track": JointTrackPolicy(
                            home_qpos=kuka_home,
                            target_qpos=(np.array(kuka_home) + np.array([-0.08, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0])).tolist(),
                        )
                    },
                )
            },
        ),
    ]

    env = RoboEnv(
        backend=MuJoCoBackend(config={"allow_actuator_name_fallback": True, "enable_viewer": False}),
        robots=robots,
        max_episode_steps=90,
    )
    try:
        obs, info = env.reset()
        print("reset ok", info.episode_id, "dof", obs.joint_positions.shape[0])
        for step in range(60):
            obs, reward, done, info = env.step()
            if step % 15 == 0:
                for rid, obs_r in env.get_processed_obs_by_robot().items():
                    print(f"  {rid} q0={float(obs_r.joint_positions[0]):.3f}")
            if done:
                break
        print("three_arm_assembly_mujoco finished")
    finally:
        env.close()


if __name__ == "__main__":
    main()
