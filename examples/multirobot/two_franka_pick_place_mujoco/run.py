"""Two Franka arms in one MuJoCo world — independent joint-space reach."""

from __future__ import annotations

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np  # noqa: E402

from robodeploy.builtins import import_builtins  # noqa: E402
from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend  # noqa: E402
from robodeploy.core.registry import use  # noqa: E402
from robodeploy.core.robot import Robot, RobotTask  # noqa: E402
from robodeploy.core.types import Pose3D  # noqa: E402
from robodeploy.env import RoboEnv  # noqa: E402
from examples.franka_pick_place_mujoco.components import ExampleFrankaMujocoDescription  # noqa: E402
from examples.policies.joint_track import JointTrackPolicy  # noqa: E402
from examples.tasks.pick_place import PickPlaceTask  # noqa: E402


def main() -> None:
    import_builtins()
    use("examples.tasks")
    use("examples.policies")
    use("examples.franka_pick_place_mujoco.components")

    home = [float(x) for x in ExampleFrankaMujocoDescription().home_qpos]
    left_target = (np.array(home) + np.array([0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])).tolist()
    right_target = (np.array(home) + np.array([-0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])).tolist()

    task = PickPlaceTask()
    robots = [
        Robot(
            robot_id="franka_left",
            description=ExampleFrankaMujocoDescription(),
            base_pose=Pose3D(position=(-0.55, 0.0, 0.4)),
            tasks={
                "pick": RobotTask(
                    task=task,
                    policies={"track": JointTrackPolicy(home_qpos=home, target_qpos=left_target)},
                )
            },
        ),
        Robot(
            robot_id="franka_right",
            description=ExampleFrankaMujocoDescription(),
            base_pose=Pose3D(position=(0.55, 0.0, 0.4)),
            tasks={
                "pick": RobotTask(
                    task=task,
                    policies={"track": JointTrackPolicy(home_qpos=home, target_qpos=right_target)},
                )
            },
        ),
    ]

    env = RoboEnv(
        backend=MuJoCoBackend(config={"allow_actuator_name_fallback": True, "enable_viewer": False}),
        robots=robots,
        max_episode_steps=120,
    )
    try:
        obs, info = env.reset()
        print("reset ok", info.episode_id, "dof", obs.joint_positions.shape[0])
        for step in range(80):
            obs, reward, done, info = env.step()
            if step % 20 == 0:
                multi = info.extra.get("multi_agent")
                states = getattr(multi, "robot_states", {}) if multi is not None else {}
                for rid, state in states.items():
                    obs_r = getattr(state, "obs", None)
                    if obs_r is not None and obs_r.joint_positions is not None:
                        print(f"  {rid} q0={float(obs_r.joint_positions[0]):.3f}")
            if done:
                break
        print("two_franka_pick_place_mujoco finished")
    finally:
        env.close()


if __name__ == "__main__":
    main()
