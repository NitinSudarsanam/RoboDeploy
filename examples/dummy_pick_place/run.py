"""Simulator-free PickPlace smoke (dummy backend + example reach policy)."""

from __future__ import annotations

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from robodeploy.builtins import import_builtins  # noqa: E402
from robodeploy.core.registry import use  # noqa: E402
from robodeploy.core.robot import Robot, RobotTask  # noqa: E402
from robodeploy.description.kuka import KukaDescription  # noqa: E402
from robodeploy.env import RoboEnv  # noqa: E402
from examples.tasks.pick_place import PickPlaceTask  # noqa: E402
from robodeploy.testing import DummyBackend  # noqa: E402

from examples.policies.reach_pick_place import ReachPickPlacePolicy  # noqa: E402


def main() -> None:
    import_builtins()
    use("examples.tasks")
    use("examples.policies")

    desc = KukaDescription()
    home = [float(x) for x in desc.home_qpos]
    task = PickPlaceTask()
    policy = ReachPickPlacePolicy(home_qpos=home, scene=task.scene_spec(), description=desc)
    robot = Robot(
        robot_id="robot0",
        description=desc,
        tasks={"pick": RobotTask(task=task, policies={"reach": policy})},
    )
    env = RoboEnv(backend=DummyBackend(), robots=[robot], max_episode_steps=50)

    obs, info = env.reset()
    print("reset ok", info.episode_id)
    for i in range(20):
        obs, reward, done, info = env.step()
        if i % 5 == 0:
            print(f"step {i} reward={reward:.3f}")
    env.close()
    print("dummy pick_place smoke finished")


if __name__ == "__main__":
    main()
