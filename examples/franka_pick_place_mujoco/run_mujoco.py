"""Run PickPlaceTask on MuJoCo with example Franka MJCF naming."""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from robodeploy.backends.simulator import backend_for_simulator  # noqa: E402
from robodeploy.builtins import import_builtins  # noqa: E402
from robodeploy.core.registry import use  # noqa: E402
from robodeploy.core.robot import Robot, RobotTask  # noqa: E402
from robodeploy.env import RoboEnv  # noqa: E402
from examples.tasks.pick_place import PickPlaceTask  # noqa: E402

from examples.franka_pick_place_mujoco.components import ExampleFrankaMujocoDescription  # noqa: E402
from examples.policies.reach_pick_place import ReachPickPlacePolicy  # noqa: F401,E402


def build_env(*, max_steps: int = 800) -> RoboEnv:
    import_builtins()
    use("examples.tasks")
    use("examples.franka_pick_place_mujoco.components")
    use("examples.policies")

    desc = ExampleFrankaMujocoDescription()
    home = [float(x) for x in desc.home_qpos]
    task = PickPlaceTask()
    policy = ReachPickPlacePolicy(
        home_qpos=home,
        scene=task.scene_spec(),
        description=desc,
    )
    robot = Robot(
        robot_id="robot0",
        description=desc,
        sensors=[],
        tasks={
            "pick": RobotTask(
                task=task,
                policies={"reach": policy},
                mode="sequential",
            )
        },
    )
    backend = backend_for_simulator(
        "mujoco",
        robots=[robot],
        config_overrides={"allow_actuator_name_fallback": True},
    )
    return RoboEnv(backend=backend, robots=[robot], max_episode_steps=max_steps)


def _attach_policy_ik(env: RoboEnv) -> None:
    robot = env.robots[0]
    policy = robot.tasks["pick"].policies["reach"]
    policy.attach_mujoco(env._backend, robot.description)


def main() -> None:
    try:
        env = build_env()
    except ImportError as exc:
        print(exc)
        print("\nInstall MuJoCo support:\n  pip install -e \".[sim]\"")
        return

    try:
        obs, info = env.reset()
        _attach_policy_ik(env)
        print("reset episode", info.episode_id)
        for i in range(800):
            obs, reward, done, info = env.step()
            if i % 50 == 0:
                ee = [float(x) for x in obs.ee_position]
                print(f"step {i:4d} reward={reward:7.3f} success={info.success} ee_z={ee[2]:.3f}")
            if done:
                print("done at step", i, "success=", info.success)
                break
            time.sleep(0.005)
    finally:
        env.close()


if __name__ == "__main__":
    main()
