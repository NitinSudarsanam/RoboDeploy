"""Kuka pick-place demo driven by sensors (prop pose + optional wrist camera).

Requires: pip install -e ".[sim]"
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

from robodeploy.backends.simulator import backend_for_simulator  # noqa: E402
from robodeploy.builtins import import_builtins  # noqa: E402
from robodeploy.core.registry import use  # noqa: E402
from robodeploy.core.robot import Robot, RobotTask  # noqa: E402
from robodeploy.core.sensor_rig import SensorRig  # noqa: E402
from robodeploy.description.kuka import KukaDescription  # noqa: E402
from robodeploy.env import RoboEnv  # noqa: E402
from examples.tasks.pick_place import PickPlaceTask  # noqa: E402

from examples.policies.sensor_reach_pick import SensorReachPickPlacePolicy  # noqa: F401,E402


def build_env(*, max_steps: int = 1500, use_camera: bool = False) -> RoboEnv:
    import_builtins()
    use("examples.tasks")
    use("examples.sensors")
    use("examples.policies")

    desc = KukaDescription()
    home = [float(x) for x in desc.home_qpos]
    task = PickPlaceTask()
    policy = SensorReachPickPlacePolicy(
        home_qpos=home,
        scene=task.scene_spec(),
        description=desc,
    )

    rig_kwargs: dict = {
        "ee_link": desc.ee_link_name,
        "prop_pose": {"prop_names": ["source", "target"]},
    }
    if use_camera:
        rig_kwargs["wrist_rgbd"] = {
            "width": 64,
            "height": 48,
            "depth": False,
            "allow_camera_fallback": True,
        }

    rig = SensorRig.robot_mounted("arm_sensors", **rig_kwargs)
    sensors = rig.materialize(is_real=False, backend_name="mujoco")

    robot = Robot(
        robot_id="robot0",
        description=desc,
        sensors=sensors,
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
        config_overrides={
            "allow_actuator_name_fallback": True,
            "enable_viewer": False,
        },
    )
    return RoboEnv(backend=backend, robots=[robot], max_episode_steps=max_steps)


def _attach_policy_ik(env: RoboEnv) -> None:
    robot = env.robots[0]
    policy = robot.tasks["pick"].policies["reach"]
    policy.attach_mujoco(env._backend, robot.description)


def main() -> None:
    try:
        env = build_env(use_camera=False)
    except ImportError as exc:
        print(exc)
        print('\nInstall MuJoCo support:\n  pip install -e ".[sim]"')
        return

    try:
        obs, info = env.reset()
        _attach_policy_ik(env)
        print("reset episode", info.episode_id, "objects", list(getattr(obs, "objects", {}).keys()))
        for i in range(1500):
            obs, reward, done, info = env.step()
            if i % 100 == 0:
                objs = getattr(obs, "objects", {})
                src = objs.get("source", ((0, 0, 0), (1, 0, 0, 0)))[0]
                print(
                    f"step {i:4d} reward={reward:7.3f} success={info.success} "
                    f"source_z={src[2]:.3f} sensors={list(getattr(obs, 'sensor_status', {}).keys())}"
                )
            if done:
                print("done at step", i, "success=", info.success)
                break
            time.sleep(0.003)
    finally:
        env.close()


if __name__ == "__main__":
    main()
