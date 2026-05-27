"""Franka Panda robomimic policy demo (structure-only).

This demo was migrated from `robodeploy.demos` to `examples/` as part of the
architecture cleanup. It now demonstrates the intended *public* API wiring:

- `RoboEnv` orchestrates
- `FrankaDescription` defines the robot
- `MuJoCoBackend` (sim) or `ROS2Backend` (real) provides execution
- `PickPlaceTask` defines the task
- `RobomimicPolicy` produces actions

Note: concrete backend implementations are currently stubs, so this example
is not expected to run yet.
"""

from __future__ import annotations

from pathlib import Path

from robodeploy import RoboEnv
from robodeploy.backends import MuJoCoBackend, ROS2Backend
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.description.franka import FrankaDescription
from robodeploy.policies.learned.robomimic import RobomimicPolicy
from robodeploy.tasks.manipulation.pick_place import PickPlaceTask


def make_env(checkpoint: Path, use_real: bool) -> RoboEnv:
    if use_real and ROS2Backend is None:
        raise ImportError("ROS2 backend is unavailable in this Python environment.")
    backend = ROS2Backend() if use_real else MuJoCoBackend()
    policy = RobomimicPolicy(checkpoint_path=checkpoint)
    robot = Robot(
        robot_id="franka0",
        description=FrankaDescription(),
        tasks={"pick_place": RobotTask(task=PickPlaceTask(), policies={"robomimic": policy})},
    )
    return RoboEnv(
        backend=backend,
        robots=[robot],
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--real", action="store_true")
    args = p.parse_args()

    env = make_env(args.checkpoint, use_real=args.real)
    obs, info = env.reset()
    print("reset:", info)
    env.close()


if __name__ == "__main__":
    main()

