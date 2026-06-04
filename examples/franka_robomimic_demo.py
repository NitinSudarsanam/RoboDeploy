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
from examples.tasks.pick_place import PickPlaceTask


def make_env(checkpoint: Path | None, use_real: bool, *, dry_run: bool = False) -> RoboEnv:
    if use_real and ROS2Backend is None:
        raise ImportError("ROS2 backend is unavailable in this Python environment.")
    backend = ROS2Backend() if use_real else MuJoCoBackend()
    if dry_run:
        import numpy as np

        def predict_fn(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
            state = obs_dict["state"]
            return np.concatenate([state[:7] * 0.0, np.array([0.0])])

        policy = RobomimicPolicy(config={"predict_fn": predict_fn, "arm_dof": 7})
    else:
        if checkpoint is None:
            raise ValueError("--checkpoint is required unless --dry-run is set.")
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
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--real", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Use injectable predict_fn (no robomimic checkpoint).")
    args = p.parse_args()

    env = make_env(args.checkpoint, use_real=args.real, dry_run=args.dry_run)
    obs, info = env.reset()
    print("reset:", info)
    env.close()


if __name__ == "__main__":
    main()

