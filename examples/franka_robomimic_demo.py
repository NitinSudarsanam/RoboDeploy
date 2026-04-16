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
from robodeploy.description.franka import FrankaDescription
from robodeploy.policies.learned.robomimic import RobomimicPolicy
from robodeploy.tasks.manipulation.pick_place import PickPlaceTask


def make_env(checkpoint: Path, use_real: bool) -> RoboEnv:
    backend = ROS2Backend() if use_real else MuJoCoBackend()
    policy = RobomimicPolicy(checkpoint_path=checkpoint)
    return RoboEnv(
        description=FrankaDescription(),
        backend=backend,
        task=PickPlaceTask(),
        policy=policy,
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--real", action="store_true")
    args = p.parse_args()

    _ = make_env(args.checkpoint, use_real=args.real)
    raise NotImplementedError(
        "Backend implementations are stubs in this migration. "
        "Once MuJoCoBackend/ROS2Backend are implemented, call env.reset()/env.step()."
    )


if __name__ == "__main__":
    main()

