"""Run the user-defined Kuka sinusoid demo on Isaac Sim (if installed)."""

from __future__ import annotations

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()
from pathlib import Path


from robodeploy.core.robot import Robot, RobotTask  # noqa: E402
from robodeploy.env import RoboEnv  # noqa: E402

# Import registers @register_* components and exposes the user classes.
from examples.user_kuka_sinusoid.components import (  # noqa: E402
    UserKukaDescription,
    UserKukaSinusoidTask,
    UserSinusoidPolicy,
)


def main() -> None:
    from robodeploy.backends.sim.isaacsim.backend import IsaacSimBackend

    backend = IsaacSimBackend(config={
            # Use a lighter experience by default to avoid optional extensions
            # failing to load on some Windows GPU/driver setups.
            "experience": "isaacsim.exp.base.python.kit",
            "headless": False,
            "renderer": "RaytracedLighting",
        })

    desc = UserKukaDescription()
    task = UserKukaSinusoidTask(max_steps=2000)
    policy = UserSinusoidPolicy(amplitude=0.35, frequency_hz=0.2)
    robot = Robot(
        robot_id="robot0",
        description=desc,
        sensors=[],
        tasks={
            "sinusoid": RobotTask(
                task=task,
                policies={"main": policy},
                mode="sequential",
            )
        },
    )
    env = RoboEnv(backend=backend, robots=[robot])

    env.reset()
    for _ in range(1000):
        env.step()
    env.close()


if __name__ == "__main__":
    main()

