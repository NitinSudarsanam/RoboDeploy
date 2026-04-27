"""
RoboDeploy — unified bridge for robot learning across simulators and real hardware.

Quickstart
----------

Level 1 — direct construction (recommended for full control):

    from robodeploy import RoboEnv, Robot, RobotTask
    from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
    from robodeploy.description.franka import FrankaDescription
    from robodeploy.tasks.manipulation.pick_place import PickPlaceTask
    from my_pkg.policies import MyPolicy

    franka = Robot(
        robot_id="franka0",
        description=FrankaDescription(),
        tasks={"pick": RobotTask(task=PickPlaceTask(), policies={"main": MyPolicy()})},
    )
    env = RoboEnv(backend=MuJoCoBackend(), robots=[franka])

Level 2 — string-based make() after registering your components:

    from robodeploy import use, RoboEnv

    use("my_project.components")
    env = RoboEnv.make(robot="franka", backend="mujoco", task="pick_place", policy="my_policy")

Level 3 — config dict / Hydra / YAML:

    env = RoboEnv.from_config({
        "robot": "franka", "backend": "mujoco", "task": "pick_place", "policy": "my_policy",
        "custom_modules": ["my_project.components"],
    })
"""

from robodeploy.env import RoboEnv
from robodeploy.core.registry import use, auto_discover_entry_points as discover
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.selectors import (
    IPolicySelector,
    ITaskSelector,
    WeightedPolicySelector,
    WeightTaskSelector,
)
from robodeploy.obs_pipeline import ObsPipeline
from robodeploy.bridge import RoboBridge

__all__ = [
    "RoboEnv",
    "Robot",
    "RobotTask",
    "RoboBridge",
    "ObsPipeline",
    "ITaskSelector",
    "IPolicySelector",
    "WeightTaskSelector",
    "WeightedPolicySelector",
    "use",
    "discover",
]
