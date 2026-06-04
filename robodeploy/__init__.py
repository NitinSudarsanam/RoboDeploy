"""
RoboDeploy — unified bridge for robot learning across simulators and real hardware.

Quickstart
----------

Level 1 — direct construction (recommended for full control):

    from robodeploy import RoboEnv, Robot, RobotTask
    from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
    from robodeploy.description.franka import FrankaDescription
    from my_pkg.tasks import MyTask
    from my_pkg.policies import MyPolicy

    franka = Robot(
        robot_id="franka0",
        description=FrankaDescription(),
        tasks={"pick": RobotTask(task=MyTask(), policies={"main": MyPolicy()})},
    )
    env = RoboEnv(backend=MuJoCoBackend(), robots=[franka])

To swap simulators without hand-built backend dicts (ROS2 imports load lazily):

    from robodeploy.backends.simulator import backend_for_simulator

    env = RoboEnv(backend=backend_for_simulator("mujoco", robots=[franka]), robots=[franka])

Level 2 — string-based make() after registering your components:

    from robodeploy import use, RoboEnv

    use("my_project.components")  # registers my_task, my_policy, ...
    env = RoboEnv.make(robot="franka", backend="mujoco", task="my_task", policy="my_policy")

Level 3 — config dict:

    env = RoboEnv.from_config({
        "robot": "franka", "backend": "mujoco", "task": "my_task", "policy": "my_policy",
        "custom_modules": ["my_project.components"],
    })

Example YAML presets (not in the installed package) live under ``examples/config/`` — use
``examples.env_from_preset("kuka_pick_mujoco")`` after ``pip install -e .`` from the repo.
"""

from robodeploy.env import RoboEnv
from robodeploy.core.registry import use, auto_discover_entry_points as discover
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.sensor_rig import SensorRig
from robodeploy.core.selectors import (
    IPolicySelector,
    ITaskSelector,
    WeightedPolicySelector,
    WeightTaskSelector,
)
from robodeploy.obs_pipeline import ObsPipeline

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "RoboBridge":
        from robodeploy.bridge import RoboBridge

        return RoboBridge
    raise AttributeError(name)

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
    "__version__",
]
