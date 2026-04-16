"""
RoboDeploy — unified bridge for robot learning across simulators and real hardware.

Quickstart
----------

Level 1 — direct injection (recommended for most users):

    from robodeploy         import RoboEnv
    from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
    from robodeploy.description.franka          import FrankaDescription
    from robodeploy.tasks.manipulation.pick_place import PickPlaceTask

    env = RoboEnv(
        description = FrankaDescription(),
        backend     = MuJoCoBackend(),
        task        = PickPlaceTask(),
    )

Level 2 — string-based make() after registering your components:

    from robodeploy import use, RoboEnv

    use("my_project.components")   # triggers @register_* decorators in your code
    env = RoboEnv.make(robot="myrobot", backend="mujoco", task="my_task")

Level 3 — config dict / Hydra / YAML:

    from robodeploy import RoboEnv

    env = RoboEnv.from_config({
        "robot": "franka", "backend": "mujoco", "task": "pick_place",
        "custom_modules": ["my_project.components"],   # optional
    })

Level 3 + third-party packages (pip-installable robots/tasks):

    from robodeploy import discover, RoboEnv

    discover()   # scans entry points from all installed packages
    env = RoboEnv.make(robot="community_robot", backend="mujoco", task="my_task")
"""

from robodeploy.env                     import RoboEnv
from robodeploy.core.registry           import use, auto_discover_entry_points as discover
from robodeploy.obs_pipeline            import ObsPipeline
from robodeploy.bridge                  import RoboBridge

__all__ = [
    "RoboEnv",
    "RoboBridge",
    "ObsPipeline",
    "use",
    "discover",
]
