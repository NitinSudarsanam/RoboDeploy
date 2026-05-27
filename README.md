# RoboDeploy

RoboDeploy is a small runtime for running the same robot task and policy code against MuJoCo, ROS2/RViz, Gazebo, Isaac Sim, or a real ROS2 hardware adapter. User code usually talks to `RoboEnv`, `Robot`, `RobotTask`, backend factories, and the shared `Observation` / `Action` dataclasses.

## Install

Editable install from the repo root:

```bash
python -m pip install -e .
```

Optional extras are split by use case:

```bash
python -m pip install -e ".[sim]"
python -m pip install -e ".[real]"
python -m pip install -e ".[dev]"
```

Isaac Sim is handled by NVIDIA's own Python environment, so the `isaacsim` extra is only a marker for now.

## Project layout

```text
robodeploy/
  backends/       Simulator and ROS2 hardware adapters
  core/           Shared types, registries, robot/task arbitration
  description/    Robot descriptions and bundled URDF/MJCF/mesh assets
  policies/       Policy interfaces plus placeholder examples
  sensors/        Camera, RGBD, force/torque, and sensor pairing code
  tasks/          Task interfaces and manipulation task stubs
  viz/            RViz marker and trace publishing
examples/         Runnable demos and structure-only examples
tests/            Unit, smoke, and hardware-gated tests
```

`robodeploy/core/interop.py` copies JAX arrays through NumPy before PyTorch conversion. NumPy-to-PyTorch may share memory through `torch.from_numpy`; the project does not currently provide a DLPack zero-copy path.

## Presets and policy chains

```python
from robodeploy import RoboEnv

env = RoboEnv.from_preset("kuka_pick_mujoco")  # YAML preset -> RoboEnv.make
session = env.demo_session()  # record explicit actions for replay
```

Register a composed policy with `policy_names` in config (see `robodeploy.policies.composition.PolicyChain`).

## CLI

After install (`python -m pip install -e .`), the `robodeploy` CLI is available:

```bash
robodeploy list-presets
robodeploy list-registry --builtins
```

## Basic use

```python
from robodeploy import RoboEnv, Robot, RobotTask
from robodeploy.backends.simulator import backend_for_simulator
from robodeploy.description.franka import FrankaDescription
from robodeploy.tasks.manipulation.pick_place import PickPlaceTask
from my_project.policies import MyPolicy

robot = Robot(
    robot_id="robot0",
    description=FrankaDescription(),
    tasks={
        "pick": RobotTask(
            task=PickPlaceTask(),
            policies={"main": MyPolicy()},
        )
    },
)

backend = backend_for_simulator("mujoco", robots=[robot])
env = RoboEnv(backend=backend, robots=[robot])

obs, info = env.reset()
obs, reward, done, info = env.step()
env.close()
```

The same `Robot` can be passed to `backend_for_simulator("ros2_rviz", ...)`, `"gazebo"`, `"isaacsim"`, or `"real_world"` when the required optional dependencies and robot-specific configuration are available.

## Configuration and registration

For string-based construction, register project components first:

```python
from robodeploy import RoboEnv, use

use("my_project.components")
env = RoboEnv.make(
    robot="franka",
    backend="mujoco",
    task="pick_place",
    policy="my_policy",
    backend_kwargs={"enable_viewer": False},
)
```

`RoboEnv.from_config()` supports the same registry names and, on this branch, can also accept already constructed robot/backend/task/policy objects for lightweight programmatic setup.

## Backends and examples

Backend setup notes live in [docs/BACKEND_SETUP.md](docs/BACKEND_SETUP.md). The example index in [examples/README.md](examples/README.md) lists the maintained smoke paths. Useful starting points:

```bash
python -m examples.user_kuka_sinusoid.run_mujoco
python -m examples.user_kuka_sinusoid.run_ros2_rviz --fake-sim
python -m examples.user_kuka_sinusoid.run_switch_simulator
```

Real SO-101 setup and calibration are documented in [docs/SO101_REAL.md](docs/SO101_REAL.md).

