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
  tasks/          TaskBase + domain randomization (concrete tasks in examples/tasks)
  viz/            RViz marker and trace publishing
examples/         Runnable demos and structure-only examples
tests/            Unit, smoke, and hardware-gated tests
```

`robodeploy/core/interop.py` copies JAX arrays through NumPy before PyTorch conversion. NumPy-to-PyTorch may share memory through `torch.from_numpy`; the project does not currently provide a DLPack zero-copy path.

## Presets and policy chains

```python
from examples.env_from_preset import env_from_preset

env = env_from_preset("kuka_pick_mujoco")  # examples/config/presets.yaml
session = env.demo_session()  # record explicit actions for replay
```

Register a composed policy with `policy_names` in config (see `robodeploy.policies.composition.PolicyChain`).

## CLI

After install (`python -m pip install -e .`), the `robodeploy` CLI handles registry listing
and simulator-free dummy smoke runs. **Demo presets** live under `examples/` only:

```bash
robodeploy list-registry --builtins
python -m examples.cli list-presets
```

Common debugging patterns:

```bash
# Load user-registered components before listing
robodeploy list-registry --custom-module examples.user_kuka_sinusoid.components

# Load pip-installed extensions (entry points) before listing
robodeploy list-registry --discover

# Machine-parseable JSON output
python -m examples.cli list-presets --json
robodeploy list-registry --builtins --json --pretty

# Simulator-free smoke runs (library dummy backend)
robodeploy run-episode --dummy --steps 10 --action sinusoid --json
robodeploy run-episode --dummy --steps 1 --json --pretty
robodeploy export-episode --dummy --steps 10 --action hold --out demo.jsonl --json

# Preset-based runs (examples/config/presets.yaml)
python -m examples.cli run-episode --preset kuka_pick_mujoco --dummy --steps 10 --json
python -m examples.mujoco_universe.run --preset mujoco_showcase_kuka
```

## Basic use

```python
from robodeploy import RoboEnv, Robot, RobotTask
from robodeploy.backends.simulator import backend_for_simulator
from robodeploy.description.franka import FrankaDescription
from robodeploy import use
from examples.tasks.pick_place import PickPlaceTask  # or your own @register_task
from my_project.policies import MyPolicy

use("examples.tasks")  # if using task="pick_place" via RoboEnv.make()
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

**Canonical:** `RoboEnv.from_config(cfg)` or `examples.env_from_preset("kuka_pick_mujoco")` for
demo YAML under `examples/config/presets.yaml` (sensor rigs, `custom_modules`, obs pipeline).

**Minimal smoke:** `RoboEnv.make(...)` — flat `sensors: list[str]` only; no `sensor_rigs` or
example presets. Register components first:

```python
from robodeploy import RoboEnv, use

use("my_project.components")  # registers tasks, policies, robots, ...
env = RoboEnv.make(
    robot="franka",
    backend="mujoco",
    task="my_task",
    policy="my_policy",
    backend_kwargs={"enable_viewer": False},
)
```

`from_config()` accepts registry names or already-constructed robot/backend/task/policy objects.
See [CONTRACTS.md](CONTRACTS.md) for the full construction hierarchy and [history.json](history.json) for current state and gaps.

## Backends and examples

Backend setup notes live in [docs/BACKEND_SETUP.md](docs/BACKEND_SETUP.md). The example index in [examples/README.md](examples/README.md) lists the maintained smoke paths. Useful starting points:

```bash
python -m examples.user_kuka_sinusoid.run_mujoco
python -m examples.user_kuka_sinusoid.run_ros2_rviz --fake-sim
python -m examples.user_kuka_sinusoid.run_switch_simulator
```

Real SO-101 setup and calibration are documented in [docs/SO101_REAL.md](docs/SO101_REAL.md).

