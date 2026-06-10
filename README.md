# RoboDeploy

**Backend-agnostic runtime for robot learning, evaluation, and deployment.**

Write tasks, policies, and sensor rigs once; run the same code on MuJoCo, Gazebo Harmonic, Isaac Sim, ROS2 + RViz, or real hardware. RoboDeploy v0.2 adds training (BC/PPO), benchmark evaluation, multimodal sensors (camera, FT, IMU, contact), safety guards, sim2real tooling, and packaging for distribution.

| | |
|---|---|
| **Version** | 0.2.0 |
| **Python** | 3.10 – 3.12 |
| **License** | MIT |
| **Status** | Beta — see [docs/PLATFORM_STATUS.md](docs/PLATFORM_STATUS.md) |

---

## Why RoboDeploy?

Most robotics codebases couple your task logic to a single simulator or to ROS message shapes. RoboDeploy separates concerns:

- **Backends** own physics, rendering, and hardware I/O.
- **Tasks** define rewards, success, and observation requirements.
- **Policies** map observations to actions (scripted, learned, or teleoperated).
- **Sensor rigs** declare cameras, force/torque, IMU, and contact sensors; the registry picks sim or real implementations per backend.

The same `RoboEnv` loop runs everywhere:

```text
reset → observe → (safety) → policy → (adapter) → step → reward / success
```

---

## Features (v0.2)

| Area | What you get |
|------|----------------|
| **Backends** | MuJoCo, Gazebo (ROS2 + Harmonic), Isaac Sim, ROS2 RViz, dummy smoke, real ROS2 |
| **Sensors** | RGB-D, wrist FT, IMU, contact, prop pose; `ObsPipeline` sync and noise |
| **Policies** | Reach DSL, joint trackers, BC/PPO checkpoints, diffusion/VLA stubs, remote serving |
| **Training** | Gym adapter, `SubprocVecEnv`, BC + PPO trainers, `examples/train_ppo_reach.py` |
| **Benchmarks** | `manipulation_v1` suite, `robodeploy eval`, HTML reports, leaderboard schema |
| **Safety** | `SafetyMonitor`, workspace/slew limits, force/velocity/collision guards, e-stop API |
| **Sim2real** | Calibration store, DR sweep, transfer metrics (dummy/mock paths tested) |
| **Multi-robot** | MuJoCo multi-arm presets and tests |
| **Distribution** | PyPI workflow ready, Docker CPU image, conda recipe smoke, plugin entry-points |

---

## Install

From the repository root:

```bash
python -m pip install -e .
```

### Optional extras

| Extra | Install | Use case |
|-------|---------|----------|
| `sim` | `pip install -e ".[sim]"` | MuJoCo backend |
| `kinematics` | `pip install -e ".[kinematics]"` | Pinocchio IK for reach policies |
| `real` | `pip install -e ".[real]"` | RealSense camera helpers |
| `dev` | `pip install -e ".[dev]"` | Tests, JAX, torch, gymnasium |
| `training` | `pip install -e ".[training]"` | BC/PPO training stack |
| `eval` | `pip install -e ".[eval]"` | Benchmark HTML reports |
| `learned` | `pip install -e ".[learned]"` | HF hub + transformers policies |
| `rl` | `pip install -e ".[rl]"` | Stable-Baselines3 smoke |
| `teleop` | `pip install -e ".[teleop]"` | Keyboard / SpaceMouse / LeRobot (WIP) |
| `docs` | `pip install -e ".[docs]"` | MkDocs site build |

**Recommended dev setup:**

```bash
python -m pip install -e ".[sim,kinematics,dev,training,eval]"
robodeploy doctor
```

Isaac Sim uses NVIDIA's Kit Python environment; the `isaacsim` extra is a marker only.

---

## Quick start (5 minutes)

```bash
# 1. Environment check
robodeploy doctor

# 2. No simulator required
robodeploy run-episode --dummy --steps 10 --json

# 3. List demo presets (examples/, not the PyPI package)
python -m examples.cli list-presets

# 4. MuJoCo pick-place smoke (Linux/macOS/Windows with [sim])
python -m examples.cli run-episode --preset kuka_pick_mujoco --steps 50

# 5. Tier-1 benchmark eval on dummy backend
robodeploy eval --benchmark manipulation_v1/reach_target --backend dummy --episodes 5
```

---

## Core API

### Preset-based (recommended for demos)

```python
from examples.env_from_preset import env_from_preset

env = env_from_preset("kuka_pick_mujoco")
obs, info = env.reset()
obs, reward, done, info = env.step()  # policy from preset runs if no action passed
env.close()
```

Presets live in [`examples/config/presets.yaml`](examples/config/presets.yaml). They wire backend, task, policy, sensor rigs, and optional `obs_pipeline` / `custom_modules`.

### Programmatic

```python
from robodeploy import RoboEnv, Robot, RobotTask, use
from robodeploy.backends.simulator import backend_for_simulator
from robodeploy.description.franka import FrankaDescription
from examples.tasks.pick_place import PickPlaceTask
from my_pkg.policies import MyPolicy

use("examples.tasks")
robot = Robot(
    robot_id="robot0",
    description=FrankaDescription(),
    tasks={
        "pick": RobotTask(task=PickPlaceTask(), policies={"main": MyPolicy()}),
    },
)
backend = backend_for_simulator("mujoco", robots=[robot])
env = RoboEnv(backend=backend, robots=[robot])
obs, info = env.reset()
obs, reward, done, info = env.step()
env.close()
```

### Config dict (canonical for apps)

```python
from robodeploy import RoboEnv, use

use("my_project.components")
env = RoboEnv.from_config({
    "backend": "mujoco",
    "robots": [{"id": "robot0", "description": "franka", "task": "pick_place", "policy": "my_policy"}],
    "sensor_rigs": [...],
})
```

See [CONTRACTS.md](CONTRACTS.md) for construction rules and [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) for the full mental model.

---

## CLI overview

| Command | Purpose |
|---------|---------|
| `robodeploy doctor` | Check MuJoCo, ROS2, Gazebo, torch, calibration dirs |
| `robodeploy list-registry` | Backends, robots, tasks, policies, sensors |
| `robodeploy run-episode --dummy` | Simulator-free smoke |
| `robodeploy eval --benchmark ...` | Run manipulation_v1 benchmarks |
| `robodeploy train bc` / `train ppo` | Train policies (dummy or preset backends) |
| `robodeploy scaffold task` | Generate task/policy boilerplate |
| `python -m examples.cli run-episode --preset ...` | Run YAML presets from `examples/` |

Full reference: [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md).

---

## Training and evaluation

```bash
# Behavioral cloning on dummy env
robodeploy train bc --dataset path/to/dataset --dummy --epochs 50

# PPO reach_target (production-scale script)
python examples/train_ppo_reach.py --backend dummy --total-steps 500000
python examples/train_ppo_kuka_pick.py --total-steps 100000

# Eval trained checkpoint on benchmark
robodeploy train eval --checkpoint bc_final.pt --dummy
robodeploy eval --benchmark manipulation_v1/reach_target --backend dummy --policy bc_final.pt
```

Details: [docs/TRAINING.md](docs/TRAINING.md), [benchmarks/README.md](benchmarks/README.md).

---

## Backends

| Name | Factory string | Notes |
|------|----------------|-------|
| MuJoCo | `mujoco` | Primary dev backend; EGL headless in CI |
| Gazebo | `gazebo` | Alias for `ros2_gazebo`; Linux + Jazzy + Harmonic |
| ROS2 RViz | `ros2_rviz` | Visualization + optional fake joint sim |
| Isaac Sim | `isaacsim` | Mock-tested in CI; GPU Kit for live |
| Real hardware | `real_world` | ROS2 controllers + sensors |
| Dummy | `dummy` | No physics; registry and CLI smoke |

Setup: [docs/BACKEND_SETUP.md](docs/BACKEND_SETUP.md).

### Gazebo multimodal pick (Linux)

```bash
pip install -e ".[kinematics,dev]"
ROBODEPLOY_LIVE_GAZEBO=1 pytest -m live_gazebo -q   # CI-equivalent smoke
python -m examples.kuka_ft_imu_pick_gazebo.run_gazebo
```

Expect `obs.images`, `obs.ft_forces`, `obs.imu_angular_velocity`, `obs.contact_state`, and `obs.objects` after reset. Troubleshooting: controller topics at `/joint_states` (not `/robot0/joint_states`).

---

## Project layout

```text
robodeploy/           Installable package
  backends/           MuJoCo, Gazebo, Isaac, ROS2, dummy
  core/               Types, registry, sensor rigs, transforms
  description/        Robot URDF/MJCF assets (Kuka, Franka, SO-101, …)
  env.py              RoboEnv control loop
  training/           Gym adapter, BC, PPO, datasets
  evaluation/         Benchmark harness, metrics, reports
  policies/learned/   BC, diffusion, VLA, robomimic adapters
  safety/             SafetyMonitor and guards
  observability/      Replay, manifests, seeding
examples/             Presets, demo tasks/policies, examples.cli
benchmarks/           manipulation_v1, sim2real suites
tests/                ~620 tests (pytest -m "not hardware")
docs/                 MkDocs guides and tutorials
plans/                Goal plans + integration status (for contributors)
```

`examples/` is **not** shipped as importable package data on PyPI; clone the repo or vendor presets for demos.

---

## Documentation

| Document | Audience |
|----------|----------|
| [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) | **Start here** — comprehensive platform guide |
| [docs/index.md](docs/index.md) | MkDocs home + doc map |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layer diagram and design principles |
| [CONTRACTS.md](CONTRACTS.md) | Public API contracts |
| [examples/README.md](examples/README.md) | Runnable demos and presets |
| [docs/PLATFORM_STATUS.md](docs/PLATFORM_STATUS.md) | What CI proves vs what is planned |
| [docs/RELEASE.md](docs/RELEASE.md) | Versioning, PyPI, PR checklist |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |

**Build docs locally:**

```bash
pip install -e ".[docs]"
mkdocs serve
```

---

## Development

```bash
python -m pytest -m "not hardware" -q          # full suite (~620 tests)
python -m pytest tests/training/ -q            # training only
python -m compileall robodeploy tests examples
```

Hardware-gated tests: [tests/HARDWARE_TESTS.md](tests/HARDWARE_TESTS.md).

---

## Roadmap and honesty

v0.2 delivers a credible integration core (~65% of strategic goals). Remaining work includes teleop/IL data collection, live Gazebo pick at ≥70% over 10 seeds, PyPI `v0.2.0` tag, Isaac GPU parity, and real-hardware benchmark automation.

- Strategic plans: [plans/README.md](plans/README.md)
- Wave 2 follow-ups: `plans/WAVE2_0N_*.md`
- Contributor integration audit: [plans/INTEGRATION_STATUS.md](plans/INTEGRATION_STATUS.md)

---

## Contributing

Issues and PRs welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before large changes. Compare branch to `main`:

https://github.com/RahulSajnani/RoboDeploy/compare/main...feat/plans-2-3-integration-core
