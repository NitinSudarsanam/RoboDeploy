# RoboDeploy Documentation

Backend-agnostic robot learning runtime. Write tasks, policies, and sensor rigs once; run on MuJoCo, Gazebo, Isaac Sim, or real hardware via ROS2.

**New to the project?** Read the [Project Guide](PROJECT_GUIDE.md) first — it covers the full platform end to end.

---

## Quick start

```bash
pip install -e ".[sim,dev]"
robodeploy doctor
robodeploy run-episode --dummy --steps 5
python -m examples.cli run-episode --preset kuka_pick_mujoco --steps 50
```

---

## Documentation map

### Start here

| Guide | Description |
|-------|-------------|
| [Project Guide](PROJECT_GUIDE.md) | Comprehensive platform overview |
| [Platform status](PLATFORM_STATUS.md) | Maturity, CI coverage, roadmap |
| [Tutorial 1 — Getting started](tutorials/01_getting_started.md) | Install and first episode |

### Tutorials

| # | Topic |
|---|-------|
| [01](tutorials/01_getting_started.md) | Hello RoboDeploy |
| [02 Teleop](tutorials/02_teleop.md) | Teleoperation concepts |
| [03 Training](tutorials/03_training.md) | BC and PPO |
| [04 Sim2Real](tutorials/04_sim2real.md) | Calibration and transfer |
| [05 Safety](tutorials/05_safety.md) | Guards and e-stop |

### How-to guides

| Guide | Topic |
|-------|-------|
| [Task creation](TASK_CREATION.md) | Custom tasks and rewards |
| [Policy creation](POLICY_CREATION.md) | Scripted and learned policies |
| [Scene definition](SCENE_DEFINITION.md) | Props, Scene IR, YAML DSL |
| [Sensor integration](SENSOR_INTEGRATION.md) | Rigs, cameras, FT, IMU |
| [Training](TRAINING.md) | Production PPO paths, nightly CI |
| [Sim2Real](SIM2REAL.md) | Calibration, DR, transfer metrics |
| [Safety](SAFETY.md) | SafetyMonitor and real-hardware |
| [Plugins](PLUGINS.md) | Entry points and extensions |
| [Cookbook](COOKBOOK.md) | Copy-paste recipes |

### Reference

| Doc | Topic |
|-----|-------|
| [CLI reference](CLI_REFERENCE.md) | All `robodeploy` commands |
| [API reference](API_REFERENCE.md) | Module index |
| [Backend setup](BACKEND_SETUP.md) | MuJoCo, Gazebo, Isaac, ROS2 |
| [SO-101 real robot](SO101_REAL.md) | Hardware bring-up |
| [Migration 0.2](MIGRATION_0.2.md) | Upgrade from 0.1.x |
| [Release](RELEASE.md) | Versioning and PyPI |

### Repository docs (outside MkDocs root)

| Doc | Topic |
|-----|-------|
| [README.md](../README.md) | Repo landing page |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | Design principles |
| [CONTRACTS.md](../CONTRACTS.md) | Public API contracts |
| [examples/README.md](../examples/README.md) | Demos and presets |
| [benchmarks/README.md](../benchmarks/README.md) | Eval suites |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Contributor guide |
| [plans/INTEGRATION_STATUS.md](../plans/INTEGRATION_STATUS.md) | CI ↔ claims audit |

---

## Build docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Open http://127.0.0.1:8000. Production site deploys via `.github/workflows/docs.yml` on push to `main`.
