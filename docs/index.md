# RoboDeploy Documentation

RoboDeploy is a backend-agnostic robot learning runtime. Write tasks, policies, and sensor rigs once; run on MuJoCo, Gazebo, Isaac Sim, or real hardware via ROS2.

## Quick start

```bash
pip install -e ".[sim,dev]"
robodeploy doctor
robodeploy run-episode --dummy --steps 5
python -m examples.cli run-episode --preset kuka_pick_mujoco --steps 50
```

## Documentation map

| Topic | Guide |
|-------|-------|
| First run | [Tutorial 1](tutorials/01_getting_started.md) |
| Custom tasks | [Task creation](TASK_CREATION.md) |
| Scripted policies | [Policy creation](POLICY_CREATION.md) |
| Worlds and props | [Scene definition](SCENE_DEFINITION.md) |
| Cameras, FT, IMU | [Sensor integration](SENSOR_INTEGRATION.md) |
| Recipes | [Cookbook](COOKBOOK.md) |
| Upgrading from 0.1.x | [Migration 0.2](MIGRATION_0.2.md) |
| All CLI commands | [CLI reference](CLI_REFERENCE.md) |

## Build docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Production site is published to GitHub Pages on push to `main` (see `.github/workflows/docs.yml`).
