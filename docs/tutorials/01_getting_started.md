# Tutorial 1 — Getting Started with RoboDeploy

**Time:** ~10 minutes  
**Goal:** Install RoboDeploy, run your first episode, and understand the core concepts.

## Install

From the repo root:

```bash
python -m pip install -e .
python -m pip install -e ".[sim,dev]"
robodeploy doctor
```

`robodeploy doctor` reports MuJoCo, PyTorch, ROS 2, and calibration directory status. Fix any `[FAIL]` lines before running on hardware.

## Hello world (5 commands)

```bash
# 1. List registered backends, robots, tasks, policies
robodeploy list-registry --builtins

# 2. Smoke test without a simulator (dummy backend)
robodeploy run-episode --dummy --steps 10 --json

# 3. List demo presets (examples/config/presets.yaml)
python -m examples.cli list-presets

# 4. Resolve a preset to see full config
robodeploy config resolve --preset kuka_pick_mujoco --json

# 5. Run a MuJoCo pick-place episode (requires [sim] extra)
python -m examples.cli run-episode --preset kuka_pick_mujoco --steps 50
```

## Core concepts

| Concept | What it is |
|---------|------------|
| **Backend** | Simulator or hardware adapter (`mujoco`, `gazebo`, `ros2`, `dummy`). Owns physics and sensor I/O. |
| **Robot** | Kinematic description + joint limits (`kuka`, `franka`, custom URDF/MJCF). |
| **Scene** | Table, objects, lighting — built with `SceneBuilder` or scene YAML. |
| **Task** | Reward, success, reset, and `obs_spec` — what the robot should accomplish. |
| **Policy** | Maps observations to actions (scripted reach DSL, BC checkpoint, teleop). |
| **Sensor rig** | Declares cameras, FT, IMU, prop pose estimators attached to the arm. |

Data flows: `backend.reset()` → `task.reset_fn()` → loop: `obs = backend.get_obs()` → `action = policy.get_action(obs)` → `backend.step(action)` → `reward = task.reward_fn(obs, action)`.

## Scaffold your first task

```bash
robodeploy scaffold task --name kitchen_pick --template pick_place --output examples/tasks/kitchen_pick.py
robodeploy lint task examples/tasks/kitchen_pick.py
```

Add `examples.tasks` to your preset `custom_modules`, then run with `python -m examples.cli run-episode --preset <yours>`.

## Next steps

- [Tutorial 2 — Teleop](02_teleop.md) — collect demonstrations with keyboard.
- [Task creation guide](../TASK_CREATION.md) — deeper API reference.
- [Examples catalog](../../examples/README.md) — runnable demos by capability.
