# Kuka Pick & Place Demo

A minimal, self-contained pick-and-place demo for the Kuka arm. The same task, policy, sensors, and scene run on **MuJoCo**, **ROS 2 + RViz**, and **Gazebo** — you only change which simulator is active.

This folder is intentionally small: `demo/` holds the entry script and YAML preset; all robot models, backends, tasks, and policies live in `robodeploy/`.

---

## What it does

1. Spawns a table with a **red source cube** and a **green target sphere** at fixed world-frame poses (see `robodeploy/demos/scenes/pick_table.py`).
2. Runs a sensor-driven reach policy (`example_sensor_reach_pick`) that:
   - reaches the cube using wrist FT, contact, prop pose, and EE pose sensors;
   - engages a kinematic carry when grasp conditions are met;
   - moves the cube toward the target and reports success when placement criteria are satisfied.
3. Prints progress every 100 steps and ends with `Success.` or `Done.`

On MuJoCo you typically see success around step 400 with the default seed. RViz and Gazebo require a Linux or WSL2 environment with ROS 2 Jazzy.

---

## File layout

| Path | Purpose |
|------|---------|
| `demo/run_pick.py` | Entry point — edit `SIMULATOR` and `SEED` here |
| `demo/config/kuka_pick.yaml` | Per-backend config (task, policy, sensors, backend kwargs) |
| `robodeploy/demos/scenes/pick_table.py` | Shared scene geometry (table, cube, target) |
| `robodeploy/demos/tasks/pick_place.py` | Task registration (`pick_place`) |
| `robodeploy/policies/reach_dsl.py` | Reach / carry / place policy logic |

---

## Prerequisites

### All backends

From the **repository root**:

```bash
python -m pip install -e ".[sim,kinematics,dev]"
```

Optional sanity check:

```bash
robodeploy doctor
```

Pinocchio (`[kinematics]`) is recommended for world-frame EE FK on ROS backends.

### Per simulator

| Simulator | OS | Python install | System packages |
|-----------|----|----------------|-----------------|
| **mujoco** (default) | Windows, Linux, macOS | `pip install -e ".[sim,kinematics]"` | None |
| **rviz** | **Linux or WSL2 only** | `pip install -e ".[sim,kinematics]"` | ROS 2 **Jazzy**, RViz2, `robot_state_publisher` |
| **gazebo** | **Linux or WSL2 only** | `pip install -e ".[sim,kinematics]"` | ROS 2 Jazzy, Gazebo Harmonic (`gz`), `ros_gz_bridge`, `gz_ros2_control`, `ros2-controllers` |

**Native Windows:** MuJoCo works out of the box. RViz and Gazebo will fail with `ImportError: ROS 2 packages not found` because `rclpy` is not available on Windows — use WSL2 (Ubuntu 24.04 + Jazzy) or Docker instead.

---

## Quick start (MuJoCo — recommended first run)

1. Clone the repo and open a terminal at the repo root.

2. Install:

   ```bash
   python -m pip install -e ".[sim,kinematics]"
   ```

3. Confirm `demo/run_pick.py` has:

   ```python
   SIMULATOR = "mujoco"
   SEED = 0
   ```

4. Run:

   ```bash
   python demo/run_pick.py
   ```

5. **Expected console output** (approximate):

   ```
   Kuka pick & place (MuJoCo)
   step 0: success=False
   step 100: success=False
   ...
   step 400: success=False
   Success.
   ```

6. A MuJoCo viewer window opens (`enable_viewer: true` in config). You should see the arm reach toward the red cube on the table, pick it up, and place it near the green target.

---

## Switching simulators

Edit the constants at the top of `demo/run_pick.py`:

```python
SIMULATOR = "mujoco"   # mujoco | rviz | gazebo
SEED = 0
```

Then run the same command:

```bash
python demo/run_pick.py
```

`run_pick.py` loads the matching block from `demo/config/kuka_pick.yaml` (`mujoco`, `rviz`, or `gazebo`). You do **not** need separate scripts per backend.

| `SIMULATOR` | Viewer | Notes |
|-------------|--------|-------|
| `mujoco` | MuJoCo GUI | Works on Windows; best for first-time setup |
| `rviz` | RViz2 | Fake joint sim + markers; needs ROS sourced in WSL |
| `gazebo` | Gazebo GUI | Full physics sim; slower startup (~60 s readiness) |

---

## RViz (WSL2 / Linux)

### One-time WSL setup (Ubuntu 24.04 + Jazzy)

If you use WSL, bootstrap once from **inside WSL** at the repo root:

```bash
bash scripts/wsl24-bootstrap.sh
```

This creates `.venv-wsl` with the right dependencies and documents ROS package installation.

### Run with RViz window

1. In WSL, source ROS and activate the venv:

   ```bash
   source /opt/ros/jazzy/setup.bash
   source .venv-wsl/bin/activate
   ```

2. Set `SIMULATOR = "rviz"` in `demo/run_pick.py`.

3. From repo root:

   ```bash
   python demo/run_pick.py
   ```

4. RViz opens with:
   - robot model from URDF + `/joint_states`;
   - scene markers (table, red cube, green target) in **world** frame;
   - the cube should **move with the gripper** during carry (live marker updates).

### Headless RViz smoke (no GUI)

```bash
bash scripts/wsl_rviz_pick_smoke.sh
```

This uses the `examples/` CLI preset path for CI-style validation. The `demo/run_pick.py` path is equivalent for interactive use.

### RViz troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ROS 2 packages not found` | ROS not sourced or running on native Windows | Use WSL; `source /opt/ros/jazzy/setup.bash` |
| `CONNECTION_LOST` / stale joint states | Stray Gazebo or bridge publishing `/clock` | Kill stale processes: `pkill -f ros_gz_bridge; pkill -f 'gz sim'` |
| Arm reaches wrong direction | Frame mismatch (old configs) | Ensure `demo/config/kuka_pick.yaml` has `robot0.base_frame: world`, `rviz.fixed_frame: world`, `prefer_fk_ee_pose: true` |
| Cube stuck on table in RViz | Markers not updating | Should be fixed in current `main`; pull latest |

---

## Gazebo (WSL2 / Linux)

### Run with Gazebo GUI

1. Source ROS and venv (same as RViz above).

2. Set `SIMULATOR = "gazebo"` in `demo/run_pick.py`.

3. Run:

   ```bash
   python demo/run_pick.py
   ```

4. Gazebo launches with the Kuka URDF, table, and props. Episode limit is **4000 steps** (longer than MuJoCo/RViz) to allow sim startup and physics settling.

### Headless Gazebo smoke

```bash
bash scripts/wsl_gazebo_pick_smoke.sh
```

### Gazebo troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `gz: command not found` | Gazebo Harmonic not installed | Install via ROS Jazzy Gazebo packages |
| Readiness timeout | Slow first launch | Increase `readiness_timeout_s` in `kuka_pick.yaml` |
| Cube jitters while carried | Physics fighting kinematic sync | Expected to be reduced with pre-step pose sync; full fix may need static/kinematic entity mode |
| Placement snap at end | Legacy behavior | Default is **off** (`gazebo_place_snap: false`); set `ROBODEPLOY_GAZEBO_PLACE_SNAP=1` to re-enable |

---

## Configuration reference

All tunables live in `demo/config/kuka_pick.yaml`. The file uses YAML anchors so **task, policy, and sensors are shared** across backends; only the `backend` / `backend_kwargs` block differs.

### Key fields

| Key | Default | Meaning |
|-----|---------|---------|
| `max_episode_steps` | `2000` (4000 for gazebo) | Hard step limit; also passed to `run_pick.py` loop |
| `policy_kwargs.config.sensor_only` | `true` | Use observation sensors only (no privileged state) |
| `policy_kwargs.config.carry_mode` | `follow` | Remapped to kinematic carry on ROS backends |
| `policy_kwargs.config.gazebo_place_snap` | `false` | Skip end-of-place teleport (matches MuJoCo) |
| `task_kwargs.grasp_success_force_min` | `1.0` | Minimum FT for grasp success signal |
| `sensor_rigs` | arm_sensors rig | FT, IMU, contact, prop_pose, ee_pose |

### MuJoCo-specific (`backend_kwargs.config`)

- `enable_viewer: true` — opens interactive viewer
- `allow_actuator_name_fallback: true` — tolerates MJCF/URDF naming differences

### RViz-specific (`backend_kwargs.config`)

- `prefer_fk_ee_pose: true` — world-frame EE from Pinocchio FK
- `robot0.base_frame: world` — align EE with scene object poses
- `rviz.fixed_frame: world` — markers and robot in same frame
- `dev_fake_sim` — internal joint simulator when no hardware is connected

### Gazebo-specific (`backend_kwargs.config`)

- `sim.kind: gazebo`
- `sim.headless: false` — GUI on
- `sim.require_sensors: true` — wait for sensor topics before stepping
- `sim.readiness_timeout_s: 60.0`

### Changing seed or step limit without editing YAML

`run_pick.py` reads `SEED` from the script and `max_episode_steps` from the loaded YAML block. To experiment:

```python
SEED = 42
```

Edit `max_episode_steps` under the relevant simulator block in `kuka_pick.yaml` if episodes time out.

---

## Scene layout (what you should see)

All backends use the same world-frame layout from `pick_table.py`:

| Object | Position (x, y, z) m | Appearance |
|--------|----------------------|------------|
| Table center | (0.55, 0.0, 0.365) | Brown box, 0.9 × 0.7 m top |
| Table top surface | z = 0.38 | — |
| Source cube | (0.55, 0.0, 0.405) | Red 50 mm cube |
| Target | (0.60, 0.20, 0.42) | Green sphere, r = 0.04 m |

The robot base sits at the world origin. If the arm appears to reach away from the cube, check that world-frame settings are active (see RViz troubleshooting).

---

## Relationship to `examples/`

This demo is a **simplified front door** to the same stack documented in [`docs/DEMO_RUNBOOK.md`](../docs/DEMO_RUNBOOK.md).

| Use case | Command |
|----------|---------|
| **This demo** (minimal) | `python demo/run_pick.py` |
| CLI with presets | `python -m examples.cli run-episode --preset kuka_ft_imu_pick --simulator mujoco --seed 0 --viewer` |
| Visual runner module | `python -m examples.kuka_ft_imu_pick.run_visual --simulator mujoco --seed 0` |

Presets in `examples/config/presets.yaml` and `examples/presets/` mirror the same task/policy; `demo/config/kuka_pick.yaml` is a standalone copy for presentations and quick edits without touching the examples tree.

---

## Automated tests

From repo root:

```bash
python -m pytest tests/test_pick_scene_parity.py tests/test_pick_parity.py -q
```

These verify cross-backend scene pose equivalence, world-frame RViz config, EE sensor behavior, and marker republish on carry.

---

## Windows workflow summary

```
┌─────────────────────────────────────────────────────────┐
│  Windows (native)                                       │
│  pip install -e ".[sim,kinematics]"                     │
│  python demo/run_pick.py   # SIMULATOR = "mujoco"       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  WSL2 Ubuntu 24.04 + ROS Jazzy                          │
│  bash scripts/wsl24-bootstrap.sh   # once               │
│  source /opt/ros/jazzy/setup.bash                       │
│  source .venv-wsl/bin/activate                          │
│  python demo/run_pick.py   # SIMULATOR = rviz|gazebo  │
└─────────────────────────────────────────────────────────┘
```

For Docker-based ROS/Gazebo on Windows without a 24.04 WSL distro, see the Docker sections in `docs/DEMO_RUNBOOK.md`.
