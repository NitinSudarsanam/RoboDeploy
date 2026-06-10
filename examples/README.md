# Examples index

Runnable demos, YAML presets, and example tasks/policies. These live **outside** the `robodeploy` PyPI package — clone the repo or vendor this tree.

**Also read:** [docs/PROJECT_GUIDE.md](../docs/PROJECT_GUIDE.md) (configuration paths), [docs/BACKEND_SETUP.md](../docs/BACKEND_SETUP.md) (MuJoCo / ROS2 / Gazebo / Isaac setup), [presets/README.md](presets/README.md) (YAML fragment structure).

---

## Quick commands

```bash
# After: pip install -e ".[sim]"  (MuJoCo presets)
python -m examples.cli list-presets
python -m examples.cli run-episode --preset kuka_pick_mujoco --steps 50
python -m examples.mujoco_universe.run --list
python -m examples.mujoco_universe.run --preset mujoco_showcase_kuka
```

Library CLI (`robodeploy run-episode --dummy`) does **not** bundle example presets — use `python -m examples.cli`.

---

## Preset reference (`config/presets.yaml`)

Presets compose shared fragments from [`presets/`](presets/) (`base_sim.yaml`, `manipulate.yaml`, `arm_sensors`, etc.).

| Preset | Backend | Task | Policy | Notes |
|--------|---------|------|--------|-------|
| `kuka_pick_mujoco` | mujoco | pick_place | example_sensor_reach_pick | **Default** sensor-first pick demo |
| `franka_pick_mujoco` | mujoco | pick_place | example_sensor_reach_pick | Franka MJCF naming |
| `kuka_sensor_pick_mujoco` | mujoco | pick_place | example_sensor_reach_pick | Explicit prop_pose rig |
| `kuka_ft_imu_pick_mujoco` | mujoco | pick_place | example_reach_pick | FT grasp + follow carry |
| `mujoco_showcase_kuka` | mujoco | showcase_scene | example_joint_track | Full sensor rig showcase |
| `mujoco_showcase_franka` | mujoco | showcase_scene | example_joint_track | Franka showcase |
| `mujoco_pick_kuka` | mujoco | pick_place | example_sensor_reach_pick | Universe catalog alias |
| `kuka_sinusoid_mujoco` | mujoco | user_kuka_sinusoid | user_sinusoid | Joint sinusoid smoke |
| `kuka_sensor_ros2_rviz` | ros2_rviz | pick_place | example_sensor_reach_pick | ROS2 sensor smoke |
| `kuka_sensor_gazebo` | gazebo | pick_place | example_sensor_reach_pick | Gazebo sensor smoke |
| `kuka_ft_imu_pick_gazebo` | gazebo | pick_place | example_reach_pick | Multimodal pick E2E (Linux) |
| `two_franka_pick_mujoco` | mujoco | pick_place | example_sensor_reach_pick | Multi-robot (2 arms) |

Benchmark-oriented presets live under `benchmarks/manipulation_v1/*/preset_*.yaml` — do not edit for leaderboard submissions.

Override presets file: `ROBODEPLOY_PRESETS_FILE` or `python -m examples.cli --presets-file path/to/presets.yaml`.

### Python API

```python
from examples.env_from_preset import env_from_preset
env = env_from_preset("kuka_pick_mujoco")

from examples.vecenv_from_presets import vecenv_from_presets
vec = vecenv_from_presets(["kuka_pick_mujoco"], n_envs=4)
```

---

## Registered example modules

Import via `custom_modules` in preset YAML or `use("examples.tasks")` in code:

| Module | Registers |
|--------|-----------|
| `examples.tasks` | `pick_place`, `pour`, `peg_insertion`, `showcase_scene`, … |
| `examples.policies` | `example_reach_pick`, `example_sensor_reach_pick`, `example_joint_track` |
| `examples.sensors` | `sim_prop_pose` oracle reader |
| `examples.user_kuka_sinusoid.components` | `user_kuka_sinusoid` task, `user_sinusoid` policy |

---

## Runnable pick-and-place demos (MuJoCo)

Pick-place demos use **`example_sensor_reach_pick`** and `SimPropPoseSensor` by default (`kuka_pick_mujoco`). Object poses come from `Observation.objects`, not `backend.get_prop_pose()`.

| Example | Command |
|---------|---------|
| Kuka + MuJoCo (sensor-first) | `python -m examples.cli run-episode --preset kuka_pick_mujoco` |
| Franka + MuJoCo | `python -m examples.cli run-episode --preset franka_pick_mujoco` |
| Kuka + sensors (alias) | `python -m examples.cli run-episode --preset kuka_sensor_pick_mujoco` |
| Legacy thin wrappers | `python -m examples.kuka_pick_place_mujoco.run_mujoco` |
| Kuka + ROS2 RViz sensors | `python -m examples.kuka_sensor_ros2_rviz.run_ros2_rviz` |
| Kuka + Gazebo sensors | `python -m examples.kuka_sensor_gazebo.run_gazebo` |
| **Kuka multimodal pick (Gazebo)** | `python -m examples.kuka_ft_imu_pick_gazebo.run_gazebo` |
| Sensor diagnostics | `python -m examples.sensor_diagnostics_demo.run` |
| Sensor showcase (PNG + JSON) | `python -m examples.sensor_showcase.run` |
| MuJoCo Universe | `python -m examples.mujoco_universe.run` |
| Dummy smoke | `python -m examples.dummy_pick_place.run` |

Gazebo multimodal pick requires Linux: ROS 2 Jazzy, Gazebo Harmonic, `ros_gz_bridge`, `gz_ros2_control`. Optional: `pip install -e ".[kinematics]"` for Pinocchio reach IK.

---

## MuJoCo Universe

One CLI for robot × task × policy × sensor rig combinations. See [catalog/README.md](catalog/README.md).

```bash
python -m examples.mujoco_universe.run --list
python -m examples.mujoco_universe.run --preset mujoco_showcase_kuka
python -m examples.mujoco_universe.run --robot kuka --task showcase_scene --policy example_joint_track --rig full --steps 300
```

Catalog YAML: [`catalog/mujoco_catalog.yaml`](catalog/mujoco_catalog.yaml). `build_config()` delegates to `presets.yaml` when a combo exists.

---

## Canonical “three stacks” entrypoints

Kuka sinusoid demos use [`backend_for_simulator`](../robodeploy/backends/simulator.py): same `Robot` list, only the simulator string changes.

| Stack | Entry |
|-------|-------|
| Switchable (`BACKEND` constant) | [user_kuka_sinusoid/run_switch_simulator.py](user_kuka_sinusoid/run_switch_simulator.py) |
| MuJoCo | [user_kuka_sinusoid/run_mujoco.py](user_kuka_sinusoid/run_mujoco.py) |
| ROS2 + RViz | [user_kuka_sinusoid/run_ros2_rviz.py](user_kuka_sinusoid/run_ros2_rviz.py) (`--fake-sim` for devtool) |
| Gazebo | [user_kuka_sinusoid/run_gazebo.py](user_kuka_sinusoid/run_gazebo.py) |
| Isaac Sim | [user_kuka_sinusoid/run_isaacsim.py](user_kuka_sinusoid/run_isaacsim.py) |
| Minimal ROS2 RViz | [ros2_rviz_minimal.py](ros2_rviz_minimal.py) |

---

## Training scripts

| Script | Purpose |
|--------|---------|
| [train_ppo_reach.py](train_ppo_reach.py) | 500k PPO on `manipulation_v1/reach_target` |
| [train_ppo_kuka_pick.py](train_ppo_kuka_pick.py) | PPO on `kuka_pick_mujoco` preset |

See [docs/TRAINING.md](../docs/TRAINING.md).

---

## Multi-robot and hardware examples

| Directory | Description |
|-----------|-------------|
| [multirobot/two_franka_pick_place_mujoco/](multirobot/two_franka_pick_place_mujoco/) | Two Franka arms, MuJoCo |
| [multirobot/franka_kuka_collaborative_mujoco/](multirobot/franka_kuka_collaborative_mujoco/) | Heterogeneous dual-arm |
| [multirobot/three_arm_assembly_mujoco/](multirobot/three_arm_assembly_mujoco/) | Three-arm scene |
| [multirobot/two_so101_real/](multirobot/two_so101_real/) | SO-101 real hardware (hardware markers) |
| [kuka_ft_imu_pick_real/](kuka_ft_imu_pick_real/) | Multimodal real ROS2 demo |
| [plugin_robot_demo/](plugin_robot_demo/) | Entry-point plugin pattern |

---

## Additional examples

- **URDF + MJCF override:** [user_urdf_asset_override/](user_urdf_asset_override/)
- **CLI smoke (no sim):** [cli_smoke_no_sim.py](cli_smoke_no_sim.py)
- **Reference / older APIs:** [franka_sim_viewer_demo.py](franka_sim_viewer_demo.py), [franka_robomimic_demo.py](franka_robomimic_demo.py), [kuka_pick_demo.py](kuka_pick_demo.py)
- **Structure only (not runnable):** [multiagent_configs.py](multiagent_configs.py)

---

## Scaffold new examples

```bash
robodeploy scaffold example my_demo --preset my_preset
robodeploy scaffold task my_task --output my_project/tasks/
robodeploy scaffold policy my_policy --output my_project/policies/
```

Then add a preset snippet under `config/presets.yaml` and register `custom_modules`.
