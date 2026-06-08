# Examples index

For environment setup (MuJoCo / ROS2+RViz / Isaac Sim), see [docs/BACKEND_SETUP.md](../docs/BACKEND_SETUP.md).

## Presets and example policies

Named YAML presets and demo policies live here, **not** in the `robodeploy` package:

- [config/presets.yaml](config/presets.yaml) — `kuka_pick_mujoco` (sensor-first default), `franka_pick_mujoco`, `mujoco_showcase_kuka`, `mujoco_showcase_franka`, `mujoco_pick_kuka`, `kuka_sensor_pick_mujoco`, `kuka_sensor_ros2_rviz`, `kuka_sensor_gazebo`, `kuka_sinusoid_mujoco`
- [catalog/](catalog/) — MuJoCo Universe catalog YAML + [README](catalog/README.md)
- [tasks/](tasks/) — `pick_place`, `pour`, `peg_insertion` (import `examples.tasks` to register)
- [policies/](policies/) — `example_reach_pick`, `example_sensor_reach_pick`, `example_joint_track` (import `examples.policies` to register)
- [sensors/](sensors/) — `sim_prop_pose` oracle prop reader (import `examples.sensors` to register)
- [env_from_preset.py](env_from_preset.py) — `env_from_preset("kuka_pick_mujoco")`
- [vecenv_from_presets.py](vecenv_from_presets.py) — batched envs from preset names

CLI (from repo root, after `pip install -e .`):

```bash
python -m examples.cli list-presets
python -m examples.cli run-episode --preset kuka_pick_mujoco --dummy --steps 5 --action sinusoid
python -m examples.mujoco_universe.run --list
python -m examples.mujoco_universe.run --preset mujoco_showcase_kuka
```

Library CLI (`robodeploy list-registry`, `robodeploy run-episode --dummy`) does not bundle example presets.

## Runnable pick-and-place demos (MuJoCo)

Pick-place demos use **`example_sensor_reach_pick`** and `SimPropPoseSensor` by default (`kuka_pick_mujoco` preset). Object poses come from `Observation.objects`, not `backend.get_prop_pose()`.

**Preferred entry:** `python -m examples.cli run-episode --preset kuka_pick_mujoco` or MuJoCo Universe (below). Per-robot `run_mujoco.py` scripts are thin legacy wrappers.

| Example | Command |
|---------|---------|
| Kuka + MuJoCo (sensor-first) | `python -m examples.cli run-episode --preset kuka_pick_mujoco` |
| Franka (MJCF names) + MuJoCo | `python -m examples.cli run-episode --preset franka_pick_mujoco` |
| Kuka + sensors + MuJoCo (alias preset) | `python -m examples.cli run-episode --preset kuka_sensor_pick_mujoco` |
| Legacy thin wrappers | `python -m examples.kuka_pick_place_mujoco.run_mujoco` (etc.) |
| Kuka + ROS2 RViz sensors | `python -m examples.kuka_sensor_ros2_rviz.run_ros2_rviz` |
| Kuka + Gazebo sensors | `python -m examples.kuka_sensor_gazebo.run_gazebo` |
| Sensor diagnostics smoke | `python -m examples.sensor_diagnostics_demo.run` |
| **Multi-sensor showcase** (PNG + JSON) | `python -m examples.sensor_showcase.run` |
| **MuJoCo Universe** (catalog + all geoms/sensors) | `python -m examples.mujoco_universe.run` |
| Simulator-free smoke | `python -m examples.dummy_pick_place.run` |

## MuJoCo Universe

One CLI for every robot × task × policy × sensor rig combination. See [catalog/README.md](catalog/README.md).

```bash
python -m examples.mujoco_universe.run --list
python -m examples.mujoco_universe.run --preset mujoco_showcase_kuka
python -m examples.mujoco_universe.run --robot kuka --task showcase_scene --policy example_joint_track --rig full --steps 300
```

| Preset | Robot | Task | Policy | Sensor rig |
|--------|-------|------|--------|------------|
| `mujoco_showcase_kuka` | kuka | showcase_scene | example_joint_track | full (6 kinds) |
| `mujoco_showcase_franka` | example_franka_mujoco | showcase_scene | example_joint_track | full |
| `mujoco_pick_kuka` | kuka | pick_place | example_sensor_reach_pick | vision |

Requires `pip install -e ".[sim]"` for MuJoCo examples. MuJoCo IK binds automatically via `PolicyBase.bind_runtime()` on first `env.reset()` (see CONTRACTS.md).

Preset-based construction (loads custom modules from YAML):

```python
from examples.env_from_preset import env_from_preset
env = env_from_preset("kuka_pick_mujoco")
```

## Canonical “three stacks” entrypoints

Kuka sinusoid demos use [`backend_for_simulator`](../robodeploy/backends/simulator.py): same `Robot` list, only the simulator string changes between MuJoCo, ROS2+RViz, and Gazebo.

- **One file, edit `BACKEND` only**: [user_kuka_sinusoid/run_switch_simulator.py](user_kuka_sinusoid/run_switch_simulator.py) (`python -m examples.user_kuka_sinusoid.run_switch_simulator`)
- **MuJoCo**: [user_kuka_sinusoid/run_mujoco.py](user_kuka_sinusoid/run_mujoco.py)
- **ROS2 + RViz**: [ros2_rviz_minimal.py](ros2_rviz_minimal.py)
- **Isaac Sim** (secondary on Windows): [user_kuka_sinusoid/run_isaacsim.py](user_kuka_sinusoid/run_isaacsim.py)
- **ROS2 + RViz (Kuka sinusoid)**: [user_kuka_sinusoid/run_ros2_rviz.py](user_kuka_sinusoid/run_ros2_rviz.py) (optional `--fake-sim` for embedded joint-position devtool)
- **Gazebo via ROS2GazeboBackend (Kuka sinusoid)**: [user_kuka_sinusoid/run_gazebo.py](user_kuka_sinusoid/run_gazebo.py)

## Additional examples

- **URDF + MJCF override / defaults**:
  - [user_urdf_asset_override/run_mujoco_default.py](user_urdf_asset_override/run_mujoco_default.py)
  - [user_urdf_asset_override/run_mujoco_override_mjcf.py](user_urdf_asset_override/run_mujoco_override_mjcf.py)
- **CLI smoke (no sim)**: [cli_smoke_no_sim.py](cli_smoke_no_sim.py)
- **Other demos** (may target older APIs; treat as reference):
  - [franka_sim_viewer_demo.py](franka_sim_viewer_demo.py)
  - [franka_robomimic_demo.py](franka_robomimic_demo.py)
  - [kuka_pick_demo.py](kuka_pick_demo.py)
  - [multiagent_configs.py](multiagent_configs.py) is structure-only. It sketches N:M:K wiring and is not a runnable smoke test.
