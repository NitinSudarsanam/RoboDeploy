# Examples index

For environment setup (MuJoCo / ROS2+RViz / Isaac Sim), see [docs/BACKEND_SETUP.md](../docs/BACKEND_SETUP.md).

## Presets and example policies

Named YAML presets and demo policies live here, **not** in the `robodeploy` package:

- [config/presets.yaml](config/presets.yaml) — `kuka_pick_mujoco`, `franka_pick_mujoco`, `kuka_sensor_pick_mujoco`, `kuka_sinusoid_mujoco`
- [tasks/](tasks/) — `pick_place`, `pour`, `peg_insertion` (import `examples.tasks` to register)
- [policies/](policies/) — `example_reach_pick`, `example_sensor_reach_pick`, `example_joint_track` (import `examples.policies` to register)
- [sensors/](sensors/) — `sim_prop_pose` oracle prop reader (import `examples.sensors` to register)
- [env_from_preset.py](env_from_preset.py) — `env_from_preset("kuka_pick_mujoco")`
- [vecenv_from_presets.py](vecenv_from_presets.py) — batched envs from preset names

CLI (from repo root, after `pip install -e .`):

```bash
robodeploy list-presets
robodeploy run-episode --preset kuka_pick_mujoco --dummy --steps 5 --action sinusoid
```

## Runnable pick-and-place demos (MuJoCo)

These wire `examples.tasks.PickPlaceTask` + `ReachPickPlacePolicy` end-to-end (no stubs):

| Example | Command |
|---------|---------|
| Kuka + MuJoCo | `python -m examples.kuka_pick_place_mujoco.run_mujoco` |
| Franka (MJCF names) + MuJoCo | `python -m examples.franka_pick_place_mujoco.run_mujoco` |
| Kuka + sensors + MuJoCo | `python -m examples.kuka_sensor_pick_mujoco.run_mujoco` |
| Simulator-free smoke | `python -m examples.dummy_pick_place.run` |

Requires `pip install -e ".[sim]"` for MuJoCo examples. The reach policy uses MuJoCo Jacobian IK (call `policy.attach_mujoco(backend, description)` after `env.reset()`). It completes pick-place in sim via kinematic carry of the `source` prop.

The sensor-driven variant (`example_sensor_reach_pick`) reads object poses from `Observation.objects` (via `SimPropPoseSensor` / `SensorRig`) instead of calling `backend.get_prop_pose()` for perception.

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
