# Examples index

For environment setup (MuJoCo / ROS2+RViz / Isaac Sim), see [docs/BACKEND_SETUP.md](../docs/BACKEND_SETUP.md).

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
- **Other demos** (may target older APIs; treat as reference):
  - [franka_sim_viewer_demo.py](franka_sim_viewer_demo.py)
  - [franka_robomimic_demo.py](franka_robomimic_demo.py)
  - [kuka_pick_demo.py](kuka_pick_demo.py)
  - [multiagent_configs.py](multiagent_configs.py) is structure-only. It sketches N:M:K wiring and is not a runnable smoke test.
