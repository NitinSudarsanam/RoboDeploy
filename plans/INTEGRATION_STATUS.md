# Integration status (honest)

What presets claim vs what CI actually exercises. Updated with Plan 2 (integration layer + CI honesty).

| Preset / benchmark | Runnable command | CI job | Status |
|--------------------|------------------|--------|--------|
| `kuka_pick_mujoco` | `python -m examples.cli run-episode --preset kuka_pick_mujoco --steps 50` | `sensor-e2e-linux` | **Smoke** — builds env, steps MuJoCo |
| `my_kitchen_pick_mujoco` | `python -m examples.cli run-episode --preset my_kitchen_pick_mujoco` | — | **Doc only** — requires tutorial task file |
| `kuka_ft_imu_pick_gazebo` | `python -m examples.kuka_ft_imu_pick_gazebo.run_gazebo` | `sensor-live-gazebo` (sensors only) | **Blocked on Plan 1** — pick-place success not claimed in CI |
| `kuka_sensor_gazebo` | `python -m examples.kuka_sensor_gazebo.run_gazebo` | `sensor-live-gazebo` | **Sensor smoke** — not pick-place |
| `manipulation_v1/*/preset_dummy` | `robodeploy eval --benchmark manipulation_v1/reach_target --backend dummy` | `unittest`, `benchmark-nightly` | **Pass** — eval harness |
| `manipulation_v1/*/preset_mujoco` | `robodeploy eval --benchmark manipulation_v1/reach_target --backend mujoco` | `eval-mujoco-smoke`, `test_benchmark_preset_builds_env` | **Build + short eval** — not full suite nightly |
| `manipulation_v1/*/preset_gazebo` | `get_backend("gazebo")` + preset load | `test_gazebo_presets_load` | **YAML + alias** — full Gazebo env build waits Plan 1 |
| `stacking_3blocks` / `cloth_fold` sim | showcase_scene + joint_track placeholder | `test_benchmark_preset_builds_env` | **Tier placeholder** — not real fold/stack tasks |
| `two_franka_pick_mujoco` | `python -m examples.cli run-episode --preset two_franka_pick_mujoco` | — | **Not E2E tested** |
| `gym.make("robodeploy/kuka_pick_mujoco-v0")` | — | — | **Not tested** |
| PyPI `pip install robodeploy` | — | — | **Not published** |
| Nightly `manipulation_v1` Gazebo eval | — | — | **Not run** — dummy-only nightly remains honest |

## Backend naming

| Context | Name |
|---------|------|
| Presets / benchmarks / `backend_for_simulator("gazebo")` | `gazebo` |
| Registered class | `ros2_gazebo` |
| Resolution | `get_backend("gazebo")` → `ROS2GazeboBackend` |
