# Kuka FT + IMU Pick (MuJoCo)

Demonstrates a fully sensor-driven pick-place pipeline without backend physics queries.

## Sensors

| Sensor | Role |
|---|---|
| `wrist_ft` | Grasp confirmation via force threshold (`grasp_detection: ft`) |
| `wrist_imu` | Phase settle gate (`imu_omega_max` in policy config) |
| `wrist_contact` | Binary touch exposed as `obs.contact_state` |
| `prop_pose` | Oracle object poses for task success distance checks |

## Policy

`example_reach_pick` (`ReachTrajectoryPolicy`) with:

- `grasp_detection: ft` — advances close-gripper phase on averaged FT force
- `force_threshold` / `grasp_force_window` — tunable grasp band
- `imu_omega_max` / `imu_settle_steps` — waits for IMU stillness before phase advance

## Task

`pick_place` with `grasp_success_force_min` — success requires FT grasp confirmation, not `backend.has_prop_contact()`.

## Run

```bash
pip install -e ".[sim]"
python examples/kuka_ft_imu_pick_mujoco/run_mujoco.py
```

## Sensor health

Each step surfaces `info.extra["sensor_status"]` and `info.extra["sensor_health"]` for monitoring stale or failed sensors.
