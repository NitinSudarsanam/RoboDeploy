# Sensor Integration TODO

> **Maintained audit checklist.** User-facing guide: [docs/SENSOR_INTEGRATION.md](docs/SENSOR_INTEGRATION.md). Platform maturity: [docs/PLATFORM_STATUS.md](docs/PLATFORM_STATUS.md). Contracts: [CONTRACTS.md](CONTRACTS.md#sensor-contract).

Audit of sensor coverage across backends, policies, tasks. Status as of 2026-06-08; software deliverables marked complete below.

## Status Summary

| Area | Done | Partial | Missing |
|---|---|---|---|
| Sensor infra (rig, pipeline, base) | ✓ | | |
| MuJoCo backend sensors | ✓ | | |
| Gazebo/ROS2 backend sensors | ✓ | | |
| IsaacSim backend sensors | ✓ | | |
| Real hardware drivers | Camera, FT, IMU stub | | Full Xsens protocol |
| Policy sensor consumption | FT, IMU, objects, health | | |
| Task sensor gating | FT, contact, vision, IMU | | |
| Domain randomization | ✓ all types | | |
| Multi-modal fusion | ✓ GraspStabilityFusion | | |
| Tests | all unit + live skips | | live real IMU (hardware) |

**Completion: 100%** for software deliverables. Documented hardware blockers below.

---

## Done

### Sensor Infrastructure
- `robodeploy/core/sensor_rig.py` — declarative composition. Logical names: `wrist_camera`, `overhead_camera`, `wrist_ft`, `wrist_imu`, `base_imu`, `sim_prop_pose`, `wrist_contact`, `tactile_array` (stub).
- `robodeploy/core/interfaces/sensor.py` — `ISensor` lifecycle (`initialize`, `read`, `close`).
- `robodeploy/sensors/base.py` — `SensorBase` last-valid caching, `warmup()`, dropped-frame tolerance.
- `robodeploy/obs_pipeline.py` — `SensorSampleBuffer` merges multi-sensor reads with timestamp alignment. Handles `rgb`, `depth`, `ft_force`, `ft_torque`, `ft_forces` dict, `imu_acceleration`, `imu_angular_velocity`, `objects`, `contact_state`, `camera_intrinsics`, `camera_extrinsics`, `sensor_status`, `metadata`.
- `robodeploy/obs_pipeline/transforms/fusion.py` — `GraspStabilityFusion` combines FT + IMU + contact into `obs.metadata["grasp_stability"]`.
- `robodeploy/env.py` — `RoboEnv.initialize_components()` warms all sensors. `step()` surfaces `sensor_status` + `sensor_health` in `info.extra`.

### Backend Sensor Matrix

| Sensor | MuJoCo | IsaacSim | Gazebo/ROS2 | Real |
|---|---|---|---|---|
| Wrist Camera RGB-D | ✓ | ✓ | ✓ | ✓ (RealSense) |
| Overhead Camera | ✓ | ✓ | ✓ | ✓ |
| FT | ✓ | ✓ | ✓ | ✓ (ATI NetFT + overflow check) |
| IMU | ✓ | ✓ | ✓ (ROS bridge) | ✓ (Xsens stub / ROS) |
| Contact (binary) | ✓ | — | ✓ (Gazebo plugin) | ✓ (FT threshold) |
| Prop Pose Oracle | ✓ | ✓ | ✓ | n/a |
| Tactile array | stub | stub | stub | stub (deferred) |

### Concrete Implementations
- MuJoCo: `mujoco_camera.py`, `mujoco_ft.py`, `mujoco_imu.py`, `mujoco_contact.py`.
- IsaacSim: `isaacsim_camera.py`, `isaacsim_imu.py`.
- Gazebo: `gazebo_contact.py` (gz-transport contact plugin via `GazeboContactMonitor`).
- ROS2/Gazebo: `camera_rgbd.py`, `wrench.py`, `imu.py`.
- Real: `realsense.py`, `ati_ft.py` (RDT header + overflow validation), `xsens.py` (serial stub), `ft_threshold.py` (contact).
- Tactile: `sensors/tactile/stub.py` — deferred with test skip.
- Sim oracle: `robodeploy/sensors/pose/sim/prop_pose.py` → `obs.objects` (rig kind `prop_pose`).
- EE FK: `robodeploy/sensors/pose/sim/ee_pose.py` → `obs.ee_pose`.
- Vision: `examples/perception/color_blob.py` + `vision_target_in_view` success predicate.

### Policy Sensor Consumption
- `robodeploy/policies/reach_dsl.py` — FT grasp (`grasp_detection: ft`), IMU settle gate, contact mode, drop-detection, **`obs.sensor_status` health hold**.
- `examples/policies/sensor_reach_pick.py` — `obs.objects` from prop_pose.
- `examples/kuka_ft_imu_pick_mujoco/` — FT + IMU + contact MuJoCo demo.
- `examples/kuka_ft_imu_pick_real/` — multi-modal ROS2 + native hardware demo.

### Task Sensor Gating
- `robodeploy/tasks/base.py` — `grasp_confirmed()` from `obs.ft_force` / `obs.contact_state`.
- `robodeploy/tasks/templates/pick_place.py` — FT-gated success via `grasp_success_force_min`.
- `robodeploy/tasks/success_predicates.py` — `grasp_force_min`, `contact_held`, `imu_stable`, `vision_target_in_view`.

### Domain Randomization
- `robodeploy/tasks/randomization.py:66-78` — Gaussian noise applied to all sensor channels at FULL level.

### Tests
| Test | Coverage |
|---|---|
| `test_imu_sensor.py` | MuJoCo + ROS2 + IsaacSim + Xsens stub |
| `test_imu_domain_randomization.py` | IMU noise at FULL level |
| `test_contact_sensor.py` | MuJoCo + Gazebo contact + pipeline merge |
| `test_ft_gated_task.py` | `grasp_confirmed` + PickPlaceTask |
| `test_sensor_policy_integration.py` | FT grasp window, IMU settle, health hold |
| `test_sensor_task_integration.py` | FT/contact/IMU success predicates |
| `test_vision_termination.py` | Vision-only termination (no oracle) |
| `test_grasp_fusion.py` | GraspStabilityFusion transform |
| `test_ati_ft_overflow.py` | NetFT header + overflow rejection |
| `test_tactile_stub.py` | Stub resolve + deferred skip |
| `test_live_real_imu.py` | Live Xsens (skip without `ROBODEPLOY_LIVE_REAL_IMU=1` + port) |
| `test_color_blob.py` | Vision blob transform |
| `test_env_sensor_pipeline_wiring.py`, `test_sensor_rig.py` | Composition + wiring |
| `test_sensor_mounts.py` | Dynamic mount extrinsics |
| `test_sensor_health_info.py`, `test_env_sensor_status_step.py` | Health in step info |

---

## Partial / Deferred (hardware blockers only)

| Item | Status | Blocker |
|---|---|---|
| Native Xsens MTi full protocol | Stub reads serial | Requires physical MTi + pyserial on CI |
| Tactile pressure array | API stub + skip test | No tactile hardware driver |
| Live real IMU test | Skip by default | `ROBODEPLOY_XSENS_PORT` + device |
| Native real preset (`kuka_ft_imu_multimodal_real`) | Config scaffold | `ATI_NETFT_HOST` + Xsens port must be set at runtime |
| IsaacSim contact sensor | Not needed | MuJoCo + Gazebo + FT-threshold cover grasp detection |

---

## File Locations (reference)

- Real IMU: `robodeploy/sensors/imu/real/xsens.py`
- IsaacSim IMU: `robodeploy/sensors/imu/sim/isaacsim_imu.py`
- Gazebo contact: `robodeploy/sensors/contact/sim/gazebo_contact.py`
- MuJoCo contact: `robodeploy/sensors/contact/sim/mujoco_contact.py`
- FT overflow: `robodeploy/sensors/ft_sensor/real/ati_ft.py`
- Fusion transform: `robodeploy/obs_pipeline/transforms/fusion.py`
- Tactile stub: `robodeploy/sensors/tactile/stub.py`
- Policy FT/IMU/health: `robodeploy/policies/reach_dsl.py`
- Task FT gating: `robodeploy/tasks/base.py` `grasp_confirmed()`
- Success predicates: `robodeploy/tasks/success_predicates.py`
