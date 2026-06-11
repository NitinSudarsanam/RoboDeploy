# Sensor Integration Guide

RoboDeploy treats sensors as first-class components: you declare a **sensor rig** in your preset, the runtime resolves concrete implementations per backend, and observations flow through a shared pipeline into tasks and policies.

## Quick start

Add a manipulation rig to your preset YAML:

```yaml
sensor_rigs:
  - rig_id: arm_sensors
    wrist_rgbd:
      width: 128
      height: 96
    wrist_ft: {}
    wrist_imu: {}
    prop_pose:
      prop_names: [source, target]
```

Run with sensors wired:

```bash
python -m examples.cli run-episode --preset kuka_sensor_pick_mujoco --steps 50
robodeploy config resolve --preset kuka_sensor_pick_mujoco --json
```

Scaffold a custom MuJoCo sensor:

```bash
robodeploy scaffold sensor --name pressure --backend mujoco \
  --output robodeploy/sensors/pressure/sim/mujoco_pressure.py
```

## Sensor rig declaration

`SensorRig` bundles logical sensors attached to one robot. Presets use shorthand keys that map to `SensorSpec` entries:

| Shorthand key | Kind | Observation fields |
|---------------|------|-------------------|
| `wrist_rgbd` | `wrist_camera` | `obs.rgb`, `obs.depth`, intrinsics/extrinsics |
| `overhead_rgbd` | `overhead_camera` | same as wrist |
| `wrist_ft` | `wrist_ft` | `obs.ft_force`, `obs.ft_torque` |
| `wrist_imu` | `wrist_imu` | `obs.imu_acceleration`, `obs.imu_angular_velocity` |
| `base_imu` | `base_imu` | same IMU fields |
| `wrist_contact` | `wrist_contact` | `obs.contact_state` |
| `prop_pose` | `sim_prop_pose` | `obs.objects` (oracle poses) |

Build rigs in Python with `SensorRig.robot_mounted()`:

```python
from robodeploy.core.sensor_rig import SensorRig

rig = SensorRig.robot_mounted(
    rig_id="arm_sensors",
    wrist_ft={"noise_std": 0.5},
    wrist_imu={},
    prop_pose={"prop_names": ["source", "target"]},
    ee_link="robot0/ee_link",
)
```

At env build time, `RoboEnv` resolves each spec to a concrete `ISensor` via `resolve_sensor_class(kind, backend)`.

## Built-in sensors

### Camera (RGB-D)

- **MuJoCo**: `robodeploy/sensors/camera/sim/mujoco_camera.py`
- **IsaacSim**: `isaacsim_camera.py`
- **Gazebo/ROS2**: `backends/real/ros2/sensors/camera_rgbd.py`
- **Real**: `realsense.py` (Intel RealSense)

Mount on the EE link with `SensorMount(parent_link=..., position=...)`. Images land in `obs.rgb` / `obs.depth` keyed by sensor name.

### Force-torque (FT)

- **MuJoCo**: `mujoco_ft.py` — reads simulated wrench at wrist.
- **IsaacSim / Gazebo / ROS2**: backend-specific wrench bridges.
- **Real**: `ati_ft.py` — ATI NetFT with overflow validation.

Policies use FT for grasp confirmation (`grasp_detection: ft` in reach DSL). Tasks call `self.grasp_confirmed(obs)` which checks `obs.ft_force` magnitude.

### IMU

- **MuJoCo**: `mujoco_imu.py`
- **ROS2**: `imu.py` bridge
- **Real**: `xsens.py` (serial stub; full protocol deferred)

Reach policies can gate phases on `imu_stable` predicates. Domain randomization adds Gaussian noise to IMU channels at `FULL` DR level.

### Contact (binary)

- **MuJoCo**: `mujoco_contact.py` — prop contact or EE proximity.
- **Gazebo**: `gazebo_contact.py` via gz-transport plugin.
- **Real**: `ft_threshold.py` — FT magnitude threshold.

Exposed as `obs.contact_state[name] -> bool`. Used by `contact_held` success predicates and grasp fusion.

### Prop pose (sim oracle)

`robodeploy/sensors/pose/sim/prop_pose.py` publishes ground-truth object poses into `obs.objects` (rig kind `prop_pose` → `sim_prop_pose`). Use for sim development; replace with vision or `perception_source` for sim2real transfer.

### EE pose (sim FK)

`robodeploy/sensors/pose/sim/ee_pose.py` publishes `obs.ee_pose` from joint encoders + FK (MuJoCo, Pinocchio, or ROS TF). Configure `prefer_fk_ee_pose` / `robot0.base_frame: world` on ROS backends for world-frame parity with scene props.

### Tactile (stub)

`sensors/tactile/stub.py` reserves the API. Deferred until a reference hardware driver exists.

## Per-backend resolution

Sensor classes register with `@register_sensor("kind_sim")` and pair to backends via `register_sensor_pair`. The resolver picks the implementation matching the active backend:

```python
from robodeploy.core.registry import resolve_sensor_class

cls = resolve_sensor_class("wrist_ft", backend="mujoco")
```

Override implementation in preset config:

```yaml
sensor_rigs:
  - rig_id: arm_sensors
    wrist_ft:
      impl: ati_ft_real  # force a specific registered sensor
```

## Custom sensor authoring

1. Subclass `SensorBase` (or `ContactSensorBase` for binary contact).
2. Implement `_init_impl(backend)`, `_read_impl() -> SensorData`, `_close_impl()`.
3. Register with `@register_sensor("my_sensor_sim")` and `register_sensor_pair(..., backend="mujoco")`.
4. Add the kind to your rig shorthand or reference by `impl` name in YAML.
5. Lint: `robodeploy lint all`

Minimal MuJoCo sensor skeleton:

```python
from robodeploy.sensors.base import SensorBase
from robodeploy.core.types import SensorData

class MySensor(SensorBase):
    def _init_impl(self, backend):
        self._backend = backend

    def _read_impl(self) -> SensorData:
        return SensorData(custom_field=1.0, timestamp=0.0)

    def _close_impl(self):
        self._backend = None
```

Map custom fields through `SensorSampleBuffer` if they should appear on `Observation`.

## Domain randomization noise

`robodeploy/tasks/randomization.py` applies Gaussian noise to sensor channels when DR level is `FULL`:

- FT force/torque
- IMU acceleration / angular velocity
- Camera RGB (optional pixel noise)

Configure per-task via `task.config["domain_randomization"]`.

## Consuming sensor observations

### In tasks

```python
def success_fn(self, obs) -> bool:
    if self.config.get("require_ft_grasp"):
        return self.grasp_confirmed(obs)
    pose = self.object_pose("source", obs)  # uses obs.objects or vision
    ...
```

Reuse predicates from `robodeploy/tasks/success_predicates.py`: `grasp_force_min`, `contact_held`, `imu_stable`, `vision_target_in_view`.

### In policies

Reach DSL YAML supports sensor-driven phases:

```yaml
phases:
  - name: grasp
    grasp_detection: ft
    ft_threshold: 8.0
  - name: settle
    imu_gate: true
```

Python policies read `obs.ft_force`, `obs.sensor_status`, and `obs.metadata["grasp_stability"]` (from `GraspStabilityFusion`).

### Health and dropped frames

`SensorBase` caches last-valid readings and tolerates warmup drops. `env.step()` surfaces `info.extra["sensor_status"]` and `sensor_health` when a sensor stalls.

## Multi-modal fusion

`robodeploy/obs_pipeline/transforms/fusion.py` provides `GraspStabilityFusion`, combining FT + IMU + contact into a single stability score. Enable in preset:

```yaml
obs_pipeline:
  transforms:
    - kind: grasp_stability_fusion
```

## Examples and presets

| Example | Sensors |
|---------|---------|
| `kuka_sensor_pick_mujoco` | wrist FT + prop_pose |
| `kuka_ft_imu_pick_mujoco` | FT + IMU + contact |
| `mujoco_showcase_kuka` | full rig (cameras, FT, IMU, prop_pose) |
| `kuka_sensor_ros2_rviz` | ROS2 camera + wrench |

See `examples/sensor_showcase/run.py` and `docs/tutorials/02_teleop.md` for teleop with sensor feedback.

## Troubleshooting

```bash
robodeploy doctor                    # check pyrealsense2, MuJoCo, ROS2
robodeploy config resolve --preset <name> --json   # verify sensor impl resolution
robodeploy lint preset presets.yaml --check <name>
```

Common issues:

- **Empty `obs.objects`**: prop_pose not in rig, or prop names mismatch scene.
- **FT always zero**: sensor not warmed up; check `sensor_status` in info.
- **Real camera missing**: install `pip install -e ".[real]"` and verify USB permissions.
