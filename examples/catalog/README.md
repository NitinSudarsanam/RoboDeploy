# MuJoCo Universe catalog

Single reference for robots, tasks, policies, sensor rigs, and geom kinds supported by the MuJoCo flagship demo.

- **Catalog YAML**: [mujoco_catalog.yaml](mujoco_catalog.yaml)
- **Loader**: [load.py](load.py) — `list_robots()`, `build_config()`, `get_combo()`
- **Runner**: `python -m examples.mujoco_universe.run`

## Sensor kind → YAML kwarg

| Sensor kind | Logical name | MuJoCo class | YAML kwarg |
|-------------|--------------|--------------|------------|
| Wrist RGB-D | `wrist_camera` | `MuJoCoRGBDCamera` | `wrist_rgbd: {width, height, depth, allow_camera_fallback}` |
| Overhead RGB-D | `overhead_camera` | `MuJoCoRGBDCamera` | `overhead_rgbd: {mount, width, height, ...}` |
| Wrist F/T | `wrist_ft` | `MuJoCoFTSensor` | `wrist_ft: {}` |
| Wrist IMU | `wrist_imu` | `MuJoCoIMUSensor` | `wrist_imu: {}` |
| Base IMU | `base_imu` | `MuJoCoIMUSensor` | `base_imu: {mount: ...}` |
| Prop poses | `prop_pose` | `SimPropPoseSensor` | `prop_pose: {prop_names: [...]}` |

## Geom kind → showcase prop

| Geom kind | Showcase prop | Task |
|-----------|---------------|------|
| `box` | `showcase_box` | `showcase_scene` |
| `cylinder` | `showcase_cylinder` | `showcase_scene` |
| `sphere` | `showcase_sphere` | `showcase_scene` |
| `capsule` | `showcase_capsule` | `showcase_scene` |
| `mesh` | (optional asset) | not in default showcase |

## Robot → EE link

| Robot | EE link | Asset |
|-------|---------|-------|
| `kuka` | `robot0/ee_link` | built-in MJCF |
| `franka` | `panda_hand` | built-in MJCF |
| `example_franka_mujoco` | `robot0/ee_link` | example MJCF naming |

## Policy → observation needs

| Policy | Needs | IK |
|--------|-------|-----|
| `example_joint_track` | proprio only | no |
| `example_sensor_reach_pick` | `obs.objects` (prop_pose sensor) | yes |
| `example_reach_pick` | backend prop poses | yes |

## Preset shortcuts

| Preset | Robot | Task | Policy | Rig |
|--------|-------|------|--------|-----|
| `mujoco_showcase_kuka` | kuka | showcase_scene | example_joint_track | full |
| `mujoco_showcase_franka` | example_franka_mujoco | showcase_scene | example_joint_track | full |
| `mujoco_pick_kuka` | kuka | pick_place | example_sensor_reach_pick | vision |

## Commands

```bash
python -m examples.mujoco_universe.run --list
python -m examples.mujoco_universe.run --preset mujoco_showcase_kuka
python -m examples.mujoco_universe.run --robot kuka --task showcase_scene --policy example_joint_track --rig full
python -m examples.mujoco_universe.run --viewer
```
