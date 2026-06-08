# Example preset fragments

Reusable YAML anchors for `examples/config/presets.yaml`:

| File | Anchors | Purpose |
|------|---------|---------|
| `base_sim.yaml` | `base_sim`, `base_kuka`, `base_franka` | MuJoCo defaults |
| `base_real.yaml` | `base_real`, `base_kuka_real` | ROS2 / real-hardware defaults |
| `manipulate.yaml` | `manipulate_pick`, `arm_sensors` | Pick-place task + sensor rig |

`presets.yaml` merges these via top-level `include:` and YAML anchors (`<<: *base_kuka_mujoco`).
