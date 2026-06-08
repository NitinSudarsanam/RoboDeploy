# Scene Definition Guide

Scenes describe static world content: manipulable props, tables, lighting, terrain, and cameras. RoboDeploy uses `SceneSpec` internally; you can author scenes in Python (`SceneBuilder`) or YAML (CLI tools).

## SceneBuilder API

```python
from robodeploy.scene_builder import SceneBuilder

scene = (
    SceneBuilder()
    .add_table(height=0.4, size=(1.2, 0.8, 0.03))
    .add_box("block", size=(0.03, 0.03, 0.03), pos=(0.55, 0.0, 0.41), mass=0.05)
    .add_mesh("mug", asset="mug.obj", pos=(0.55, 0.0, 0.41), mass=0.08)
    .add_target("tray", pos=(0.65, 0.2, 0.41))
    .set_lighting("bright")
    .set_cameras("overview")
    .set_terrain("procedural", size=(4.0, 4.0), procedural_params={"generator": "ridge", "seed": 1})
    .validate(backend="mujoco")
    .build_spec()
)
```

Methods chain and return `self`. Call `build_spec()` to produce a `SceneSpec` for `TaskBase.scene_spec()`.

## YAML scene format

For standalone validation and inspection, use a YAML file:

```yaml
table_height: 0.0
lighting: default
props:
  - name: source
    position: [0.55, 0.0, 0.38]
    mass: 0.05
    is_fixed: false
    geom:
      kind: box
      size: [0.025, 0.025, 0.025]
  - name: target
    position: [0.60, 0.20, 0.38]
    is_fixed: true
    geom:
      kind: box
      size: [0.04, 0.04, 0.003]
```

CLI:

```bash
robodeploy scene validate scene.yaml --backend mujoco
robodeploy scene inspect scene.yaml --backend mujoco --json
```

Non-zero exit code on validation errors (duplicate names, missing geometry, missing mesh files).

## Cross-backend validation

`SceneValidator` checks:

- Unique prop names
- Geometry kind supported per backend (e.g. capsule warnings on Gazebo)
- Mesh asset paths exist on disk
- Positive mass for non-fixed bodies
- Friction sanity warnings

Pass `--backend mujoco|gazebo|isaacsim` to tailor warnings.

## Asset resolution

Resolve robot and mesh paths for a target backend:

```bash
robodeploy assets list --robot
robodeploy assets resolve kuka --backend mujoco
robodeploy assets info franka
```

Built-in robots live under `robodeploy/description/<name>/assets/<format>/`. The catalog also indexes `examples/catalog/mujoco_catalog.yaml`.

## Lighting and cameras

Lighting presets: `minimal`, `bright`, `dark`, `randomized` via `SceneBuilder.set_lighting()` or `SceneSpec.lighting`.

Camera presets: `overview`, `tabletop`, `overhead`, `wrist` via `SceneBuilder.set_cameras()`. Presets resolve to `CameraSpec` entries in `WorldSpec.cameras` and emit on MuJoCo, Gazebo, and IsaacSim builders.

## Physics groups and collision masks

`PropConfig` exposes `collision_layer` (bit index) and `collision_mask` (which layers collide). SceneBuilder `.add_box(..., layer=1, mask=0xFF)` maps to unified IR and MuJoCo `contype`/`conaffinity`. Optional `friction_dist: [min, max]` supports domain-randomization sampling at the spec level.

## Procedural terrain

Generators: `perlin`, `ridge`, `stairs`. Use procedural terrain on any backend:

```python
.set_terrain("procedural", size=(4, 4), procedural_params={"generator": "stairs", "num_steps": 8})
```

Backends resolve procedural specs to heightfield PNGs shared across MuJoCo, Gazebo, and IsaacSim.

## Common scene layouts

| Layout | Props | Notes |
|--------|-------|-------|
| Table-top pick | `source`, `target`, optional `table` | Default pick_place layout |
| Peg insertion | `peg`, `hole` | Cylinder + box fixture |
| Showcase | all geom kinds | Geometry compatibility testing |

## Physics notes

- Fixed props: `is_fixed: true`, mass ignored by most backends
- Manipulable objects: set `mass` and procedural `geom` or `asset_path`
- Domain randomization modifies poses at reset — declare props in `scene_spec()` first

## Pitfalls

- **Three geom styles:** prefer `geom=GeomSpec(...)` over legacy `asset_path` + `objects` alias
- **Missing backend variant:** Gazebo needs URDF paths; MuJoCo prefers MJCF
- **validate() skipped:** call `.validate(backend=...)` in SceneBuilder before long runs

## Next steps

- [TASK_CREATION.md](TASK_CREATION.md) — attach scene to a task
- [BACKEND_SETUP.md](BACKEND_SETUP.md) — install simulators
- `robodeploy lint task` / `robodeploy scene validate` — catch errors early
