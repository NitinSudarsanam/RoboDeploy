# Migrating from RoboDeploy 0.1.x to 0.2.0

This guide covers breaking and recommended changes when upgrading from 0.1.x. Version 0.2.0 focuses on typed configs, scene builders, sensor rigs, and CLI tooling.

## Install

```bash
pip install --upgrade robodeploy
# or from source:
pip install -e ".[sim,dev]"
robodeploy doctor
```

## Summary of renames

| 0.1.x | 0.2.0 | Notes |
|-------|-------|-------|
| `PropConfig` lists in tasks | `SceneBuilder.add_*()` | Fluent scene API |
| `ReachPickPlacePolicy` (Python) | `ReachTrajectoryPolicy` + YAML DSL | Policy <50 lines |
| Raw `dict` env configs | `EnvConfig` / `RoboEnv.from_config()` | Typed + validated |
| `obs.objects` oracle only | Sensor rig + `prop_pose` / vision | Sim2real path |
| Manual preset copy-paste | `robodeploy scaffold preset` | Inheritance via YAML anchors |
| No lint/validate CLI | `robodeploy lint`, `config validate`, `scene validate` | CI-friendly |

## Scene definition: PropConfig → SceneBuilder

**Before (0.1.x):**

```python
from robodeploy.core.types import PropConfig, GeomSpec, SceneSpec

def scene_spec(self):
    return SceneSpec(props=[
        PropConfig(name="source", position=(0.55, 0, 0.41), geom=GeomSpec(kind="box", size=(0.03, 0.03, 0.03))),
    ])
```

**After (0.2.0):**

```python
from robodeploy.scene_builder import SceneBuilder

def scene_spec(self):
    return (
        SceneBuilder()
        .add_table(height=0.4)
        .add_box("source", pos=(0.55, 0, 0.41), mass=0.08)
        .add_target("target", pos=(0.65, 0.2, 0.41))
        .validate(backend="mujoco")
        .build_spec()
    )
```

Validate standalone YAML scenes:

```bash
robodeploy scene validate scene.yaml --backend mujoco
```

## Policy: ReachPickPlacePolicy → ReachTrajectoryPolicy YAML

**Before:** Subclass or copy `ReachPickPlacePolicy` (~260 lines) with embedded phase logic.

**After:** Keep a thin loader (`examples/policies/reach_pick_place.py`, ≤50 lines) and put phases in YAML:

```yaml
example_sensor_reach_pick:
  home: [0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0]
  phases:
    - name: pregrasp
      target_frame: source
      offset: [0.0, 0.0, 0.10]
```

Scaffold new policies:

```bash
robodeploy scaffold policy --name my_reach --template reach_dsl --output examples/policies/my_reach.yaml
```

FT grasp confirmation moves to YAML (`grasp_detection: ft`) instead of hard-coded Python.

## Config: dict → EnvConfig

**Before:**

```python
env = RoboEnv.make(robot="kuka", backend="mujoco", task="pick_place", policy="reach")
```

**After:**

```python
from robodeploy.env import RoboEnv

cfg = {
    "robot": "kuka",
    "backend": "mujoco",
    "task": "pick_place",
    "policy": "example_sensor_reach_pick",
    "sensor_rigs": [...],
}
env = RoboEnv.from_config(cfg)
```

Or use example presets:

```python
from examples.env_from_preset import env_from_preset
env = env_from_preset("kuka_pick_mujoco")
```

Inspect resolved config (sensor implementations, merged kwargs):

```bash
robodeploy config resolve --preset kuka_pick_mujoco --json
robodeploy config validate examples/config/presets.yaml
```

## Preset inheritance

0.2.0 presets use YAML anchors and includes under `examples/presets/`:

- `base_sim.yaml` — shared sim defaults
- `base_real.yaml` — real robot defaults
- `manipulate.yaml` — pick-place bundles

**Before:** Duplicate full preset blocks per backend.

**After:**

```yaml
include:
  - ../presets/base_sim.yaml
  - ../presets/manipulate.yaml

kuka_pick_mujoco:
  <<: *manipulate_pick
  backend: mujoco
```

Scaffold new presets:

```bash
robodeploy scaffold preset --name my_pick --robot kuka --template manipulate --output snippet.yaml
```

Internal anchor names (`_base_kuka`, `base_kuka_mujoco`) are not listable presets.

## Observations: oracle → sensor-driven

**Before:** Tasks read object poses from backend oracle helpers.

**After:** Declare a sensor rig with `prop_pose` for sim, or vision/perception for transfer:

```yaml
sensor_rigs:
  - rig_id: arm_sensors
    prop_pose:
      prop_names: [source, target]
```

In policies and tasks, use `self.object_pose(name, obs)` and `obs.ft_force` / `obs.contact_state` instead of calling `backend.get_prop_pose()` directly.

For deployment, remove `prop_pose` and add a camera + `vision_target_in_view` predicate.

## Multi-robot (new in 0.2.0)

Presets may use a `robots:` list instead of a single `robot` key:

```yaml
two_franka_pick_mujoco:
  backend: mujoco
  robots:
    - robot_id: left
      robot: franka
      task: pick_place
      policy: example_sensor_reach_pick
```

See `examples/multirobot/` and `CHANGELOG.md`.

## CLI commands (new)

| Command | Purpose |
|---------|---------|
| `robodeploy scaffold` | Generate task/policy/preset/robot/sensor/example |
| `robodeploy lint` | Static checks on tasks, policies, presets |
| `robodeploy scene` | Validate / inspect scene YAML |
| `robodeploy config` | Show, resolve, validate, diff presets |
| `robodeploy assets` | List / resolve robot and mesh assets |
| `robodeploy doctor` | Environment health check |

`robodeploy run-episode` still works; use `--preset` via `python -m examples.cli` for example presets.

## Plugin discovery (new)

Third-party packages register via entry points. Call `robodeploy list-registry --discover` or `robodeploy.discover()` at startup. See `docs/PLUGINS.md`.

## Deprecated patterns

- Calling `backend.has_prop_contact()` from tasks/policies — use `obs.contact_state` or `TaskBase.grasp_confirmed()`.
- Hard-coded sensor init inside policies — declare `sensor_rigs` in preset.
- Copying entire example files — use `robodeploy scaffold example`.

## Verification checklist

After migrating:

```bash
robodeploy doctor
robodeploy config validate examples/config/presets.yaml
robodeploy lint all
robodeploy run-episode --dummy --steps 5
python -m pytest tests/test_presets.py tests/test_cli.py -q
```

## Getting help

- Tutorials: `docs/tutorials/`
- Task guide: `docs/TASK_CREATION.md`
- Policy guide: `docs/POLICY_CREATION.md`
- Sensor guide: `docs/SENSOR_INTEGRATION.md`
- Cookbook: `docs/COOKBOOK.md`
