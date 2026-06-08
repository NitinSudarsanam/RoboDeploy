# Task Creation Guide

RoboDeploy tasks declare **what** the robot should accomplish: scene layout, required observations, rewards, and success conditions. This guide walks through creating a pick-and-place task in under 30 lines using `SceneBuilder` and `TaskBase`.

## Quick start

Scaffold a starter file:

```bash
robodeploy scaffold task --name kitchen_pick --template pick_place --output examples/tasks/kitchen_pick.py
robodeploy lint task examples/tasks/kitchen_pick.py
```

Register the module in your preset `custom_modules` list, then run via `python -m examples.cli run-episode --preset <your_preset>`.

## TaskBase contract

Every task subclasses `TaskBase` and implements:

| Method | Purpose |
|--------|---------|
| `obs_spec()` | Declare required sensors (`rgb`, `depth`, `objects`, `ft_sensor`, …) |
| `scene_spec()` | Return a `SceneSpec` describing props, lighting, terrain |
| `language_instruction()` | Natural-language goal string for language-conditioned policies |
| `reset_fn(backend)` | Bind backend, apply domain randomization, reposition objects |
| `reward_fn(obs, action)` | Scalar reward each step |
| `success_fn(obs)` | Episode success when `True` |

Optional: `failure_fn(obs)`, `max_steps`, domain randomization via `task.config`.

## Multi-phase task choreography (pour / insertion)

Pour and insertion tasks support YAML choreography for implicit phases (reach → tilt → verify):

```yaml
phases:
  - reach: {target: cup_source, threshold: 0.05}
  - tilt: {axis: y, angle: 1.2, hold_steps: 30}
  - verify: {predicate: liquid_in_target, source: cup_source, target: cup_target}
```

Shipped examples: `examples/tasks/choreography/pour.yaml` and `insertion.yaml`. `PourTemplate` and `InsertionTemplate` load them automatically; override via `task_kwargs.choreography_path` or inline `choreography` dict.

## SceneBuilder (recommended)

Use the fluent API instead of hand-building `PropConfig` lists:

```python
from robodeploy.scene_builder import SceneBuilder

def scene_spec(self):
    return (
        SceneBuilder()
        .add_table(height=0.4)
        .add_box("source", pos=(0.55, 0.0, 0.41), mass=0.08)
        .add_target("target", pos=(0.65, 0.2, 0.41))
        .validate(backend="mujoco")
        .build_spec()
    )
```

Validate before runtime with:

```bash
robodeploy scene validate my_scene.yaml --backend mujoco
```

## Reward design patterns

**Dense reach:** negative EE-to-object distance. Good for early training signal.

**Sparse success:** +1 on success, 0 otherwise. Hard to learn without shaping.

**Shaped transport:** combine reach + transport + lift bonus (see `examples/tasks/pick_place.py`).

Use `self.object_pose(name, obs)` for sensor-driven poses; avoid reading oracle backend state in policies.

## Success and failure

Keep success predicates geometrically interpretable: distance thresholds, height checks, contact via sensors. Reuse distance helpers from existing tasks or add predicates in `success_fn`.

Declare `require_objects: true` in `task_kwargs` when using `prop_pose` sensors so `obs_spec` matches the sensor rig.

## Domain randomization

Configure via preset `task_kwargs`:

```yaml
task_kwargs:
  domain_randomization:
    level: LIGHT
    object_position_range: 0.02
```

`TaskBase._apply_domain_randomization()` runs on each `reset_fn` when the backend supports scene edits.

## Common pitfalls

- **Oracle dependence:** reading prop poses only from the backend breaks sim2real. Prefer `object_pose()` with `prop_pose` sensors.
- **Reward hacking:** lift bonus without transport constraint lets the agent cheat by lifting in place.
- **obs_spec mismatch:** enabling `objects=True` without a `prop_pose` rig triggers `obs_spec_policy: raise` errors at startup.
- **Duplicate prop names:** run `robodeploy scene validate` before long training runs.

## Linting

```bash
robodeploy lint task examples/tasks/pick_place.py
robodeploy lint all
```

The linter checks `@register_task`, `TaskBase` inheritance, required methods, and deprecated API usage.

## Next steps

- [POLICY_CREATION.md](POLICY_CREATION.md) — bind a reach policy to your scene
- [SCENE_DEFINITION.md](SCENE_DEFINITION.md) — cross-backend scene YAML and validation
- [ARCHITECTURE.md](../ARCHITECTURE.md) — full framework overview
