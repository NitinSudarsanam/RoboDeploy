# Tutorial 2 — Your First Custom Task

**Time:** ~20 minutes  
**Goal:** Define a small pick task with `SceneBuilder`, run it in MuJoCo, and understand task hooks.

## Prerequisites

```bash
python -m pip install -e ".[sim,dev]"
```

## Minimal custom task

Create `examples/tasks/my_kitchen_pick.py`:

```python
from robodeploy.core.registry import register_task
from robodeploy.core.types import SceneSpec
from robodeploy.scene_builder import SceneBuilder
from robodeploy.tasks.templates.pick_place import PickPlaceTemplate


@register_task("my_kitchen_pick")
class MyKitchenPickTask(PickPlaceTemplate):
  def scene_spec(self) -> SceneSpec:
    return (
      SceneBuilder()
      .add_box(self.source_name, size=(0.03, 0.03, 0.03), pos=(0.55, 0.05, 0.38), mass=0.06, rgba=(0.9, 0.2, 0.1, 1.0))
      .add_target(self.target_name, pos=(0.62, -0.12, 0.38))
      .build_spec()
    )
```

This is under 20 lines: scene layout, source cube, and placement target come from `SceneBuilder`; rewards and success logic inherit from `PickPlaceTemplate`.

## Register and run

Register the task in a preset (see `my_kitchen_pick_mujoco` in `examples/config/presets.yaml`) or add the same block after creating `examples/tasks/my_kitchen_pick.py`, then run via the examples CLI:

```bash
python -m examples.cli run-episode \
  --preset my_kitchen_pick_mujoco \
  --steps 400
```

Example presets live under `examples/` — use `python -m examples.cli`, not `robodeploy run-episode --robot/--backend`, for MuJoCo sims.

## What each piece does

| Piece | Role |
|-------|------|
| `PickPlaceTemplate` | Dense reach + grasp rewards, success when source is near target |
| `SceneBuilder` | Backend-agnostic props (box, sphere, cylinder, …) |
| `register_task` | Makes the task discoverable by name |
| `example_reach_pick` | YAML reach DSL policy bound to `source` / `target` props |

## Next steps

- Tutorial 3 (`03_training.md`) — train BC on recorded demos.
- Tutorial 2 teleop (`02_teleop.md`) — keyboard control and demo recording.
- `docs/TASK_CREATION.md` — reward design, success predicates, domain randomization.
