# Policy Creation Guide

Policies map observations to actions. RoboDeploy supports three main patterns: **YAML reach DSL**, **subclassing `PolicyBase`**, and **learned policies** (`TrainablePolicyBase`).

## Quick start

Scaffold a reach DSL file:

```bash
robodeploy scaffold policy --name custom_reach --template reach_dsl --output examples/policies/custom_reach.yaml
robodeploy lint policy examples/policies/custom_reach.yaml
```

Or scaffold a Python policy:

```bash
robodeploy scaffold policy --name my_policy --template custom --output examples/policies/my_policy.py
```

## PolicyBase contract

| Requirement | Details |
|-------------|---------|
| `@register_policy("name")` | Registers for preset / `get_policy()` lookup |
| `action_space` | Declared in `__init__` via `super().__init__(action_space=ActionSpace.JOINT_POS, …)` |
| `get_action(obs)` | Returns `Action` each control step |
| `reset()` | Inherited; override `_reset_impl()` for per-episode state |

`RoboEnv` validates `action_space` against the backend at startup.

## Pattern 1: Reach trajectory YAML

Define phases with targets derived from scene prop names:

```yaml
custom_reach:
  home: [0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0]
  action_hz: 50.0
  carry_mode: kinematic
  phases:
    - name: pregrasp
      target_frame: source
      offset: [0.0, 0.0, 0.10]
    - name: grasp
      offset: [0.0, 0.0, 0.015]
```

Load from a policy module or the thin shim in `examples/policies/reach_pick_place.py` (≤50 lines — all phase tuning in `reach_pick_place.yaml`). Lint with `robodeploy lint policy <file.yaml>`.

## Pattern 2: Custom PolicyBase subclass

See scaffolded `examples/policies/my_policy.py`. Key steps:

1. Choose `ActionSpace` (`JOINT_POS`, `JOINT_VEL`, `EE_POSE`, …).
2. Implement `get_action` using proprioception and sensor fields on `Observation`.
3. Call `bind_runtime(backend, description)` when you need IK or scene access (MuJoCo).

## Backend binding

Policies that need IK or prop poses must bind after env construction:

```python
policy.bind_runtime(env.backend, env.primary_robot.description)
```

Without binding, fall back to joint-space tracking (see `ReachPickPlacePolicy`).

## Sensor consumption

| Obs field | Typical use |
|-----------|-------------|
| `obs.ft_force` | Grasp confirmation, contact-rich manipulation |
| `obs.imu_acceleration` | Mobile base / vibration cues |
| `obs.objects` | Prop poses from `prop_pose` sensor rig |
| `obs.rgb` / `obs.depth` | Vision policies |

Align policy expectations with task `obs_spec()` and preset `sensor_rigs`.

## Learned policies

Wrap a PyTorch module with `TrainablePolicyBase` or use built-in loaders (`RobomimicPolicy`, `DiffusionPolicy`, `VLAPolicy`). Train with:

```bash
robodeploy train bc --dataset demos.jsonl --dummy  # see Goal 2
```

## Linting

```bash
robodeploy lint policy examples/policies/reach_pick_place.py
```

Checks: `@register_policy`, `PolicyBase` subclass, `get_action`, `action_space` declaration.

## Common pitfalls

- **Missing action_space:** linter error; env will also reject mismatched spaces.
- **Scene/policy mismatch:** phase `target_frame` must match prop names in `scene_spec()`.
- **Hz mismatch:** set `action_hz` in policy config for real-time bridges.

## Next steps

- [TASK_CREATION.md](TASK_CREATION.md) — define the scene your policy manipulates
- [SCENE_DEFINITION.md](SCENE_DEFINITION.md) — prop layout and validation
- [CONTRACTS.md](../CONTRACTS.md) — `IPolicy` interface details
