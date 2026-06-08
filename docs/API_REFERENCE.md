# RoboDeploy API Reference

Curated reference for the public Python API. For the full module index, run `pdoc robodeploy` after `pip install pdoc`.

## Package entry (`robodeploy`)

| Symbol | Description |
|--------|-------------|
| `RoboEnv` | Main environment: backends, robots, stepping, recording |
| `Robot`, `RobotTask` | Robot description + task/policy bindings |
| `SensorRig` | Declarative sensor attachment to links |
| `ObsPipeline` | Observation transform chain |
| `use(module)` | Import and register user components |
| `discover()` | Load pip entry-point extensions |
| `RoboBridge` | Sim/real observation bridge (lazy import) |

## RoboEnv

```python
from robodeploy import RoboEnv, Robot, RobotTask, use

# Level 1 — direct construction
env = RoboEnv(backend=..., robots=[robot])

# Level 2 — registered names
use("my_pkg.components")
env = RoboEnv.make(robot="kuka", backend="mujoco", task="pick_place", policy="my_policy")

# Level 3 — config dict / EnvConfig
env = RoboEnv.from_config({"robot": "kuka", "backend": "mujoco", ...})
```

Key methods:

| Method | Purpose |
|--------|---------|
| `reset()` | Reset backend and tasks |
| `step(action)` | Apply action, return `(obs, reward, done, info)` |
| `run_episode(steps, record=False)` | Roll out until done or step limit |
| `demo_session()` | Context for explicit action injection / recording |
| `trip_estop(reason)` | Operator e-stop |
| `reset_safety()` | Clear safety latch after hazard cleared |

## Core types (`robodeploy.core.types`)

| Type | Fields (high level) |
|------|---------------------|
| `Observation` | `joint_positions`, `joint_velocities`, `ee_pose`, `rgb`, `depth`, `objects`, sensor extras |
| `Action` | `joint_positions`, `joint_velocities`, `ee_delta`, gripper |
| `ObsSpec` | Declares required observation channels |
| `SceneSpec` | Props, lighting, terrain for scene build |
| `EpisodeInfo` | `episode_id`, `reward`, `success`, `extra` metadata |

## Registry (`robodeploy.core.registry`)

```python
from robodeploy.core.registry import register_task, register_policy, register_backend

@register_task("my_task")
class MyTask(TaskBase): ...

list_registered()  # backends, robots, tasks, policies, sensors
```

## Task API (`robodeploy.tasks.base.TaskBase`)

Implement: `obs_spec`, `scene_spec`, `language_instruction`, `reset_fn`, `reward_fn`, `success_fn`.

Templates (`robodeploy.tasks.templates`): `PickPlaceTemplate`, `PourTemplate`, `InsertionTemplate` — override `source_name`, `target_name`, `scene_spec`.

## Policy API (`robodeploy.policies.base.PolicyBase`)

Implement `get_action(obs) -> Action`. Declare `action_space` in `__init__`.

Reach DSL: YAML loaded by `ReachTrajectoryPolicy` (`robodeploy.policies.reach_dsl`).

Learned: subclass `TrainablePolicyBase` / use `robodeploy.policies.learned.adapter`.

## Scene (`robodeploy.scene_builder.SceneBuilder`)

Fluent builder: `.add_table()`, `.add_box()`, `.add_mesh()`, `.add_target()`, `.validate(backend)`, `.build_spec()`.

## Config (`robodeploy.core.env_config.EnvConfig`)

Validated preset schema. Load via `EnvConfig.from_dict`, `resolve_preset(name)`, or `examples.config.load_example_preset`.

## Safety (`robodeploy.safety`)

`SafetyMonitor`, `JointLimitGuard`, `VelocityLimitGuard`, `ForceLimitGuard`, `WorkspaceGuard`, `EstopGuard`, `SafetyError`.

## Sim2Real (`robodeploy.sim2real`)

`merge_preset_with_dr`, transfer evaluation helpers. CLI: `robodeploy dr-sweep`, `robodeploy transfer-eval`.

## Training (`robodeploy.training`)

`robodeploy train bc` — behavior cloning CLI wrapping `robodeploy.training.bc`.

## Teleop (`robodeploy.teleop`)

`run_teleop_session(env, device="keyboard", record_path=...)`. Devices: keyboard, spacemouse, gamepad.

## Observability CLI modules

| Module | Commands |
|--------|----------|
| `robodeploy.cli_scene` | `scene validate`, `scene inspect` |
| `robodeploy.cli_config` | `config show/resolve/validate/diff` |
| `robodeploy.cli_assets` | `assets list/resolve/info/verify` |
| `robodeploy.cli_doctor` | `doctor` |
| `robodeploy.scaffold` | `scaffold task/policy/preset` |
| `robodeploy.linter` | `lint task/policy/preset/all` |

## Examples (not installed)

| Helper | Purpose |
|--------|---------|
| `examples.env_from_preset.env_from_preset(name)` | Build `RoboEnv` from YAML preset |
| `python -m examples.cli` | Preset-based run/export/list |

See [CLI_REFERENCE.md](CLI_REFERENCE.md) for every subcommand and flag.
