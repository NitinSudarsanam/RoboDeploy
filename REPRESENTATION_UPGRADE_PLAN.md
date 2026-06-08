# Representation Upgrade Plan

Plan to upgrade policies, tasks, and scene/3D env representations. Goal: complete, validated, easy for users. State as of 2026-06-08.

## Pain Point Summary

| Layer | Pain |
|---|---|
| Policy | Phase machines hardcoded in Python. `ReachPickPlacePolicy` = 261 lines. No DSL. No builder. Carry mode = brittle string enum. |
| Task | No DSL. Reward monolithic. `pick_place`/`peg_insertion`/`pour` ~identical structure, duplicated. No success-predicate library. |
| Scene | Per-backend builders (MuJoCo/Gazebo/IsaacSim). Geom fragmented (`geom`+`asset_path`+`asset` dict). No validation. No physics groups. Heightfield MuJoCo-only. |
| Env Config | `RoboEnv.make()` crippled — no sensor_rigs. Presets untyped YAML. No inheritance. No CLI scaffolder. |
| UX | New task = 86 lines boilerplate. New policy = 261 lines. No `scaffold`, no `lint`, no `validate-scene` CLI. |

---

## A. Policy Layer

### Current
- `IPolicy` + `PolicyBase` clean two-level interface (`core/interfaces/policy.py:34-133`, `policies/base.py:31-134`).
- Class-based, config-driven. `@register_policy` decorator. Explicit `action_space`.
- `ReachPickPlacePolicy` hardcodes 8 phases (SETTLE_HOME, PREGRASP, GRASP, LIFT, TRANSIT, PLACE, RETREAT, HOLD). Waypoints extracted from `SceneSpec` at init.
- Carry mode = string enum (`kinematic|follow|contact|weld`), no validation helper.
- Builtin learned policy loaders exist (`RobomimicPolicy`, `VLAPolicy`, `DiffusionPolicy`).

### Proposed

1. **Reach trajectory DSL** — `robodeploy/policies/reach_dsl.py` (new)
   ```yaml
   reach_pick_place:
     home: [0, -0.6, 0, -1.8, 0, 1.2, 0]
     phases:
       - {name: pregrasp, target_frame: ee_frame, offset: [0,0,0.10], tracking_blend: 0.22, settle_threshold: 0.025}
       - {name: grasp,    offset: [0,0,0.015]}
       - {name: lift,     offset: [0,0,0.14]}
   ```
   `ReachTrajectoryPolicy` loads YAML + scene → auto state machine. Cuts `ReachPickPlacePolicy` 261 → ~40 lines + YAML.

2. **PolicyBuilder** — `robodeploy/policies/builder.py` (new). Fluent API:
   ```python
   policy = (PolicyBuilder()
       .with_action_space(ActionSpace.JOINT_POS)
       .add_phase("settle_home", settle_time=40)
       .add_reach_phase("pregrasp", ee_height_offset=0.10)
       .add_grasp_phase()
       .add_carry(mode="follow")
       .build())
   ```

3. **PolicyConfig schema** — `robodeploy/core/policy_config.py` (new). Pydantic/dataclass. Validate `carry_mode`, action space, action_hz. `PolicyBase.__init__` consumes validated config.

4. **Native batched action API** — `IPolicy.get_action_batch()` default loop → vectorized hook. Required for VecEnv RL training.

---

## B. Task Layer

### Current
- `ITask` + `TaskBase` (`core/interfaces/task.py:43-248`, `tasks/base.py:42-254`).
- Required: `obs_spec`, `scene_spec`, `language_instruction`, `reset_fn`, `reward_fn`, `success_fn`, optional `failure_fn`.
- Domain randomization built-in (`TaskBase._domain_randomizer()`, line 202). Configured via `task.config.domain_randomization`.
- Example tasks (`pick_place.py:86`, `peg_insertion.py`, `pour.py`) ~identical structure. Reward = -distance + lift bonus repeated.
- `object_pose(name, obs)` reads from sensors or backend.

### Proposed

1. **Task templates** — `robodeploy/tasks/templates/` (new). Abstract `PickPlaceTemplate`, `PourTemplate`, `InsertionTemplate`:
   ```python
   class PickPlaceTemplate(TaskBase):
       source_name: str
       target_name: str
       reward_weights: dict = {"ee_to_source": 0.35, "source_to_target": 1.0, "lift_bonus": 0.1}
       def reward_fn(self, obs, action): ...  # shared impl
   ```
   Cuts ~70% duplication.

2. **Composable RewardBuilder** — `robodeploy/tasks/reward_builder.py` (new):
   ```python
   reward = (RewardBuilder()
       .distance("ee", "source", scale=0.35, name="reach")
       .distance("source", "target", scale=1.0, name="transport")
       .bonus_lift("source", initial_z=0.38, max_bonus=0.1)
       .build())
   ```
   Composes named scalar predicates → `reward_fn(obs, action)`.

3. **Success predicate registry** — `robodeploy/tasks/success_predicates.py` (new). `@register_success("dist_below_threshold")`. Reusable distance/orientation/contact predicates.

4. **TaskConfig schema** — `robodeploy/core/task_config.py` (new). Validated dataclass with `scene`, `obs_spec`, `domain_randomization`, `reward_schema`, `success_threshold`, `language_instruction`. Replace untyped `task.config` dict.

5. **Multi-phase task DSL** — for pour/insertion implicit phases (reach → tilt → verify). YAML choreography:
   ```yaml
   phases:
     - reach: {target: cup, threshold: 0.05}
     - tilt:  {axis: y, angle: 1.2, hold_steps: 30}
     - verify: {predicate: liquid_in_target}
   ```

---

## C. Scene & 3D Representation

### Current
- `SceneSpec` + `WorldSpec` (`core/types.py:326-370`). `PropConfig`: name, asset_path, pose, mass, is_fixed, geom, material, `asset` multi-format dict.
- Per-backend builders:
  - MuJoCo: `MjcfSceneBuilder` (`backends/sim/mujoco/scene_builder.py:17-375`). MJCF includes, procedural geoms, lights, cameras, sensors, grasp welds (line 136).
  - Gazebo: `GazeboSceneBuilder` (`backends/sim/gazebo/scene_builder.py:47-80`). SDF from WorldSpec, quat→RPY.
  - IsaacSim: USD stage (not shown).
- `AssetFormat` enum (MJCF, URDF, USD).
- No unified IR. No validation. No collision groups.

### Proposed

1. **Unified Scene IR** — `robodeploy/core/scene_ir.py` (new). Backend-agnostic:
   ```python
   @dataclass
   class UnifiedPropSpec:
       name: str
       geometry: UnifiedGeom        # box|cylinder|sphere|capsule|mesh|heightfield
       physics: UnifiedPhysics      # mass, friction, damping, collision_groups
       visual: UnifiedVisual        # material, shader, texture
       pose: Pose3D
       variants: dict[AssetFormat, AssetPath]
   ```
   Each backend implements `SceneIR → BackendNative` converter. Validates buildable on target backend before init.

2. **SceneBuilder fluent API** — `robodeploy/scene_builder.py` (new):
   ```python
   scene = (SceneBuilder()
       .add_box("table", size=(1.2,0.8,0.03), pos=(0,0,0), fixed=True)
       .add_mesh("source", asset="cube.obj", mass=0.05, pos=(0.55,0,0.41))
       .add_target("target", pos=(0.6,0.2,0.41))
       .set_lighting("bright")
       .set_terrain("flat", size=(4,4))
       .validate()
       .build_spec())
   ```
   Reduces prop boilerplate 6 → 2 lines.

3. **AssetLoader** — `robodeploy/core/asset_loader.py` (new). Format priority list (MJCF, URDF, USD), fallback chains, auto URDF→MJCF conversion (via mj_parse).

4. **SceneValidator** — `robodeploy/core/scene_validator.py` (new). Pre-flight checks: unique names, geom/backend compat, asset paths exist, physics param sanity. Returns `ValidationReport(errors, warnings, suggestions)`.

5. **Lighting/camera presets** — `robodeploy/backends/lighting_presets.py` (new). YAML presets `minimal|bright|dark|randomized`. Reference via `SceneSpec.lighting = "bright"`.

6. **Physics groups + collision masks** — extend `PropConfig` with `collision_layer`, `collision_mask`, `friction_dist` for randomization at spec level.

7. **Procedural terrain** — extend `TerrainSpec` beyond `heightfield_path`. Add generators (Perlin, ridge, stairs). LOD + sparse heightfield. Backport to Gazebo + IsaacSim (currently MuJoCo-only).

---

## D. Env Configuration Layer

### Current
- 4 construction paths: `RoboEnv.from_config(cfg)`, `env_from_preset("kuka_pick_mujoco")`, `RoboEnv.make(...)` (flat sensors only — broken), `RoboEnv(backend=..., robots=[...])`.
- `examples/config/presets.yaml`: untyped YAML. `sensor_rigs`, `obs_pipeline`, `custom_modules` declared but no schema validator.
- `Robot` holds `tasks: {task_id: RobotTask}`, `RobotTask` holds `task` + `policies` dict.
- SensorRig late-bound resolution via backend-aware registry pairs.

### Proposed

1. **EnvConfig schema** — `robodeploy/core/env_config.py` (new):
   ```python
   @dataclass
   class EnvConfig:
       robot: str | RobotDescription
       backend: str | IBackend
       task: str | ITask
       policy: str | IPolicy
       sensors: list[str | ISensor]
       sensor_rigs: list[SensorRig] | None = None
       obs_pipeline: ObsPipeline | None = None
       backend_kwargs: dict | None = None
       task_kwargs: dict | None = None
       policy_kwargs: dict | None = None
       obs_spec_policy: Literal["warn","raise","off"] = "warn"
       max_episode_steps: int | None = None
   ```
   `from_yaml()`, `from_dict()`, `to_yaml()` helpers. Pydantic validation.

2. **ConfigBuilder** — `robodeploy/core/config_builder.py` (new). Fluent:
   ```python
   config = (EnvConfigBuilder()
       .with_robot("franka").with_backend("mujoco")
       .with_task("pick_place", object_mass=0.1)
       .with_policy("example_sensor_reach_pick")
       .add_sensor("wrist_rgbd", width=128, height=128)
       .add_sensor("wrist_ft")
       .validate().build())
   env = RoboEnv.from_config(config)
   ```

3. **Preset inheritance** — extend `examples/config/presets.yaml` with YAML anchors:
   ```yaml
   base_kuka_mujoco: &base_kuka
     robot: kuka
     backend: mujoco
     backend_kwargs: {config: {allow_actuator_name_fallback: true}}
   kuka_pick_mujoco:
     <<: *base_kuka
     task: pick_place
     policy: example_sensor_reach_pick
   ```

4. **Fix `RoboEnv.make()`** — accept `sensor_rigs`, `custom_modules`, `obs_pipeline`. Align with `from_config()`.

5. **Sensor resolution debug** — `env.debug_resolution()` prints selected sensor implementation per logical name + backend pair.

---

## E. CLI & Tooling

### Current
- `robodeploy list-registry`, `run-episode --dummy`, `serve-policy`.
- No scaffolder, no linter, no scene validator.

### Proposed

1. **Scaffold** — `robodeploy scaffold task --name my_task --template pick_place`. Generates boilerplate file with docstrings, method stubs, `@register_*`.

2. **Lint** — `robodeploy lint task examples/tasks/pick_place.py`. Checks `@register_*` present, all abstract methods implemented, action_space declared.

3. **Scene validate/inspect** — `robodeploy scene validate my_scene.yaml --backend mujoco`. Lint scene errors, resolved asset paths, geom/backend warnings.

4. **Config resolve** — `robodeploy config resolve --preset kuka_pick_mujoco --json`. Show fully-resolved backend/sensor implementations.

5. **Asset catalog** — `robodeploy assets list [--robot|--mesh|--mjcf]`. Index `robodeploy/description/` + `examples/catalog/mujoco_catalog.yaml`.

---

## F. Docs & Examples

1. `docs/TASK_CREATION.md` — step-by-step: `TaskTemplate` → `SceneBuilder` → `RewardBuilder` → success predicate. Target: pick_place in ~20 lines.
2. `docs/POLICY_CREATION.md` — `ReachTrajectoryPolicy` YAML or `PolicyBuilder`. Target: reach policy in ~15 lines.
3. `docs/SCENE_DEFINITION.md` — `SceneBuilder` API, cross-backend validation, asset resolution.
4. `examples/presets/` — base templates: `base_sim.yaml`, `base_real.yaml`, `manipulate.yaml`. Users `include:` them.

---

## Implementation Plan

| Layer | File(s) | Change | Impact |
|---|---|---|---|
| Policy | `policies/reach_dsl.py` | YAML reach DSL | `ReachPickPlacePolicy` 261→40 |
| Policy | `policies/builder.py` | `PolicyBuilder` fluent API | Reduce custom-policy boilerplate |
| Policy | `core/policy_config.py` | Validated `PolicyConfig` | Catch errors early |
| Task | `tasks/templates/` | Reusable task bases | 70% dedup |
| Task | `tasks/reward_builder.py` | Composable reward | Reward 30→10 lines |
| Task | `tasks/success_predicates.py` | Registered predicates | Reusable |
| Task | `core/task_config.py` | Validated `TaskConfig` | Replace dict |
| Scene | `core/scene_ir.py` | Unified backend-agnostic IR | Single source of truth |
| Scene | `scene_builder.py` | `SceneBuilder` fluent API | Prop 6→2 lines |
| Scene | `core/asset_loader.py` | Asset resolution + fallback | Auto URDF→MJCF |
| Scene | `core/scene_validator.py` | Pre-flight validation | Catch mismatches before run |
| Scene | `backends/lighting_presets.py` | Lighting presets | Reusable |
| Env | `core/env_config.py` | `EnvConfig` schema | Replace untyped dicts |
| Env | `core/config_builder.py` | `EnvConfigBuilder` | Programmatic |
| Env | `examples/config/presets.yaml` | YAML anchors | Kill duplication |
| Env | `env.py` `make()` fix | Add `sensor_rigs` support | Unblock minimal path |
| CLI | `cli.py` | `scaffold`, `lint`, `scene`, `config`, `assets` | 50% onboarding cut |
| Docs | `docs/TASK_CREATION.md` | Task guide | Hours → minutes |
| Docs | `docs/POLICY_CREATION.md` | Policy guide | Hours → minutes |
| Docs | `docs/SCENE_DEFINITION.md` | Scene guide | |

## Suggested Order (Highest ROI First)

1. **SceneBuilder + SceneValidator** — foundation. Unblocks tasks + presets.
2. **Unified Scene IR + AssetLoader** — backend-agnostic source of truth. Auto URDF→MJCF.
3. **Task templates + RewardBuilder** — 70% dedup, biggest user-facing win.
4. **EnvConfig schema + ConfigBuilder + fix `RoboEnv.make()`** — kill untyped YAML errors.
5. **Reach trajectory DSL + PolicyBuilder** — slash policy boilerplate 6×.
6. **CLI scaffold/lint/scene/assets** — onboarding tools.
7. **Multi-phase task DSL, procedural terrain, lighting presets** — polish.
8. **Docs + preset templates** — close the loop.

**Effort**: ~200–300 hrs total. Highest ROI: SceneBuilder + task templates (~60% friction reduction).
