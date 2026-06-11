# Goal 1 — Cut Representation Boilerplate

**Priority**: Tier 1 (highest ROI). **Effort**: ~100h. **Touches**: every user.

## Problem

- `ReachPickPlacePolicy` = 261 lines, 8 hardcoded phases (SETTLE_HOME, PREGRASP, GRASP, LIFT, TRANSIT, PLACE, RETREAT, HOLD).
- `PickPlaceTask` = 86 lines. `peg_insertion.py`, `pour.py` ~identical structure.
- Scene = per-backend Python: `MjcfSceneBuilder` (MuJoCo), `GazeboSceneBuilder` (SDF), Isaac USD.
- `PropConfig` geometry fragmented: `geom`, `asset_path`, `asset` dict — three ways to declare one prop.
- Carry mode = brittle string enum (`kinematic|follow|contact|weld`). No validation helper.
- No fluent builders. No YAML DSL. No scaffolder.

## Current State (Audit)

### Policy Layer
- `robodeploy/core/interfaces/policy.py:34-133` — `IPolicy` abstract contract.
- `robodeploy/policies/base.py:31-134` — `PolicyBase` shared scaffolding.
- `@register_policy("<name>")` decorator (registry pattern).
- `examples/policies/reach_pick_place.py:64-107` — carry-mode dispatch as if/elif chain.
- `examples/policies/reach_pick_place.py:29` — `_waypoints_from_scene()` extracts targets at init.

### Task Layer
- `robodeploy/core/interfaces/task.py:43-248` — `ITask` contract.
- `robodeploy/tasks/base.py:42-254` — `TaskBase` with `_domain_randomizer()` at line 202.
- `examples/tasks/pick_place.py:18` — `require_objects` flag declares object obs requirement.

### Scene Layer
- `robodeploy/core/types.py:326-370` — `SceneSpec`, `WorldSpec`, `PropConfig`.
- `robodeploy/backends/sim/mujoco/scene_builder.py:17-375` — MJCF builder.
- `robodeploy/backends/sim/gazebo/scene_builder.py:47-80` — SDF builder.
- IsaacSim scene = inline in `backends/sim/isaacsim/backend.py:307-325` (geom dispatch).

---

## Deliverables

### D1. Unified Scene IR — `robodeploy/core/scene_ir.py` (NEW, ~250 lines)

Backend-agnostic representation. All backends consume this IR.

```python
from dataclasses import dataclass, field
from typing import Literal

GeomKind = Literal["box","sphere","cylinder","capsule","mesh","plane","heightfield"]

@dataclass(frozen=True)
class UnifiedGeom:
    kind: GeomKind
    size: tuple[float, ...]              # box: (sx,sy,sz); sphere: (r,); cyl: (r,h); mesh: ()
    mesh_path: str | None = None         # for kind=="mesh"
    heightfield_path: str | None = None  # for kind=="heightfield"
    convex_decomp: bool = False          # auto-decompose concave meshes

@dataclass(frozen=True)
class UnifiedPhysics:
    mass: float = 0.1
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)  # slide, spin, roll
    restitution: float = 0.0
    damping: float = 0.0
    collision_layer: int = 0             # bit index
    collision_mask: int = 0xFFFF         # which layers this collides with
    is_fixed: bool = False

@dataclass(frozen=True)
class UnifiedVisual:
    rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    material: str | None = None          # named material from material library
    texture_path: str | None = None

@dataclass(frozen=True)
class Pose3D:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w,x,y,z

@dataclass(frozen=True)
class UnifiedPropSpec:
    name: str
    geometry: UnifiedGeom
    physics: UnifiedPhysics = field(default_factory=UnifiedPhysics)
    visual: UnifiedVisual = field(default_factory=UnifiedVisual)
    pose: Pose3D = field(default_factory=Pose3D)
    variants: dict[str, str] = field(default_factory=dict)  # {"mjcf": path, "urdf": path, "usd": path}
    parent_frame: str | None = None      # for attached props

@dataclass(frozen=True)
class UnifiedLighting:
    preset: Literal["minimal","bright","dark","randomized"] | None = None
    lights: tuple = ()                    # explicit LightSpec tuple

@dataclass(frozen=True)
class UnifiedTerrain:
    kind: Literal["flat","heightfield","procedural"] = "flat"
    size: tuple[float, float] = (4.0, 4.0)
    heightfield_path: str | None = None
    procedural_params: dict | None = None  # for Perlin/ridge/stairs

@dataclass(frozen=True)
class SceneIR:
    props: tuple[UnifiedPropSpec, ...]
    lighting: UnifiedLighting = field(default_factory=UnifiedLighting)
    terrain: UnifiedTerrain = field(default_factory=UnifiedTerrain)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
```

Add `SceneSpec.to_ir() -> SceneIR` in `core/types.py` for back-compat. Existing `PropConfig.geom`+`asset_path`+`asset` collapse into single `UnifiedGeom` + `variants` map.

### D2. Backend IR Converters

Each backend exposes `build_from_ir(ir: SceneIR) -> NativeScene`:

| Backend | File | Function |
|---|---|---|
| MuJoCo | `backends/sim/mujoco/scene_builder.py` | `MjcfSceneBuilder.from_ir()` |
| Gazebo | `backends/sim/gazebo/scene_builder.py` | `GazeboSceneBuilder.from_ir()` |
| IsaacSim | `backends/sim/isaacsim/scene_builder.py` (NEW, extract from backend.py:307-325) | `IsaacSceneBuilder.from_ir()` |

Each converter handles geom-kind dispatch + missing-feature fallback (e.g., Gazebo capsule → cylinder + 2 spheres compound).

### D3. SceneValidator — `robodeploy/core/scene_validator.py` (NEW, ~150 lines)

```python
@dataclass
class ValidationIssue:
    level: Literal["error","warning","info"]
    message: str
    prop_name: str | None = None
    suggested_fix: str | None = None

@dataclass
class ValidationReport:
    issues: list[ValidationIssue]
    @property
    def ok(self) -> bool: return not any(i.level == "error" for i in self.issues)

class SceneValidator:
    def validate(self, ir: SceneIR, backend_name: str) -> ValidationReport:
        # Check: unique prop names
        # Check: geom kind supported by backend (capsule on Gazebo → warning)
        # Check: asset variant present for backend (mjcf for MuJoCo, urdf for Gazebo, usd for Isaac)
        # Check: heightfield path exists if kind="heightfield"
        # Check: mass > 0 unless is_fixed
        # Check: parent_frame references existing prop
        # Warn: friction outside [0.1, 2.0] range
        # Warn: collision_mask=0 (no collisions)
        ...
```

CLI: `robodeploy scene validate scene.yaml --backend mujoco`.

### D4. SceneBuilder Fluent API — `robodeploy/scene_builder.py` (NEW, ~300 lines)

```python
class SceneBuilder:
    def __init__(self):
        self._props: list[UnifiedPropSpec] = []
        self._lighting = UnifiedLighting()
        self._terrain = UnifiedTerrain()
        self._gravity = (0.0, 0.0, -9.81)

    # Geom shortcuts
    def add_box(self, name, *, size, pos=(0,0,0), quat=(1,0,0,0), mass=0.1, fixed=False, rgba=None, layer=0) -> "SceneBuilder":
        ...
    def add_sphere(self, name, *, radius, pos, mass=0.1, fixed=False, rgba=None) -> "SceneBuilder":
        ...
    def add_cylinder(self, name, *, radius, height, pos, mass=0.1, fixed=False) -> "SceneBuilder":
        ...
    def add_capsule(self, name, *, radius, length, pos, mass=0.1, fixed=False) -> "SceneBuilder":
        ...
    def add_mesh(self, name, *, asset, pos, mass=0.1, fixed=False, convex_decomp=True) -> "SceneBuilder":
        ...
    def add_plane(self, name="ground", *, size=(4,4), pos=(0,0,0)) -> "SceneBuilder":
        ...

    # Semantic shortcuts
    def add_table(self, name="table", *, size=(1.2, 0.8, 0.03), height=0.4) -> "SceneBuilder":
        # Common preset: fixed box at z=height
        ...
    def add_target(self, name="target", *, pos, radius=0.04) -> "SceneBuilder":
        # Visual-only sphere marker
        ...

    # Composition
    def set_lighting(self, preset: str | UnifiedLighting) -> "SceneBuilder": ...
    def set_terrain(self, kind: str, *, size=(4,4), heightfield_path=None) -> "SceneBuilder": ...
    def set_gravity(self, gx, gy, gz) -> "SceneBuilder": ...

    # Output
    def validate(self, backend: str | None = None) -> "SceneBuilder":
        report = SceneValidator().validate(self.build_ir(), backend or "mujoco")
        if not report.ok:
            raise SceneValidationError(report)
        return self
    def build_ir(self) -> SceneIR: ...
    def build_spec(self) -> SceneSpec: ...   # back-compat path
```

**Example user code (reduces 6 lines/prop → 2):**

Before:
```python
PropConfig(name="cube", asset_path="", position=(0.55, 0, 0.41),
           orientation=(1,0,0,0), mass=0.05, is_fixed=False,
           geom=GeomSpec(kind="box", size=(0.04, 0.04, 0.04)),
           material=MaterialSpec(rgba=(0.8, 0.2, 0.2, 1.0)))
```

After:
```python
.add_box("cube", size=(0.04,0.04,0.04), pos=(0.55, 0, 0.41), mass=0.05, rgba=(0.8,0.2,0.2,1))
```

### D5. Asset Loader + Format Resolution — `robodeploy/core/asset_loader.py` (NEW, ~200 lines)

```python
class AssetLoader:
    def __init__(self, search_paths: list[Path] | None = None,
                 format_priority: list[AssetFormat] | None = None):
        self._paths = search_paths or [Path("robodeploy/description"), Path("examples/assets")]
        self._priority = format_priority or [AssetFormat.MJCF, AssetFormat.URDF, AssetFormat.USD]

    def resolve(self, name: str, backend: str) -> Path:
        """Find best matching asset variant for backend. Auto-convert if needed."""
        # 1. Try preferred format for backend (mujoco→mjcf, gazebo→urdf, isaac→usd)
        # 2. Fall back to priority chain
        # 3. Auto-convert URDF→MJCF via `mujoco.MjModel.from_xml_path` w/ urdf compile
        # 4. Auto-convert URDF→USD via `isaacsim.urdf_importer`
        ...

    def catalog(self) -> list[dict]:
        """List all known assets with formats present."""
        ...
```

CLI: `robodeploy assets list`, `robodeploy assets resolve cube --backend mujoco`.

### D6. Reach Trajectory DSL — `robodeploy/policies/reach_dsl.py` (NEW, ~350 lines)

YAML schema:

```yaml
reach_pick_place:
  action_space: JOINT_POS
  action_hz: 50.0
  home: [0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0]
  carry: {mode: follow, follow_blend: 0.6}
  phases:
    - {name: settle_home, kind: settle, hold_steps: 40}
    - {name: pregrasp,    kind: reach, target: source, offset: [0,0,0.10], tracking_blend: 0.22, settle_threshold: 0.025}
    - {name: grasp,       kind: reach, target: source, offset: [0,0,0.015], settle_threshold: 0.015}
    - {name: close_gripper, kind: gripper, command: close, hold_steps: 10}
    - {name: lift,        kind: reach, target: source, offset: [0,0,0.14]}
    - {name: transit,     kind: reach, target: target, offset: [0,0,0.14]}
    - {name: place,       kind: reach, target: target, offset: [0,0,0.02]}
    - {name: open_gripper,kind: gripper, command: open, hold_steps: 10}
    - {name: retreat,     kind: reach, target: target, offset: [0,0,0.18]}
    - {name: hold,        kind: hold,  steps: 30}
```

Implementation:

```python
class ReachTrajectoryPolicy(PolicyBase):
    @classmethod
    def from_yaml(cls, path: str | Path, *, action_space: ActionSpace | None = None) -> "ReachTrajectoryPolicy":
        spec = yaml.safe_load(Path(path).read_text())
        return cls(_compile_phases(spec), action_space=action_space or ActionSpace[spec["action_space"]])

    def reset(self, obs): self._phase_idx = 0; self._step_in_phase = 0
    def get_action(self, obs):
        phase = self._phases[self._phase_idx]
        action = phase.compute(obs, self._waypoints, self._ik_solver)
        if phase.settled(obs) or self._step_in_phase > phase.max_steps:
            self._phase_idx = min(self._phase_idx + 1, len(self._phases) - 1)
            self._step_in_phase = 0
        else:
            self._step_in_phase += 1
        return action
```

Cuts `ReachPickPlacePolicy` 261 → ~40 lines + 25-line YAML.

### D7. PolicyBuilder — `robodeploy/policies/builder.py` (NEW, ~250 lines)

```python
class PolicyBuilder:
    def with_action_space(self, space: ActionSpace) -> "PolicyBuilder": ...
    def with_config(self, **kwargs) -> "PolicyBuilder": ...
    def add_phase(self, name: str, **kwargs) -> "PolicyBuilder": ...
    def add_reach_phase(self, name, *, target, offset, blend=0.22) -> "PolicyBuilder": ...
    def add_grasp_phase(self, *, settle_steps=10) -> "PolicyBuilder": ...
    def add_release_phase(self) -> "PolicyBuilder": ...
    def add_carry(self, *, mode: Literal["kinematic","follow","contact","weld","none"]) -> "PolicyBuilder": ...
    def add_hold(self, *, steps: int) -> "PolicyBuilder": ...
    def build(self) -> ReachTrajectoryPolicy: ...
```

### D8. Task Templates — `robodeploy/tasks/templates/` (NEW)

Files:
- `pick_place.py` — `PickPlaceTemplate(TaskBase)` with `source_name`, `target_name`, `reward_weights` class attrs.
- `pour.py` — `PourTemplate` with `cup_name`, `target_zone`, tilt phase.
- `insertion.py` — `InsertionTemplate` with `peg_name`, `hole_pose`.
- `stacking.py` — `StackingTemplate` with N-cube list.

Each template implements `reward_fn`, `success_fn`, `failure_fn` using composable predicates. Subclass overrides only `scene_spec()` + class attrs.

User code (replaces 86-line `pick_place.py` task):

```python
@register_task("kitchen_pick")
class KitchenPick(PickPlaceTemplate):
    source_name = "mug"
    target_name = "tray"
    reward_weights = {"ee_to_source": 0.4, "source_to_target": 1.0, "lift_bonus": 0.15}
    def scene_spec(self) -> SceneSpec:
        return (SceneBuilder()
                .add_table(height=0.4)
                .add_mesh("mug", asset="mug.obj", pos=(0.55, 0, 0.41), mass=0.08)
                .add_target("tray", pos=(0.65, 0.2, 0.41))
                .build_spec())
```

~15 lines vs 86.

### D9. RewardBuilder — `robodeploy/tasks/reward_builder.py` (NEW, ~200 lines)

```python
class RewardBuilder:
    def distance(self, source: str, target: str, *, scale=1.0, name=None) -> "RewardBuilder":
        """Negative L2 distance term."""
    def orientation(self, source: str, target_quat, *, scale=1.0) -> "RewardBuilder": ...
    def bonus_lift(self, source: str, *, initial_z: float, max_bonus=0.1, threshold=0.05) -> "RewardBuilder": ...
    def bonus_in_zone(self, source: str, zone_center, zone_radius, *, scale=1.0) -> "RewardBuilder": ...
    def penalty_action_norm(self, *, scale=0.001) -> "RewardBuilder": ...
    def penalty_collision(self, props: list[str], *, scale=1.0) -> "RewardBuilder": ...
    def penalty_force_above(self, threshold_N: float, *, scale=0.1) -> "RewardBuilder": ...
    def build(self) -> Callable[[Observation, Action], float]: ...
    def build_components(self) -> Callable[[Observation, Action], dict[str, float]]: ...  # for logging
```

### D10. SuccessPredicate Registry — `robodeploy/tasks/success_predicates.py` (NEW, ~150 lines)

```python
@register_success("object_at_target")
def object_at_target(obs, *, source: str, target_pos, threshold=0.04) -> bool: ...

@register_success("gripper_holding")
def gripper_holding(obs, *, source: str, ee_distance_max=0.04) -> bool: ...

@register_success("force_above_threshold")
def force_above_threshold(obs, *, threshold_N=5.0) -> bool: ...

@register_success("object_lifted")
def object_lifted(obs, *, source: str, initial_z: float, lift_min=0.05) -> bool: ...

class CompoundSuccess:
    """AND/OR composition of registered predicates."""
    @classmethod
    def all_of(cls, *predicates) -> Callable[[Observation], bool]: ...
    @classmethod
    def any_of(cls, *predicates) -> Callable[[Observation], bool]: ...
```

### D11. PolicyConfig + TaskConfig Schemas — `robodeploy/core/policy_config.py`, `core/task_config.py` (NEW)

Pydantic dataclasses with `__post_init__` validation. Replace untyped `dict` configs.

### D12. Lighting Presets — `robodeploy/backends/lighting_presets.py` (NEW, ~100 lines)

YAML library with 4 presets (`minimal`, `bright`, `dark`, `randomized`). Reference via `SceneSpec.lighting = "bright"` or `SceneBuilder.set_lighting("bright")`.

---

## Phased Rollout

### Phase 1.1 — Foundation (~30h)
- D1 Scene IR + D5 AssetLoader + D3 SceneValidator.
- Migrate `PropConfig` → `UnifiedPropSpec` (back-compat shim).
- Update `MjcfSceneBuilder.from_ir()` (MuJoCo first; Gazebo + Isaac in Phase 1.4).
- Add `tests/test_scene_ir.py`, `tests/test_scene_validator.py`, `tests/test_asset_loader.py`.

### Phase 1.2 — Fluent SceneBuilder (~20h)
- D4 SceneBuilder + D12 lighting presets.
- Migrate 5 example scenes to builder syntax. Keep raw `SceneSpec` for back-compat.
- Add `tests/test_scene_builder.py` with golden output assertions.

### Phase 1.3 — Task Templates + Builders (~25h)
- D8 templates (pick_place, pour, insertion, stacking).
- D9 RewardBuilder + D10 SuccessPredicate registry.
- D11 TaskConfig schema.
- Migrate `examples/tasks/*.py` to templates. Verify 70% LOC reduction.
- Add `tests/test_reward_builder.py`, `tests/test_task_templates.py`.

### Phase 1.4 — Policy DSL + Builder (~20h)
- D6 reach_dsl YAML loader + D7 PolicyBuilder.
- D11 PolicyConfig schema.
- Migrate `ReachPickPlacePolicy` to YAML-driven `ReachTrajectoryPolicy`. Keep old class as thin shim.
- Add `tests/test_reach_dsl.py`, `tests/test_policy_builder.py`.

### Phase 1.5 — Cross-Backend Parity (~15h)
- IR converters for Gazebo (`GazeboSceneBuilder.from_ir`) + Isaac (`IsaacSceneBuilder.from_ir`).
- Cross-backend equivalence tests: same SceneIR yields equivalent geom counts + poses in each backend.

---

## Acceptance Criteria

- [x] `examples/policies/reach_pick_place.py` ≤ 50 lines (loads YAML; 36 lines).
- [x] `examples/tasks/pick_place.py` ≤ 25 lines (uses `PickPlaceTemplate`; 17 lines).
- [x] A new prop = 1 line: `.add_box("X", size=..., pos=..., mass=...)` (`robodeploy/scene_builder.py`; `tests/test_scene_builder.py`, `test_representation_gaps.py::test_camera_and_lighting_presets_in_scene_builder`).
- [x] `robodeploy scene validate scene.yaml --backend mujoco` exits non-zero on bad scene (`tests/test_cli.py::test_scene_validate_bad_scene_exits_nonzero`).
- [x] `SceneIR` round-trips MuJoCo + Gazebo + Isaac with identical prop counts (`tests/test_backend_parity.py` — MuJoCo/Gazebo prop counts + poses; Isaac `from_ir` offline emits prop paths, live USD counts need GPU Kit runtime, see GOAL_06).
- [x] Reward function composable in ≤ 5 lines for pick-place (`RewardBuilder().distance(...).bonus_lift(...).build()`; `tests/test_reward_builder.py`, templates use it in `robodeploy/tasks/templates/`).
- [x] All existing example scenes work unchanged (back-compat shim: `SceneSpec.to_ir()`; `examples/*` are thin re-exports of `robodeploy.demos.*`, full suite green 2026-06-11).
- [x] `tests/` adds: scene_ir, scene_validator, scene_builder, asset_loader, reach_dsl, policy_builder, reward_builder, task_templates.

## Risks

- **Back-compat**: existing user scenes use `PropConfig` directly. Solution: `SceneSpec.to_ir()` shim + deprecation warning, 1-release migration window.
- **Asset auto-conversion**: URDF→MJCF via `mujoco.MjModel.from_xml_path` works for simple URDFs only. Complex URDFs (multi-joint, custom plugins) may fail. Mitigation: fallback to manual variant in `UnifiedPropSpec.variants`.
- **DSL expressiveness ceiling**: YAML cannot encode arbitrary Python logic. Mitigation: `phase_fn: "module:func"` escape hatch.

## Out of Scope

- Visual scene editor (web GUI). Defer to Goal 7.
- Procedural mesh generation (CAD-like). External tool problem.
- Animation/keyframe authoring.
