# RoboDeploy — Sensor & 3D Environment Capability Plan

> **Superseded for execution tracking** — see `history.json` for completed subtasks. Kept as design reference only.

**Date**: 2026-05-12
**Scope**: Audit of repo flaws relevant to (a) adding robot sensors, and (b) adding 3D environments / scene composition through the library structure. Concrete plan follows the audit.
**Companion**: read alongside `AUDIT_REPORT.md` (2026-04-22) for general architecture compliance findings. This document does not repeat those; it focuses on the specific layers that block sensor + environment work and the contract changes those features will need.

---

## 1. Findings — flaws, contract breaches, spaghetti

### 1.1 Sensor stack is structurally broken

**Two unrelated sensor systems coexist.**

- Top-level `robodeploy/sensors/*` — `ISensor` / `SensorBase` lifecycle. This is what `Robot.sensors` carries and what `RoboEnv` plumbs.
- `robodeploy/backends/real/ros2/sensors/*` — a separate `IRos2Sensor` `Protocol`, with its own registry (`register_ros2_sensor`) and its own concrete class (`Ros2RgbdCameraSensor`).

They share **no common interface**, no common config, no common pairing. The "sim/real pair" convention documented in `ARCHITECTURE.md` (`wrist_camera_sim`, `wrist_camera_real`) is therefore aspirational — the working ROS2 RGBD sensor is *not* registered as `wrist_camera_real`, so `RoboEnv.make(..., sensors=["wrist_camera"])` cannot select it.

Citations:
- `robodeploy/sensors/camera/sim/mujoco_camera.py:16-19` — `_init_impl`, `_read_impl` both raise `NotImplementedError`.
- `robodeploy/sensors/camera/real/realsense.py:15-19` — same: stub.
- `robodeploy/sensors/ft_sensor/sim/mujoco_ft.py`, `…/real/ati_ft.py` — same.
- `robodeploy/backends/real/ros2/sensors/camera_rgbd.py:75` — real implementation, inherits `Ros2NodeAdapter, IRos2Sensor`, never `SensorBase`.
- `robodeploy/backends/real/ros2/sensors/registry.py` — second registry alongside `core/registry.py`.

**Backends silently drop `Robot.sensors`.**

- `robodeploy/backends/sim/mujoco/backend.py:48-49` — `del scene; del sensors` at the top of `_load()`. The sensor list is thrown away before any rendering attachment.
- `robodeploy/backends/sim/isaacsim/backend.py:68` — stores `self._sensors = list(sensors)` but never invokes `sensor.initialize(...)` and never composes their output into `Observation`.
- `robodeploy/backends/real/ros2/backend.py:176-283` — completely ignores `robot.sensors`; only consults `self.config[f"{robot_id}.sensors"]` (a parallel config-dict pathway) to instantiate `IRos2Sensor` objects.

So today: a user passing `sensors=[MyCamera()]` to `Robot(...)` gets silent no-op on every backend.

**Sensor lifecycle invariant is broken in `RoboEnv`.**

`SensorBase.read()` (`robodeploy/sensors/base.py:84-88`) raises `RuntimeError` if `initialize()` has not been called. But `RoboEnv._initialize_components` (`robodeploy/env.py:226-229`) calls `sensor.warmup()` without ever calling `sensor.initialize(backend)`. `warmup()` calls `_read_impl()` directly, sidestepping the guard. The first real `read()` would then trip the guard. Conclusion: there is no call site in the codebase that actually walks `ISensor` through its declared lifecycle.

**`SensorData → Observation` merge has no implementation.**

`Observation` has `rgb`, `depth`, `ft_force`, `ft_torque`, `imu_*` fields. No backend produces them from sensors:

- MuJoCo backend's `_build_obs()` returns `Observation(...)` with the vision/FT/IMU slots all unset.
- ROS2 backend (`backend.py:312-316`, `:348-354`) does the merge by **directly mutating a frozen-ish dataclass instance**: `obs.rgb = sd.rgb`, `obs.depth = sd.depth`. No protocol, no per-sensor naming, no synchronization, type-ignored.

**Single-camera assumption is baked in.**

`Observation.rgb : Optional[ndarray]` is a single tensor. There is no `images: dict[str, ndarray]` keyed by sensor name. Two wrist cameras + an overhead camera + a depth camera cannot be represented in the current `Observation` schema without overloading semantics. `ObsSpec` mirrors this — flat booleans (`rgb: bool`, `depth: bool`) with a single `image_width / image_height` pair.

**No sensor → robot link binding.**

`Robot.sensors` is a flat `list[ISensor]`. There is no mount metadata (parent link, pose, mount type). The MuJoCo backend cannot know where to anchor a wrist camera; the Isaac backend cannot construct an `XFormPrim` under the right articulation link. This is the missing piece that prevents sim cameras from being implementable.

**Other smaller issues.**

- `RoboEnv.make()` resolves sensors by `name + "_sim" / "_real"` suffix (`env.py:114-117`). No kwargs are forwarded — sensor configs (resolution, framerate, exposure) are lost.
- `shared_sensors` are accepted by `RoboEnv` but every backend either ignores them (`isaacsim`, `mujoco` raise `NotImplementedError` on non-empty `shared_sensors`) or silently does nothing.
- `SensorData.timestamp_source` exists (`core/types.py:135`) but no sensor actually populates it, so the documented `TIME_WINDOW` widening logic has no signal to act on.
- `ObsPipeline.sync_policy()` is a stub (`obs_pipeline.py:108-116`) — `ObsSyncMode` enum exists but no actual buffering is implemented.
- `Ros2RgbdCameraSensor` `get_diagnostics()` is fine, but `read()` returns `timestamp = wall_time_s` from the cache — i.e. host receive time labeled as both `timestamp_hw` and `timestamp_recv`. The hardware stamp from the ROS message header is discarded.

### 1.2 3D environment / scene layer is cosmetic

**`SceneSpec` is passed everywhere and loaded nowhere.**

The signature reaches every backend:

- `MuJoCoBackend._load(...)` — `del scene` on `mujoco/backend.py:48`.
- `IsaacSimBackend._load(...)` — scene is passed only to `RvizPublisher.publish_scene(scene)` (line 111). Isaac stage is not modified.
- `ROS2RealBackend.initialize_multi` — scene is passed only to `RvizPublisher.publish_scene(scene)` (line 298).

So the entire scene path produces RViz **markers** and nothing else. Props have no physics, no collision, no mass, no contact. The "table_height" field is read by nobody.

**Two competing scene primitives.**

`SceneSpec` carries both `props: list[PropConfig]` (newer) and `objects: list[ObjectSpec]` (legacy, kept for backward compat). `RoboEnv._merged_scene` (`env.py:196-213`) merges both. Every consumer must handle both, none currently do. This duplication will propagate as soon as anyone implements scene loading — a strong sign it should be collapsed before adding capability.

**`PropConfig` schema is insufficient for real scenes.**

```python
PropConfig:
    name: str
    asset_path: str                # plain string, no AssetFormat
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    mass: float = 0.1
    is_fixed: bool = False
```

Missing for a real 3D scene library:

- Procedural primitive shape (box / cylinder / sphere / mesh) so users can declare "0.05 m red cube" without shipping an asset file.
- Material / color / texture / friction — needed for both visual rendering and sim-to-real domain randomization.
- Format selection (MJCF include vs URDF vs USD) per backend.
- Mount parent (attach to world vs robot link).
- Joint type / DoF (free body vs fixed weld vs articulated drawer).
- Inertia diag (engines without it default to mass-only-sphere assumptions that destabilize).
- Asset bundle layout (mesh dir, texture dir).

**Concrete tasks ship empty scenes.**

`tasks/manipulation/pick_place.py:24-27`, `pour.py:18-23`, `peg_insertion.py:18-23` all declare `PropConfig(name=..., asset_path="", ...)`. `asset_path` is the empty string — even if a backend tried to load these, there is nothing to load. The reward / success functions reduce to "drive EE toward fixed coordinate", reducing pick-and-place to free-space waypoint tracking with no actual object.

**`SupportsSceneEdit` / `SupportsPayload` / `SupportsPhysicsRandomization` protocols exist but no backend implements any.**

`backends/capabilities.py:37-56` declares the protocols. `Grep` over `robodeploy/backends/**` finds zero implementations of `set_prop_pose`, `get_prop_pose`, `get_prop_names`, `set_payload`, or `set_physics_params`. `DomainRandomizer._randomize_object_poses` (`tasks/randomization.py:120-138`) calls `backend.teleport_object()` inside a `try/except NotImplementedError: pass` — randomization is therefore a silent no-op on every backend that exists today.

**No environment hierarchy.**

There is no `WorldSpec` / `TerrainSpec` / `LightSpec` / `CameraSpec`. The scene is implicitly a tabletop:

- `SceneSpec.table_height: float` baked in as a top-level field.
- `SceneSpec.lighting: str` is a free-form string ("default" | "random" | "dark"). No way to declare a directional + ambient + N point lights, or to sweep ranges for domain randomization.
- No way to declare a free-form floor / non-flat terrain. Mujoco's URDF auto-import (`mujoco/backend.py:262-298`) injects a fixed 2×2 plane.
- No way to declare environment-level cameras (overhead, third-person, evaluation) independent of robot-mounted sensors. `shared_sensors` on `RoboEnv` is the place for that but nothing consumes it.

**Asset library does not exist.**

There is no `robodeploy/world/` or `robodeploy/assets/` containing prop meshes or asset bundles. Users must supply paths to external files in their own project. For a library that targets sim-to-real transfer, the absence of a reference prop set is a serious onboarding gap.

### 1.3 Cross-cutting flaws that will bite the sensor/env work

- **`is_real` leakage**: `RoboEnv.make` (`env.py:114`) uses `backend.is_real` to pick the sensor suffix. Acceptable per ARCHITECTURE.md as a wiring-time read, but it means sensor pairing depends on backend identity *string-matching*, which is fragile.
- **Backend god-class growth**: `ROS2RealBackend.initialize_multi` is already ~180 lines doing transport setup + RSP + controller spawning + sensor instantiation + RViz. Adding scene-prop spawning here will push it past readable.
- **`MuJoCoBackend._compile_mjcf_with_position_actuators`** (`mujoco/backend.py:178-324`) already does substantial XML synthesis. Adding prop XML generation here will mix robot-actuator wiring with scene composition. Should be extracted into a `MjcfSceneBuilder` before adding props, not after.
- **`Observation` dataclass is not frozen**, so mutating `obs.rgb` works at runtime, but it's a contract violation that hides bugs. The merge protocol should produce a new `Observation` (or a builder), not mutate.
- **`DomainRandomizer` keys props by `object_name: str`**, but `SceneSpec` may carry both `props` and `objects` lists with overlapping names. Identity resolution between randomizer, backend, and task is undefined.
- **Sensor pairing strategy is string-suffix coupling**, which forces every sensor type to use the same naming convention. Replace with explicit pairing metadata on a registry entry.

---

## 2. Plan — adding sensor + 3D environment capability

The order matters. Sensors need scene loading (camera with no scene = uninteresting), and scene loading needs the schema cleanup. Each phase ends with one concrete demo that exercises only the new capability.

### 2.0 Execution model

Use one main agent and a small number of follower subagents.

- The main agent owns the plan, keeps the subtask order honest, watches for scope drift, updates `history.json`, and does the final verification before a subtask is marked done.
- Follower subagents take only the parts that can genuinely run in parallel, usually isolated backend-specific or test-specific slices with clear file boundaries.
- Follower subagents should report code changes and verification back to the main agent. The main agent decides what to merge, what to reject, and whether the subtask is actually complete.
- If a subtask still has tight coupling or contract risk, keep it single-agent. Only split work once the boundaries are clear enough that parallel edits will not fight each other.

### Phase 0 — schema unification (prerequisite, ~1 day)

Goal: remove the dead branches so later work doesn't fork.

1. **Collapse `ObjectSpec`/`PropConfig`** in `core/types.py`:
   - Mark `ObjectSpec` and `SceneSpec.objects` deprecated; `RoboEnv._merged_scene` keeps the read path for one release, but stop emitting `objects` from any task in this repo. Update `pick_place.py`, `pour.py`, `peg_insertion.py` to use `props` only.
2. **Expand `PropConfig`** (additive, default-compatible):
   ```python
   @dataclass
   class GeomSpec:
       kind: Literal["box", "cylinder", "sphere", "capsule", "mesh"]
       size: tuple[float, ...]            # box: (sx,sy,sz); sphere: (r,); mesh: ()
       mesh_path: Optional[str] = None    # required when kind=="mesh"

   @dataclass
   class MaterialSpec:
       rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0)
       friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
       texture: Optional[str] = None

   @dataclass
   class PropConfig:
       name: str
       position: tuple[float, float, float] = (0.0, 0.0, 0.0)
       orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
       mass: float = 0.1
       is_fixed: bool = False
       geom: Optional[GeomSpec] = None              # NEW — procedural primitives
       material: MaterialSpec = field(default_factory=MaterialSpec)  # NEW
       asset: Optional[dict[AssetFormat, str]] = None  # NEW — per-format paths
       parent_link: Optional[str] = None            # NEW — None = world
       inertia_diag: Optional[tuple[float, float, float]] = None
   ```
   Keep `asset_path: str = ""` as a forwarded alias so existing callers do not break.
3. **Introduce `WorldSpec`** wrapping scene-level state:
   ```python
   @dataclass
   class LightSpec:
       position: tuple[float, float, float] = (0.0, 0.0, 2.0)
       direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
       diffuse: tuple[float, float, float] = (0.8, 0.8, 0.8)
       kind: Literal["directional", "point", "spot"] = "directional"

   @dataclass
   class CameraSpec:
       name: str
       position: tuple[float, float, float]
       orientation: tuple[float, float, float, float]
       fov_deg: float = 60.0
       resolution: tuple[int, int] = (640, 480)

   @dataclass
   class TerrainSpec:
       kind: Literal["flat", "heightfield"] = "flat"
       size: tuple[float, float] = (4.0, 4.0)
       heightfield_path: Optional[str] = None

   @dataclass
   class WorldSpec:
       props: list[PropConfig] = field(default_factory=list)
       lights: list[LightSpec] = field(default_factory=list)
       cameras: list[CameraSpec] = field(default_factory=list)
       terrain: TerrainSpec = field(default_factory=TerrainSpec)
       gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
   ```
   `SceneSpec` becomes a thin wrapper (`SceneSpec.world: WorldSpec`) with backward-compat shims that copy old `props/objects/table_height/lighting` into the `world`. This is the only schema break and it's still source-compatible for one release.

4. **Drop dead state**: `BackendBase._task` was already removed; remove `BackendBase._sensors` storage attribute and rely on `Robot.sensors` carried per-tick. Stops the parallel state in `initialize` vs `initialize_multi`.

5. **Decision needed from user**: do we want to keep ROS-side `Ros2RgbdCameraSensor` as a separate `IRos2Sensor` implementation, or refactor it to subclass `SensorBase` so it can be passed as a normal `Robot.sensors[i]` entry? Recommendation: refactor (Phase 2, see §2.3) because the parallel registry is the root cause of the broken pairing convention.

### Phase 1 — 3D environment loading (~3–5 days)

Goal: tasks can declare props/lights/terrain and `MuJoCoBackend` actually loads them.

1. **Extract `MjcfSceneBuilder`** from `MuJoCoBackend._compile_mjcf_with_position_actuators`:
   ```python
   # robodeploy/backends/sim/mujoco/scene_builder.py
   class MjcfSceneBuilder:
       def __init__(self, robot_mjcf_xml: str): ...
       def attach_world(self, world: WorldSpec) -> None: ...
       def attach_actuators(self, joint_names: list[str], kp: float) -> None: ...
       def emit(self) -> str: ...                  # final XML
   ```
   Move the existing actuator injection, light/floor/camera injection, and inertia clamping out of the backend. `MuJoCoBackend._load` becomes ~30 lines.
2. **Implement `MjcfSceneBuilder.attach_world`**:
   - For `GeomSpec.kind="box"` etc. emit `<body><geom type="box" size=... rgba=... mass=.../><freejoint/></body>`; `is_fixed=True` skips `<freejoint/>`.
   - For `kind="mesh"` emit `<asset><mesh file=.../></asset>` + `<geom type="mesh" mesh=.../>`.
   - For `asset` dict containing an `AssetFormat.MJCF` entry, emit `<include file="..."/>`.
   - For `LightSpec`, emit `<light .../>` in `<worldbody>`.
   - For `CameraSpec`, emit `<camera .../>` (note: this is in addition to the backend's auto-injected default camera; suppress the auto-injected one if user provides any).
   - For `TerrainSpec.kind="flat"`, emit the existing 2×2 plane sized from `terrain.size`; for `heightfield`, emit `<asset><hfield .../></asset>`.
   - For `world.gravity`, set on `<option gravity=.../>`.
3. **Implement `MuJoCoBackend.SupportsSceneEdit`**:
   - `get_prop_names`, `get_prop_pose` use `mj_name2id(BODY)` + `data.xpos`.
   - `set_prop_pose` writes to `data.qpos[freejoint_addr]` for free bodies, or `data.mocap_pos / mocap_quat` for mocap bodies. Decision: free-body for movable, mocap for "teleportable but otherwise simulated" — match `is_fixed=False` to free, `is_fixed=True` to mocap when randomization is requested, plain weld otherwise.
   - `set_prop_mass` writes `model.body_mass[id]`. Document this requires `mj_setM` recomputation; defer that until DomainRandomizer needs it.
4. **Implement `MuJoCoBackend.SupportsPhysicsRandomization`**:
   - `set_physics_params(gravity=..., friction=...)` mutates `model.opt.gravity` / `model.geom_friction`.
5. **Wire `DomainRandomizer`**: now non-no-op. Add an integration test that asserts a prop pose actually moves between two calls.
6. **Concrete tasks**:
   - Rebuild `PickPlaceTask` with `props=[PropConfig(name="cube", geom=GeomSpec("box",(0.025,0.025,0.025)), material=MaterialSpec(rgba=(1,0,0,1)))], success_fn`: cube z above threshold AND xy within target radius. Reward: distance(ee, cube) + bonus once grasped.
   - Same shape for `PourTask`, `PegInsertionTask`. Keep them simple but real.
7. **Asset library bootstrap**: `robodeploy/world/assets/objects/` containing 3–5 sample MJCF prop files (cube, cylinder, mug, peg, hole_plate). Mirror in URDF for ROS-side use.

After Phase 1, an end-to-end test exists that does not involve sensors: `RoboEnv.make(robot="franka", backend="mujoco", task="pick_place")`, run 100 steps, assert cube position changes.

### Phase 2 — Sensor stack rewrite (~5–7 days)

Goal: `Robot.sensors=[wrist_camera, ft_sensor]` actually produces populated `Observation.images["wrist"]`, `obs.ft_force`.

1. **Fix the lifecycle in `RoboEnv`** (`env.py:215-243`):
   ```python
   def _initialize_components(self):
       scene = self._merged_scene()
       self._backend.initialize_multi(self._robots, scene, self._shared_sensors)
       for sensor in self._all_sensors():
           sensor.initialize(self._backend)    # ← missing today
           sensor.warmup()
       ...
   ```
2. **Add `Observation.images: dict[str, ndarray]`** and `Observation.depths: dict[str, ndarray]` keyed by sensor name. Keep `rgb`/`depth` as single-sensor shorthand pointing at the primary camera. Update `Observation` to be immutable-by-convention (use `replace()` for merges).
3. **Add `ObsSpec.cameras: list[CameraRequest]`** where `CameraRequest = (name, width, height, fields={"rgb","depth"})`. Backends only render what is requested.
4. **Define `SensorMount`**:
   ```python
   @dataclass
   class SensorMount:
       parent_link: Optional[str] = None    # None = world frame
       position: tuple[float, float, float] = (0.0, 0.0, 0.0)
       orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
   ```
   `SensorBase.__init__` gains an optional `mount: SensorMount` argument. Backends use it to attach sim sensors to the correct link.
5. **Replace string-suffix pairing with explicit pairing metadata**:
   ```python
   # robodeploy/sensors/registry.py
   @register_sensor_pair("wrist_camera")
   class WristCameraPair:
       sim:  type[ISensor] = MuJoCoCameraSensor
       real: type[ISensor] = RealSenseCamera
       default_mount: SensorMount = SensorMount(parent_link="panda_hand")
   ```
   `RoboEnv.make` resolves `sensors=["wrist_camera"]` via this. Kill the `_sim`/`_real` suffix path.
6. **Concrete sim sensors** (use MuJoCo's built-in renderer; depend on `mujoco>=3.1`):
   - `MuJoCoCameraSensor`: uses `mujoco.MjvScene` + `MjrContext` headless rendering at the requested resolution, attached to the camera prim emitted by the scene builder.
   - `MuJoCoFTSensor`: reads `data.sensor[]` for a `<force>` / `<torque>` sensor element placed at the wrist by the description.
   - `MuJoCoIMUSensor`: reads `data.sensor[]` for accelerometer + gyro elements.
   - `MuJoCoTouchSensor`: reads `data.sensor[]` for `<touch>` sites placed on the gripper fingers.
7. **Concrete real sensors**:
   - `RealSenseCamera`: wrap `pyrealsense2.pipeline()`; hardware timestamp via `frame.get_timestamp()`; `timestamp_source = "hardware"`.
   - `Ros2Camera`: refactor `Ros2RgbdCameraSensor` to inherit `SensorBase` (keeps `Ros2NodeAdapter` as a delegate). Now passes through the unified `ISensor` registry; existing ROS topic-config path stays as an alternate constructor.
   - `AtiNetFT`: simple TCP/UDP NetFT client; warmup tares the sensor.
8. **Backend merge protocol**:
   ```python
   def _merge_sensor_data(obs: Observation, sensors: list[ISensor]) -> Observation:
       images, depths = dict(obs.images or {}), dict(obs.depths or {})
       ft_force, ft_torque = obs.ft_force, obs.ft_torque
       for s in sensors:
           sd = s.read()
           if sd.rgb is not None:   images[s.name] = sd.rgb
           if sd.depth is not None: depths[s.name] = sd.depth
           ft_force  = sd.ft_force  if sd.ft_force  is not None else ft_force
           ft_torque = sd.ft_torque if sd.ft_torque is not None else ft_torque
       return replace(obs, images=images, depths=depths,
                      ft_force=ft_force, ft_torque=ft_torque)
   ```
   Called inside each backend's `get_obs_multi()` and `step_multi()` after the proprioceptive `Observation` is built. Removes the ad-hoc mutation in `ros2/backend.py:312-316`.
9. **Wire `ObsPipeline.sync_policy`**: implement `DROP_LATEST` (default, current behavior) and `TIME_WINDOW` (drop a frame whose `timestamp_hw` falls outside the window of the latest proprioceptive `timestamp_hw`). Adds the `timestamp_source` widening rule.
10. **End-to-end demo**:
    ```python
    env = RoboEnv.make(
        robot="franka", backend="mujoco", task="pick_place",
        sensors=["wrist_camera", "overhead_camera"],
    )
    obs, _ = env.reset()
    assert obs.images["wrist"].shape == (480, 640, 3)
    assert obs.images["overhead"].shape == (480, 640, 3)
    ```

### Phase 3 — Multi-backend parity (~ongoing)

1. **`IsaacSimBackend`**: implement `WorldSpec` loading via `XFormPrim`/`RigidPrim` for primitives, `UsdGeom.Mesh` for mesh props, USD reference for `asset[AssetFormat.USD]`. Wire `SensorBase.initialize` to create `isaacsim.sensors.camera.Camera` prims using the sensor mount metadata. Implement `SupportsSceneEdit` via `prim.GetAttribute("xformOp:translate").Set(...)`.
2. **`ROS2GazeboBackend`**: extend `GazeboLauncher` to spawn one `<model>` per prop via `ros_gz_sim create` after world load (already partially present; needs prop iteration). Mount cameras as Gazebo sensors and bridge via `ros_gz_bridge`. The existing `Ros2RgbdCameraSensor` (now `Ros2Camera : SensorBase`) consumes the bridged topics; no separate code path.
3. **`ROS2RealBackend`**: scene loading is a no-op (props are physical). `SupportsSceneEdit.get_prop_pose` may be implemented by subscribing to a perception topic if configured. `set_prop_pose` raises (real props are not teleportable). Real sensors come from the same registry; configuration uses `RobotSensorSpec.real` so the user does not change code between sim and real.

### Phase 4 — Polish / hardening (optional)

- `RoboEnv.shared_sensors` wired through to backend (overhead/world cameras).
- DomainRandomizer extended to randomize `LightSpec.diffuse`, `MaterialSpec.rgba`, `MaterialSpec.friction` ranges.
- Reference `.rviz` config that subscribes to `/robodeploy/scene/markers` + each registered camera topic so RViz shows the new scene props live.
- `WorldSpec.from_yaml(path)` for declarative scenes outside Python.

---

## 3. Interface contract changes summary

| File | Change | Phase |
|---|---|---|
| `core/types.py` | Add `GeomSpec`, `MaterialSpec`, `LightSpec`, `CameraSpec`, `TerrainSpec`, `WorldSpec`; extend `PropConfig`; add `Observation.images/depths`; deprecate `ObjectSpec` & `SceneSpec.objects` | 0 |
| `core/interfaces/sensor.py` | Add `SensorMount`; document mount semantics in `initialize()` docstring | 2 |
| `core/interfaces/backend.py` | Promote `set_prop_pose`, `get_prop_pose`, `get_prop_names`, `set_prop_mass`, `set_payload`, `set_physics_params` from optional protocols to default-`NotImplementedError` methods on `IBackend` so capability detection works via `hasattr` + protocol check | 1 |
| `backends/base.py` | Drop `_sensors` storage; new `_merge_sensor_data` helper | 2 |
| `backends/sim/mujoco/scene_builder.py` | New file, ~200 lines extracted from `backend.py` | 1 |
| `backends/sim/mujoco/backend.py` | `_load` uses `MjcfSceneBuilder`; implement scene-edit + physics-randomization methods; merge sensor data into `Observation` | 1, 2 |
| `backends/sim/isaacsim/backend.py` | Implement `WorldSpec` loading + Isaac camera prims | 3 |
| `backends/sim/gazebo/backend.py` | Spawn props via `ros_gz_sim create`; bridge per-prop topics | 3 |
| `backends/real/ros2/backend.py` | `_load`-side: ingest `Robot.sensors` through the unified `ISensor` lifecycle instead of the `config.sensors` dict; keep config-dict as a backward-compatible alternate | 2 |
| `sensors/registry.py` | New explicit pairing decorator; remove suffix-coupling in `RoboEnv.make` | 2 |
| `sensors/camera/sim/mujoco_camera.py` | Concrete implementation using `mujoco.MjvScene` | 2 |
| `sensors/camera/real/realsense.py` | Concrete implementation using `pyrealsense2` | 2 |
| `sensors/ft_sensor/*` | Concrete implementations (`<force>` site for sim, NetFT/CANopen for real) | 2 |
| `obs_pipeline.py` | Implement `sync_policy` (`DROP_LATEST`, `TIME_WINDOW`) | 2 |
| `env.py` | Call `sensor.initialize(backend)` before `warmup()`; route `shared_sensors` to backend | 2 |
| `tasks/manipulation/*` | Rewrite with real props + reward shaping; use `ObsSpec.cameras` | 1, 2 |
| `world/assets/` | New directory with reference prop assets (MJCF/URDF/USD triples) | 1 |
| `tests/` | Add `test_scene_loading_mujoco.py`, `test_sensor_lifecycle.py`, `test_camera_sim.py`, `test_domain_randomization.py` | each phase |

---

## 4. Risks / open questions

1. **MuJoCo MJCF assembly via `ElementTree` already shows brittleness** (`mujoco/backend.py:178-324`). Phase 1 doubles the surface. If we hit limits, switch to `mjcf.RootElement` from `dm_control` (heavier dep) before XML grows further.
2. **`Observation` schema change is the only source-incompatible change** in Phase 0. Confirm acceptable. Alternative: leave `rgb`/`depth` as primary and stash multi-camera under `Observation.extra: dict` — keeps source compat at the cost of typing.
3. **Sensor pairing decorator** competes with the existing per-name registry. Migration path: keep `@register_sensor` working, add `@register_sensor_pair` as the higher-level idiom; sensor pairing entries internally re-register the legacy names so `Level 2 / Level 3` config paths still resolve.
4. **Real-hardware sensor reading from the same `ISensor` instance the inference loop uses** needs thread-safety review (multi-process bridge in `bridge.py`). For Phase 2 keep sensors single-threaded inside the inference process; the seqlock work in `bridge.py` is a separate prerequisite for production, not for adding the capability.
5. **USD path for IsaacSim** has been "planned" since at least the prior audit (`backends/sim/isaacsim/backend.py:74-85`); plumbing for it is partial. Phase 3 should not block on it — accept URDF-only Isaac initially and gate USD on a separate ticket.

---

## 5. Definition of done

A user can write the following and have it work end-to-end on MuJoCo, with all proprioception and sensor fields populated, and props that physically interact with the robot:

```python
env = RoboEnv.make(
    robot="franka",
    backend="mujoco",
    task="pick_place",
    sensors=["wrist_camera", "overhead_camera", "wrist_ft"],
    backend_kwargs={"enable_viewer": True},
)
obs, info = env.reset()
assert obs.images["wrist"].shape    == (480, 640, 3)
assert obs.images["overhead"].shape == (480, 640, 3)
assert obs.ft_force.shape           == (3,)
assert info.extra["multi_agent"].robot_states["robot0"].obs is obs

for _ in range(200):
    obs, reward, done, info = env.step()
    if done: break

assert info.success or info.failure   # cube manipulated or fell
```

Swapping `backend="mujoco"` → `backend="isaacsim"` and `backend="ros2"` should require no other code changes.

---

## 6. Other repo-wide issues (broader sweep)

Surveyed beyond the sensor/env focus. Not all of these block sensor/env work, but they are real defects, hazards, or spaghetti the user should know about. Grouped by area.

### 6.1 Real-time bridge — does not match `ARCHITECTURE.md`

`robodeploy/bridge.py` is documented as a multiprocessing seqlock-backed control bridge, but is not actually that.

- `RoboBridge.run()` runs `env.step()` in the calling async coroutine — the inference *and* hardware command path execute together in one thread. The "decoupled control + inference" diagram from `ARCHITECTURE.md` is not implemented.
- `_start_control_process()` does spawn a `multiprocessing.Process` and creates an `ActionTrajectory`, but `_control_process_entry` (bridge.py:241-257) **only reads** the latest joint positions from shared memory and discards them (`del _q`). It never calls `backend.step()` or any hardware driver. The control process is therefore a no-op heartbeat; all real hardware writes still happen from the main thread inside `env.step()`.
- `ControlLoop._loop` (the thread variant) calls `self._env.backend.step_multi(...)` in parallel with `RoboBridge.run()` also calling `env.step()` → two threads simultaneously call `backend.step_multi`, which the ROS2 driver is not documented to support. Race condition during `_thread.start()`/`run()` overlap.
- `EStopFlag`, `InferenceLoop`, watchdog, gravity-compensation table, ε velocity clamp, action-space-aware empty-buffer behavior, hard-pin e-stop — none implemented despite being described in detail in `ARCHITECTURE.md` §Real Hardware.
- `RoboBridge` accesses `self._control._hz` (bridge.py:186) — private attribute on its own composed component. `control_hz` property exists at line 196 right next door; just shadows accidentally.

### 6.2 `ActionTrajectory` seqlock has correctness gaps

`robodeploy/action_trajectory.py`:

- The seqlock pattern is well-formed in shape, but uses Python's `struct.pack_into` writes. CPython does not emit memory barriers between successive `pack_into` calls on `multiprocessing.SharedMemory`, so on weakly-ordered architectures the reader could observe an updated seq with stale `q` bytes. On x86-TSO this is fine in practice; on ARM-based dev machines (Apple Silicon, Raspberry Pi, Jetson) this is undefined.
- No spin timeout. `read_latest_joint_positions` loops `while True` and only breaks on a consistent read. Writer crash mid-write (seq stays odd) → reader spins forever. `ARCHITECTURE.md:154` specifically calls out a 500 µs spin timeout requirement.
- Time field uses `time.time()` (wall clock) inside the writer, not `time.monotonic()`. NTP slew while running can produce non-monotonic timestamps in the buffer.
- `write()` writes the header twice (once with odd seq+dof, once with even seq+dof) but the second write overwrites `wall_time_s` with a *fresh* `time.time()` — reader sees the second timestamp, not the timestamp at which payload was committed. Off by a few µs, harmless, but inconsistent with the documented protocol.
- Only `joint_positions` actions are supported. Velocity, torque, EE, gripper are silently dropped by `write()` (returns when `q is None`).
- `tests/test_action_trajectory.py` is one test (single-threaded, single-write/read). No cross-process test, no contention test, no torn-write test.

### 6.3 `IPolicy` interface has a duplicate definition

`robodeploy/core/interfaces/policy.py:120-127` defines `notify_rejected`. Lines 134-136 define `notify_rejected` again with the same no-op body. Last one wins — harmless functionally, but a clear copy-paste survivor. Same file: `action_hz` (line 129) is on `IPolicy` directly but listed under "Optional overrides", returns `0.0` default — meaning a policy that forgets to set it gets a divide-by-zero in any rate-adaptive caller.

### 6.4 `infer_action_space` is incomplete

`robodeploy/core/spaces.py:41-55` defines `infer_action_space()` claiming to "keep RoboEnv open/closed". It checks `joint_positions`, `joint_velocities`, `joint_torques`, `ee_position` — but `DELTA_EE` and `CARTESIAN_POSE` both populate `ee_position` and are indistinguishable. There is no way to mark an `Action` as "delta" vs "absolute" ee pose. Cartesian policies will mismap.

Also `Robot.infer_action_space` (`core/robot.py:260`) and `RobotTask.action_space()` (`core/robot.py:78-80`) are two different paths to discover the same fact; the latter ignores `infer_action_space()` entirely.

### 6.5 `SafetyFilter` Cartesian fallthrough

`robodeploy/kinematics/safety.py:79-90`: for `CARTESIAN_POSE` / `DELTA_EE`, the filter passes the action through **unchanged**. Comment says "backend-specific", but no backend implements Cartesian workspace clamping. A Cartesian policy can therefore command the EE outside the workspace with no clamping anywhere.

Additionally, `_filter_joint_pos` updates `self._prev_pos` to the *clamped* value, but `_filter_joint_vel` and `_filter_joint_torque` never update `self._prev_pos` — `_freeze_action()` will then freeze at a stale position the next time e-stop fires after a torque-mode episode. Mixed-mode usage is broken.

### 6.6 ROS2 controllers — three stubs and a long fork

`robodeploy/backends/real/ros2/controllers/`:

- `gripper.py` (10 lines): `raise NotImplementedError`.
- `joint_effort.py` (10 lines): `raise NotImplementedError`.
- `joint_velocity.py` (10 lines): `raise NotImplementedError`.

All three are decorated with `@register_controller` and look "supported" from the registry, but throw on first use. Either remove them or implement them.

`joint_trajectory.py` inherits from `joint_position.py` and overrides `_on_node_ready` and `_publish_joint_positions` — but **does not** call `super()._on_node_ready`, instead re-implementing the JointState subscription and TF listener with subtly different error handling (`except Exception` vs `except ImportError`). DRY violation that will diverge.

### 6.7 Backend god-classes still exist

Despite the prior split (`ROS2RealBackend` + `ROS2GazeboBackend`):

- `ROS2RealBackend.initialize_multi` is 178 lines (`real/ros2/backend.py:122-300`) and does: backend config parsing, fake-sim spawning, robot_state_publisher launching, per-robot controller wiring, sensor instantiation, RViz initialization, diagnostics setup. Each is a distinct lifecycle. Should be a sequence of `_setup_xxx` methods.
- `RoboEnv` is 522 lines doing routing + obs aggregation + viz payload build + diagnostics collection + arbitration-event consumption + episode counter wiring. The previous audit flagged it; still applies.
- `MuJoCoBackend._compile_mjcf_with_position_actuators` is 147 lines and synthesizes MJCF XML in addition to actuator wiring; will balloon further once scene props are added.

### 6.8 `is_real` honesty regression

`ROS2GazeboBackend(ROS2RealBackend)` sets `is_real = False` (class attribute). But `ROS2RealBackend` *also* declares `is_real = True` as a class attribute. Class-attribute inheritance with override works, but a subclass that quietly flips the parent's `is_real` is exactly the kind of trap `ARCHITECTURE.md §Sim-to-Real Data Flow` warns against. Worth documenting prominently or making the override an `__init_subclass__` check (e.g. require an explicit `is_simulated_via_ros: bool` field).

`backend_for_simulator("ros2_rviz", ...)` returns a `ROS2RealBackend` with `is_real = True` (`backends/simulator.py:208`). But "ros2_rviz" is described as a sim path — RViz visualization without real hardware. If a user enables `local_ros_graph=True` they get a fake joint sim driving the topics; the backend still reports `is_real=True`. Tasks/policies branching on `backend.is_real` will misroute.

### 6.9 Dead / placeholder components

- `robodeploy/policies/scripted/waypoint.py` — `raise NotImplementedError("WaypointPolicy placeholder only.")`
- `robodeploy/policies/scripted/joint_pd.py` — `raise NotImplementedError("JointPDPolicy placeholder only.")`
- `robodeploy/policies/learned/diffusion.py` — `raise NotImplementedError("DiffusionPolicy placeholder only.")`
- `robodeploy/policies/learned/vla.py` — `raise NotImplementedError("VLAPolicy placeholder only.")`
- `robodeploy/policies/remote/transport.py` — `GrpcTransport` is unconditionally `raise NotImplementedError` until proto stubs are generated. Listed in user-facing docs and `ARCHITECTURE.md` as "production". A user who picks it gets a runtime crash.

All five are registered via `@register_policy(...)` or instantiable from user code. They look supported via `list_registered()` and `RoboEnv.make()`. Either implement them, mark them `@register_policy("..._stub")`, or delete and let users implement their own.

### 6.10 `Observation` / `Action` are JAX-typed but rarely JAX-backed

`core/types.py:21-23`:
```python
try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp
```

Fields annotated `jnp.ndarray`. Some backends produce `jnp` (e.g. `_build_obs` in `mujoco/backend.py`), others produce raw NumPy (`joint_position.py:235-239`). `ros2/backend.py:312-316` later mutates `obs.rgb = sd.rgb` where `sd.rgb` is plain `np.ndarray`. The "zero-copy" claim in `core/interop.py` does not hold cross-backend, and type assertions in user code will fail intermittently. Document the actual contract: "ndarray-like, either NumPy or JAX, callers must coerce".

Also: `Observation.timestamp: float = 0.0` is mutable default (it's a dataclass scalar), but the dataclass is not frozen — `ros2/backend.py` mutates fields after construction. Decide: frozen + replace(), or document that mutation is allowed and document the ownership model.

### 6.11 ROS2 `Ros2RgbdCameraSensor.read()` loses hardware timestamp

`backends/real/ros2/sensors/camera_rgbd.py:144-154`:

```python
now = float(rgb_lv.wall_time_s or depth_lv.wall_time_s or 0.0)
return SensorData(
    rgb=rgb_lv.value,
    depth=depth_lv.value,
    timestamp=now,
    timestamp_hw=now,
    timestamp_recv=now,
)
```

`timestamp_hw` is set to host wall-clock time — the same value as `timestamp_recv`. The actual `msg.header.stamp` from the ROS message is discarded in `_on_rgb` / `_on_depth`. `timestamp_source` is never set (defaults to `"unspecified"`). Defeats the whole point of separate hardware/receive timestamps.

`LastValueCache.read()` uses `time.time()` (wall clock); like the ActionTrajectory issue, NTP slew can cause non-monotonic timestamps.

### 6.12 Default sensor and registry behavior

- `register_*` decorators raise `KeyError` on duplicate registration. Hot-reloading or running tests in the same process twice will fail with a confusing "already registered" error. Should either no-op when re-registering the same class object or include a `replace=True` flag.
- `register_sensor` / `register_policy` etc. have no unregister API. Tests that register fake components leak into other tests.
- `RoboEnv.make(...)` calls `SensorClass()` with no kwargs (`env.py:117`). Real sensors require config (resolution, port, framerate). Currently no way to pass it through `make()` other than going to Level 1 direct injection.

### 6.13 Gazebo launcher swallows failures

`backends/real/ros2/sim_launchers/gazebo.py`:

- `UrdfSpawner.spawn()` (`urdf_spawner.py:32-59`) runs `subprocess.run(..., check=False)` and never inspects the return code or stdout. If `ros_gz_sim create` fails (URDF malformed, world not ready), the call returns silently and Gazebo runs with no robot. Caller cannot tell.
- Same pattern at `gazebo.py:91-92`: `try/except: self._rsp = None` — robot_state_publisher launch errors swallowed.
- `RosGzBridgeLauncher.start()` only bridges `/clock` by default (`ros_gz_bridge.py:27`). No joint_states, no TF, no camera image — minimal usability.
- `time.sleep(0.5)` at `gazebo.py:82` is the only readiness signal — Gazebo Harmonic on a cold start can take 5+ s.

These survived the prior audit because the recommendation was "add readiness gates"; they were not implemented.

### 6.14 Tests are very thin

`tests/`:

- `test_action_trajectory.py` — 1 single-process test.
- `test_behavior_profile.py` — covers behavior preset math but does not actually start any backend.
- `test_env_refactor.py` — uses a `DummyBackend` that does not exercise scene loading, sensors, or any real backend.
- `test_so101_real.py` — calibration roundtrip + helper checks; the only "hardware smoke" gate skips unless `ROBODEPLOY_SO101_PORT` is set.
- `test_so101_urdf.py` (not read) — URDF parsing only.

Missing test categories: backend physics sanity, sensor read-after-init, ObsPipeline+NormalizeTransform end-to-end, domain randomization causes a state change, RoboBridge actually decouples control from inference, SafetyFilter shape rejection, multi-robot isolation, IK convergence, KinematicsSolver.plan(), Gazebo readiness gates.

### 6.15 `RobotDescription` variant arg is dead

`description/base.py` documents `variant: "default" | "sim" | "real"` for `asset_path`. `FrankaDescription.asset_path` does `del variant` — ignored. `KukaDescription`, `SO101Description`, `URDFRobotDescription` all do the same. The MuJoCo backend passes `variant="sim"` (`mujoco/backend.py:63`) — silently ignored. Either delete the parameter or wire it through to the asset directory layout (`assets/mjcf/sim/`, `assets/mjcf/real/`).

### 6.16 `KinematicsSolver.plan()` silent straight-line fallback

`kinematics/solver.py:194-220`: `plan(q_start, q_goal)` returns a straight-line joint-space interpolation. Comment says "real deployments should replace this with a collision-aware planner". `Arbitrator` and any future task-switching path will silently use this — straight-line in joint space can drive through singularities and through scene props with no warning. `ARCHITECTURE.md:380-394` requires OMPL/MoveIt2.

At minimum: log a warning the first time `plan()` is called and the default fallback is in use. Better: raise unless `obstacles is None` (i.e., explicit "I know I'm using the dumb interpolator").

### 6.17 Pervasive `except Exception: pass`

Grep across `robodeploy/`: 109 bare-exception swallows across 26 files. Concentrations:

- `backends/sim/isaacsim/backend.py` (17) — physics init failures, articulation init failures, RViz failures.
- `backends/real/ros2/backend.py` (10) — driver stops, sensor stops, RViz stop, launcher stop.
- `backends/real/ros2/controllers/so101_feetech.py` (17) — hardware lifecycle.
- `backends/sim/mujoco/backend.py` (13) — viewer launch, viz publish, MJCF cache write.
- `bridge.py` (4) — control proc spawn / shutdown.

Most of these swallow real errors. Even where the intent is "best-effort cleanup", they should call `warnings.warn(...)` so the user sees something during development. Current behavior is silent malfunctions.

### 6.18 `core/interop.py` "zero-copy" is misleading

Top-level docstring claims "Bridge: JAX is handed to PyTorch Policy via to_torch() (Zero-copy)". Implementation `to_torch()`:
```python
if jnp is not None and isinstance(data, jnp.ndarray):
    data = np.array(data)        # ← this is a copy
return torch.from_numpy(data)
```

`np.array(jax_array)` is a host copy; `torch.from_numpy(np_array)` is the only zero-copy step. JAX → torch is not zero-copy via this path. Either implement DLPack-based `to_torch` (`torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))`), or correct the documentation.

### 6.19 Examples directory uses the placeholder stack

`examples/franka_robomimic_demo.py` and `examples/franka_sim_viewer_demo.py` exist as user references but depend on the same placeholder pieces (empty `scene_spec`, `MuJoCoCameraRenderer` raises `NotImplementedError`, etc.). New users following the README arrive at non-functional demos.

The `examples/so101/run_switch_simulator.py` path is more complete (uses `backend_for_simulator`) and likely works on real SO-101 hardware via Feetech. But that is also the only fully-working example in the repo, despite Franka being the headline robot.

### 6.20 Minor / smaller issues

- `RemotePolicy.__init__` does not call `super().__init__(action_space=...)` consistently with later code paths — `_episode_count` is initialized but `self._transport` could be `None` at `close()` if construction failed mid-way. Defensive `close()` is missing.
- `Robot.task_action_resolver` falls back to "last candidate wins" when called (`core/robot.py:222`) — silently. Multi-task users may not realize their resolver is the no-op default.
- `RoboEnv.make()` requires `policy` (`env.py:104-108`) but the docstring elsewhere implies you can construct with `policy=None` for external action injection. Mismatch.
- `BackendBase.config` merges nested `{"config": {...}}` keys but only one level deep — three-level nesting like `config={"sim": {"config": {...}}}` is not handled.
- `ROS2RealBackend.step_multi` requires `len(actions) == len(self._drivers)` (`backend.py:329-331`) — but `RoboEnv.step` always passes one action per `Robot`, never per-driver. Multi-driver-per-robot setups will mismatch.
- `Ros2Runtime.shutdown()` (`runtime.py:71-91`) shuts down the entire `rclpy` context — destroying it inside one of multiple test files would tear down rclpy for everything else in the same Python process.
- `IsaacSimBackend._reset_impl` catches `Exception` and silently returns the existing `_build_obs()` (`isaacsim/backend.py:328-329`) — a physics-restart failure on reset returns "looks fine" obs while the simulator is in an unknown state.
- `FakeJointPosSim` `_publish_loop` runs in a thread and writes joint state from outside the ROS executor — works because rclpy publishers are threadsafe, but the contract is not documented.

---

## 7. Updated priority recommendation

To the existing Phase 0–4 plan (§2), add as out-of-band cleanup tickets:

| Issue | Severity | Phase to bundle |
|---|---|---|
| §6.1 bridge does not decouple | **P0 (correctness)** — any real-hardware test will be unsafe | separate ticket before Phase 2 |
| §6.2 ActionTrajectory seqlock gaps | **P0** — would not flag during testing, only under load | bundle with bridge work |
| §6.3 duplicate `notify_rejected` | trivial | Phase 0 cleanup |
| §6.4 ActionSpace inference incomplete | **P1** | Phase 0 |
| §6.5 SafetyFilter Cartesian / prev_pos | **P1** — silent hazard | Phase 0 |
| §6.6 three stub ROS2 controllers | P2 | discrete ticket; don't ship as registered |
| §6.7 god-classes | **P1** — blocks Phase 1/2 work | Phase 1 prep |
| §6.8 `is_real` regression | **P1** | Phase 0 |
| §6.9 placeholder policies | P2 — unregister or implement | Phase 4 |
| §6.10 Observation typing inconsistency | **P1** | Phase 0 (frozen + replace()) |
| §6.11 Ros2 RGBD hardware timestamp lost | **P1** | Phase 2 (sensor rewrite) |
| §6.12 registry replace API | P3 | low-priority |
| §6.13 Gazebo launcher failure swallowing | **P1** | Phase 3 |
| §6.14 thin tests | **P1** ongoing | each phase adds tests |
| §6.15 dead `variant` arg | P3 — decide and remove or wire | Phase 0 |
| §6.16 `KinematicsSolver.plan` silent fallback | **P1** | discrete ticket |
| §6.17 except-Exception sprawl | P2 — convert to `warnings.warn` | gradual |
| §6.18 `interop.py` zero-copy claim | trivial doc fix | Phase 0 |
| §6.19 examples not runnable | P2 | After Phase 2 sensors work |
| §6.20 misc | each small | as touched |

**Effective P0 stack** (must clear before any real-hardware deployment):

1. Implement actual decoupled `RoboBridge` (multiprocessing inference + control) — §6.1.
2. Fix `ActionTrajectory` seqlock spin timeout, memory ordering on weak archs, monotonic clock — §6.2.
3. Fix `SafetyFilter` mixed-mode state + Cartesian clamp — §6.5.
4. Drop the silent `KinematicsSolver.plan` fallback (raise or warn) — §6.16.
5. Honest `is_real` everywhere — §6.8.

After P0, the sensor + 3D environment plan in §2 becomes safe to land incrementally without piling on hazards.
