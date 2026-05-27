# RoboDeploy — Backend Sensor & 3D-Environment Evaluation

**Date**: 2026-05-12
**Scope**: How well each backend handles sensors and 3D environments across simulation and real-hardware deployment. Concrete next-step plan follows the scorecard.
**Companion docs**: `AUDIT_REPORT.md`, `SENSORS_AND_ENV_PLAN.md`, `DEFECT_CATALOG.md`.

---

## 1. Scorecard

Rating per capability per backend. Scale: ❌ none / ⚠ stub or partial / ◐ partial-but-usable / ✅ working.

### 1.1 Sensors

| Backend | Sim cameras (RGB/depth) | Real cameras | FT sensor (sim) | FT sensor (real) | IMU | Touch / contact | Hardware timestamps | Sensor lifecycle wired | Multi-camera schema |
|---|---|---|---|---|---|---|---|---|---|
| `MuJoCoBackend` | ❌ | n/a | ❌ | n/a | ❌ | ❌ | n/a | ❌ | ❌ |
| `IsaacSimBackend` | ❌ | n/a | ❌ | n/a | ❌ | ❌ | n/a | ❌ | ❌ |
| `ROS2GazeboBackend` | ⚠ (via Ros2 RGBD topic, no spawn) | n/a | ❌ | n/a | ❌ | ❌ | ⚠ (drops `msg.header.stamp`) | ⚠ (config-dict path only) | ❌ |
| `ROS2RealBackend` | ⚠ (RGBD topic) | ⚠ (subscribe only) | ❌ | ❌ | ❌ | ❌ | ⚠ (same loss) | ⚠ (config-dict path only) | ❌ |

### 1.2 3D environments / scenes

| Backend | Loads `SceneSpec.props` | Procedural primitives (box/cyl/sphere) | Mesh prop import | Per-prop material/color/friction | Per-prop pose query/set | Per-prop mass set | Lighting | Terrain | Domain randomization wired |
|---|---|---|---|---|---|---|---|---|---|
| `MuJoCoBackend` | ❌ (scene discarded) | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠ (one auto-injected light) | ⚠ (one auto-injected 2×2 plane) | ❌ |
| `IsaacSimBackend` | ❌ (scene → RViz only) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠ (default ground plane) | ❌ |
| `ROS2GazeboBackend` | ❌ (scene → RViz only) | ❌ | ⚠ (single `robot_urdf` spawn via `ros_gz_sim create`, not props) | ❌ | ❌ | ❌ | ⚠ (from world SDF) | ⚠ (from world SDF) | ❌ |
| `ROS2RealBackend` | n/a (real props are physical) | n/a | n/a | n/a | ⚠ (would need perception) | n/a | n/a | n/a | n/a |

### 1.3 Sim-to-real parity

| Capability | Sim has it | Real has it | Same code path? |
|---|---|---|---|
| Proprioception (joint pos/vel/torque) | ✅ MuJoCo, ⚠ Isaac (zero torques), ⚠ ROS2Gz | ✅ ROS2Real | ✅ via `Observation` |
| EE pose | ✅ MuJoCo, ⚠ Isaac (zero), ⚠ ROS2Gz | ✅ ROS2Real (TF lookup with identity-on-failure) | ✅ |
| RGB camera | ❌ | ⚠ (ROS2 topic) | ❌ (different registry) |
| Depth camera | ❌ | ⚠ (ROS2 topic) | ❌ |
| FT sensor | ❌ | ❌ | ❌ |
| Scene props | ❌ | n/a | ❌ |
| Domain randomization | ❌ (no `teleport_object` impl) | n/a | ❌ |

**Bottom line**: every cell that lights up green is proprioception. Everything sensor-related and everything scene-related is either ❌ or ⚠ that does not actually run.

---

## 2. Per-backend deep evaluation

### 2.1 `MuJoCoBackend` — sim

Location: `robodeploy/backends/sim/mujoco/backend.py` (429 lines).

**Sensors — capability: none.**

- `_load()` discards the sensor list outright: `del scene; del sensors` (line 48-49). `Robot.sensors` never reaches MuJoCo.
- No call site invokes `sensor.initialize(backend)` for sim sensors. The MuJoCo backend exposes `self._model`, `self._data`, and `self._mujoco` as private members; even a user who manually instantiated `MuJoCoCameraRenderer` has no documented API to attach it.
- `Observation._build_obs` (line 400-428) returns proprioception only. RGB/depth/FT/IMU fields are never populated.
- The two registered "sim sensor" classes (`MuJoCoCameraRenderer`, `MuJoCoFTSensor`) raise `NotImplementedError` in `_init_impl` and `_read_impl`.

**3D environments — capability: implicit empty tabletop only.**

- `SceneSpec` parameter is `del`'d in `_load`. Props, lighting beyond default, terrain, world cameras — all dropped.
- The backend *does* synthesize a minimal MJCF when only a URDF is provided (`_compile_mjcf_with_position_actuators`, line 178-324): adds one light, one 2×2 floor plane, one camera. That is the entirety of "environment" — fixed and unconfigurable except via the `urdf_*` config keys.
- No implementation of `set_prop_pose`, `get_prop_pose`, `get_prop_names`, `set_prop_mass`, `teleport_object`, or `set_physics_params`. `DomainRandomizer` silently no-ops.
- RViz scene markers *are* published when `rviz.enabled=true` (line 173-176) — cosmetic only, no physics.

**What's salvageable**:

- The URDF → MJCF compilation path is real and works (URDF in, runnable MuJoCo model out, with auto-injected actuators).
- Position-actuator wiring is solid.
- The `MjcfSceneBuilder` extraction recommended in `SENSORS_AND_ENV_PLAN.md` would replace ≈ 150 lines of inline XML synthesis with a unit-testable composer and unlock prop / camera / light injection in the same pipeline.

**Score: sensors 0/10, environments 1/10.**

### 2.2 `IsaacSimBackend` — sim

Location: `robodeploy/backends/sim/isaacsim/backend.py` (429 lines).

**Sensors — capability: none.**

- `_load()` stores `self._sensors = list(sensors)` (line 68) but never calls `sensor.initialize(...)` and never merges sensor data into `Observation`.
- `_build_obs` (line 396-428) is proprioception-only. RGB/depth/FT/IMU all left as `None`.
- No use of Isaac's native sensor types (`isaacsim.sensors.camera.Camera`, RTX lidar, IMU).
- `shared_sensors` raises `NotImplementedError` outright (line 61-62).

**3D environments — capability: ground plane + URDF robot only.**

- `_create_world` (line 192-205) adds Isaac's default ground plane, then the URDF robot. SceneSpec props are not loaded into the USD stage.
- The `usd_prefer` / `usd_fallback_to_urdf` flag pair (line 76-85) is misleading — even when `usd_prefer=True` and USD exists, the backend appends to `_warnings` and falls back to URDF. There is no working USD code path.
- Article-style RViz scene publishing (line 110-113) is cosmetic; Isaac's own stage is not modified to match.
- No `SupportsSceneEdit` implementation; `set_prop_pose` not available.
- `_reset_impl` catches `Exception` and returns the prior `_build_obs()` (line 328-329) — physics-restart failures hidden, sim left in unknown state.
- Hardcoded `"robot0"` prim path and articulation name (line 333, 343, 365) — multi-robot Isaac is impossible without source edits.

**What's salvageable**:

- The Kit launch + URDF import path works (when Isaac is installed). The lazy-import pattern is correct.
- World creation hooks (`_create_world`, `_ensure_physics_ready`) are stable extension points.

**Score: sensors 0/10, environments 2/10.**

### 2.3 `ROS2GazeboBackend` — sim via ROS2 transport

Location: `robodeploy/backends/sim/gazebo/backend.py` (59 lines) + `backends/real/ros2/sim_launchers/gazebo.py` (160 lines).

**Sensors — capability: partial via parallel `IRos2Sensor` registry.**

- The same `ROS2RealBackend.initialize_multi` codepath handles sensors. It reads per-robot config (e.g. `robot0.sensors`) and instantiates `Ros2RgbdCameraSensor` for `type: "rgbd"`. RGB and depth flow over real ROS topics from Gazebo's `ros_gz_image` plugin.
- These sensors do **not** implement `ISensor` and cannot be passed via `Robot.sensors` — only via config dict.
- `msg.header.stamp` is discarded (`camera_rgbd.py:144-154`); `timestamp_hw` set to host wall-clock.
- FT / IMU / touch / lidar — no ROS sensor adapters registered.

**3D environments — capability: world SDF loads but props don't.**

- `GazeboLauncher` spawns one URDF model via `ros_gz_sim create` (`urdf_spawner.py`). That's the *robot* — not scene props. Scene props would need additional `create` calls per-prop, which are not wired.
- The world SDF (passed via `config.sim.world`) is opaque to RoboDeploy. Whatever Gazebo finds in it appears; nothing in `SceneSpec` flows in.
- `ros_gz_bridge` defaults bridge only `/clock` (`ros_gz_bridge.py:27`). No `/joint_states`, no TF, no camera image unless the user passes explicit `bridge_rules`. The Kuka example does pass these (`gazebo_ros2_extra_config`), but other users have to learn the convention.
- `time.sleep(0.5)` is the only readiness gate before declaring Gazebo ready (`gazebo.py:82`). Gazebo Harmonic cold-start is ~2–5 s.
- `UrdfSpawner.spawn()` runs `subprocess.run(check=False)` and does not inspect return code (`urdf_spawner.py:32-59`). If Gazebo refuses the model, world has no robot and caller cannot tell.
- `ROS2GazeboBackend.is_real = False` overrides parent's `is_real = True` via class-attribute inheritance — brittle.

**What's salvageable**:

- Subprocess lifecycle helpers (`GazeboLauncher.start/stop`) are functional.
- `wait_for_topics` polling exists for explicitly-named topics (`gazebo.py:113-129`).
- `Ros2RgbdCameraSensor` is genuinely useful — the only working camera path in the entire repo.

**Score: sensors 4/10, environments 3/10.**

### 2.4 `ROS2RealBackend` — real hardware

Location: `robodeploy/backends/real/ros2/backend.py` (438 lines).

**Sensors — capability: working via parallel registry, but not via `ISensor`.**

- Same as Gazebo path: `Ros2RgbdCameraSensor` works when the ROS graph publishes the topics. `Robot.sensors` is ignored.
- `step_multi`/`reset_multi` mutate `obs.rgb`/`obs.depth` directly on returned `Observation` (line 312-316, 348-354) — frozen-by-convention dataclass mutated, no merge protocol.
- FT / IMU / encoder-loaded current — no real-hardware adapters registered.
- `_get_ee_pose_from_tf` returns identity pose `(0,0,0,1)` on TF lookup failure (`joint_position.py:214-226`). Downstream sees plausible-looking EE pose that is wrong.
- Joint-state stamps are read (`joint_position.py:159-164`) and exposed via `last_joint_state_stamp_s` diagnostic, but `Observation.timestamp_hw` is **not** set from them — it stays default 0.0.

**3D environments — capability: not applicable.**

- Real props are physical. The backend correctly does not try to teleport them.
- `set_prop_pose` would need a perception subsystem (e.g. AprilTags, ChArUco, OptiTrack) to make sense. None exists.
- `get_prop_pose` could be implemented by subscribing to a fiducial topic; not implemented.

**What's salvageable**:

- Per-robot namespacing (`/robot0`), joint-state subscription, and command publishing are mature.
- Safety primitives (`Watchdog`, `EStop`, `JointLimitGuard`, `TemperatureGuard` — `safety.py`) are real, working, well-tested.
- `Commander` + `slew_limit_command` give proper command pacing.
- SO-101 Feetech path (`so101_feetech.py`) is the most complete robot driver in the repo.

**Score: sensors 3/10, environments n/a (real world).**

---

## 3. Cross-cutting gaps

### 3.1 No unified sensor contract

Two parallel sensor systems coexist:

```
robodeploy/sensors/*                              robodeploy/backends/real/ros2/sensors/*
├── ISensor / SensorBase                          ├── IRos2Sensor (Protocol)
├── @register_sensor                              ├── @register_ros2_sensor
└── all stubs (NotImplementedError)               └── Ros2RgbdCameraSensor (works)
```

User code paths:

- `Robot(sensors=[...])` → flows through `RoboEnv` → backends ignore.
- `backend_kwargs={"robot0.sensors": [{"type": "rgbd", ...}]}` → only path that actually instantiates a working sensor.

These never meet. The "sim/real pairing" advertised in `ARCHITECTURE.md` does not exist.

### 3.2 No common scene-loading API

Each backend would need its own scene loader. Currently zero have one. The shared contract is `IBackend.initialize(description, scene, sensors)` but only `description` is consumed.

`SupportsSceneEdit` protocol is declared (`backends/capabilities.py:37-42`). Zero implementations.

### 3.3 No sensor mount metadata

`ISensor.initialize(backend)` is the only attach hook. There is no `SensorMount(parent_link, pose)` field anywhere. Backends therefore have no way to know:

- Which robot link a wrist camera attaches to.
- The camera's pose relative to that link.
- Camera intrinsics (FOV, resolution, distortion) for sim rendering.

The `ARCHITECTURE.md` spec mentions `intrinsics()` / `extrinsics()` methods — these are not on `ISensor`.

### 3.4 No multi-camera schema

`Observation.rgb: Optional[ndarray]` is single-camera. A wrist + overhead + depth setup cannot be represented without per-step dict mutation that escapes the type system.

### 3.5 No environment-side cameras (third-person / evaluation)

`RoboEnv.shared_sensors: list[ISensor]` exists as the documented home for overhead / world cameras. Every backend either rejects non-empty `shared_sensors` (MuJoCo, Isaac) or ignores them silently (ROS2). The path is dead.

### 3.6 No timestamp discipline

- `SensorData.timestamp_hw` and `timestamp_recv` are never populated meaningfully:
  - MuJoCo (if anyone wired it) would use sim time.
  - ROS2 RGBD discards `msg.header.stamp`.
- `SensorData.timestamp_source` (e.g. `"hardware"`, `"software"`, `"sim"`) is never set.
- `ObsPipeline.SyncPolicy` (`DROP_LATEST`, `TIME_WINDOW`) is enum-only; `sync_policy()` method on the pipeline is a stub.

Without these, sensor-fusion policies and the `TIME_WINDOW` widening rule cannot run.

### 3.7 No domain randomization that actually randomizes

`DomainRandomizer` calls `backend.teleport_object()` and `backend.set_physics_params()` inside `try/except NotImplementedError: pass`. Every backend raises. So `DomainRandomizer` is global no-op. No warning to the user.

---

## 4. What needs to happen — concrete next steps

Ordered so each step unblocks the next. Each item is small enough to be one PR.

### Step 1 — Stop the silent failures (1 day)

These are the highest signal-to-noise fixes. Each is a behaviour change, not a feature.

1. **Fail loudly when a sensor is dropped**: `MuJoCoBackend._load` and `IsaacSimBackend._load` raise `NotImplementedError("Backend X does not handle ISensor instances yet")` instead of `del sensors`. Same for `ROS2RealBackend.initialize_multi` when `Robot.sensors` is non-empty. Users discover they need the config-dict path *immediately*, not on step 1000 of an RL run.
2. **Fail loudly on `DomainRandomizer` no-op**: `tasks/randomization.py:135-137` and `:155-157` change `except NotImplementedError: pass` → `warnings.warn(..., stacklevel=2)` once per backend.
3. **Fail loudly when scene props are dropped**: `MuJoCoBackend._load` and `IsaacSimBackend._load` check `len(scene.props) + len(scene.objects) > 0` and `warnings.warn` if non-empty (since they can't load them yet).
4. **Honest `is_real`**: `backend_for_simulator("ros2_rviz", local_ros_graph=True, ...)` returns a backend whose `is_real` reports `False`. Add an explicit `is_simulated_transport: bool` to `IBackend` so subclasses can declare "real protocol, sim physics" without lying.

Outcome: every silent failure flagged in `DEFECT_CATALOG.md §16` becomes a visible warning or error. No new capability yet, but the gaps stop hiding.

### Step 2 — Schema unification (1 day)

Prerequisites for everything that follows. Source-compatible.

1. **`PropConfig` extension** (additive — defaults preserve old behaviour):
   ```python
   @dataclass
   class GeomSpec:
       kind: Literal["box", "cylinder", "sphere", "capsule", "mesh"]
       size: tuple[float, ...]
       mesh_path: Optional[str] = None

   @dataclass
   class MaterialSpec:
       rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
       friction: tuple[float, float, float] = (1.0, 0.005, 1e-4)
       texture: Optional[str] = None

   @dataclass
   class PropConfig:
       # existing fields preserved
       geom: Optional[GeomSpec] = None
       material: MaterialSpec = field(default_factory=MaterialSpec)
       asset: Optional[dict[AssetFormat, str]] = None
       parent_link: Optional[str] = None
       joint_type: Literal["free", "fixed"] = "free"
       inertia_diag: Optional[tuple[float, float, float]] = None
   ```
2. **Introduce `WorldSpec`** wrapping props + lights + cameras + terrain + gravity. `SceneSpec` becomes a thin adapter holding a `WorldSpec` + the legacy `table_height` / `lighting` for back-compat.
3. **Introduce `SensorMount`** + add optional `mount: SensorMount | None` to `SensorBase.__init__`.
4. **Expand `Observation`**: add `images: dict[str, ndarray]`, `depths: dict[str, ndarray]`. Keep flat `rgb`/`depth` as primary-camera alias.
5. **Expand `ObsSpec`**: add `cameras: list[CameraRequest]`. Each request carries (name, width, height, modalities).
6. **Drop dupes**: kill duplicate `IPolicy.notify_rejected` definition (`core/interfaces/policy.py:134-136`). Deprecate `ObjectSpec`.

Outcome: types are in place. No backend behaviour changes yet.

### Step 3 — MuJoCo scene loader (3 days)

The most-used sim backend gets real scenes first.

1. **Extract `MjcfSceneBuilder`** from `MuJoCoBackend._compile_mjcf_with_position_actuators` into `backends/sim/mujoco/scene_builder.py`. Move actuator injection, inertia clamping, default light/floor/camera. ~200 lines out of `backend.py`.
2. **Implement `MjcfSceneBuilder.attach_world(world: WorldSpec)`**:
   - `GeomSpec("box", (sx, sy, sz))` → `<body><geom type="box" .../><freejoint/></body>`.
   - `GeomSpec("mesh", (), mesh_path=...)` → `<asset><mesh file=.../></asset>` + `<geom type="mesh" .../>`.
   - `PropConfig.asset[AssetFormat.MJCF]` → `<include file=.../>`.
   - `LightSpec` → `<light/>` in `<worldbody>`.
   - `TerrainSpec("heightfield", ...)` → `<asset><hfield .../></asset>` + plane geom.
   - `world.gravity` → `<option gravity=.../>`.
3. **Implement `MuJoCoBackend.SupportsSceneEdit`**:
   - `get_prop_names` → list of body names registered as props.
   - `get_prop_pose(name)` → read `data.xpos[id]` + `data.xquat[id]`.
   - `set_prop_pose(name, pose)` → write to `data.qpos[freejoint_addr]` for free bodies, `data.mocap_*` for fixed-but-teleportable.
   - `set_prop_mass(name, mass)` → write `model.body_mass[id]`. Call `mj_setM` on next reset.
4. **Implement `MuJoCoBackend.SupportsPhysicsRandomization`**: `set_physics_params(gravity=..., friction=...)` writes `model.opt.gravity`, `model.geom_friction`.
5. **DomainRandomizer non-no-op test**: assert prop position differs between two `reset()` calls when `RandomLevel.LIGHT`.

Outcome: MuJoCo loads scenes, props interact with the robot, randomization works.

### Step 4 — MuJoCo sensor stack (3 days)

1. **Scene-builder camera emission**: when `Robot.sensors` includes a camera, append `<camera name="..." pos=... xyaxes=..."/>` in the appropriate body (parent_link from `SensorMount`).
2. **`MuJoCoCameraSensor` real implementation** using `mujoco.MjvScene` + `mujoco.MjrContext` offscreen renderer at the requested resolution. Output `SensorData(rgb=..., depth=..., timestamp=data.time, timestamp_source="sim")`.
3. **`MuJoCoFTSensor`** reads `data.sensor[]` entries for `<force>`/`<torque>` elements at the wrist site. SceneBuilder injects the elements at robot composition time.
4. **`MuJoCoTouchSensor`** reads `<touch>` site data at gripper fingers.
5. **`MuJoCoIMUSensor`** reads accelerometer + gyro site data.
6. **Wire sensor lifecycle in `RoboEnv._initialize_components`**:
   ```python
   for sensor in self._all_sensors():
       sensor.initialize(self._backend)
       sensor.warmup()
   ```
7. **Backend-side merge**: new helper `_merge_sensor_data(obs, sensors) -> Observation` called from `step_multi` and `get_obs_multi`. Returns a new `Observation` via `dataclasses.replace`.
8. **Pairing registry**: `@register_sensor_pair("wrist_camera")` returns `(MuJoCoCameraSensor, RealSenseCamera, default_mount)`. `RoboEnv.make` picks sim or real by `backend.is_real`.

Outcome: `obs.images["wrist"]` populated on MuJoCo. End-to-end sensor path proven on one backend.

### Step 5 — Real-hardware sensor path (2 days)

1. **`RealSenseCamera` real implementation** (`sensors/camera/real/realsense.py`). Use `pyrealsense2.pipeline`. Hardware timestamps via `frame.get_timestamp()`. Set `timestamp_source="hardware"`.
2. **Refactor `Ros2RgbdCameraSensor` to subclass `SensorBase`** (delegates to existing `Ros2NodeAdapter`). Preserve `msg.header.stamp` → `timestamp_hw`. Register as `wrist_camera_real` so the pairing works.
3. **`ROS2RealBackend.initialize_multi` consumes `Robot.sensors`**. Keep config-dict path as alternate constructor for one release.
4. **`AtiNetFT` real FT sensor** via NetFT UDP. Warmup tares.
5. **Sync policy implementation in `ObsPipeline`**:
   - `DROP_LATEST` (default, current behaviour).
   - `TIME_WINDOW(window_ms=15.0)` — drop sensor reads whose `timestamp_hw` is outside ±window of the proprioceptive `timestamp_hw`. Widen window when `timestamp_source == "software"`.

Outcome: same `Robot(sensors=[...])` works on MuJoCo and on real hardware. Sim-to-real for perception is real.

### Step 6 — Isaac + Gazebo parity (3 days each)

Done last because they unlock advanced features but are not on the critical path.

**Isaac**:

1. `WorldSpec` loading: `XFormPrim` + `RigidPrim` for primitives, `UsdGeom.Mesh` for meshes, USD reference for `asset[AssetFormat.USD]`.
2. Camera prims via `isaacsim.sensors.camera.Camera` at sensor mount poses.
3. `SupportsSceneEdit` via `prim.GetAttribute("xformOp:translate").Set(...)`.
4. Un-hardcode `"robot0"` prim path (`/World/<robot_id>`).
5. Drop `_reset_impl` exception swallowing — re-raise with context.
6. Implement actual USD prefer path (currently always falls back).

**Gazebo**:

1. Spawn one `<model>` per prop after world load. Need `ros_gz_sim create` per prop, or pre-merge SDF.
2. Add per-camera bridge rules so camera topics auto-bridge.
3. Replace `time.sleep(0.5)` readiness with `/clock` topic poll + `controller_manager/list_controllers` service check.
4. Inspect `subprocess.run` return codes; raise with readable diagnostics.
5. Sensor data flows through unified `ISensor` path (reusing the refactored `Ros2RgbdCameraSensor`).

Outcome: same task definition runs on MuJoCo, Isaac, Gazebo, and Real — only the backend constructor changes.

### Step 7 — Real-hardware perception loop for scene state (stretch, 5 days)

Optional but enables sim-to-real for tasks that require knowing where props are on real hardware.

1. New `ScenePerception` interface independent of any backend:
   ```python
   class IScenePerception(ABC):
       def get_prop_pose(self, name: str) -> Optional[Pose]: ...
       def get_prop_names(self) -> list[str]: ...
   ```
2. `AprilTagPerception` implementation subscribing to `/apriltag_detections`.
3. `ROS2RealBackend.get_prop_pose(name)` delegates to attached `IScenePerception` (None → `NotImplementedError`).
4. `set_prop_pose` on real backend raises `RuntimeError("real props cannot be teleported — raise HumanInterventionRequired in your task reset_routine instead")`.

Outcome: real-hardware tasks can read prop state where sim teleports.

---

## 5. What does *not* need to happen

Time-box explicitly. Tempting but out of scope for sensor+env:

- Implementing `WaypointPolicy`/`JointPDPolicy`/`DiffusionPolicy` placeholders. These are policy concerns; orthogonal.
- Implementing `GrpcTransport`. Distributed inference is separate.
- `KinematicsSolver.plan()` OMPL integration. Required for `Arbitrator.switch()` but not for sensors/env.
- `RoboBridge` multiprocess decoupling. Required for production real-hardware but does not block sensors or scenes.
- USD-first Isaac. URDF works for Phase 1; defer USD until Isaac has a real user.

These are all in `AUDIT_REPORT.md` / `DEFECT_CATALOG.md` and should be tracked there.

---

## 6. Definition of done — one acceptance test

A single test exercises every capability touched by this plan:

```python
# tests/test_sensors_and_scenes_e2e.py

def test_sensor_and_scene_capabilities_on_mujoco():
    from robodeploy import RoboEnv, Robot, RobotTask
    from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
    from robodeploy.description.franka import FrankaDescription
    from robodeploy.tasks.manipulation.pick_place import PickPlaceTask
    from robodeploy.sensors.camera.sim.mujoco_camera import MuJoCoCameraSensor
    from robodeploy.sensors.ft_sensor.sim.mujoco_ft import MuJoCoFTSensor
    from robodeploy.core.types import SensorMount
    from robodeploy.policies.scripted.joint_pd import JointPDPolicy  # noqa  — needs to be implemented

    robot = Robot(
        robot_id="franka",
        description=FrankaDescription(),
        tasks={"pick": RobotTask(
            task=PickPlaceTask(),
            policies={"hold": JointPDPolicy(...)},
        )},
        sensors=[
            MuJoCoCameraSensor("wrist", mount=SensorMount(parent_link="panda_hand")),
            MuJoCoFTSensor("wrist_ft",  mount=SensorMount(parent_link="panda_hand")),
        ],
    )
    env = RoboEnv(backend=MuJoCoBackend(config={"enable_viewer": False}), robots=[robot])
    obs, info = env.reset()

    # Sensors populated
    assert obs.images["wrist"].shape == (240, 320, 3)
    assert obs.ft_force.shape == (3,)

    # Scene props present
    backend = env.backend
    assert "cube" in backend.get_prop_names()
    pose0 = backend.get_prop_pose("cube")

    # Step interacts with the cube; pose changes
    for _ in range(50):
        env.step()
    pose1 = backend.get_prop_pose("cube")
    assert not np.allclose(pose0, pose1)

    # Domain randomization moves the cube between episodes
    env.reset()
    pose2 = backend.get_prop_pose("cube")
    assert not np.allclose(pose0[:3], pose2[:3])

    env.close()
```

Swapping `MuJoCoBackend` for `ROS2RealBackend` and swapping `MuJoCoCameraSensor` for `RealSenseCamera` should pass the same proprio + image assertions on real hardware (cube assertions skipped — physical props don't teleport).

When this test passes, the sensor + 3D environment story is real across the documented surface.

---

## 7. Effort summary

| Step | Days | Outcome |
|---|---|---|
| 1. Stop silent failures | 1 | No more invisible no-ops |
| 2. Schema unification | 1 | Types in place |
| 3. MuJoCo scene loader | 3 | Props physically interact |
| 4. MuJoCo sensor stack | 3 | `obs.images["wrist"]` populated in sim |
| 5. Real-hardware sensor path | 2 | Same code in sim and real |
| 6. Isaac + Gazebo parity | 6 | All backends consistent |
| 7. Real-hardware perception (optional) | 5 | Real-hardware scene state |
| **Critical path** (1-5) | **10 days** | Sensor + env story works end-to-end on MuJoCo + real |
| **Full plan** | **21 days** | All four backends at parity |

---

*End of evaluation.*
