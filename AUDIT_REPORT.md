# RoboDeploy — Architecture Audit & Integration Report

**Date**: 2026-04-22
**Scope**: SOLID/OOP compliance, ARCHITECTURE.md conformance, generality, and Linux Gazebo/RViz/IsaacSim integration strategy.
**Status**: Repo is ~60–70% aligned with the documented architecture. Several foundational abstractions drift; core real-time machinery is missing; Linux-side sim launchers need hardening.

> Current note: this audit is historical. Since it was written, the branch has added robot-centric `RoboEnv`, explicit multi-backend contracts, `ActionTrajectory`, process-owned `RoboBridge`, sensor diagnostics/pairing, Gazebo launcher hardening, and the `ROS2GazeboBackend` / `ROS2RvizBackend` split. Treat individual findings below as a checklist to verify, not as guaranteed current defects.

---

## 1. Executive Summary

| Area | Verdict |
|---|---|
| Core contract (`core/`) purity | Mostly clean — no backend leakage |
| Backend layer conformance | **Broken** — task injected into backend, hardcoded `"robot0"`, simulator launch mixed into "real" backend |
| Policies layer | Sparse — mostly stubs; no `TeleopPolicy`, no `IInputDevice` |
| Tasks layer | Interface clean, but all concrete manipulation tasks are placeholders |
| Real-time bridge | **Incomplete** — no seqlock `ActionTrajectory`, `ControlLoop` uses threads not processes, no `InferenceLoop`, no watchdog |
| Multi-robot plumbing | Half-wired — `RoboEnv` routes, but backends only shim single→multi |
| Arbitrator / task switching | **Stub** — no OMPL/MoveIt2 planning |
| `set_payload`, `swap_sensor`, sync policies | Missing |
| Linux integration (Gazebo/RViz/IsaacSim) | Partially working on paper; several brittle points |
| Tests | Only 2 test classes; no physics/randomization coverage |

The principle "one axis of variation per layer" is violated in two high-impact places: `ROS2Backend` bundles Gazebo launch + robot-state-publisher + RViz bridge + controller-spawning, and `RoboEnv` spans orchestration + routing + viz + diagnostics.

---

## 2. Deviation From ARCHITECTURE.md — Top Offenders

### 2.1 IBackend holds `ITask` — breaks principle 2 ("no shared state between layers")

ARCHITECTURE.md §Interface Contracts / §IBackend:
> SceneSpec is passed directly … so the backend can load props **without any dependency on ITask**. This breaks the circular init: backend does not import or call tasks.

Reality:
- `robodeploy/core/interfaces/backend.py:47-67` — `initialize(description, task, sensors)` still requires `ITask`.
- `robodeploy/backends/base.py:71,86` — `BackendBase` stores `self._task: ITask | None`.
- `robodeploy/backends/sim/mujoco/backend.py:117` — `self._rviz_bridge.publish_scene(task.scene_spec())`.
- `robodeploy/backends/sim/isaacsim/backend.py:89` — same pattern.
- `robodeploy/backends/real/ros2/backend.py:56` — `initialize_multi([robot], scene=task.scene_spec(), …)`.

Fix direction: change `initialize()` signature to `initialize(description, scene: SceneSpec, sensors)`; let `RoboEnv` pre-merge scene; drop `self._task` from `BackendBase`.

### 2.2 `ROS2Backend` fuses hardware transport + Gazebo + RSP + controller spawning

ARCHITECTURE.md: "Each layer solves exactly one problem."
- `robodeploy/backends/real/ros2/backend.py:128-141` launches Gazebo via `GazeboLauncher`.
- `:151-165` launches `robot_state_publisher`.
- `:controllers/*` spawns ros2_control controllers.
- `is_real = True` is hard-declared, yet a Gazebo subprocess means physics is simulated.

Consequence: `backend.is_real` lies. Tasks' `reset_routine` and `DomainRandomizer` dispatches on `is_real`; when Gazebo is running they behave as if on real hardware. Safety-critical.

Fix direction: split into `ROS2RealBackend` (pure transport) and `ROS2GazeboBackend(BackendBase)` with `is_real=False`. Keep launchers but make them the concern of the sim-side subclass.

### 2.3 No `ActionTrajectory` seqlock; `ControlLoop` is a thread

ARCHITECTURE.md §Real Hardware: Decoupled Control and Inference (lines 142-193):
- "ControlLoop **must** run in a `multiprocessing.Process` (not a thread) to escape the GIL."
- "`ActionTrajectory` is a **seqlock** shared-memory ring buffer …"
- "InferenceLoop watchdog process monitors … heartbeat."

Reality (`robodeploy/bridge.py`):
- `ActionBuffer` uses `threading.Lock`, not seqlock.
- `ControlLoop` is a `threading.Thread`.
- No `InferenceLoop` subclass, no heartbeat, no watchdog.
- No spin-timeout; no empty-buffer action-space-aware decay; no ε clamp; no quintic decel profile.
- No pre-computed gravity-compensation torques.
- No hardware e-stop pin assertion path.

Fix direction: net-new module `robodeploy/action_trajectory.py` + rewrite `bridge.py` around `multiprocessing.Process` + `multiprocessing.shared_memory`.

### 2.4 `Arbitrator.switch()` is a stub

ARCHITECTURE.md §1 Robot N Tasks: Arbitrator must drain, plan collision-free, inject trajectory, wait completion, then swap.
Reality: `robodeploy/core/arbitrator.py` contains only state-tracking and event emission. No `KinematicsSolver.plan()` exists (`kinematics/solver.py` exposes only `fk/ik/jacobian`). No OMPL / MoveIt2 wrapper.

### 2.5 Missing contract surface

| Spec requirement | Present? | Location |
|---|---|---|
| `IBackend.set_payload(robot_id, mass, com)` | No | n/a |
| `IBackend.set_prop_pose/set_prop_mass/get_prop_pose/get_prop_names` | No | n/a |
| `RobotConfig.swap_sensor()` | Stub `NotImplementedError` | `core/robot_config.py:38-40` |
| `TeleopPolicy` | No | should be `policies/scripted/teleop.py` |
| `IInputDevice` (spacemouse/VR) | No | should be `sensors/input/*` |
| `ObsPipeline.sync_policy` (DROP_LATEST / TIME_WINDOW) | No | `obs_pipeline.py` |
| `SensorData.timestamp_source` field | Unverified | inspect `core/types.py` |
| `policy.notify_rejected(obs, action)` | No | `policies/base.py` |
| `ActionChunkTransform` p99 sizing + min-depth invariant | Partial | `action_adapter.py:196-244` (exists, lacks p99 + depth guard integration) |
| Sensor hot-swap validation | Stub | `core/robot_config.py` |
| `HumanInterventionRequired` → global e-stop broadcast | No | `env.py:275-281` only prompts local |

---

## 3. SOLID Violations (by principle)

### SRP

- **`RoboEnv` god class** (`env.py`, 539 lines). Mixes: lifecycle, single-vs-multi branching, observation aggregation, action routing, viz payload construction, info assembly, diagnostics, pause/resume hooks. Split into `SingleAgentRouter`, `MultiAgentRouter`, `ObsAggregator`, `InfoAssembler`.
- **`ROS2Backend`** (`backends/real/ros2/backend.py`). Hardware transport + Gazebo launcher + RSP launcher + RViz publisher + controller spawner + preset resolver. Split as in §2.2 and hoist RViz/RSP out to optional composition.
- **`GazeboLauncher`** (`backends/real/ros2/sim_launchers/gazebo.py`). Starts `gz sim`, `ros_gz_bridge`, RSP, `ros_gz_sim create`, and multiple `controller_manager` spawners. Each is a distinct lifecycle. Extract `RosGzBridgeLauncher`, `UrdfSpawner`, `ControllerSpawner` so each can be tested/replaced.
- **`BackendBase._resolve_asset_path`** (`backends/base.py:110-150`). Mixes override-lookup + selection-telemetry + resolution.

### OCP

- `env.py:521-532 _infer_action_space` is a hardcoded if/elif cascade. New action space = edit `RoboEnv`. Move to `ActionSpace.from_action(action)` dispatcher or a registry.
- `env.py:160-163` hardcodes `_real/_sim` suffix to pick sensor class. Add sensor pairing metadata instead.
- Single vs multi-agent branching threads through `env.py:67, 221-225, 287-290, 318-320`. Replace with polymorphic router classes.

### LSP

- `IBackend.initialize_multi/reset_multi/step_multi/get_obs_multi/teleport_object/set_physics_params` all raise `NotImplementedError` by default. Callers must hedge. Either promote to required, or fold into capability protocols (`SupportsMultiRobot`, `SupportsScene`, already used for `SupportsDiagnostics` in `backends/capabilities.py`).
- `backends/base.py:161,171` provides silent single-robot shims for `reset_multi` / `step_multi` / `get_obs_multi`. A caller believing the backend supports N robots will silently get N=1 behavior. Remove shims; require explicit capability.

### ISP

- `IBackend` mixes core control, scene manipulation, physics randomization, diagnostics. The capability pattern in `backends/capabilities.py` (`SupportsDiagnostics` protocol) is the right direction — extend it: `SupportsSceneEdit`, `SupportsPhysicsRandomization`, `SupportsPayload`.

### DIP

- `env.py:_initialize_components` instantiates concrete sensor/policy/task objects from `RobotConfig`/`TaskConfig` directly. Acceptable because `RoboEnv` is the composition root, but there's no explicit factory seam — makes injecting test doubles for physics sanity hard.
- `backends/sim/isaacsim/backend.py:80` imports `from robodeploy.backends.real.ros2.rviz import RvizPublisher`. Sim backend depends on real-backend subpackage. Should hoist `RvizPublisher` to a neutral module (e.g., `robodeploy/viz/rviz_publisher.py`) so both sim and real can compose it without a cross-layer import.

---

## 4. OOP Anti-Patterns

- **Inappropriate intimacy**: both sim backends call `task.scene_spec()` from inside `_load` for a purely optional RViz side-effect (`mujoco/backend.py:117`, `isaacsim/backend.py:89`). Violates separation and leaks task into backend.
- **Stale reference**: `BackendBase._task` never cleared post-load (`backends/base.py:86`). Long-lived backends hold dead task objects through task swaps.
- **Hardcoded identity**: `"robot0"` literal threaded through single-agent path (`mujoco/backend.py:50`, `isaacsim/backend.py:65,67`, `ros2/backend.py:56`, `isaacsim/backend.py:322,343`). Breaks multi-robot audits and diagnostics.
- **Silent fallback guessing** (`mujoco/backend.py:76-82`): if `mj_name2id` misses by joint name it retries with `"robot0/act{idx}"`. Should fail loudly — silent mapping guesses corrupt action delivery.
- **Swallowed exceptions everywhere** (`isaacsim/backend.py:89-91, 161-168, 178-182, 218-220, 245-246, 253-255, 257-261`, `gazebo.py:99-102, 108-141, 145-158, 163-180`): `try/except Exception: pass` hides real failures. This is the main cause of brittle Linux launches.
- **Feature envy**: `env.py` reaches into `RobotConfig.obs_pipeline`, `RobotConfig.action_adapter`, `RobotConfig.description.get_safety_filter()` directly instead of asking `RobotConfig` for a pre-wired step function.

---

## 5. Generality Breaks

- **Hardcoded `robot0`** string in asset resolution, actuator fallback, obs publishing, home-pose seeding (see list above). Blocks multi-robot with real IDs.
- **Franka/Kuka naming inside presets** (`backends/real/ros2/presets.py:40-72`) with no automatic selection. Presets should be indexed by `RobotDescription.ros2_preset_name` (add to description API), otherwise "robot-agnostic" claim in README.md is false.
- **URDF-only Isaac import**: `isaacsim/backend.py:65` uses URDF + `URDFImportRobot`; no USD path. Isaac's native format is USD; URDF import has known unit/mass quirks.
- **MJCF actuator assumption**: MuJoCo backend assumes `robot0/act{idx}` naming (`mujoco/backend.py:82`). Any user-supplied MJCF that uses different actuator names silently mismaps.
- **ROS2 bridge hardcoded topic pair** (`gazebo.py:82-85`): only `/clock` is bridged by default. Real Gazebo integration needs `/joint_states`, camera image, clock, TF — all missing from defaults.
- **Single-agent compatibility shims** (`env.py:67-89, 221-225, 287-290`) fork behavior for N=1. Better: build a single `MultiAgentRouter` and let N=1 be the degenerate case; return a proxy that unwraps lists/dicts.
- **`_infer_action_space`**: hardcoded 4-way cascade. No `CARTESIAN_VEL`, `FORCE`, `GRIPPER`, `DELTA_POS`, `JOINT_IMPEDANCE` paths.
- **`examples/user_kuka_sinusoid/assets/gazebo_world.sdf`** has only ground+sun — robot must be spawned externally. Brittle.

---

## 6. Missing Critical Features (Blocking Real-Hardware Deployment)

1. `ActionTrajectory` seqlock ring buffer with spin timeout (ARCH 145-179).
2. `ControlLoop` running in `multiprocessing.Process` (ARCH 148).
3. `InferenceLoop` process + heartbeat to watchdog (ARCH 154-156).
4. `EStopFlag` in `multiprocessing.shared_memory` + HW pin driver hook.
5. Empty-buffer action-space-aware behavior table (ARCH 226-240) with ε clamp.
6. Pre-computed gravity-compensation torques at init.
7. `IPolicy.notify_rejected()` + `ActionAdapter` NaN guard sequence per spec.
8. `Arbitrator.switch()` motion planning via OMPL/MoveIt2.
9. `KinematicsSolver.plan(q_start, q_goal, obstacles)` added.
10. `SafetyFilter` Tier 1 vs Tier 2 split (scalar clamp in `ControlLoop`, collision/IK in `InferenceLoop`).
11. `HumanInterventionRequired` → global e-stop broadcast.
12. Sensor sync policy selector on `ObsPipeline`.
13. `IBackend.set_payload()` + FCI integration for Franka.
14. `RobotConfig.swap_sensor()` validation.
15. `TeleopPolicy` + `IInputDevice` + `SpaceMouseInputDevice` + `VRControllerInputDevice`.
16. Camera `intrinsics()` / `extrinsics()` on camera sensors.
17. `UndistortTransform` in the transform library.
18. `SensorData.timestamp_source` flag + `TIME_WINDOW` widening logic.

---

## 7. Dead / Stub Code

- `robodeploy/backends/sim/demo_franka_pick.py` — imports non-existent `MujocoEngine` / `BasicFrankaPickTask`.
- `robodeploy/policies/scripted/waypoint.py:1` — header literally says "placeholder matching architecture layout."
- `robodeploy/tasks/manipulation/{pick_place,pour,peg_insertion}.py` — all have empty `scene_spec`, always-False `success_fn`, no reward shaping.
- `tests/test_env_refactor.py:180` — asserts on stale `info` from a previous step (copy-paste).
- `robodeploy/backends/real/ros2/_driver.py` (deleted per `git status`) — verify nothing imports it.

---

## 8. Architectural Re-shape Checklist

| Change | Files touched | Priority |
|---|---|---|
| Drop `ITask` from `initialize()`; pass `SceneSpec` instead | `core/interfaces/backend.py`, `backends/base.py`, all backend subclasses, `env.py` | P0 |
| Split `ROS2Backend` into `ROS2RealBackend` + `ROS2GazeboBackend` | `backends/real/ros2/backend.py`, new `backends/sim/gazebo/backend.py` | P0 |
| Move `RvizPublisher` to neutral `robodeploy/viz/` module | `backends/real/ros2/rviz.py` → `viz/rviz_publisher.py` | P1 |
| Replace single-agent mode branching with polymorphic router | `env.py` | P1 |
| Replace `threading`-based `ControlLoop` with `multiprocessing` + seqlock | `bridge.py` + new `action_trajectory.py` | P0 |
| Add `IBackend` capability protocols (`SupportsSceneEdit`, `SupportsPayload`, `SupportsMultiRobot`) | `backends/capabilities.py` | P1 |
| Remove hardcoded `"robot0"` strings; thread real `robot_id` | every backend `_load`, obs builder | P1 |
| Promote `set_payload` / `set_prop_*` to interfaces | `core/interfaces/backend.py` | P1 |
| Implement `KinematicsSolver.plan()` + `Arbitrator.switch()` | `kinematics/solver.py`, `core/arbitrator.py` | P2 |
| Add `TeleopPolicy`, `IInputDevice`, spacemouse driver | `policies/scripted/teleop.py`, `sensors/input/*` | P2 |
| Flesh out concrete manipulation tasks | `tasks/manipulation/*` | P2 |
| Expand test coverage: physics sanity, randomization, ROS2 contracts | `tests/` | P2 |

---

## 9. Linux Integration Plan — Gazebo, RViz, Isaac Sim

All three target **ROS 2 Jazzy on Ubuntu 24.04** per ARCHITECTURE.md.

### 9.1 Gazebo (Harmonic / gz-sim 8.x)

Current state (`backends/real/ros2/sim_launchers/gazebo.py`):
- Boots `gz sim <world>`, optionally `ros_gz_bridge parameter_bridge`, optionally `robot_state_publisher`, optionally `ros_gz_sim create` for URDF spawn, optionally `controller_manager spawner` per controller.
- Problems identified:
  - Everything under swallowed `except`; silent failure cascade.
  - Default bridge only forwards `/clock` — no `/joint_states`, no camera, no TF.
  - URDF spawn runs `subprocess.run([...], check=False)` — returns success even if Gazebo never actually spawned the model.
  - No `ros_gz` version check — Harmonic needs `ros_gz_bridge` from the Jazzy apt repo, not Humble.
  - No world SDF ships a robot model — users must supply URDF separately; the combination "world + URDF + controllers.yaml" is not bundled.
  - `time.sleep(0.5)` startup is optimistic; `gz sim` often takes ≥2s to become responsive.

Integration steps for Linux (no implementation):
1. Package a reference `controllers.yaml` per supported robot under `robodeploy/ros2_assets/<robot>/controllers.yaml`.
2. Bundle a functional `empty_world.sdf` + `table_world.sdf` under `ros2_assets/worlds/`.
3. Replace the hand-rolled launcher with a generated `ros2 launch` file: cleaner process tree, respects ROS 2 lifecycle, and `rclpy`/`launch_testing` can assert readiness.
4. Add explicit readiness gates:
   - `/controller_manager/list_controllers` returns expected controller as `active`.
   - `/joint_states` publishing at ≥ nominal `control_hz`.
   - `/clock` published.
5. Bridge rules must default to a usable set: `/clock`, `/joint_states`, `/tf`, `/tf_static`, camera image/info pairs. Parametrize per robot.
6. Separate `ROS2GazeboBackend` (subclass of `BackendBase`, `is_real=False`, `supported_action_spaces` identical to real) so downstream code can detect sim vs real honestly.
7. Provide `apt install` list in docs: `ros-jazzy-ros-gz ros-jazzy-ros-gz-bridge ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-controller-manager ros-jazzy-joint-state-broadcaster ros-jazzy-joint-state-publisher-gui gz-harmonic`.

### 9.2 RViz 2

Current state:
- `backends/real/ros2/rviz.py` publishes `/robodeploy/scene/markers`, `/robodeploy/tasks/markers`, `/robodeploy/<robot_id>/ee_pose` on a timer thread.
- `IsaacSimBackend` and `MuJoCoBackend` also compose `RvizPublisher` — acceptable, but creates a cross-layer import (`sim/isaacsim` → `real/ros2/rviz`).

Linux steps:
1. Move `RvizPublisher` to `robodeploy/viz/rviz_publisher.py`; all three backends import from a shared module.
2. Ship a reference `.rviz` layout in `robodeploy/ros2_assets/rviz/default.rviz` with `/robodeploy/**` subscribers pre-configured.
3. Document `rviz2 -d path/to/default.rviz`.
4. Ensure `robot_state_publisher` is optional — when absent, RViz still shows markers but the robot mesh will be missing. Log a clear warning.
5. For the MuJoCo path, publish a synthetic `/joint_states` topic so RViz can co-visualize MuJoCo-driven robots with ROS-RViz panels — enables sim-to-real visual parity.

### 9.3 Isaac Sim

Current state (`backends/sim/isaacsim/backend.py`):
- Lazy imports; requires launching inside Isaac's `python.sh` (Kit runtime).
- Uses URDF importer extension. Not USD-native.
- `_enable_extension_best_effort` silently swallows failure.
- `_seed_home_pose`, `_ensure_physics_ready`, `_initialize_articulation` — heavy error handling for known Windows breakage; Linux needs its own path.
- Hardcoded `"robot0"` prim path and articulation name.
- No sensor support: cameras, depth, FT — `_load(…)` discards the `sensors` arg (`del sensors`).
- Only `JOINT_POS` action space.

Linux steps:
1. Pin to Isaac Sim 4.2+ (USD-native pipeline, Kit 106). Document Ubuntu 22.04/24.04 with NVIDIA driver ≥ 550, CUDA toolchain, `vulkan-tools`, `glibc ≥ 2.35`.
2. Add USD import path alongside URDF: prefer USD when `description.asset_path(AssetFormat.USD)` exists. Keep URDF as fallback.
3. Wire Isaac sensors: `isaacsim.sensors.camera.Camera`, `RTXLidar`, `IMU` — map to `ISensor` pairs so the same `MuJoCoCamera`/`RealSenseCamera` selection mechanism works.
4. Add `JOINT_VEL`, `JOINT_TORQUE` action spaces — Isaac supports all three via `ArticulationAction(joint_efforts=...)`.
5. Un-hardcode `"robot0"` prim path — use `/World/<robot_id>`.
6. Drop silent `except` blocks around physics init — let failures surface; current behavior swallows `omni.physx.tensors` load failures which is a known blocker.
7. Use `SimulationApp` livestream mode with WebRTC for headless servers; document `ISAACSIM_HEADLESS=1` env toggle.
8. Emit RViz-compatible joint state via `isaac_ros` bridge or the same `RvizPublisher` — give Isaac + RViz parity with Gazebo + RViz for visualization.

### 9.4 Integration Matrix (target)

| Backend | `is_real` | Physics | RViz support | Joint cmd | Launch cost |
|---|---|---|---|---|---|
| `MuJoCoBackend` | False | MuJoCo native | via `RvizPublisher` + synthetic `/joint_states` | `JOINT_POS/VEL/TORQUE` | fast |
| `ROS2GazeboBackend` (new) | False | Gazebo Harmonic via ros2_control | native topics | `JOINT_POS/TRAJ/VEL/EFFORT` | slow (~5s) |
| `IsaacSimBackend` | False | PhysX / USD | via shared `RvizPublisher` | `JOINT_POS/VEL/TORQUE` | slowest (~30s) |
| `ROS2RealBackend` (new) | True | hardware | native topics | whatever preset supports | n/a |

All four satisfy the same `IBackend` surface once `initialize(..., scene: SceneSpec, ...)` is adopted.

### 9.5 Linux-Specific CI

- GitHub Actions matrix: `ubuntu-24.04` + ROS 2 Jazzy, headless Gazebo Harmonic smoke test, `rclpy` wait-for-topics gate.
- Isaac Sim cannot run in standard GH runners — use a self-hosted GPU runner or mark as optional nightly.
- MuJoCo + RViz pairs run fine on stock Ubuntu runners with Xvfb.

---

## 10. Priority Roadmap

**P0 — unblocks multi-backend + any real deployment**
1. Remove `ITask` from `IBackend.initialize`. Pass merged `SceneSpec`.
2. Un-hardcode `"robot0"` throughout backends.
3. Split `ROS2Backend` into `ROS2RealBackend` + `ROS2GazeboBackend`.
4. Replace threading `ControlLoop` with `multiprocessing` + seqlock `ActionTrajectory`.

**P1 — correctness & OOP hygiene**
5. Hoist `RvizPublisher` to neutral `viz/` module.
6. Collapse single-agent shims in `env.py` into a polymorphic router.
7. Add `SupportsMultiRobot`, `SupportsSceneEdit`, `SupportsPayload` capability protocols.
8. Fail-fast on actuator name misses; remove swallowed `except`s in Isaac/Gazebo code.
9. Add Linux Gazebo readiness gates + bundled `controllers.yaml` / worlds.
10. Complete `IBackend.set_payload`, `set_prop_pose`, `set_prop_mass`.

**P2 — feature completeness**
11. `KinematicsSolver.plan()` + working `Arbitrator.switch()`.
12. `TeleopPolicy` + `IInputDevice` + SpaceMouse/VR drivers.
13. `SensorData.timestamp_source`, `TIME_WINDOW` sync policy, `UndistortTransform`.
14. Flesh out concrete tasks (pick_place, pour, peg_insertion) with real reward/success.
15. Physics-sanity and randomization test coverage.
16. Isaac Sim: USD path, sensors, multi-action-space.

---

## Appendix A — Specific file:line citations

- `robodeploy/core/interfaces/backend.py:47-143` — `ITask` coupling & `NotImplementedError` multi-robot methods.
- `robodeploy/backends/base.py:71,86,161-175` — stale task, silent multi-robot shims.
- `robodeploy/backends/base.py:110-150` — asset resolution mixes concerns.
- `robodeploy/backends/sim/mujoco/backend.py:50,76-82,117` — hardcoded `robot0`, fallback guessing, task intimacy.
- `robodeploy/backends/sim/isaacsim/backend.py:59-91` — task injected, RViz import from `real/`, URDF-only.
- `robodeploy/backends/sim/isaacsim/backend.py:161-261` — swallowed exceptions around physics init.
- `robodeploy/backends/real/ros2/backend.py:56,128-141,151-165` — single→multi shim hardcoded `robot0`; Gazebo + RSP inside "real" backend.
- `robodeploy/backends/real/ros2/sim_launchers/gazebo.py:72-180` — launcher mixes 5 responsibilities; silent failures; weak readiness.
- `robodeploy/backends/real/ros2/presets.py:40-72` — Franka/Kuka names with no auto-selection.
- `robodeploy/core/arbitrator.py:40-67` — stub `switch()`.
- `robodeploy/core/robot_config.py:38-40` — `swap_sensor` stub.
- `robodeploy/env.py:67,160-163,221-225,287-290,318-320,521-532` — branching + hardcoded dispatch + god-class scope.
- `robodeploy/bridge.py:14-175` — threading instead of multiprocessing; no seqlock; no watchdog.
- `robodeploy/policies/scripted/waypoint.py:1` — placeholder.
- `robodeploy/tasks/manipulation/*.py` — empty concrete tasks.
- `tests/test_env_refactor.py:180` — stale `info` reference.
- `examples/user_kuka_sinusoid/assets/gazebo_world.sdf` — minimal, no robot.

---

*End of report.*
