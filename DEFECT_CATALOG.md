# RoboDeploy — Defect Catalog

**Date**: 2026-05-12
**Scope**: Comprehensive enumeration of points of failure, contract drift, and spaghetti across the repository. Single document, no narrative; each entry is a discrete defect with severity, file/line citation, and a one-line fix direction.
**Reads alongside**: `AUDIT_REPORT.md` (2026-04-22) — original architecture audit; `SENSORS_AND_ENV_PLAN.md` — sensor/3D-environment capability plan.

Severity legend:
- **P0** — silent safety/correctness hazard, or code that does not do what its name/docs claim.
- **P1** — broken or missing capability that callers depend on; bug or contract breach.
- **P2** — design smell, dead code, spaghetti, OOP violation; not actively wrong but blocks future work.
- **P3** — cosmetic, docs, stale examples.

---

## 1. Real-time bridge (`bridge.py`, `action_trajectory.py`)

| ID | Sev | File:line | Defect | Fix direction |
|---|---|---|---|---|
| BR-1 | P0 | `robodeploy/bridge.py:158-192` | `RoboBridge.run()` runs `env.step()` and ControlLoop in the same process; no actual decoupling of inference vs control. | Move inference to its own `multiprocessing.Process`; control proc reads `ActionTrajectory` and writes to backend. |
| BR-2 | P0 | `robodeploy/bridge.py:241-257` | `_control_process_entry` only reads shm and discards (`del _q`). Never calls `backend.step()`. The "decoupled control loop" is a no-op heartbeat. | Implement actual hardware-command path in the control process; share backend handle or RPC to a backend-process. |
| BR-3 | P0 | `robodeploy/bridge.py:91-101` + `:158-191` | Thread `ControlLoop._loop` and async `run()` can both call `backend.step_multi` simultaneously. Race against ROS2 driver state. | Pick exactly one path (process-based) and delete the thread fallback once equivalent. |
| BR-4 | P1 | `robodeploy/bridge.py:138-141` | `env.set_pause_hooks(...)` only wires pause/resume to the *thread* control loop, not to the spawned process. Pause path becomes silently inactive when process loop is used. | Move pause/resume into the shared-memory `EStopFlag`. |
| BR-5 | P1 | `robodeploy/bridge.py:186` | Reads private `self._control._hz` instead of the public `control_hz` property defined ten lines later. | Use `self.control_hz`. |
| AT-1 | P0 | `robodeploy/action_trajectory.py:105-116` | Reader spin loop has no spin timeout. Writer crash mid-write (odd seq) → reader spins forever. `ARCHITECTURE.md:154` mandates 500 µs cap. | Add monotonic-time deadline; on timeout return last-valid copy + escalate to E-stop. |
| AT-2 | P1 | `robodeploy/action_trajectory.py:78-101` | Seqlock uses naive `struct.pack_into`; no memory barriers. Safe on x86-TSO, undefined on ARM (Apple Silicon, Jetson). | Insert explicit fences via `ctypes` or use `multiprocessing.Value` with cas-style ops; document target arch. |
| AT-3 | P1 | `robodeploy/action_trajectory.py:93,101` | Both header writes use `time.time()` (wall clock) — NTP slew breaks monotonicity inside the buffer. Second write overwrites first's timestamp. | Switch to `time.monotonic_ns()`; write timestamp exactly once. |
| AT-4 | P1 | `robodeploy/action_trajectory.py:78-81` | Only `Action.joint_positions` survives. `joint_velocities`, `joint_torques`, `ee_*`, `gripper` silently dropped. | Document JOINT_POS-only or extend slot layout. |
| AT-5 | P2 | `tests/test_action_trajectory.py` | Single-process, single-write test only. No contention, no DOF mismatch, no torn-write, no crash-during-write. | Add cross-process contention tests; force-kill writer mid-write. |

---

## 2. Sensor layer

| ID | Sev | File:line | Defect | Fix direction |
|---|---|---|---|---|
| SN-1 | P0 | `robodeploy/env.py:225-229` | `RoboEnv._initialize_components` calls `sensor.warmup()` without ever calling `sensor.initialize(backend)`. `SensorBase.read()` guard means the first real read raises `RuntimeError`. | Add `sensor.initialize(backend)` before warmup. |
| SN-2 | P0 | `robodeploy/backends/sim/mujoco/backend.py:48-49` | `_load()` does `del scene; del sensors`. `Robot.sensors` is silently discarded. | Iterate sensors, attach to renderer via MJCF camera elements + offscreen render context. |
| SN-3 | P0 | `robodeploy/backends/sim/isaacsim/backend.py:68` | Stores `self._sensors` but never invokes `initialize()` or merges `SensorData` into `Observation`. | Add Isaac-side camera/IMU prim creation + per-step merge. |
| SN-4 | P0 | `robodeploy/backends/real/ros2/backend.py:176-283` | `initialize_multi` ignores `Robot.sensors`; only reads `self.config[f"{robot_id}.sensors"]` dict. Passed `ISensor` list is silently dropped. | Drive sensor instantiation from `Robot.sensors`; keep config dict as alternate constructor. |
| SN-5 | P1 | `robodeploy/sensors/camera/sim/mujoco_camera.py:15-19`, `…/real/realsense.py:15-19`, `robodeploy/sensors/ft_sensor/sim/mujoco_ft.py`, `…/real/ati_ft.py` | All four "pair" stubs raise `NotImplementedError`. The advertised sim/real pairing convention has zero working implementations. | Implement at least the MuJoCo camera + RealSense pair; FT pair gated by hardware availability. |
| SN-6 | P1 | `robodeploy/backends/real/ros2/sensors/camera_rgbd.py:75-170` | Parallel `IRos2Sensor` protocol with own registry (`register_ros2_sensor`); not derived from `SensorBase`. Cannot be passed in `Robot.sensors`. | Refactor to subclass `SensorBase`; remove parallel registry once migration done. |
| SN-7 | P1 | `robodeploy/backends/real/ros2/sensors/camera_rgbd.py:144-154` | Discards `msg.header.stamp`; sets `timestamp_hw = wall_time_s` (host clock). `timestamp_source` never populated. | Read `msg.header.stamp.sec/nanosec`; set `timestamp_source = "hardware"`. |
| SN-8 | P1 | `robodeploy/backends/real/ros2/backend.py:312-316,348-354` | Merges `SensorData → Observation` by mutating a dataclass instance: `obs.rgb = sd.rgb`. No protocol, type-ignored, races with frozen-by-convention expectation. | Use `dataclasses.replace(obs, ...)`; centralize in a `_merge_sensor_data` helper. |
| SN-9 | P1 | `robodeploy/core/types.py:55-67` | Single-camera assumption in `Observation` (`rgb: Optional[ndarray]`). No `images: dict[str, ndarray]`. Multi-camera setups cannot be represented. | Add `images`/`depths` dicts keyed by sensor name; keep `rgb`/`depth` as primary alias. |
| SN-10 | P1 | `robodeploy/core/interfaces/sensor.py` | No mount metadata (`SensorMount` / parent_link / pose). Sim cameras have no link to attach to. | Add `mount: SensorMount` argument to `SensorBase.__init__`. |
| SN-11 | P1 | `robodeploy/env.py:114-117` | `RoboEnv.make` resolves sensors by `name + "_sim"/"_real"` suffix and passes no kwargs. Sensor config (resolution, port) is lost. | Replace with explicit pairing decorator; accept `sensors_kwargs` dict. |
| SN-12 | P1 | `robodeploy/env.py:223,225-229` | `shared_sensors` is passed to backend but every backend either ignores it or raises `NotImplementedError`. | Implement at least overhead-camera path for MuJoCo. |
| SN-13 | P1 | `examples/so101/run_switch_simulator.py:221` | Imports `MuJoCoOverheadCameraRenderer` from `robodeploy.sensors.camera.sim.mujoco_camera` — module exposes only `MuJoCoCameraRenderer` (and that one raises NotImplementedError). `--camera` flag crashes. | Either implement the class or remove the flag wiring. |
| SN-14 | P2 | `robodeploy/core/types.py:142-156` | `ObsSpec` only has flat `rgb/depth/ft_sensor/imu` booleans and a single `image_width/height`. Cannot declare per-camera resolution or per-modality lists. | Add `cameras: list[CameraRequest]`, `force_sensors: list[str]`. |
| SN-15 | P2 | `robodeploy/sensors/base.py:113-116` | `warmup()` catches *every* exception during init reads. On a real driver fault the user sees silence. | `warnings.warn` on each warmup exception. |
| SN-16 | P2 | `robodeploy/backends/real/ros2/sensors/base.py:25-28` | `LastValueCache.write` stamps via `time.time()` — NTP slew breaks ordering for downstream sync logic. | `time.monotonic()`. |

---

## 3. 3D environment / scene layer

| ID | Sev | File:line | Defect | Fix direction |
|---|---|---|---|---|
| SC-1 | P0 | `robodeploy/backends/sim/mujoco/backend.py:48` | `_load()` does `del scene`. SceneSpec is dropped before any prop loading. | Compose props into MJCF via a new `MjcfSceneBuilder`. |
| SC-2 | P0 | `robodeploy/backends/sim/isaacsim/backend.py:67-113` | Only uses `scene` to call `RvizPublisher.publish_scene`. Isaac stage is never populated. | Add `XFormPrim` / `RigidPrim` / USD reference loading per prop. |
| SC-3 | P0 | `robodeploy/backends/real/ros2/backend.py:296-298` | Same — `scene` is only forwarded to RViz markers, never to Gazebo spawn. | Pipe scene to `GazeboLauncher` and spawn per-prop URDFs. |
| SC-4 | P0 | `robodeploy/tasks/randomization.py:120-138` | `DomainRandomizer._randomize_object_poses` catches `NotImplementedError` silently — and *no* backend implements `teleport_object`. Domain randomization is a global no-op. | Either implement on at least MuJoCo + Isaac, or remove the silent skip so failures surface. |
| SC-5 | P1 | `robodeploy/backends/capabilities.py:37-56` | `SupportsSceneEdit`, `SupportsPayload`, `SupportsPhysicsRandomization` protocols declared; zero implementations. | Implement at least on MuJoCo. |
| SC-6 | P1 | `robodeploy/core/types.py:168-196` | Two competing scene primitives (`PropConfig` + legacy `ObjectSpec`). Both merged in `RoboEnv._merged_scene`. | Deprecate `ObjectSpec`; collapse. |
| SC-7 | P1 | `robodeploy/core/types.py:168-180` | `PropConfig` schema insufficient: `asset_path: str` only (no `AssetFormat`); no `geom`/material/parent_link/joint_type/inertia. | Expand schema; see `SENSORS_AND_ENV_PLAN.md §2`. |
| SC-8 | P1 | `robodeploy/tasks/manipulation/pick_place.py:24-27`, `pour.py:18-23`, `peg_insertion.py:18-23` | All concrete tasks ship `PropConfig(asset_path="")` — empty strings. Reward reduces to "drive EE toward fixed coordinate". | Rebuild tasks with real props once SC-1 lands. |
| SC-9 | P1 | `robodeploy/core/types.py:184-196` | `SceneSpec` is tabletop-only: `table_height: float`, `lighting: str` free-form. No `WorldSpec`, no `TerrainSpec`, no `LightSpec`, no environment `CameraSpec`. | Introduce `WorldSpec`; deprecate `table_height`. |
| SC-10 | P2 | — | No `robodeploy/world/` or `assets/objects/` library. Users must bring their own asset paths from outside the package. | Bootstrap with 5 reference props (MJCF/URDF/USD triples). |
| SC-11 | P2 | `robodeploy/tasks/randomization.py:54-67` | Randomizer addresses props by `object_name: str`; identity contract across reset, backend, randomizer is undefined. | Document the resolution order; switch to PropID once SC-6 collapses. |

---

## 4. Backend layer

| ID | Sev | File:line | Defect | Fix direction |
|---|---|---|---|---|
| BK-1 | P0 | `robodeploy/backends/sim/gazebo/backend.py:24-27` | `ROS2GazeboBackend(ROS2RealBackend)` overrides parent `is_real=True` → `False` via class attribute. Subtle trap: any downstream code branching on `is_real` may pick wrong path during inheritance hierarchy walk. | Replace inheritance with composition; or enforce via `__init_subclass__`. |
| BK-2 | P0 | `robodeploy/backends/simulator.py:208` | `backend_for_simulator("ros2_rviz", ...)` returns a `ROS2RealBackend` with `is_real=True` even under `local_ros_graph=True` (fake-joint sim). Tasks branching on `is_real` misroute. | Either flip to `False` via wrapper, or rename the simulator option. |
| BK-3 | P1 | `robodeploy/backends/base.py:160-174` | `BackendBase.reset_multi` / `step_multi` / `get_obs_multi` are silent single-robot shims. A caller assuming N-robot support gets N=1 with no signal. | Remove shims; require explicit `SupportsMultiRobot`. (Already flagged in prior audit; still present.) |
| BK-4 | P1 | `robodeploy/backends/base.py:91-102` | `initialize_multi` raises `NotImplementedError` by default, but `initialize` is the abstract method — so subclasses pass the abstract gate without implementing the multi-robot path. | Make at least one of the two abstract. |
| BK-5 | P1 | `robodeploy/backends/sim/mujoco/backend.py:33-37` | `if shared_sensors: raise NotImplementedError`. Shared sensors are documented as supported. | Implement or document explicit unsupported status. |
| BK-6 | P1 | `robodeploy/backends/sim/isaacsim/backend.py:59-65` | Same — Isaac backend rejects `shared_sensors`. | Same. |
| BK-7 | P1 | `robodeploy/backends/sim/mujoco/backend.py:117-140` | Hardcoded actuator-name fallback `"{robot_id}/act{idx}"` (`mujoco/backend.py:137`). Silent mismap for user MJCFs. Currently gated behind `allow_actuator_name_fallback=False` default — but `backend_for_simulator` sets `allow_actuator_name_fallback=True` (`simulator.py:126`), re-enabling silent guessing. | Make the fallback fail loudly even when enabled, log which actuators came from fallback. |
| BK-8 | P1 | `robodeploy/backends/sim/mujoco/backend.py:332-336,360-362` | `_rviz_bridge.publish_robot_state("robot0", obs)` — hardcoded literal `"robot0"`. Multi-robot RViz views collapse. | Use `self._robot_id`. |
| BK-9 | P1 | `robodeploy/backends/sim/isaacsim/backend.py:343-345,365-367` | Same hardcoded `"robot0"`. | Same. |
| BK-10 | P1 | `robodeploy/backends/sim/isaacsim/backend.py:323-329` | `_reset_impl` swallows `Exception` and returns the current `_build_obs()`. Physics-restart failure → reset returns "looks fine" obs while sim is in unknown state. | Re-raise with context; let `RoboEnv.reset()` surface the failure. |
| BK-11 | P1 | `robodeploy/backends/sim/isaacsim/backend.py:161-261` | Numerous `try/except: pass` blocks around extension load, physics init, articulation init. Known Windows-VC++ failure paths produce silent zero-DOF observations. | Demote to `warnings.warn`; re-raise for fatal cases. |
| BK-12 | P1 | `robodeploy/backends/sim/isaacsim/backend.py:74-85` | USD path "best-effort" — if `usd_prefer=True` and USD exists, it falls back to URDF anyway (`usd_fallback_to_urdf=True`) and only logs to `self._warnings`. The flag combination is misleading. | Either implement USD import or remove the flag. |
| BK-13 | P1 | `robodeploy/backends/real/ros2/backend.py:329-331` | `step_multi` requires `len(actions) == len(self._drivers)`. `RoboEnv` always passes one action per `Robot`. Multi-driver-per-robot setups will mismatch. | Document one-driver-per-robot constraint; or accept dict keyed by `robot_id`. |
| BK-14 | P1 | `robodeploy/backends/real/ros2/backend.py:122-300` | `initialize_multi` is 178 lines doing 7 distinct lifecycles (config parse, fake-sim spawn, RSP launch, per-robot controller + sensors, RViz, diagnostics). | Split into `_setup_*` helpers. |
| BK-15 | P2 | `robodeploy/backends/base.py:109-149` | `_resolve_asset_path` mixes override lookup + telemetry record + path resolution. | Pull telemetry into a dedicated recorder. |
| BK-16 | P2 | `robodeploy/backends/sim/mujoco/backend.py:178-324` | `_compile_mjcf_with_position_actuators` is 147 lines synthesizing MJCF XML inline with actuator wiring + light/floor/camera injection + inertia clamping. Will explode further when scene props are added. | Extract `MjcfSceneBuilder`. |
| BK-17 | P2 | `robodeploy/backends/base.py:91-102` | Variant arg dead: `_resolve_asset_path` passes `variant=variant` but every `RobotDescription.asset_path` does `del variant`. | Either wire to per-variant asset directories or remove the parameter. |
| BK-18 | P2 | `robodeploy/backends/base.py:55-62` | Config-merge only handles a single nested `{"config": {...}}` level. Three-level nesting (`{"sim": {"config": {...}}}`) silently kept un-merged. | Either deep-merge or reject nesting depth > 1 explicitly. |

---

## 5. ROS2 / hardware path

| ID | Sev | File:line | Defect | Fix direction |
|---|---|---|---|---|
| RO-1 | P1 | `robodeploy/backends/real/ros2/controllers/gripper.py`, `joint_effort.py`, `joint_velocity.py` | Three controllers registered via `@register_controller` but raise `NotImplementedError` on first use. Surface as supported in `make_controller` registry. | Mark with `..._stub` suffix or remove registration. |
| RO-2 | P1 | `robodeploy/backends/real/ros2/runtime.py:71-91` | `Ros2Runtime.shutdown()` is process-global. Tearing down inside one test breaks rclpy for every other test in the process. | Reference-count nodes; only shutdown when refcount hits zero. |
| RO-3 | P1 | `robodeploy/backends/real/ros2/sim_launchers/gazebo.py:75-92` | `subprocess.Popen(gz sim ...)` with `stdout/stderr=DEVNULL` — Gazebo crash on startup is invisible. `time.sleep(0.5)` for readiness; Gazebo Harmonic cold-start is ≥ 2 s. | Drain stderr to log; replace sleep with `/clock` topic poll. |
| RO-4 | P1 | `robodeploy/backends/real/ros2/sim_launchers/gazebo.py:91-92,144-158` | `try/except: self._rsp = None` and similar `try/except: pass` around bridge / RSP / Gazebo shutdown swallows real errors. | `warnings.warn`. |
| RO-5 | P1 | `robodeploy/backends/real/ros2/sim_launchers/urdf_spawner.py:32-59` | `subprocess.run(..., check=False)` and never inspects return code. If `ros_gz_sim create` fails, world has no robot and caller cannot tell. | Check returncode; raise on non-zero. |
| RO-6 | P1 | `robodeploy/backends/real/ros2/sim_launchers/ros_gz_bridge.py:27` | Default bridge only forwards `/clock`. No `/joint_states`, no TF, no camera image — minimal usability. | Add a default ruleset including joint_states + tf + camera_info. |
| RO-7 | P1 | `robodeploy/backends/real/ros2/sim_launchers/controller_spawner.py:42-48` | `subprocess.run` without return-code check or stdout capture. Spawner failures silent. | Same — surface return code. |
| RO-8 | P2 | `robodeploy/backends/real/ros2/controllers/joint_trajectory.py:27-57` | `JointTrajectoryControllerAdapter` inherits from `JointPositionControllerAdapter` but reimplements `_on_node_ready` without calling `super()`. Subscription + TF wiring duplicated with subtly different exception handling. | Extract shared setup into a protected helper. |
| RO-9 | P2 | `robodeploy/backends/real/ros2/sim_launchers/robot_state_publisher.py:33-40` | Writes URDF text into a YAML params file via newline-prepend hack. Any line in URDF containing only `---` or other YAML control chars breaks the file silently. | Use `yaml.safe_dump`. |
| RO-10 | P2 | `robodeploy/backends/real/ros2/controllers/joint_position.py:214-226` | `_get_ee_pose_from_tf` returns identity pose on any exception. Downstream "ee_pose looks reasonable" but is actually `(0,0,0,1)`. | Return `None` and have `get_obs` populate Observation with `ee_position=NaN`; surface a diagnostic. |
| RO-11 | P2 | `robodeploy/backends/real/ros2/presets.py:39-72` | Robot-specific Franka/Kuka/UR presets live inside the ROS2 backend folder rather than per-description, contradicting the "robot-agnostic ROS2 layer" stated goal in the same file's docstring. | Move presets to `description/<robot>/ros2_preset.py`. |
| RO-12 | P2 | `robodeploy/ros2/devtools/fake_jointpos_sim.py:1-15` | Thin re-export shim that imports from `backends.real.ros2.dev.fake_joint_sim` — the very "internal" backend path the file claims to abstract over. | Promote the underlying implementation up to `robodeploy/ros2/devtools/`. |

---

## 6. Safety / kinematics

| ID | Sev | File:line | Defect | Fix direction |
|---|---|---|---|---|
| SF-1 | P0 | `robodeploy/kinematics/safety.py:79-90` | Cartesian and Delta-EE actions pass through filter unchanged. No backend implements Cartesian workspace clamping. | Add a workspace-bounds check; or refuse Cartesian until clamping lands. |
| SF-2 | P1 | `robodeploy/kinematics/safety.py:118-163` | `_filter_joint_pos` updates `_prev_pos` but `_filter_joint_vel` / `_filter_joint_torque` do not. After a torque-mode episode, e-stop freeze uses a stale position. | Update `_prev_pos` from any mode that has access to the current joint state, or freeze from observation. |
| SF-3 | P1 | `robodeploy/kinematics/solver.py:194-220` | `KinematicsSolver.plan(q_start, q_goal)` returns a straight-line joint-space interpolation. Doc says "real deployments should replace this". `Arbitrator` uses it silently. | Raise unless caller acknowledges via `unsafe_straight_line=True`; or fail unless `obstacles is None` *and* an explicit flag is set. |
| SF-4 | P2 | `robodeploy/kinematics/solver.py:55-68` | `_ensure_loaded` lazy-imports Pinocchio per instance — fine — but the URDF path is read every time `_ensure_loaded` is called *until* `_model` is set. Re-raises on disk read error; no cache. | Cache (model, urdf_path) on the description; share across consumers. |

---

## 7. Policy / action layer

| ID | Sev | File:line | Defect | Fix direction |
|---|---|---|---|---|
| PL-1 | P0 | `robodeploy/core/interfaces/policy.py:120-127` and `:134-136` | `notify_rejected` defined twice (the second definition wins; identical no-op body). Copy-paste survivor. | Delete the duplicate. |
| PL-2 | P0 | `robodeploy/core/spaces.py:41-55` | `infer_action_space` cannot distinguish `CARTESIAN_POSE` from `DELTA_EE` — both set `ee_position`. Falls through to `CARTESIAN_POSE` for delta actions. | Add an explicit `action_space` field to `Action` or a `delta: bool` marker. |
| PL-3 | P0 | `robodeploy/policies/scripted/waypoint.py:17`, `joint_pd.py:17`, `policies/learned/diffusion.py:17`, `policies/learned/vla.py:17` | All four registered policies raise `NotImplementedError("placeholder only")`. Visible via `list_registered()` as available. | Either implement or unregister with `..._stub` suffix. |
| PL-4 | P1 | `robodeploy/policies/remote/transport.py:199-263` | `GrpcTransport.connect/send_obs_recv_action/send_reset` all raise `NotImplementedError`. Documented as "production" path. | Generate proto stubs and wire, or remove from the `serve(transport="grpc")` dispatch. |
| PL-5 | P1 | `robodeploy/policies/remote/server.py:118-120` | `socket.bind(getattr(self._transport, "_addr", ...))` reaches into transport's private attribute. Tight coupling defeats the IPolicyTransport abstraction. | Add `IPolicyTransport.serve_loop(handler)` so the server is transport-agnostic. |
| PL-6 | P1 | `robodeploy/policies/remote/remote_policy.py:80-109` | `get_action` retries on `(TimeoutError, ConnectionError, RuntimeError)` but never on `zmq.Again` / `zmq.ZMQError` from ZmqTransport — silent crash on stale REQ socket. | Catch transport-specific errors via the transport layer; expose a normalized error class. |
| PL-7 | P1 | `robodeploy/policies/learned/robomimic.py:42-71` | `_load_policy()` runs in `__init__` — torch + cuda allocation before user has decided to use the policy. | Defer load to first `reset()` or first `get_action()`; or accept a `lazy_load` flag. |
| PL-8 | P1 | `robodeploy/core/interfaces/policy.py:129-132` | `action_hz` default returns `0.0`. Any rate-adaptive caller will divide by zero. | Make `action_hz` abstract; or default to a clearly sentinel `math.nan` and check. |
| PL-9 | P2 | `robodeploy/action_adapter.py:213-235` | `ActionChunkTransform` accepts `Action.joint_positions` as `[chunk_size, dof]`. But `Action.joint_positions: Optional[jnp.ndarray]` is typed as 1-D in the dataclass docstring (`[dof]`). Implicit 2-D contract is undocumented. | Document or use a dedicated `ActionChunk` dataclass. |
| PL-10 | P2 | `robodeploy/action_adapter.py:118-129` | `ScaleActionTransform.forward` only touches `joint_positions` + `gripper`. If the policy outputs velocities or torques, they pass through unscaled with no warning. | Either route per-field or raise on unexpected fields. |
| PL-11 | P2 | `robodeploy/core/robot.py:201-229` | `Robot.step()` calls `_arbitrator.evaluate()` then `task.compute_action()` then `task.action_space()` (which reads the first policy's space). For multi-policy tasks with mixed action spaces, this picks one space and silently scope-creeps. | Validate uniform action_space at `RobotTask.__post_init__`. |
| PL-12 | P2 | `robodeploy/core/robot.py:216-226` | `task_action_resolver` default behavior is "last candidate wins". Multi-task concurrent users may not realize. | Raise `RuntimeError` if no resolver is supplied and there are multiple candidates. |

---

## 8. Observation / types

| ID | Sev | File:line | Defect | Fix direction |
|---|---|---|---|---|
| TY-1 | P1 | `robodeploy/core/types.py:20-23` | `jnp` is `jax.numpy` if available, else `numpy`. Fields annotated `jnp.ndarray`. Some backends produce JAX, others raw NumPy, some have raw NumPy mutated in (`ros2/backend.py:312-316`). Type assertions in user code will fail intermittently. | Document the actual contract: "ndarray-like; callers coerce". Or fix all producers to a single type. |
| TY-2 | P1 | `robodeploy/core/types.py:30-79` | `Observation` is not frozen; `ros2/backend.py` mutates fields post-construction; `_merge_sensor_data` (to be added) will too. | Either freeze + use `replace()` everywhere, or document mutation contract. |
| TY-3 | P1 | `robodeploy/core/interop.py:20-40` | `to_torch` docstring claims "zero-copy"; implementation does `np.array(data)` on a JAX array (host copy). | Implement via DLPack (`torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))`), or correct the docstring. |
| TY-4 | P2 | `robodeploy/core/types.py:142-156` | `ObsSpec.image_width / image_height` are single values — all cameras must share resolution. | Per-camera resolution (see SN-14). |
| TY-5 | P2 | `robodeploy/core/types.py:159-180` | `ObjectSpec` and `PropConfig` both exist; SceneSpec carries both lists. (Duplicates SC-6.) | Same fix. |
| TY-6 | P2 | `robodeploy/core/types.py:283-312` | `MultiAgentInfo.rejected_actions: list[dict[str, Any]]` is loosely typed. Producer/consumer agreement is implicit. | Define a `RejectedAction` dataclass. |

---

## 9. RoboEnv orchestration

| ID | Sev | File:line | Defect | Fix direction |
|---|---|---|---|---|
| EN-1 | P1 | `robodeploy/env.py:104-110` | `RoboEnv.make` requires `policy` (raises `ValueError`). Docstrings elsewhere imply `policy=None` is acceptable for external action injection. | Either allow `None` and document, or remove the contradictory docstrings. |
| EN-2 | P1 | `robodeploy/env.py:308-372` | `step()` accepts `action: Action | list[Action] | dict[str, Action]`. If user passes 3 actions for 2 robots, the third is silently dropped (`_normalize_explicit_actions:367`). | Raise on length mismatch. |
| EN-3 | P1 | `robodeploy/env.py:289-302` | `_run_task_reset_routine` blocks on `input()` for `HumanInterventionRequired`. Cannot be used in headless tests or CI. | Make the prompt overridable via the `on_pause` hook; default still `input()` but injectable. |
| EN-4 | P1 | `robodeploy/env.py:196-213` | `_merged_scene` merges `props` and `objects` lists by name, but on collision the *first* prop wins and conflicting fields (mass, asset_path) are silently lost. | Raise on name collision. |
| EN-5 | P1 | `robodeploy/env.py:266-277` | Fallback `Observation` for empty `raw_obs_list` constructs ee_position by slicing `home_qpos[:3]` and ee_orientation by `home_qpos[:4]` of the *joint angles* — these are dimensional nonsense; the slot is sized to fit shapes, not values. | Fail loudly when the backend returns no obs. |
| EN-6 | P1 | `robodeploy/env.py:521-525` | `_infer_action_space` is a hardcoded if/elif cascade. New action space requires editing `RoboEnv`. (Prior audit flagged.) | Delegate to `infer_action_space()` in `core/spaces.py`. |
| EN-7 | P2 | `robodeploy/env.py:441-474` | `_build_multi_info` is 30+ lines hand-rolling state-aggregation across robots. Each scope addition will require an edit here. | Extract into `MultiAgentBuilder`. |
| EN-8 | P2 | `robodeploy/env.py:488-503` | `_maybe_send_viz_to_backend` and `_backend_diagnostics` use duck-typing (`getattr ... callable`) before isinstance protocol check. Either path alone would suffice. | Pick one. |
| EN-9 | P2 | `robodeploy/env.py:518-519` | `render()` forwards to `backend.render()` unconditionally — but `IBackend.render` is documented as no-op for real backends. Calling `render()` on a real backend is silently a no-op rather than rejected. | Document or branch. |
| EN-10 | P3 | `robodeploy/env.py:266-277` | `EpisodeInfo` constructed twice in `reset()` (line 266 and line 279). The first is discarded. | Build once. |

---

## 10. Registry / plugin system

| ID | Sev | File:line | Defect | Fix direction |
|---|---|---|---|---|
| RG-1 | P1 | `robodeploy/core/registry.py:67-70,82-83,etc.` | `register_*` raises `KeyError` on duplicate registration. Running the same test file twice (pytest reload) fails. | Treat re-registration of the *same class object* as a no-op; raise only on collision with a different class. |
| RG-2 | P1 | `robodeploy/core/registry.py` | No unregister API. Test fixtures leak across tests. | Add `unregister_*` with a `replace=True` shorthand. |
| RG-3 | P1 | `robodeploy/backends/real/ros2/sensors/registry.py:13-20` | Second sensor registry parallel to `core/registry.py`'s `register_sensor`. (Duplicates SN-6.) | Collapse during sensor rewrite. |
| RG-4 | P1 | `robodeploy/backends/real/ros2/controllers/base.py:100-107` | Third registry — `register_controller`. Same shape, same defect. | Acceptable as a distinct layer, but consolidate with the others via a shared `Registry` base class. |
| RG-5 | P2 | `robodeploy/core/registry.py:269-278` | `auto_discover_entry_points` swallows every entry-point load error with `warnings.warn`. A misconfigured entry point silently disables a feature on user machines. | Raise on first failure unless `silent=True`. |

---

## 11. Tests

| ID | Sev | File:line | Defect | Fix direction |
|---|---|---|---|---|
| TS-1 | P1 | `tests/` | Test suite is ~5 files. No backend physics sanity, no sensor lifecycle, no IK convergence, no DomainRandomizer effect, no Gazebo readiness, no multi-robot isolation, no SafetyFilter shape rejection, no RoboBridge decoupling. | Phase-aligned test additions per `SENSORS_AND_ENV_PLAN.md §2`. |
| TS-2 | P1 | `tests/test_env_refactor.py:50-99` | `DummyBackend` does not exercise scene loading, sensors, or any real backend behavior. The "multi-agent routing" test does not actually verify routing behaviorally — just that keys exist in `info.extra`. | Add a `MuJoCoBackend` integration test that asserts step changes joint positions. |
| TS-3 | P1 | `tests/test_action_trajectory.py:11-23` | Single test, single-process, single write. | Add cross-process test with forced writer crash. |
| TS-4 | P2 | `tests/test_so101_real.py:155-168` | Tests config-builder shape only; never actually starts a backend. | Mark as unit; add an integration test gated on `pytest.importorskip("rclpy")`. |
| TS-5 | P2 | `tests/test_so101_real.py:171-186` | Hardware smoke test calls `bus.connect(handshake=True)` and `disconnect(disable_torque=True)` — leaving real motors energized for the duration of any test failure between those calls. | Wrap in `try/finally: disable_torque()` first. |

---

## 12. Examples

| ID | Sev | File:line | Defect | Fix direction |
|---|---|---|---|---|
| EX-1 | P1 | `examples/franka_robomimic_demo.py:30-35` | Calls `RoboEnv(description=..., backend=..., task=..., policy=...)`. The current `RoboEnv.__init__` requires `robots=[...]`; this signature is gone. The example raises `TypeError`. | Rewrite using `Robot` + `RobotTask`. |
| EX-2 | P1 | `examples/franka_robomimic_demo.py:21` | Imports `MuJoCoBackend, ROS2Backend` from `robodeploy.backends` — `ROS2Backend` is a renamed re-export of `ROS2RealBackend` (`backends/__init__.py:7-9`). Fine when ROS2 is installed; on a Windows dev box without ROS2, `ROS2Backend = None`, and the import-time `ROS2Backend()` call (line 28) crashes with `TypeError: 'NoneType' object is not callable`. | Lazy import in `make_env`. |
| EX-3 | P1 | `examples/franka_sim_viewer_demo.py:13-15`, `kuka_pick_demo.py:13-15` | Both are `def main(): raise NotImplementedError(...)`. Headline demos are unrunnable. | Replace with working `RoboEnv.make(...)` examples once backends compose. |
| EX-4 | P1 | `examples/multiagent_configs.py:11,46,84,103` | Imports `DiffusionPolicy`, `PourTask`, `PegTask`, `PickPlaceTask`, `RobomimicPolicy("pick.pt")` etc. — all of which are placeholder/`NotImplementedError` at runtime. Building any of these envs and calling `step()` crashes. | Replace with a working scripted policy + minimal real task. |
| EX-5 | P2 | `examples/so101/run_switch_simulator.py:221` | References `MuJoCoOverheadCameraRenderer` which does not exist in the named module. (Duplicates SN-13.) | Fix or remove. |
| EX-6 | P2 | `examples/user_kuka_sinusoid/components.py:22-50` | User-kuka description re-points URDF/MJCF paths into the `robodeploy/` source tree (`Path(__file__).resolve().parents[2] / "robodeploy" / ...`). Breaks if installed via pip (parent path is site-packages, not the source tree). | Bundle assets next to the example file. |
| EX-7 | P2 | `examples/franka_robomimic_demo.py:1-14` + `franka_sim_viewer_demo.py:1-11` + `kuka_pick_demo.py:1-9` | All three example docstrings tell the user that the demo "does not run". README's headline examples on a fresh clone are non-functional. | Either complete or remove from `examples/`. |
| EX-8 | P3 | `examples/multiagent_configs.py:18-24` | `average_joint_position_actions` returns `Action()` (all-None) when no valid `joint_positions` exist. Downstream backend will then receive an empty Action and silently no-op. | Raise. |

---

## 13. SO-101 path

| ID | Sev | File:line | Defect | Fix direction |
|---|---|---|---|---|
| SO-1 | P1 | `robodeploy/backends/real/ros2/controllers/so101_feetech.py:23-43` | `_import_feetech_stack` tries new + legacy lerobot import paths; on old layout, returns `Motor=None, MotorNormMode=None` but does not surface to the caller. `_build_motors_dict` then raises "too old" — confusing because the import "succeeded". | Raise immediately on legacy layout. |
| SO-2 | P1 | `robodeploy/description/so101/_urdf_assets.py:101-111` | `_copy_meshes_next_to_urdf` silently swallows IO errors. Missing meshes → cached URDF with valid `<mesh>` refs but no files next to it; backend then fails on read with no breadcrumb back to the silent copy. | Surface failure with the mesh name. |
| SO-3 | P2 | `robodeploy/description/so101/calibration.py:80-125` | `_from_lerobot_style` assumes 4096 ticks per 2π for Feetech STS3215. Hardcoded — not a doc, not a constant. If the description changes models, silently wrong. | Move to a per-model table. |
| SO-4 | P2 | `examples/so101/run_switch_simulator.py:198,201-203` | `time.sleep(0.2)` to "let ros2_rviz settle" — magic-number readiness gate. | Replace with topic existence check. |
| SO-5 | P3 | `robodeploy/description/so101/description.py:39` | `joint_order=["1","2","3","4","5","6"]` — joint names are bare integers (per URDF). Works, but every consumer must remember the URDF uses string-of-int. | Document at the description docstring. |

---

## 14. Visualization / RViz

| ID | Sev | File:line | Defect | Fix direction |
|---|---|---|---|---|
| VZ-1 | P1 | `robodeploy/viz/rviz_publisher.py:42-44` | `Ros2Runtime.ensure_started()` started inside `RvizPublisher.start()`. If the backend separately initialized ROS2, the executor + spin thread is shared — fine. But if the user creates `RvizPublisher` standalone, it spins a daemon thread that survives process exit until `Ros2Runtime.shutdown()` is called. No documented owner. | Document the global lifecycle; offer an explicit `RvizPublisher.shutdown_runtime()`. |
| VZ-2 | P1 | `robodeploy/viz/rviz_publisher.py:49-68` | If `fixed_frame == "world"`, publishes a static `world → base_link` transform — but base_link may not be the robot's actual base (SO-101 uses `base`). Multi-robot view collapses both robots onto identity. | Read `robot_description.ros_base_frame_id()` instead. |
| VZ-3 | P2 | `robodeploy/backends/sim/mujoco/ros2_bridge.py:38-39` | `publish_scene(scene)` swallows every `Exception` in the caller (`mujoco/backend.py:173-176`). Bridge import path is correct, but missing ROS2 environment will cause the bridge to silently no-op. | Demote to `warnings.warn`. |

---

## 15. Cross-cutting / repository hygiene

| ID | Sev | File:line | Defect | Fix direction |
|---|---|---|---|---|
| CC-1 | P1 | repo-wide | 109 `except Exception:` swallows across 26 files. Concentrations in IsaacSim (17), so101_feetech (17), MuJoCo (13), ROS2 backend (10). Real driver / hardware failures invisible. | Per-file conversion to `warnings.warn`; revisit each. |
| CC-2 | P1 | `robodeploy/__init__.py:43-67` | Top-level imports `RoboBridge` from `robodeploy.bridge`. `bridge.py` imports `mp = multiprocessing` at module load — fine. But its `ActionTrajectory` import touches `multiprocessing.shared_memory`, which on some restricted environments (sandboxed CI) is unavailable. Top-level `import robodeploy` would then fail. | Lazy-import `RoboBridge`. |
| CC-3 | P2 | repo-wide | "Placeholder" docstrings are pervasive: `tasks/manipulation/*.py`, `policies/learned/{diffusion,vla}.py`, `policies/scripted/{waypoint,joint_pd}.py`, `examples/franka_*.py`, `examples/kuka_pick_demo.py`. Every "headline" demo type advertised in README is a stub. | Implement at least one per category before next release. |
| CC-4 | P2 | `robodeploy/policies/__init__.py` (not read here) and `tasks/__init__.py` | No re-exports of registered types; users must import deeply (`from robodeploy.tasks.manipulation.pick_place import PickPlaceTask`). | Add public `__all__` at each package root. |
| CC-5 | P2 | `pyproject.toml` (not read) | No CI configured beyond what tests run locally. Audit reports request matrix CI. | Add GH Actions matrix per `AUDIT_REPORT.md §9.5`. |
| CC-6 | P3 | `robodeploy/backends/real/ros2/rviz.py` | Now a shim to `robodeploy/viz/rviz_publisher`. Once all callers migrate, delete the shim. | Delete after migration. |
| CC-7 | P3 | `robodeploy/backends/sim/__init__.py:5-8` | Wraps Isaac import in `except Exception` (not `ImportError`) — catches any user-side error during Isaac module evaluation, silently disabling the backend. | `except ImportError` only. |
| CC-8 | P3 | `ARCHITECTURE.md` vs reality | Document describes features that are stubs (Arbitrator switch planning, Teleop, IInputDevice, decoupled bridge, gravity comp). New readers expect these to exist. | Mark each section "implemented / planned / stubbed". |
| CC-9 | P3 | `robodeploy/demos/` (in git, pyc-only) | `robodeploy/demos/__pycache__/franka_sim_viewer_demo.cpython-312.pyc` exists but no source — `pip install` artifact left behind. | `.gitignore` `__pycache__/`. |

---

## 16. Contract drift summary

Cross-references between `ARCHITECTURE.md` and the code where the spec and implementation disagree. Drift is the highest-impact category — users build mental models from the spec.

| Doc section | Reality | Severity |
|---|---|---|
| §IBackend "Backend does NOT … know about tasks" | Backends pass `task.scene_spec()` directly to `RvizPublisher` (cosmetic only now; doc says forbidden). | P2 |
| §Real Hardware: Decoupled Control and Inference | `bridge.py` runs both in one thread. Process variant is a no-op. | **P0** |
| §Real Hardware: ActionTrajectory seqlock | Implemented partially (BR/AT-* above). No spin timeout, no memory barriers. | **P0** |
| §1 Robot N Tasks: Arbitrator switch with `KinematicsSolver.plan()` (OMPL) | `LocalArbitrator` does not plan; `plan()` is straight-line interpolation. | **P0** |
| §IBackend.set_payload / set_prop_pose / set_prop_mass / set_physics_params | Declared as protocols; zero implementations. | P1 |
| §ISensor.intrinsics() / extrinsics() | Not implemented anywhere. | P1 |
| §ISensor sim/real pairing convention | Suffix-based; the only working real ROS2 sensor isn't registered under any `_real` name. | P1 |
| §`Observation.images` (implied by multi-camera architecture) | Single-camera `rgb`/`depth` only. | P1 |
| §SyncPolicy DROP_LATEST / TIME_WINDOW | Enum exists; pipeline is no-op. | P1 |
| §TeleopPolicy + IInputDevice (Spacemouse, VR) | Not in repo. | P2 |
| §HumanInterventionRequired → global e-stop broadcast | `env.py` only prompts locally. | P1 |
| §Gravity-compensation torques precomputed at init | Not implemented. | P1 |
| §EStopFlag (shared memory) + HW e-stop pin | Not implemented. | **P0** for real hardware |
| §Empty-buffer action-space-aware behavior with ε clamp | Not implemented. | P1 |
| §`RobotConfig.swap_sensor` validation | Now `Robot` (not `RobotConfig`); no swap API. | P2 |
| §RemotePolicy GrpcTransport | `NotImplementedError`. | P1 |
| §Domain randomizer FULL level | Calls `teleport_object` / `set_physics_params` that no backend implements. Silent no-op. | **P0** (silent) |

---

## 17. Defect counts by category

| Category | P0 | P1 | P2 | P3 | Total |
|---|---|---|---|---|---|
| Real-time bridge | 4 | 3 | 1 | 0 | 8 |
| Sensor layer | 4 | 9 | 3 | 0 | 16 |
| Scene / environment | 4 | 5 | 2 | 0 | 11 |
| Backend layer | 2 | 11 | 4 | 0 | 17 |
| ROS2 / hardware path | 0 | 7 | 5 | 0 | 12 |
| Safety / kinematics | 1 | 2 | 1 | 0 | 4 |
| Policy / action | 3 | 5 | 4 | 0 | 12 |
| Observation / types | 0 | 3 | 3 | 0 | 6 |
| RoboEnv | 0 | 6 | 3 | 1 | 10 |
| Registry | 0 | 4 | 1 | 0 | 5 |
| Tests | 0 | 3 | 2 | 0 | 5 |
| Examples | 0 | 4 | 3 | 1 | 8 |
| SO-101 | 0 | 2 | 2 | 1 | 5 |
| Visualization | 0 | 2 | 1 | 0 | 3 |
| Cross-cutting | 0 | 2 | 3 | 4 | 9 |
| Contract drift | 5 | 8 | 2 | 0 | 15 |
| **Total** | **23** | **76** | **40** | **7** | **146** |

---

## 18. Recommended sequencing

Reading order for fixes, designed to clear blockers before touching dependents:

1. **Stop-the-bleeding P0 silent hazards** — SC-4 (DomainRandomizer silent no-op), AT-1 (seqlock infinite spin), BR-1/BR-2/BR-3 (bridge concurrency), SF-1 (Cartesian SafetyFilter passthrough), SF-3 (`plan()` silent straight line), BK-10 (Isaac silent reset failure), BK-1/BK-2 (`is_real` honesty). Replace silent failures with warnings or raises.
2. **Schema cleanup** — collapse `ObjectSpec`/`PropConfig` (SC-6), fix duplicate `notify_rejected` (PL-1), fix `infer_action_space` cartesian/delta ambiguity (PL-2), fix `Observation` mutation contract (TY-2).
3. **Sensor lifecycle** — SN-1 (initialize before warmup), SN-8 (use `replace()`), SN-9 (multi-camera schema), SN-10 (mounts).
4. **Scene loading** — extract `MjcfSceneBuilder` (BK-16), implement at least MuJoCo `SupportsSceneEdit` (SC-5).
5. **Real bridge rewrite** — BR-1..BR-4, AT-1..AT-4.
6. **Concrete content** — implement at least one task that uses real props (SC-8), one working sim camera (SN-5), one working real sensor (SN-5).
7. **Test backfill** — TS-1..TS-5.
8. **Examples + docs** — EX-1..EX-7 then CC-8.

After step 1, the codebase no longer fails silently. After step 5, real-hardware deployment is safe. After step 6, the library is end-to-end demonstrable in sim.

---

*End of catalog.*
