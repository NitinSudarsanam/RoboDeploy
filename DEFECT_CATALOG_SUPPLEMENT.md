# RoboDeploy — Defect Catalog Supplement

**Date**: 2026-05-12
**Scope**: Additional points of failure, contract drift, and spaghetti found after `DEFECT_CATALOG.md`. Focused on areas under-examined in prior sweeps: packaging, docs, `__init__.py` exports, example-runner ordering, asset hygiene, planning-doc rot. Defects here are *new*; no duplication of the 146 entries in `DEFECT_CATALOG.md`.

Severity scale: **P0** silent hazard / does-not-work-as-named; **P1** broken or missing; **P2** smell / dead code / drift; **P3** cosmetic / stale.

---

## 1. Packaging — repo is unpackageable

| ID | Sev | File:line | Defect | Fix |
|---|---|---|---|---|
| PK-1 | **P0** | repo root | **No `pyproject.toml`, no `setup.py`, no `setup.cfg`, no `requirements.txt`** exist at repo root. README's "Quick Start" `uv sync --extra sim --extra real` therefore cannot work. Repo is not installable. | Author `pyproject.toml` with `[project.optional-dependencies]` for `sim`, `real`, `isaacsim`. |
| PK-2 | P1 | `robodeploy/__init__.py` | No `__version__` defined. Tooling that depends on it (e.g. `importlib.metadata`, `pip show`) reports nothing. | Add `__version__ = "0.x.y"`. |
| PK-3 | P2 | repo root | No `MANIFEST.in` / asset inclusion config. Bundled MJCF / URDF / STL files would not ship with a wheel build. | Add asset patterns to `pyproject.toml` `[tool.setuptools.package-data]`. |
| PK-4 | P2 | `.venv-wsl/` committed | `.venv-wsl/` contains a full Linux venv with site-packages including pandas, numpy. Massive repo bloat; OS-specific paths in repo. | `.gitignore` `.venv*/`. |

---

## 2. README / CONTRIBUTING / docs — severe drift

`README.md` and `CONTRIBUTING.md` describe a different repo than the one in the file tree.

| ID | Sev | File:line | Defect | Fix |
|---|---|---|---|---|
| DC-1 | **P0** | `README.md:13-40` | Project structure section claims `robots/` directory with `franka/`, `ur5/` subfolders. Reality: `robodeploy/description/franka`, `…/kuka`, `…/so101`. `robots/` does not exist. | Rewrite the layout block. |
| DC-2 | **P0** | `README.md:17-20` | Claims `core/bridge.py` and `core/robot.py`. `bridge.py` is at top-level (`robodeploy/bridge.py`); `core/robot.py` exists but has nothing matching the doc ("Universal Robot Policy interface"). Cross-reference is wrong. | Same. |
| DC-3 | **P0** | `README.md:53-75` | "Usage Example: Multi-Robot, Multi-Task" imports `MujocoEngine` (does not exist), `tasks.DepthAlignmentTask` / `RGBSortingTask` (do not exist), `BatchedObservation` (does not exist), `brain.plan(obs)` (no such API), `sim_engine.apply_action(actions)` (no such method). The canonical example in the README cannot run. | Replace with `RoboEnv` + `Robot` + `RobotTask` shape. |
| DC-4 | P1 | `README.md:9` | "DLPack-based memory sharing between JAX (Sim/Perception) and PyTorch (Inference)" — but `robodeploy/core/interop.py:20-40` `to_torch` does a host copy (`np.array(jax_array)`). Not DLPack. | Either implement DLPack or correct the claim. (Repeats TY-3 from main catalog with new context.) |
| DC-5 | P1 | `CONTRIBUTING.md:37-46` | "Adding a New Robot (`/robots/<name>`)" with `/sim` and `/real` subfolders. Reality is `description/<name>/assets/` with no sim/real split. | Rewrite or delete section. |
| DC-6 | P1 | `CONTRIBUTING.md:54-58` | References `obs.task_mask` (does not exist on `Observation`), `MujocoEngine` (does not exist), `sim_engine.teleport_object`/`set_physics` as user-facing API (only `IBackend.teleport_object` exists, no backend implements). | Rewrite. |
| DC-7 | P1 | `CONTRIBUTING.md:74` | "Run `pytest tests/test_physics.py`" — `test_physics.py` does not exist. | Either add or remove reference. |
| DC-8 | P1 | `CONTRIBUTING.md:25` | "If a file exceeds 300 lines, it must be split into logical sub-modules." Violators: `env.py` (522), `mujoco/backend.py` (429), `isaacsim/backend.py` (430), `ros2/backend.py` (438), `core/types.py` (313), `action_adapter.py` (296 — borderline). | Either enforce or relax the rule. |
| DC-9 | P1 | `CONTRIBUTING.md:27-28` | "Suffix non-SI units: `timeout_ms`, `frequency_hz`." Violators: `Observation.timestamp` (no `_s`/`_ns`), `friction_range` (no unit), `BehaviorProfile.kp`/`kp_scale`/`joint_damping` (no units), `control_hz_by_robot_id` follows. Mixed compliance. | Either enforce or rewrite. |
| DC-10 | P1 | `CONTRIBUTING.md:28` | "Booleans must be prefixed: `is_calibrated`, `has_gripper`, `should_reset`." Violators: `enable_viewer`, `apply_motor_limits`, `allow_uncalibrated`, `headless`, `publish_state`, `publish_command_echo`, `cache_compiled_mjcf`, `_initialized`, `usd_prefer`, `usd_fallback_to_urdf`. Mixed. | Either fix names or rewrite. |
| DC-11 | P2 | `docs/BACKEND_SETUP.md:124-128` | "MuJoCo requires an MJCF model path from the robot description (or an override)." But `MuJoCoBackend._load` (line 67-109) auto-compiles URDF into MJCF when no MJCF is available. Doc lags by a refactor. | Update. |
| DC-12 | P2 | `docs/BACKEND_SETUP.md:30-41` | Codifies the double-nested `{"config": {"config": ...}}` workaround as a supported "less common but seen in some wrappers" pattern, rather than removing the workaround. | Either remove the merge in `BackendBase` (and the doc) or document it as deprecated. |
| DC-13 | P2 | `docs/BACKEND_SETUP.md:232-241` | Documents RViz topics `/robodeploy/<robot_id>/trace` as published. `RvizPublisher` does publish them — but only when `publish_robot_state` is called with a non-zero `ee_position`. Real ROS2 backend's `_get_ee_pose_from_tf` returns identity `(0,0,0,1)` on TF lookup failure (`joint_position.py:225-226`), so the trace ends up a single point at origin with no signal that anything is wrong. | Mark `trace` empty when EE pose is the identity fallback. |
| DC-14 | **P0** | `docs/REFACTOR_PLAN.md` | Authoritative-looking "Refactor Plan" describes work as if it is the next step: "Delete `core/robot_config.py`, `core/task_config.py`, env-wide `Arbitrator`". `core/arbitrator.py` was indeed not deleted but exists (verified by file listing). Plan describes `orchestration/env_router.py` and `orchestration/env_evaluator.py` files that **do not exist anywhere**. The plan is partially-applied: `Robot`/`LocalArbitrator` shipped; `Arbitrator` deletion did not happen; orchestration directory was never created. | Either complete the deletion (`core/arbitrator.py` — verify nothing imports it) or mark the doc as historical. |
| DC-15 | P2 | `docs/SO101_REAL.md:27-29` | Doc says the bundled template `calibration/example.json` exists. `SO101Calibration.locate(allow_template=False)` then refuses it at runtime — sensible safety but the doc points users at the template path *and* the runtime rejects it. Confusing for a user following the doc. | Doc should mention the rejection up front. |
| DC-16 | P2 | `.github/copilot-instructions.md:11-13` | Tells coding assistants "Use JAX for sim, NumPy for real, PyTorch for policies." Codebase mixes JAX and NumPy in sim (`mujoco/backend.py:401-428` uses jnp opportunistically with numpy fallback). Real backends use NumPy + JAX mix. Rule never enforced. | Either enforce or rewrite to "any ndarray-like; coerce at the boundary." |
| DC-17 | P3 | `CONTRIBUTING.md:80-85` | "Pro-Tip: Using AI with this Guide" — Gemini Pro / Copilot specific instructions. Tightly coupled to a specific tool; will rot. | Either generalize or move to a separate AI-tooling doc. |

---

## 3. Package layout — inconsistent re-exports

`__init__.py` files do not consistently surface what they should.

| ID | Sev | File:line | Defect | Fix |
|---|---|---|---|---|
| PX-1 | P1 | `robodeploy/backends/sim/gazebo/__init__.py` | Empty except for docstring. `from robodeploy.backends.sim.gazebo import ROS2GazeboBackend` fails. Compare with `backends/sim/mujoco/__init__.py` which exports `MuJoCoBackend`. Inconsistent. | Add `from .backend import ROS2GazeboBackend`. |
| PX-2 | P1 | `robodeploy/policies/__init__.py` | Re-exports only `RobomimicPolicy`. `WaypointPolicy`, `JointPDPolicy`, `DiffusionPolicy`, `VLAPolicy` are registered (`@register_policy`) but not exported. Users can't `from robodeploy.policies import DiffusionPolicy`. (They are placeholders, but their presence is part of the advertised surface.) | Add the missing re-exports or delete the policy stubs. |
| PX-3 | P1 | `robodeploy/tasks/__init__.py` | Only re-exports `PickPlaceTask`. `PourTask`, `PegTask` are registered but not exported. | Same. |
| PX-4 | P1 | `robodeploy/core/__init__.py:1-23` | Imports *modules* into the namespace (`from . import interop, types`) and lists module names in `__all__`. But `robodeploy/__init__.py` re-exports *classes*. Inconsistent convention: `from robodeploy.core import types` works but `from robodeploy.core import Observation` does not. | Pick one and apply uniformly. |
| PX-5 | P2 | `robodeploy/backends/__init__.py:7-9` | `ROS2Backend = None` when ROS2 import fails. Downstream `ROS2Backend()` calls crash with `TypeError: 'NoneType' object is not callable` — the import "succeeded" so the user has no breadcrumb. (Repeats EX-2 with broader scope.) | Either delete the alias on ImportError, or raise a wrapped `ImportError("install rclpy ...")` on first use. |
| PX-6 | P2 | repo | `ROS2RealBackend` reachable via four paths: `robodeploy.backends.real.ros2.backend.ROS2RealBackend` (canonical), `robodeploy.backends.real.ros2.ROS2Backend` (aliased), `robodeploy.backends.ROS2Backend` (re-export), and after `import_builtins()` it's also in the string registry as `"ros2"`. Four routes, two names, one class. | Document one canonical import path; mark others deprecated. |
| PX-7 | P2 | `robodeploy/ros2/devtools/__init__.py` | Re-exports `FakeJointPosSim`/`FakeJointPosSimConfig` from `backends/real/ros2/dev/fake_joint_sim.py`. Three modules expose the same two classes (`backends...dev`, `robodeploy.ros2.devtools`, `examples/user_kuka_sinusoid/ros2_fake_jointpos_sim.py`). | Pick one public path; mark the rest internal or deprecated. |
| PX-8 | P3 | `robodeploy/sensors/__init__.py:1-3` | Exports `SensorBase` only. Concrete sensor stubs not exported — also fine since they raise `NotImplementedError`. But once they're implemented, the convention will need a refresh. | Address when SN-5 lands. |

---

## 4. Examples — runner-ordering bugs

Found by reading every example file end-to-end.

| ID | Sev | File:line | Defect | Fix |
|---|---|---|---|---|
| EX-9 | **P0** | `examples/user_kuka_sinusoid/run_mujoco.py:13` | Imports `from examples.user_kuka_sinusoid.components import ...` at top of module, **before** `_ensure_repo_on_path()` is defined and called (lines 20-26). Direct execution `python examples/user_kuka_sinusoid/run_mujoco.py` raises `ModuleNotFoundError`. Only `python -m examples.user_kuka_sinusoid.run_mujoco` works. The path-injection function is dead code as written. | Move the function call and definition before any user-import. |
| EX-10 | **P0** | `examples/user_kuka_sinusoid/run_gazebo.py:17`, `run_isaacsim.py:13`, `run_ros2_rviz.py:21` | Same ordering bug as EX-9 in three more sibling files. | Same fix. |
| EX-11 | P1 | `examples/user_kuka_sinusoid/run_isaacsim.py:8` | `from robodeploy.backends.sim.isaacsim.backend import IsaacSimBackend` at module top. The backend module attempts to lazy-import Isaac inside `_launch_kit`, but importing the class still triggers `from robodeploy.backends.sim.isaacsim.backend import ...` — which is fine itself, but on a non-Isaac machine running `python -m examples.user_kuka_sinusoid.run_isaacsim --help` or any path that hits this file will succeed at import (Isaac imports are deferred) → user gets confusing failure later. Worse: the example never gates the import behind `--help` parsing. | Move the import inside `main()`. |
| EX-12 | P2 | `examples/user_kuka_sinusoid/run_switch_simulator.py:20` | `BACKEND: SimulatorName = "ros2_rviz"` at line 20 uses `SimulatorName` annotation **before** importing it (line 32). Works only because of `from __future__ import annotations` at line 11. Removing that import silently breaks the file. | Either move import above the annotation, or document the dependency on `__future__.annotations`. |
| EX-13 | P1 | `examples/user_urdf_asset_override/run_mujoco_default.py:28-29` | Passes `backend_kwargs={"config": {"enable_viewer": False}}` — the deprecated double-nested shape. This example is shipped as a reference. It teaches the wrong pattern. | Flatten. |
| EX-14 | **P0** | `examples/user_urdf_asset_override/components.py:62-63` | ```python
   def get_action(self, obs: Observation) -> Action:
       return Action(joint_positions=obs.joint_positions)
   ```
   "Hold" policy that feeds proprioception back as a joint-position command. Encoder noise creates a positive feedback loop. On real hardware with any servo deadband this drifts. Shipped as a user-style reference policy. | Replace with `home_qpos` constant. |
| EX-15 | P1 | `examples/so101/calibrate_so101.py:79,87,88,93-96` | `bus.connect(handshake=True)` then two `input()` blocks before `bus.disconnect(disable_torque=True)`. Ctrl-C during the operator-prompt window leaves the serial bus open and motors torque-off-but-bus-connected. Next run fails to connect on the same port until power-cycle. | Wrap in `try/finally` with `bus.disconnect(disable_torque=True)`. |
| EX-16 | P2 | `examples/multiagent_configs.py:11,46,84,103,128,131` | Builds envs out of `DiffusionPolicy()`, `RobomimicPolicy("pick.pt")`, `PourTask`, `PegTask`, `PickPlaceTask` — all of which are either placeholders raising `NotImplementedError` or have empty `scene_spec`. Running any of the returned envs crashes. The file ships as "structure-only" but is imported and invoked by other examples (`many_robots_many_tasks_many_policies()` returns an env that crashes on first `env.reset()`). | Mark as type-stub `if False:` block, or fix to use working components. |
| EX-17 | P3 | `examples/README.md:21-25` | Lists `franka_sim_viewer_demo.py`, `franka_robomimic_demo.py`, `kuka_pick_demo.py`, `multiagent_configs.py` as "Other demos (may target older APIs; treat as reference)." README explicitly ships non-functional demos. | Move to `examples/_legacy/` or delete. |

---

## 5. Visualization — bug surface

| ID | Sev | File:line | Defect | Fix |
|---|---|---|---|---|
| VZ-4 | P1 | `robodeploy/viz/rviz_publisher.py:104` | `world = scene.to_world() if hasattr(scene, "to_world") else scene` — defensive call for a method that does not exist on `SceneSpec`. The fallback hides the missing migration from `SceneSpec` → `WorldSpec` (called out in `SENSORS_AND_ENV_PLAN.md §2`). When migration lands and SceneSpec gains `to_world`, the data shape will change underneath the fallback silently. | Either implement `SceneSpec.to_world` now, or delete the defensive call until the migration ships. |
| VZ-5 | P2 | `robodeploy/viz/rviz_publisher.py:192-197` | `_trace_by_robot[robot_id]` grows up to 2000 points, then truncates to the last 2000 — but the cap is applied only inside `publish_robot_state`. Caller that calls `publish_scene` repeatedly without ever calling `publish_robot_state` does not trigger the cap; the trace dict accumulates state from prior episodes across `RoboEnv.reset()` boundaries. | Clear `_trace_by_robot` on `RoboEnv.reset()` via a new `RvizPublisher.reset()` hook. |
| VZ-6 | P2 | `robodeploy/viz/rviz_publisher.py:128-133` | `m.scale.x = float(size[0]) * 2.0` — assumes MJCF half-extent convention for `GeomSpec.size`. `GeomSpec.size` is undocumented. If `size` is full-extent (URDF convention), the marker is twice the real prop. | Document `GeomSpec.size` as half-extent (MJCF) and adapt URDF input upstream. |
| VZ-7 | P2 | `robodeploy/viz/rviz_publisher.py:49-68` | Hardcoded `world → base_link` static transform on `start()` when `fixed_frame == "world"`. SO-101 uses `base`, not `base_link`. Multi-robot views collapse onto the identity at `base_link`. | Read base frame from each `RobotDescription.ros_base_frame_id()`. (Repeats VZ-2 with stronger consequence.) |

---

## 6. Asset hygiene

| ID | Sev | File:line | Defect | Fix |
|---|---|---|---|---|
| AS-1 | P2 | `robodeploy/description/so101/assets/urdf/so101.urdf:4` | `<robot name="so100">` — model name is `so100` despite directory and class name being `so101`. Downstream tools that read the `robot name` attribute (RViz RobotModel, ros2_control config matching) see `so100` and may fail to map. | Rename to `so101` in the URDF. |
| AS-2 | P2 | `robodeploy/description/so101/assets/urdf/assets/README.md` | A `README.md` lives inside the *assets* directory next to STL files. Build / install tooling that ships `package_data` matching `*.md` would pick up this file. | Either omit `*.md` from `package_data` patterns or move the doc. |
| AS-3 | P2 | `robodeploy/description/kuka/assets/mjcf/kuka.xml` | Single-file demo MJCF with no companion URDF (`description/kuka/description.py:46-49` raises `FileNotFoundError` for any non-MJCF request). `KukaDescription` therefore cannot drive `KinematicsSolver` (which requires URDF). Multiple examples that build envs with `KukaDescription` + behavior-translator features that touch `KinematicsSolver` will crash. | Generate a URDF or document `KukaDescription` as MuJoCo-only. |
| AS-4 | P3 | `robodeploy/demos/__pycache__/` | Compiled pyc files committed without corresponding .py source: `franka_pick.cpython-312.pyc`, `panda_oscillation.cpython-312.pyc`, `franka_sim_viewer_demo.cpython-312.pyc`. (Already in main catalog as CC-9; bundling this here for asset-hygiene grouping.) | Delete; `.gitignore` `__pycache__/`. |

---

## 7. Edge-case bugs in core logic

| ID | Sev | File:line | Defect | Fix |
|---|---|---|---|---|
| LO-1 | P1 | `robodeploy/core/robot.py:106-119` | `RobotTask.compute_action` when `len(self.policies) == 1` bypasses `self.policy_selector` entirely — even if a 1-policy task was constructed with a `policy_weights={p: 0.0}` or a `policy_selector` that always returns `Action()` to disable the task. The selector code path is unreachable for single-policy tasks. | Always invoke the selector if one is configured. |
| LO-2 | P1 | `robodeploy/core/selectors.py:58-67` | `WeightTaskSelector.select`: when all weights are absent or tie at 0, picks the first candidate (`best_id = candidates[0]; best_w = 0.0`). Caller setting `task_weights={"pick": 0.0, "peg": 1.0}` works; caller setting `task_weights={"pick": 0.0}` and expecting "pick" to *not* run silently still picks "pick". Counterintuitive zero-as-disable semantics. | Document that zero is a valid weight; offer an `EnabledTaskSelector` for true disable. |
| LO-3 | P1 | `robodeploy/core/robot.py:140-180` | `Robot` dataclass uses `field(init=False)` for `_arbitrator` and `_safety` without `default` / `default_factory`. Works only because `__post_init__` assigns immediately. Python dataclass ordering: any `field(init=False)` without a default must follow init-with-defaults; subclasses adding new init=True fields will break. | Use `default=None` + post-init assignment; or move into `__post_init__` without `field()`. |
| LO-4 | P1 | `robodeploy/env.py:266-277` | `EpisodeInfo` is constructed twice during `reset()` — line 266 builds one with `episode_id = previous + 1` and assigns to `self._episode_info`, then line 279 builds another with the same id for the return value. The first is then discarded after the `extra` keys are populated on the second. (Repeats EN-10 with confirmed second build.) | Build once; mutate `extra` in place. |
| LO-5 | P1 | `robodeploy/description/so101/calibration.py:237-247` | `_looks_like_lerobot_calibration` heuristic: returns True if ≥4 of the dict's values contain `{"id", "range_min", "range_max"}` keys. A user calibration file accidentally containing those keys with different meaning silently routes to `_from_lerobot_style`. Loose typing; should be explicit. | Add a top-level format discriminator (`"format": "lerobot" | "robodeploy"`). |
| LO-6 | P2 | `robodeploy/env.py:135-157` | `RoboEnv.from_config(cfg)` requires `cfg["robot"]`, `cfg["backend"]`, `cfg["task"]` as **strings** (it calls `make`). No way to pass already-constructed objects via `from_config`. Hydra users with `_target_:` patterns are blocked. | Accept both: if value is a string, registry-resolve; if it's a class or instance, use directly. |
| LO-7 | P2 | `robodeploy/core/local_arbitrator.py:47-48` | `_active_task_id = self._sequential_ids[0] if self._sequential_ids else None` — order-dependent picking. Whichever sequential task happens to be first in dict iteration order becomes active by default. With Python 3.7+ this is insertion order — works — but users with task weights expecting the highest-weight task to be active at startup are surprised. | Apply `task_selector` once at init when present, before defaulting to index 0. |
| LO-8 | P2 | `robodeploy/core/robot.py:67-73` | `RobotTask.__post_init__` raises `ValueError` if multiple policies are supplied without weights or selector. Reasonable, but the error message says "or policy_selector" — when a *user* tries `RobotTask(task=..., policies={"p1": ..., "p2": ...})` they get a `ValueError` at construction with a long string. No corresponding error for a *missing* policy on a task that *requires* one. | Same exhaustive check at `Robot.__post_init__`. |

---

## 8. Test brittleness

| ID | Sev | File:line | Defect | Fix |
|---|---|---|---|---|
| TS-6 | P1 | `tests/test_env_refactor.py:144-180` | "Multi-agent routing test" asserts `reward == 2.0` and `"task0" in info.extra["viz"]["tasks"]`. Does not assert anything about routing behavior beyond reward shape. A regression that ignored `robot1` would not fail this test. | Add an assertion that both robots' actions reached the backend. |
| TS-7 | P1 | `tests/test_so101_real.py:179-185` | `test_hardware_smoke_so101_port` calls `bus.connect(handshake=True)` then `disable_torque()` then `disconnect(disable_torque=True)` — no `try/finally`. Failure between connect and disconnect leaves motors energized and bus connected. | Wrap in try/finally. |
| TS-8 | P2 | `tests/test_so101_real.py:124-136` | `test_watchdog_fires_once` sleeps 0.35 s and asserts watchdog fired exactly once. On a heavily loaded CI runner the watchdog could miss its 0.15 s window or fire twice. Flake risk. | Use a `threading.Event` from the timeout callback. |
| TS-9 | P2 | `tests/test_so101_real.py:139-152` | `test_temperature_guard_calls_violation` similar timing dependency. | Same. |

---

## 9. Stale planning artifacts kept as docs

| ID | Sev | File:line | Defect | Fix |
|---|---|---|---|---|
| PD-1 | P2 | `docs/REFACTOR_PLAN.md` | Reads as authoritative spec. References files that don't exist (`orchestration/env_router.py`, `orchestration/env_evaluator.py`). Describes deletions that didn't happen (`core/arbitrator.py` still exists; `core/robot_config.py` / `core/task_config.py` are gone but the plan doesn't mark it). Mixing "plan" and "spec" mode. | Either dated archive (`docs/archive/REFACTOR_PLAN_2026Q1.md`) or replace with status notes. |
| PD-2 | P2 | `ARCHITECTURE.md` | Describes features as if implemented: decoupled bridge, seqlock timeout, Arbitrator OMPL planning, TeleopPolicy, IInputDevice, gravity-compensation precompute, `set_payload`, swap_sensor. All are stubs or missing. New readers form an inflated mental model. (Already in main catalog as CC-8; reiterated because the cost is high.) | Add a "Status" column to each subsection: implemented / partial / planned. |
| PD-3 | P3 | `AUDIT_REPORT.md` (2026-04-22) | Audit from one month prior; some items addressed since (`viz/` extracted, `Robot`+`RobotTask` shipped, ROS2 split into `Real` + `Gazebo`); most P0/P1 items still open. Document does not mark which items have closed. | Add a "Status as of 2026-05-12" column or strikethrough closed items. |

---

## 10. Defect counts by category

| Category | P0 | P1 | P2 | P3 | Total |
|---|---|---|---|---|---|
| Packaging | 1 | 1 | 2 | 0 | 4 |
| Docs / README / CONTRIBUTING | 3 | 7 | 6 | 1 | 17 |
| Package layout / re-exports | 0 | 4 | 3 | 1 | 8 |
| Examples | 3 | 3 | 1 | 1 | 8 |
| Visualization | 0 | 1 | 3 | 0 | 4 |
| Assets | 0 | 0 | 3 | 1 | 4 |
| Edge-case core logic | 0 | 5 | 3 | 0 | 8 |
| Tests | 0 | 2 | 2 | 0 | 4 |
| Stale planning docs | 0 | 0 | 2 | 1 | 3 |
| **Total** | **7** | **23** | **25** | **5** | **60** |

Combined with prior `DEFECT_CATALOG.md` (146 entries): **206 defects** total enumerated.

---

## 11. Recommended sequencing (delta-only)

Add to existing roadmap in `DEFECT_CATALOG.md §18`:

**Pre-P0 (1 day):**
- **PK-1** Author `pyproject.toml`. Repo cannot ship without it. Single biggest blocker for adoption.
- **EX-9..EX-11** Fix example-runner ordering bugs. Three demo files break under direct execution.
- **EX-14** Replace `UserHoldPolicy` proprio-feedback shape.

**P0 doc realignment (1 day, in parallel):**
- **DC-1..DC-3** Rewrite README usage example and project layout block.
- **DC-14** Either complete or archive `REFACTOR_PLAN.md`.

**P1 hygiene (next sprint):**
- **PX-1..PX-3** Re-export missing classes from `__init__.py`.
- **LO-3..LO-5** Edge-case core-logic fixes.
- **AS-1** Rename URDF model `so100` → `so101`.
- **VZ-4** Resolve `scene.to_world()` defensive call before `WorldSpec` lands.

**Ongoing:**
- **TS-6..TS-9** Test hardening: real routing assertions, try/finally on hardware tests, replace sleep-based timing checks.

---

## 12. What is genuinely working

For balance — to avoid the impression that everything is broken:

- **`Robot` + `RobotTask` aggregate** (`core/robot.py`) is well-shaped and unit-tested.
- **`LocalArbitrator`** (`core/local_arbitrator.py`) is small, focused, testable. The arbitration story works.
- **`SafetyFilter` joint-mode clamping** is correct math (only failure mode is the Cartesian passthrough + `_prev_pos` staleness).
- **`Watchdog`, `EStop`, `TemperatureGuard`** (`ros2/safety.py`) — actually correct concurrency primitives.
- **`SO-101 Feetech adapter`** is the only end-to-end working hardware path in the repo, including calibration.
- **`Ros2RgbdCameraSensor`** is the only working camera adapter, even if it's in the wrong registry.
- **`URDFRobotDescription`** parses URDF correctly and exposes joint limits.
- **`MjcfSceneBuilder`-precursor** in `MuJoCoBackend._compile_mjcf_with_position_actuators` is a working URDF→MJCF pipeline that gets robots into MuJoCo without hand-written MJCF.
- **`BehaviorProfile`** preset → backend-specific config translation is a clean idea cleanly executed.
- **`ros2/safety.py` `SafetyError` class hierarchy** plus the `Watchdog` + `EStop` composition pattern.

Treat the working pieces as templates for fixing the broken ones. Most of the rot is in glue code, not in the well-formed components.

---

*End of supplement.*
