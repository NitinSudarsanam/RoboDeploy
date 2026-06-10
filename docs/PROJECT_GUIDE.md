# RoboDeploy Project Guide

This guide is the single entry point for understanding how RoboDeploy is organized, how data flows through the stack, and which workflows to use for training, evaluation, and deployment.

**Related:** [ARCHITECTURE.md](../ARCHITECTURE.md) (design principles), [CONTRACTS.md](../CONTRACTS.md) (API contracts), [PLATFORM_STATUS.md](PLATFORM_STATUS.md) (CI and maturity).

---

## 1. What RoboDeploy is

RoboDeploy is a **runtime**, not a simulator. It does not replace MuJoCo or ROS; it wraps them behind a shared interface so your task and policy code stays backend-agnostic.

You typically work with:

| Layer | Responsibility |
|-------|----------------|
| **Backend** | Physics, rendering, ROS topics, hardware drivers |
| **Robot** | Kinematic description, joint limits, optional per-robot sensors |
| **Task** | Reward, success, failure, `obs_spec`, scene props |
| **Policy** | `get_action(obs)` → `Action` |
| **Sensor rig** | Declarative list of sensors resolved per backend |
| **Obs pipeline** | Transforms, sync, noise before the policy sees observations |
| **Safety** | Filters and guards before actions reach the backend |

The **`RoboEnv`** class (`robodeploy/env.py`) owns the episode loop: reset, step, close, demo recording, and safety integration.

---

## 2. Package vs examples

| Location | Contents | PyPI |
|----------|----------|------|
| `robodeploy/` | Library: backends, core types, training, eval, safety | Yes (`pip install robodeploy`) |
| `examples/` | Presets, demo tasks/policies, `examples.cli`, catalog | No — clone repo |
| `benchmarks/` | `manipulation_v1`, `sim2real` task specs | Yes (files in repo) |

**Rule of thumb:** ship production code against `robodeploy.*` imports; use `examples/` as templates and smoke demos.

---

## 3. Configuration paths

Four ways to construct an environment, from most explicit to most convenient:

### 3.1 Direct injection

```python
env = RoboEnv(backend=backend, robots=[robot])
```

Use when you build `Robot`, `IBackend`, and tasks in application code.

### 3.2 `RoboEnv.from_config(cfg)`

Canonical for applications. Accepts registry names or constructed objects:

```yaml
backend: mujoco
robots:
  - id: robot0
    description: kuka
    task: pick_place
    policy: example_sensor_reach_pick
sensor_rigs:
  - mount: ee_link
    sensors: [...]
obs_pipeline: [...]
custom_modules:
  - examples.tasks
  - examples.policies
```

Supports `sensor_rigs`, `obs_pipeline`, `safety`, and `custom_modules` import paths.

### 3.3 Presets (`examples/config/presets.yaml`)

```python
from examples.env_from_preset import env_from_preset
env = env_from_preset("kuka_pick_mujoco")
```

Loads YAML, imports `custom_modules`, builds via `from_config` internally.

### 3.4 `RoboEnv.make(...)` (minimal smoke)

Flat `sensors: list[str]` only — no sensor rigs or example presets. For quick registry tests:

```python
use("my_project.components")
env = RoboEnv.make(robot="franka", backend="mujoco", task="my_task", policy="my_policy")
```

---

## 4. Episode lifecycle

```text
RoboEnv.reset()
  ├─ backend.initialize_multi(robots)
  ├─ materialize sensors from SensorRig
  ├─ bind_runtime() on policies (IK, ROS handles)
  ├─ negotiate action spaces (DELTA_EE → JOINT_POS adapters)
  ├─ build SafetyMonitor guards
  ├─ task.reset_fn(backend)
  └─ optional reset_routine() actions (real hardware homing)

RoboEnv.step(action=None)
  ├─ get_obs_multi() → ObsPipeline per robot
  ├─ policy.get_action(obs) if action is None
  ├─ SafetyMonitor.check_action()
  ├─ backend.step_multi(actions)
  ├─ task.reward_fn / success_fn / failure_fn
  └─ info.extra (safety, sensor_health, policy diagnostics)

RoboEnv.close()
  └─ backend teardown, sensor cleanup
```

Pass an explicit `Action` (or dict per robot) to override the policy for teleop or open-loop tests.

---

## 5. Core types

Defined in `robodeploy/core/types.py`:

- **`Observation`** — joint state, EE pose, images, depth, FT, IMU, `objects`, `contact_state`, etc.
- **`Action`** — `joint_positions`, `joint_velocities`, `ee_position`, delta-EE flags, gripper
- **`ActionSpace`** — `JOINT_POS`, `JOINT_VEL`, `DELTA_EE`, `CARTESIAN_POSE`, …
- **`SceneSpec`** — props, lights, terrain for backend scene builders
- **`ObsSpec`** — which observation fields a task requires

Interop: JAX arrays are copied via NumPy at boundaries (`robodeploy/core/interop.py`).

---

## 6. Robots and descriptions

`RobotDescription` subclasses under `robodeploy/description/` provide:

- Joint names, limits, EE frame
- Asset paths (URDF, MJCF, meshes)
- Optional `gazebo_ros2_extra_config()` for controller topics
- `get_safety_filter()`, `get_kinematics_solver()`

`Robot` bundles a description with one or more `RobotTask` entries (task + policies + selectors).

---

## 7. Tasks

Tasks implement `ITask` / `TaskBase`:

| Method | Role |
|--------|------|
| `obs_spec()` | Required observation fields |
| `scene_spec()` | Props and layout |
| `reward_fn(obs, action)` | Scalar reward |
| `success_fn(obs)` | Episode success |
| `failure_fn(obs)` | Early termination |
| `reset_routine(backend)` | Optional homing sequence (real hardware) |

Example tasks: `pick_place`, `pour`, `peg_insertion`, `showcase_scene` in `examples/tasks/`.

Register with `@register_task("name")` and import via `use("examples.tasks")`.

---

## 8. Policies

Policies implement `IPolicy` / `PolicyBase`:

- **`get_action(obs)`** — primary interface
- **`bind_runtime(backend, description)`** — attach IK, ROS, model weights after backend init
- **`action_space`** — declared space; may differ from backend after negotiation

**Built-in / package policies:** learned adapters (`robodeploy/policies/learned/`), remote serving, composition chains.

**Example policies** (`examples/policies/`):

| Name | Role |
|------|------|
| `example_reach_pick` | Reach DSL pick-place with FT grasp option |
| `example_sensor_reach_pick` | Sensor-driven reach (prop pose / objects) |
| `example_joint_track` | Joint-space ramp for benchmarks |

**IK order** (`robodeploy/kinematics/policy_ik.py`):

1. MuJoCo damped least squares (if backend exposes model)
2. Pinocchio URDF solver (`pip install -e ".[kinematics]"`)
3. Fallback joint tracking (logged warning)

---

## 9. Sensors and observation pipeline

### Sensor rigs

Declare in YAML or Python (`robodeploy/core/sensor_rig.py`):

```yaml
sensor_rigs:
  - robot_id: robot0
    sensors:
      - name: wrist_camera
        type: camera_rgbd
        mount: ee_link
      - name: wrist_ft
        type: ft_sensor
```

`resolve_sensor_class()` picks sim vs ROS implementations per backend.

### Obs pipeline

`ObsPipeline` applies transforms: color blob centroid → `obs.objects`, latency buffers, Gaussian noise (domain randomization), sync across async sensors.

**Perception-first rule:** policies should read `obs.objects` and camera data, not `backend.get_prop_pose()` in new code. Gazebo may use bookkeeping poses for `prop_pose` sensors — see [SENSOR_INTEGRATION.md](SENSOR_INTEGRATION.md).

### Vision

`ColorBlobTracker` requires camera **extrinsics** (position + orientation) by default. Heuristic `world_origin` scaling is opt-in via `fallback_mode=True`.

---

## 10. Backends

Factory: `backend_for_simulator(name, robots=[...])` or `get_backend(name)`.

| Backend | Module | Live CI |
|---------|--------|---------|
| MuJoCo | `backends/sim/mujoco/` | Yes — sensor e2e, eval smoke |
| Gazebo | `backends/sim/gazebo/` (`gazebo` → `ros2_gazebo`) | Linux live sensors + pick E2E |
| Isaac Sim | `backends/sim/isaacsim/` | Mock smoke only |
| ROS2 RViz | `backends/real/ros2/` | ROS2 sensor live test |
| Real | `backends/real/ros2/` | Hardware markers |
| Dummy | `backends/dummy/` | Always |

**Gazebo notes:**

- Controllers: `/joint_states`, `/joint_trajectory_controller/joint_trajectory`
- Sensors: `/wrist_camera/*`, `/wrist_ft/wrench`, `/wrist_imu/imu` (not under `/robot0/`)
- Grasp follow mode is kinematic bookkeeping — not physics weld

Full setup: [BACKEND_SETUP.md](BACKEND_SETUP.md).

---

## 11. Training

Module: `robodeploy/training/`

| Component | File / entry |
|-----------|----------------|
| Gym adapter | `gym_adapter.py` — `GymRoboEnv` |
| Registration | `gym_register.py` — `robodeploy/kuka_pick_mujoco-v0` |
| BC | `bc.py`, `trainer.py` |
| PPO | `ppo.py`, `PPOTrainer` |
| Parallel envs | `parallel_vec_env.py` — `SubprocVecEnv` |

**CLI:**

```bash
robodeploy train bc --dataset ... --dummy
robodeploy train ppo --dummy --total-steps 100000
```

**Scripts:**

```bash
python examples/train_ppo_reach.py --backend dummy --total-steps 500000
python examples/train_ppo_kuka_pick.py --preset kuka_pick_mujoco
```

**CI:** fast unit tests on all platforms; slow PPO convergence on Linux; optional `ppo-nightly.yml` (50k proxy, `continue-on-error`).

Details: [TRAINING.md](TRAINING.md), [tutorials/03_training.md](tutorials/03_training.md).

---

## 12. Evaluation and benchmarks

```bash
robodeploy eval --benchmark manipulation_v1/reach_target --backend dummy --episodes 20
robodeploy eval --benchmark manipulation_v1/reach_target --backend mujoco --episodes 3
robodeploy eval-compare --baseline a.json --candidate b.json
```

Suite layout: `benchmarks/manipulation_v1/<task>/spec.json`, `preset_dummy.yaml`, `preset_mujoco.yaml`, `reference_scores.json`.

**Honesty:** tiers 2–8 are marked `task_status: placeholder` — they use `benchmark_reach_scripted` or `showcase_scene`, not full manipulation policies. Tier 1 `reach_target` is the primary real eval task.

Read: [../benchmarks/README.md](../benchmarks/README.md).

---

## 13. Safety

`SafetyMonitor` composes guards before `backend.step_multi()`:

- **SafetyFilterGuard** — workspace, slew, joint limits (uses post-negotiation action space)
- **ForceGuard** — FT limits, three-strike critical
- **VelocityGuard** — joint velocity warnings
- **CollisionGuard** — disallowed contact pairs (configure `disallowed_pairs`)
- **EStopGuard** — software e-stop latch

```python
env.emergency_stop()
info.extra["safety"]  # per-step diagnostics
```

Real-hardware recovery and ROS2 ack paths: [SAFETY.md](SAFETY.md), [tutorials/05_safety.md](tutorials/05_safety.md).

---

## 14. Sim2real

Tools in `robodeploy/sim2real/` (and CLI subcommands):

- **CalibrationStore** — persist intrinsics/extrinsics/timing offsets
- **LatencyTransform** — sim sensor delay injection
- **TransferEvaluator** — sim vs noisy-sim metrics
- **DR sweep** — domain randomization reports

CLI examples and dry-run tests exist; automated real-hardware transfer is not CI-gated.

Read: [SIM2REAL.md](SIM2REAL.md), [tutorials/04_sim2real.md](tutorials/04_sim2real.md).

---

## 15. Multi-robot

MuJoCo supports multiple `Robot` instances with independent tasks/policies. Tests: `tests/test_multirobot_mujoco.py`, presets `two_franka_pick_mujoco`.

Gazebo currently raises if `len(robots) > 1` — single-arm only until multi-spawn lands.

---

## 16. Plugins and distribution

**Entry points** (`pyproject.toml`):

- `robodeploy.backends`, `robodeploy.robots`, `robodeploy.tasks`, `robodeploy.policies`, `robodeploy.sensors`

```bash
robodeploy list-registry --discover
```

**Docker:** `docker/Dockerfile.cpu` — CI smoke with `robodeploy --help`.

**PyPI:** `.github/workflows/publish.yml` on `v*` tags — configure trusted publishing before first release.

Read: [PLUGINS.md](PLUGINS.md), [RELEASE.md](RELEASE.md).

---

## 17. Observability and replay

- Deterministic seeding utilities
- Episode JSONL export (`robodeploy export-episode`)
- Run manifests for benchmark reproducibility
- HTML eval reports with optional video embed

Dashboard UI deferred — see `robodeploy/observability/DASHBOARD_DEFERRAL.md`.

---

## 18. Testing

```bash
python -m pytest -m "not hardware" -q     # ~620 tests
python -m pytest -m slow -q             # PPO convergence (minutes)
ROBODEPLOY_LIVE_GAZEBO=1 pytest -m live_gazebo -q   # Linux + Gazebo
```

Markers: `hardware`, `slow`, `optional_nightly`, `live_gazebo`.

Contributor integration map: [../plans/INTEGRATION_STATUS.md](../plans/INTEGRATION_STATUS.md).

---

## 19. Common workflows

### New task + policy

1. `robodeploy scaffold task my_task`
2. Implement `reward_fn` / `success_fn` in `my_project/tasks/`
3. Register with `@register_task`
4. Add preset YAML under `examples/config/` or app config
5. Test: `robodeploy run-episode --dummy` or preset CLI

### Train and benchmark

1. Collect dataset (teleop WIP) or use dummy reach dataset
2. `robodeploy train bc ...` or PPO script
3. `robodeploy eval --benchmark manipulation_v1/reach_target --policy ckpt.pt`

### Gazebo multimodal demo

1. Linux + Jazzy + Harmonic + `[kinematics]`
2. `python -m examples.kuka_ft_imu_pick_gazebo.run_gazebo`
3. Verify topics per [BACKEND_SETUP.md](BACKEND_SETUP.md)

### Release

1. Merge to `main`, green CI
2. `git tag v0.2.0 && git push origin v0.2.0`
3. PyPI publish via `publish.yml`

---

## 20. Known limitations (v0.2)

| Area | Status |
|------|--------|
| Teleop / IL data collection | Contract + stub only |
| Gazebo pick success | CI ≥50% / 10 seeds; target 70% |
| Isaac live EE/torques | Mocked CI |
| PyPI install | Workflow ready; no published tag yet |
| Benchmark tiers 2–8 | Placeholder tasks |
| Learned policy LOC budget | Deferred refactor |
| Real hardware benchmarks | Manual lab runs |

Wave 2 plans: `plans/WAVE2_0N_*.md`.

---

## 21. Package modules (`robodeploy/`)

Every installable subpackage and its role:

| Module | Responsibility |
|--------|----------------|
| `env.py` | `RoboEnv` — reset/step/close, safety, obs validation, demo recording |
| `bridge.py` | `RoboBridge` — separate control and inference processes for real hardware |
| `vec_env.py` | Sequential vector env wrapper (training uses `training/parallel_vec_env.py` for parallelism) |
| `core/` | `Observation`, `Action`, registry, `Robot`, `SensorRig`, scene validation |
| `backends/` | `IBackend` implementations: MuJoCo, Gazebo, Isaac, ROS2, dummy |
| `description/` | `RobotDescription` assets: Kuka, Franka, SO-101, user_kuka, … |
| `tasks/` | `TaskBase`, templates (`pick_place`, `pour`, `insertion`), `randomization.py` |
| `policies/` | `PolicyBase`, `reach_dsl`, `learned/`, `remote/` serving |
| `sensors/` | `ISensor` drivers: camera, FT, IMU, contact (`sim/` + `real/`) |
| `obs_pipeline/` | `ObsPipeline`, sync buffers, `transforms/` (noise, fusion, color blob) |
| `perception/` | Vision predicates (`vision_predicates.py`) |
| `kinematics/` | MuJoCo damped LS, Pinocchio, `attach_policy_ik` |
| `training/` | Gym adapter, BC, PPO, datasets, `SubprocVecEnv` |
| `evaluation/` | `robodeploy eval` harness, HTML reports, leaderboard helpers |
| `safety/` | `SafetyMonitor`, workspace/force/velocity/collision/e-stop guards |
| `sim2real/` | Calibration store, latency injection, transfer evaluator |
| `calibration/` | CLI-backed kinematic, extrinsic, hand-eye, system-ID |
| `teleop/` | Session contract; keyboard stub (`pip install -e ".[teleop]"`) |
| `observability/` | Episode export, run manifests, deterministic seeding, health |
| `ros2/` | `Ros2Runtime`, namespaced topics, dev fake joint sim |
| `multirobot/` | Multi-arm MJCF/scene helpers |
| `viz/` | RViz marker publishers |
| `testing/` | `DummyBackend`, `DummyTask`, `DummyPolicy` for CI smoke |
| `config/` | Env config parsing used by `from_config` |
| `presets_loader.py` | Shared preset YAML loading (library side; demo presets in `examples/`) |
| `scaffold.py` | `robodeploy scaffold` code generation |
| `builtins.py` | Built-in registry population |

**Not in the package:** `examples/` (demos), `benchmarks/` (eval specs), `plans/` (roadmap).

### Backend parity matrix

| Capability | MuJoCo | Gazebo | Isaac Sim | ROS2 RViz | Real |
|------------|--------|--------|-----------|-----------|------|
| Physics step | Yes | Yes | Partial (mock CI) | Fake joint sim | ROS2 controllers |
| RGB-D camera | Yes | Yes | Yes | Yes | RealSense / ROS |
| FT sensor | Yes | Yes | Yes | Yes | ATI / ROS wrench |
| IMU | Yes | Yes | Yes | Yes | ROS / Xsens stub |
| Contact | Yes | Yes | — | FT threshold | FT threshold |
| Multi-robot | Yes | Single-arm only | Partial | Per-robot topics | Per-robot |
| Grasp assist (examples) | follow/contact/weld | kinematic bookkeeping | follow | — | — |

Details: [BACKEND_SETUP.md](BACKEND_SETUP.md), [../plans/INTEGRATION_STATUS.md](../plans/INTEGRATION_STATUS.md).

---

## 22. Documentation index

| Doc | Topic |
|-----|-------|
| [tutorials/01_getting_started.md](tutorials/01_getting_started.md) | First run |
| [TASK_CREATION.md](TASK_CREATION.md) | Custom tasks |
| [POLICY_CREATION.md](POLICY_CREATION.md) | Scripted policies |
| [SCENE_DEFINITION.md](SCENE_DEFINITION.md) | Props and Scene IR |
| [SENSOR_INTEGRATION.md](SENSOR_INTEGRATION.md) | Rigs and pipelines |
| [TRAINING.md](TRAINING.md) | BC/PPO production paths |
| [COOKBOOK.md](COOKBOOK.md) | Recipes |
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | All commands |
| [API_REFERENCE.md](API_REFERENCE.md) | Module index |
| [MIGRATION_0.2.md](MIGRATION_0.2.md) | Upgrade from 0.1.x |
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | Design principles |
| [../CONTRACTS.md](../CONTRACTS.md) | Public API contracts |
| [../benchmarks/README.md](../benchmarks/README.md) | Eval suites |
