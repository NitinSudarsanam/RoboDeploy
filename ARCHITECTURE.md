# RoboDeploy Architecture

Modular runtime for robot learning and deployment. Swap simulators, hardware stacks,
and policies without rewriting user code. Primary goal: **sim-to-real transfer** through
shared `Observation` / `Action` contracts.

**Canonical references:** [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) (comprehensive guide),
[CONTRACTS.md](CONTRACTS.md) (public API behavior),
[docs/PLATFORM_STATUS.md](docs/PLATFORM_STATUS.md) (maturity and CI),
[examples/README.md](examples/README.md) (demos and presets),
[plans/INTEGRATION_STATUS.md](plans/INTEGRATION_STATUS.md) (contributor audit).

---

## Design principles

1. **One axis of variation per layer** — backend, task, policy, and sensor each solve one problem.
2. **Shared types only** — `robodeploy/core/types.py` (`Observation`, `Action`, `SensorData`, `SceneSpec`).
3. **Base classes absorb boilerplate** — `*Base` classes handle lifecycle; subclasses implement specifics.
4. **Sim-to-real by configuration** — same `ObsPipeline` / `SafetyFilter` code; backend and pipeline config differ.
5. **Extension via registry** — `@register_*` + `use("module")` for string-based wiring; direct injection preferred for apps.

---

## Layer diagram

```
User code
  │
  ├─ RoboEnv.from_config(cfg)          ← canonical (sensor_rigs, obs_pipeline, custom_modules)
  ├─ examples.env_from_preset(name)    ← loads examples/config/presets.yaml
  ├─ RoboEnv(backend=..., robots=[...]) ← direct injection
  └─ RoboEnv.make(...)                 ← minimal registry smoke (flat sensors list only)
  │
  ▼
RoboEnv (env.py)
  reset() / step() / close() / demo_session()
  Owns episode state; routes actions via EnvRouter; merges sensor reads into ObsPipeline
  │
  ├──────────────────────────────┬──────────────────────────────────────┐
  ▼                              ▼                                      ▼
IBackend                    list[Robot]                          ObsPipeline (per robot)
reset_multi /                  RobotTask: task + policies + selectors   transforms, sync, buffers
get_obs_multi /                SensorRig → materialized ISensors
step_multi
```

**Library vs examples:** `robodeploy/` is the installable package. Demo tasks, policies,
presets, catalog, and `python -m examples.cli` live under `examples/`.

---

## RoboEnv and robot model

- **`RoboEnv`** — episode loop, backend lifecycle, observation validation (`obs_spec_policy`),
  policy runtime binding (`bind_runtime` on first `reset()`), demo recording helpers.
- **`Robot`** — `robot_id`, `RobotDescription`, `tasks`, optional per-robot `sensors` / `sensor_rigs`,
  `obs_pipeline`, `action_adapter`.
- **`RobotTask`** — one `ITask`, one or more `IPolicy` instances, optional `ITaskSelector` /
  `IPolicySelector`, task-scoped `language_instruction` stamped into observations.

Multi-robot backends implement `*_multi` methods explicitly. Do not silently fall back from
multi to single-robot APIs unless the backend documents that behavior.

---

## Backends

| Name | Module role |
|------|-------------|
| `mujoco` | MuJoCo physics, MJCF scene build from `SceneSpec`, mounted sensor XML |
| `isaacsim` | Isaac Sim USD stage (mock-tested in CI when Kit unavailable) |
| `ros2_rviz` | ROS2 transport + RViz viz; fake joint sim devtool optional |
| `gazebo` | `ROS2GazeboBackend` — Gazebo via ROS2, URDF spawn, ros_gz_bridge |
| `real_world` | `ROS2RealBackend` — hardware via ROS2 controllers |

Factory: `backend_for_simulator("mujoco" | "ros2_rviz" | "gazebo" | "isaacsim" | "real_world", robots=[...])`.

**Scene:** backends receive merged `SceneSpec` from tasks (`to_world()`). Gazebo can synthesize
a temporary SDF when `config.sim.world` is omitted; explicit world files stay authoritative.

**ROS2:** `robodeploy/ros2/` provides `Ros2Runtime` (single init + executor), `Ros2NodeAdapter`,
and namespaced topic resolution. Robot descriptions may set `ros2_preset_name` for controller topics.

---

## Sensors and perception

Users declare **`SensorRig`** entries (YAML or Python). Each `SensorSpec` resolves through the
registry to a backend-appropriate `ISensor` implementation.

```
ISensor.read() → SensorData → BackendBase._merge_sensor_data()
  → Observation → ObsPipeline.process() → policy.get_action(obs)
```

- **Backend-aware pairs** — e.g. `wrist_camera` → MuJoCo renderer, ROS2 RGBD, or Isaac camera
  depending on active backend (not a simple sim/real bit).
- **Perception-first policies** — consume `obs.objects`, `images`, `depths`; avoid `backend.get_prop_pose()` in new code.
- **Metadata** — `camera_intrinsics`, `camera_extrinsics` (sim live pose or TF lookup), `sensor_status`,
  per-sensor `ft_forces` / `ft_torques`.
- **Example oracle** — `examples.sensors.sim_prop_pose` for sim prop poses (not for real deployment).

Domain randomization at `RandomLevel.FULL` can append `GaussianNoiseTransform` via `ObsPipeline`.

---

## Observation pipeline

`ObsPipeline` normalizes backend output before policies run:

- Mirrors named `images` / `depths` into legacy `rgb` / `depth` when needed.
- `ObsSyncMode` strategies (e.g. time window) and `SensorSampleBuffer` for async sensor merge.
- `RoboEnv` drains pending sensor reads into the pipeline each step when configured.

`is_real` may be read only during env wiring (sensor resolution, pipeline config) and in task
`reset_routine` — not inside shared transforms or policies.

---

## Policies

All policies implement `IPolicy` / `PolicyBase`:

- `get_action(obs)` — primary interface.
- `get_action_batch(obs_list)` — optional; env batches when one shared policy serves multiple robots.
- `bind_runtime(backend, description)` — attach sim solvers, ROS handles, etc. after backend init.
- `notify_rejected(obs, action)` — sequence models replan when safety/IK rejects an action.
- `action_hz` — nominal rate; `RoboBridge` may run inference slower than control loop.

Built-in / registered examples: joint trackers, `VLAPolicy`, `DiffusionPolicy`, `PolicyChain`,
`RobomimicPolicy`, remote `PolicyServer` (ZMQ/gRPC) and `HttpRemotePolicyClient`.

Example reach policies (`examples/policies`) support MuJoCo IK via `bind_runtime` and
`carry_mode` (`kinematic`, `none`, `follow`, `contact`, `weld`) for grasp assist.

### IK resolution order

Policies call `robodeploy.kinematics.policy_ik.attach_policy_ik` from `bind_runtime`:

1. **MuJoCo damped LS** — when the backend exposes `_model` (`robodeploy.kinematics.mujoco_ik`).
2. **Pinocchio URDF** — when the robot description provides a kinematics solver (`robodeploy.kinematics.pin_ik`).
3. **Delta-home fallback** — logged warning; joint tracking toward home without position IK.

MuJoCo attach failure falls through to Pinocchio when both are available.

### Safety paths

- **`step()`** — `SafetyMonitor.check_action()` then per-task `transform_action()` before `backend.step_multi()`.
- **`reset_routine()`** — skipped when the monitor is tripped (e-stop); otherwise each yielded action passes through the description safety filter and `SafetyMonitor.check_action()` before `step_multi()`.
- **Gazebo multi-robot** — `ROS2GazeboBackend.initialize_multi()` raises if `len(robots) > 1` until multi-spawn exists.

---

## Tasks and scenes

- **`ITask` / `TaskBase`** — `reset_fn`, `reward_fn`, `success_fn`, optional `reset_routine` for real hardware.
- **`SceneSpec`** — props (position, geom, material), lights, terrain; merged for backend init.
- **`ObjectSpec`** on `SceneSpec.objects` is deprecated; use `SceneSpec.props` instead.
- Object-aware helpers read `observation.objects` when `ObsSpec.objects` is set.

Example tasks: `pick_place`, `pour`, `peg_insertion`, `showcase_scene` under `examples/tasks/`.

---

## Real hardware bridge

`RoboBridge` decouples a fast **control loop** (process-owned) from a slower **inference loop**:

- `ActionTrajectory` — seqlock shared-memory buffer between processes.
- `EStopFlag` — software e-stop path; control loop applies last-safe or decay behavior on empty buffer.
- Inference: `get_obs` → `ObsPipeline` → policy → chunk into trajectory.

See `robodeploy/bridge.py` and `robodeploy/action_trajectory.py` for the concrete implementation.
Hardware setup notes: [docs/BACKEND_SETUP.md](docs/BACKEND_SETUP.md).

---

## Registration and discovery

```python
from robodeploy import use
use("examples.tasks")   # registers @register_task, @register_policy, etc.
```

- `robodeploy list-registry --builtins` — built-in components.
- `robodeploy list-registry --discover` — pip entry points.
- `--custom-module` on CLI commands imports user packages before lookup.

---

## Testing and CI

- **Unit/smoke:** `pytest tests/` (dummy backend, mocked Gazebo/Isaac/ROS2 where needed).
- **Sensor e2e:** Linux CI job with MuJoCo EGL; vision tests skip on Windows GLFW issues.
- **Hardware:** `pytest.mark.hardware` — local only; see [tests/HARDWARE_TESTS.md](tests/HARDWARE_TESTS.md).

---

## Package layout

```text
robodeploy/
  env.py            RoboEnv episode loop
  bridge.py         RoboBridge (control vs inference decoupling)
  cli.py            robodeploy CLI entry point
  backends/         MuJoCo, Gazebo, Isaac Sim, ROS2 RViz, real ROS2, dummy
  core/             types, registry, Robot/RobotTask, sensor_rig, scene_validator
  description/      URDF/MJCF assets per robot family
  tasks/            TaskBase, templates, randomization, success_predicates
  policies/         PolicyBase, reach_dsl, learned/, remote/
  sensors/          camera, ft_sensor, imu, contact (sim/ and real/)
  obs_pipeline/     ObsPipeline, transforms/, fusion
  perception/         vision_predicates, color-blob helpers
  kinematics/       mujoco_ik, pin_ik, policy_ik attachment
  training/         gym_adapter, bc, ppo, dataset, parallel_vec_env
  evaluation/       benchmark harness, metrics, render, video
  safety/           SafetyMonitor, guards, violation types
  sim2real/         calibration store, DR, transfer metrics
  calibration/      kinematic, extrinsic, handeye, system_id
  teleop/           session contract (keyboard stub; IL WIP)
  observability/    replay, manifests, seeding, health, snapshots
  ros2/             Ros2Runtime, adapters, devtools
  multirobot/       multi-arm scene helpers
  viz/              RViz markers and traces
  testing/          DummyBackend, DummyTask, DummyPolicy
  config/           env config parsing helpers
  _assets/          bundled meshes, manifest SHA256
examples/           presets, catalog, demo tasks/policies, examples.cli
benchmarks/           manipulation_v1, sim2real, leaderboard
tests/              unit, smoke, hardware-gated
docs/               MkDocs site (PROJECT_GUIDE, tutorials, guides)
plans/              GOAL_0N + WAVE2 plans, INTEGRATION_STATUS
```

### Module responsibilities (quick reference)

| Module | Owns | Does not own |
|--------|------|--------------|
| `backends/` | Physics step, sensor read merge, scene build | Task rewards, policy logic |
| `core/` | Shared types, registry, robot model | Simulator SDK calls |
| `description/` | Assets, joint names, limits, launch metadata | Opening simulators or serial ports |
| `tasks/` | Reward/success, scene props, obs_spec | Backend I/O |
| `policies/` | `get_action`, action_space, bind_runtime | Physics |
| `sensors/` | `ISensor.read()` → `SensorData` | Policy decisions |
| `obs_pipeline/` | Transforms before policy | Backend physics |
| `safety/` | Pre-step action checks | Post-contact physics repair |
| `training/` | BC/PPO loops, Gym registration | Benchmark HTML (see `evaluation/`) |
| `evaluation/` | `robodeploy eval`, reports, leaderboard | Training loss |
| `bridge.py` | Fast control loop + trajectory buffer | Single-process `RoboEnv.step` |
