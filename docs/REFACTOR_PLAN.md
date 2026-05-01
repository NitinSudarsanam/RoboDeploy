# Robot-Centric Refactor + ROS2 User-Code Cleanup

## Context

RoboDeploy today is **env-centric**: `RoboEnv` owns the arbitrator, the action router, and global lists of `RobotConfig` + `TaskConfig`. `TaskConfig` lists which robots execute it (inverted from the natural mental model) and binds a single policy. Task switching is explicit/sequential — no weight or scoring system exists.

Two pain points to fix:

1. **ROS2 leaks into user code.** The standard `RoboEnv.make()` flow is clean, but: (a) custom controller/sensor adapters force users to import `rclpy.node.Node` and call `Ros2Runtime.ensure_started()` themselves (see `robodeploy/backends/real/ros2/controllers/base.py:50–66` and `examples/user_kuka_sinusoid/ros2_fake_jointpos_sim.py:57–82`); (b) the fake-joint-position sim helper currently lives in user-example space and is full of rclpy code.

2. **UX is env-centric, not robot-centric.** Users construct a global task list and a global robot list separately, then `TaskConfig.robot_ids` glues them. We want users to construct **Robot** objects that own their own task↔policy mapping, with **weights or a meta-policy** controlling which task and which policy run at each step. `RoboEnv` becomes a thin orchestrator over a list of robots.

Outcome: cleaner mental model (`RoboEnv(backend, robots=[Robot(...), Robot(...)])`), zero ROS boilerplate even for custom adapters, and a uniform arbitration story (weights are the default; a `TaskSelector`/`PolicySelector` is a drop-in override).

User decisions captured:
- Task arbitration: weights default, meta-policy override (both supported via uniform selector internal repr).
- Policy arbitration (multi-policy on one task): meta-policy picks/blends; weights are sugar over a constant selector.
- ROS2 scope: close all three leaks (custom controllers, custom sensors, fake-sim).
- Backward compatibility: **hard break.** Drop the single-agent `RoboEnv(description=..., task=..., policy=...)` kwargs entirely; migrate examples/tests.

---

## New Concept Model

```
RoboEnv(backend, robots: list[Robot], shared_sensors=[], max_episode_steps=None)

Robot(
    robot_id,
    description: RobotDescription,
    tasks: dict[task_id, RobotTask],      # task_id -> RobotTask
    task_weights: dict[task_id, float] | None = None,
    task_selector: ITaskSelector | None = None,
    sensors=[], obs_pipeline=..., action_adapter=...,
)

RobotTask(
    task: ITask,
    policies: dict[policy_id, IPolicy],
    policy_weights: dict[policy_id, float] | None = None,
    policy_selector: IPolicySelector | None = None,
    mode: "sequential" | "concurrent" = "sequential",
    preserve_policy_state_on_deactivate: bool = False,
)
```

Weights and selectors are mutually exclusive at construction; if both omitted and exactly one task / one policy, it's used directly. Internally **weights wrap into a `WeightSelector`** so the runtime only deals with selectors — keeps step path uniform.

### Selector protocols (`robodeploy/core/selectors.py`, new)

```python
class ITaskSelector(Protocol):
    def select(self, *, robot_id, obs, candidates: list[str]) -> str: ...
    # Returns chosen task_id. Concurrent tasks are not selected — they always run.

class IPolicySelector(Protocol):
    def select(self, *, robot_id, task_id, obs, candidate_actions: dict[str, Action]) -> Action: ...
    # Picks one or blends. Default WeightedPolicySelector implements winner-takes-all.
```

`WeightTaskSelector(weights)` and `WeightedPolicySelector(weights)` are stock implementations. Users supplying a meta-policy implement these protocols (or pass any callable with the same signature).

---

## Files to Modify / Add

### New

- `robodeploy/core/robot.py` — `Robot` dataclass + `RobotTask` dataclass. Replaces externally-visible `RobotConfig` and `TaskConfig`. `Robot` holds an internal `_LocalArbitrator` instance.
- `robodeploy/core/selectors.py` — `ITaskSelector`, `IPolicySelector`, `WeightTaskSelector`, `WeightedPolicySelector` (stock).
- `robodeploy/core/local_arbitrator.py` — per-robot arbitration. Replaces env-wide `Arbitrator`. Owns `active_task_id`, applies `task_selector` each step for sequential pool, emits `ArbitrationEvent`s consumed by env.
- `robodeploy/backends/real/ros2/adapters_base.py` — `Ros2NodeAdapter` base class. Subclasses get a managed `self._node`, auto-call `Ros2Runtime.ensure_started()` + `add_node()` in `__init__`, auto-cleanup in `stop()`. User-written controllers/sensors no longer import `rclpy` directly.
- `robodeploy/backends/real/ros2/dev/fake_joint_sim.py` — moved from `examples/user_kuka_sinusoid/ros2_fake_jointpos_sim.py`. Now invoked via `backend_kwargs={"dev_fake_sim": FakeJointPosSimConfig(...)}` on the ROS2 backend; users never touch rclpy.

### Modify

- `robodeploy/env.py:46–533`
  - Drop `description`, `task`, `policy`, `sensors`, `obs_pipeline`, `action_adapter`, `tasks=`, `action_resolvers=` kwargs. New signature: `RoboEnv(backend, robots: list[Robot], *, shared_sensors=[], max_episode_steps=None)`.
  - Delete `_single_agent_mode` branches; collapse `_step_single` and `_step_multi` into one path (`step` always routes via robots).
  - Replace `self._arbitrator = Arbitrator(...)` with iteration over `robot._arbitrator` per step.
  - `_step_multi` simplifies: `for robot in self._robots: action = robot.step(obs_by_robot[id])` then `backend.step_multi(actions_in_order)`.
  - `switch_task(robot_id, task_id)` → delegates to `self._robot_by_id[robot_id].switch_task(task_id)`.
  - Drop `_primary_task` indirection — primary becomes `robots[0]`'s active task; expose for back-compat in `EpisodeInfo`.
- `robodeploy/env.py:131–199` `RoboEnv.make()` / `from_config()`: rebuild around a single robot wrapper `Robot(description, tasks={"task0": RobotTask(task, policies={"p0": policy})})`. Config-dict format adds `robots: [...]` array; legacy single-key form auto-promoted.
- `robodeploy/orchestration/env_router.py` — gut. Action selection moves into `Robot.step()`. Keep `normalize_explicit_actions` (still needed for explicit-action path) but delete `resolve_task_candidates` and `resolve_robot_actions`.
- `robodeploy/core/arbitrator.py` — keep `ISwitchPlanner` + `ArbitrationEvent`; delete env-wide `Arbitrator` class (replaced by per-robot `LocalArbitrator`). Move events emit into `LocalArbitrator`.
- `robodeploy/core/robot_config.py` and `robodeploy/core/task_config.py` — delete. Internal-only equivalents (used to call `backend.initialize_multi()`) move to a private `_BackendRobotSpec` struct in `robodeploy/core/robot.py`.
- `robodeploy/core/interfaces/backend.py:39–221` — `initialize_multi()` and `step_multi()` keep current signatures (they take a list of robot specs / list of actions); env calls them by translating `Robot` → backend spec. Delete the `initialize()` / `step()` / `reset()` single-agent shape from `IBackend` since hard-break.
- `robodeploy/backends/real/ros2/controllers/base.py:43–72` — keep `IControllerAdapter` protocol. Add `Ros2NodeAdapter` base class users subclass; existing built-in controllers migrate to it (no behavior change).
- `robodeploy/backends/real/ros2/backend.py` — accept `dev_fake_sim: FakeJointPosSimConfig | None` in backend_kwargs; if set, instantiate the (now-internal) fake sim during `initialize_multi`.
- All built-in controllers/sensors under `robodeploy/backends/real/ros2/controllers/*` and `.../sensors/*` — migrate to `Ros2NodeAdapter` (mechanical change, no logic change).

### Examples / tests to migrate (hard break)

- `examples/user_kuka_sinusoid/run_mujoco.py`, `run_ros2_rviz.py`, `run_gazebo.py`
- `examples/user_kuka_sinusoid/ros2_fake_jointpos_sim.py` — **delete** (folded into backend)
- `examples/user_kuka_sinusoid/components.py` — adjust registration usage if it touched env-level kwargs
- `examples/user_urdf_asset_override/run_mujoco_override_mjcf.py`
- `examples/ros2_rviz_minimal.py`
- Any tests under `tests/` using the deprecated single-agent constructor (sweep with grep).

---

## Reusable Bits (don't reinvent)

- `RobotDescription.get_safety_filter()` (`description/base.py:42–160`) — keep call site, just move into `Robot.step()`.
- `ActionAdapter.process()` (used in `env_router.py:137`) — same, moves into `Robot.step()`.
- `infer_action_space` (`core/spaces.py`) — still needed for safety filter when policy doesn't expose one.
- `ArbitrationEvent`, `RobotStepState`, `TaskStepState`, `MultiAgentInfo` (`core/types.py:266–312`) — schema unchanged; only the producer moves.
- `evaluate_active_tasks_impl` (`orchestration/env_evaluator.py`) — keep; called by env after backend step.
- `build_multi_agent_extra`, `build_viz_payload`, etc. (`core/extra_schemas.py`, `orchestration/viz.py`) — unchanged.
- `Ros2Runtime` (`backends/real/ros2/runtime.py:39–90`) — keep as the singleton; `Ros2NodeAdapter` wraps usage.

---

## User-Facing Example (target shape)

```python
from robodeploy import RoboEnv, Robot, RobotTask
from robodeploy.backends.sim.mujoco import MuJoCoBackend
from robodeploy.description.kuka import KukaDescription
from my_pkg import PickTask, PlaceTask, DiffusionPolicy, ScriptedWaypointPolicy

kuka = Robot(
    robot_id="kuka0",
    description=KukaDescription(),
    tasks={
        "pick": RobotTask(
            task=PickTask(),
            policies={"diffusion": DiffusionPolicy(), "scripted": ScriptedWaypointPolicy()},
            policy_weights={"diffusion": 0.7, "scripted": 0.3},   # winner-takes-all by default
        ),
        "place": RobotTask(task=PlaceTask(), policies={"main": ScriptedWaypointPolicy()}),
    },
    task_weights={"pick": 1.0, "place": 0.0},   # start on pick; runtime can switch
)

env = RoboEnv(backend=MuJoCoBackend(), robots=[kuka])
obs, info = env.reset()
for _ in range(200):
    obs, reward, done, info = env.step()   # robot picks task + policy via weights internally
```

ROS2 fake-sim usage (no rclpy in user code):

```python
env = RoboEnv.make(
    robot="user_kuka", backend="ros2", task="user_kuka_sinusoid", policy="user_sinusoid",
    backend_kwargs={"rviz": {"enabled": True}, "dev_fake_sim": {"robot_ns": "/robot0"}},
)
```

Custom ROS2 controller (no `rclpy` import):

```python
from robodeploy.backends.real.ros2 import Ros2NodeAdapter, register_controller, ControllerConfig

@register_controller("my_custom")
class MyController(Ros2NodeAdapter):
    controller_type = "my_custom"
    def _on_node_ready(self, node):
        self._pub = node.create_publisher(...)   # `node` is auto-managed
    def get_obs(self): ...
    def send_action(self, action): ...
```

---

## Implementation Order

1. Add `core/selectors.py`, `core/local_arbitrator.py`, `core/robot.py` (new code, nothing breaks yet).
2. Refactor `env.py` to require `robots: list[Robot]`. Delete single-agent path. Update `make()` / `from_config()`.
3. Delete `core/robot_config.py`, `core/task_config.py`, env-wide `Arbitrator`. Trim `orchestration/env_router.py`.
4. Update `core/interfaces/backend.py` (drop single-agent methods); update sim/real backends accordingly.
5. Add `Ros2NodeAdapter`; migrate built-in ROS2 controllers/sensors. Move `ros2_fake_jointpos_sim.py` into `ros2/dev/`. Wire `dev_fake_sim` kwarg.
6. Migrate every example + test. Run them all.

---

## Verification

- **Unit**: existing `tests/` (after migration) must pass. Add: weighted task selection picks highest, weighted policy selection picks highest, custom selector overrides weights, `LocalArbitrator.switch()` emits one event with correct from/to.
- **Sim smoke**: `python examples/user_kuka_sinusoid/run_mujoco.py` runs an episode without error; `run_ros2_rviz.py` likewise; `run_gazebo.py` likewise.
- **ROS2 cleanup proof**: `grep -rn "import rclpy" examples/` returns nothing. `grep -rn "Ros2Runtime" examples/` returns nothing.
- **Multi-robot smoke**: write a 2-robot example (one Kuka + one Franka, two tasks each, weights set); run under MuJoCo; confirm both robots step, weights select expected task, switching at runtime works.
- **Multi-policy smoke**: define two policies for one task with weights {0.9, 0.1}; confirm policy 0 actions match those of an equivalent single-policy run within tolerance.

---

## Critical Files Touched (quick index)

- New: `robodeploy/core/robot.py`, `core/selectors.py`, `core/local_arbitrator.py`, `backends/real/ros2/adapters_base.py`, `backends/real/ros2/dev/fake_joint_sim.py`
- Modify: `robodeploy/env.py`, `core/interfaces/backend.py`, `orchestration/env_router.py`, `backends/real/ros2/backend.py`, `backends/real/ros2/controllers/*`, all examples, all tests
- Delete: `robodeploy/core/robot_config.py`, `robodeploy/core/task_config.py`, `robodeploy/core/arbitrator.py` (replaced), `examples/user_kuka_sinusoid/ros2_fake_jointpos_sim.py`
