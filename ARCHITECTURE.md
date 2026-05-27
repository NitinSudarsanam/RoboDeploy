# RoboDeploy Architecture

> Status: partially current. The active code now uses `RoboEnv(backend=..., robots=[Robot(...)])`, `RobotTask`, per-robot selectors, explicit multi-backend methods, `WorldSpec`, process-owned `RoboBridge`, and sensor pairing. Older sections that mention `RobotConfig`, `TaskConfig`, `MujocoEngine`, env-wide task lists, or planned bridge internals should be read as historical design notes until this document is fully rewritten.

A modular bridge library for robot learning — swap simulators, robots, and policies
without changing user code. Primary design goal: **sim-to-real transfer**.

---

## Design Principles

1. **One axis of variation per layer.** Each layer solves exactly one problem.
   Swapping any layer (robot, backend, policy, task) does not require changes in any other layer.

2. **No shared state between layers.** Layers communicate only through the types
   defined in `core/types.py` (`Observation`, `Action`, `SensorData`).

3. **Base classes absorb boilerplate.** Each interface has a `*Base` class that handles
   lifecycle guards and shared state. Concrete subclasses implement only the logic
   unique to them.

4. **Sim-to-real by construction.** `ObsPipeline` and `SafetyFilter` run identically
   in simulation and on real hardware. There is no separate "real mode" — only the
   backend changes.

5. **Extension without modification.** New robots, backends, policies, and tasks are
   added by creating a new file and applying a `@register_*` decorator. Nothing else changes.

---

## System Diagram (current)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER / RESEARCHER                             │
│  # String-based wiring (registry):                                   │
│  env = RoboEnv.make(robot=\"franka\", backend=\"mujoco\",             │
│                     task=\"pick_place\", policy=\"robomimic\")        │
│                                                                      │
│  # Object-based wiring (current architecture):                       │
│  env = RoboEnv(backend=backend, robots=[robot0, robot1])             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          RoboEnv  (env.py)                           │
│  reset() / step() / close()                                          │
│  Owns episode state. Calls backend multi-robot methods.              │
└──────┬─────────────────────────────────┬────────────────────────────┘
       │                                 │
       ▼                                 ▼
┌──────────────────────┐   ┌─────────────────────────────────────────┐
│  IBackend (shared)   │   │  robots: list[Robot]                     │
│                      │   │  ┌───────────────────────────────────┐  │
│  One physics world   │   │  │ Robot robot_id=\"robot0\"          │  │
│  or hardware fleet.  │   │  │  description, obs_pipeline,        │  │
│  Knows robots/assets │   │  │  action_adapter, sensors, tasks    │  │
│  but not tasks/pols. │   │  └───────────────────────────────────┘  │
│                      │   │  ┌───────────────────────────────────┐  │
│  reset_multi()       │   │  │ RobotTask task_id=\"pick\"         │  │
│  get_obs_multi()     │   │  │  task + 1..N policies + selector   │  │
│  step_multi(actions) │   │  └───────────────────────────────────┘  │
└──────────────────────┘   └─────────────────────────────────────────┘

Built-in components are registered lazily (see `robodeploy/builtins.py`).
```

---

## Sim-to-Real Data Flow

The pipelines are different *configurations* of the same transform library —
not different code paths. `is_real` must not appear inside `ObsPipeline`,
`ActionAdapter`, `ITask`, or `IPolicy`. Those layers are backend-agnostic by design.

```
SIM PIPELINE (training)                REAL PIPELINE (deployment)
──────────────────────────────────────────────────────────────────
ObsPipeline([                          ObsPipeline([
  GaussianNoiseTransform(),              NormalizeTransform(real_stats), ← per-domain
  NormalizeTransform(sim_stats),         UndistortTransform(intrinsics), ← lens correction
])                                     ])
```

**`is_real` escape hatch**: `backend.is_real` is available and may be read in
exactly two places: (1) `RoboEnv` during component wiring (e.g., sensor suffix
resolution, choosing which `ObsPipeline` config to build), and (2) `ITask.reset_fn`
/ `reset_routine` to choose between teleport and `HumanInterventionRequired`.
Anywhere else — logging, friction compensation, network drop handling — belongs
either in the backend subclass itself or in a backend-specific `ITransform` added
to the pipeline at construction time, not as branching in shared code.

Full step-by-step data flow:

```
MuJoCoBackend.get_obs()  /  ROS2Backend.get_obs()
  timestamp, timestamp_hw, timestamp_recv all populated
        │
        ▼
  ObsPipeline.process(obs)          ← no is_real flag; pipeline config is the only difference
        │
        ▼
  Policy.get_action(obs)            ← identical Observation structure, sim or real
        │
        ▼
  ActionAdapter.process(action)     ← space conversion, chunk buffering, scaling
        │
        ▼
  SafetyFilter.filter(action)       ← always active, sim and real; e-stop here
        │
        ┌────────────────┴──────────────────────┐
        ▼                                       ▼
MuJoCoBackend.step(action)             ROS2Backend / ControlLoop
  MJX physics step                       ControlLoop thread at 100Hz
                                         InferenceLoop calls policy async
```

## Real Hardware: Decoupled Control and Inference

On real hardware a slow policy must never block the hardware command stream.
`RoboBridge` separates the two loops. The `ControlLoop` **must** run in a
`multiprocessing.Process` (not a thread) to escape the Python GIL. `ActionTrajectory`
is a **seqlock** shared-memory ring buffer — writer increments sequence counter
before and after write; reader retries if counters don't match. No mutex means
`ControlLoop` never blocks on `InferenceLoop`.

**Seqlock crash safety**: if `InferenceLoop` crashes mid-write, the sequence counter
stays odd and `ControlLoop` would spin forever. `ActionTrajectory.pop_interpolated()`
has a hard **spin timeout** (default 500μs). If the seqlock is still inconsistent
after the timeout, it treats the buffer as empty and applies decay behavior. A
separate **InferenceLoop watchdog** process monitors the InferenceLoop heartbeat and
signals `EStopFlag` if no heartbeat is received within 2× the expected inference period.

```
┌─────────────────────────────────────────────────────────────────┐
│  ControlLoop  (100Hz, isolated OS process)                      │
│    1. Check EStopFlag  ← if set: inject decel, signal HW estop │
│    2. Read joint state                                          │
│    3. NaN/Inf guard on incoming state  ← sensor fault check    │
│    4. ActionTrajectory.pop_interpolated()  ← seqlock + timeout │
│       → see "Empty buffer behavior" below                       │
│    5. SafetyFilter Tier 1: scalar clamping  ← <0.1ms           │
│    6. NaN/Inf guard on outgoing action  ← last line of defense  │
│    7. backend.step()                                            │
└─────────────────────────────────────────────────────────────────┘
                        ▲
          ActionTrajectory  (seqlock shared-memory ring buffer)
          Bounded size = MAX_CHUNKS * chunk_size.
          Full → drop oldest chunk, never block.
          EStopFlag set → ControlLoop clears buffer and injects a
            joint-space deceleration trajectory immediately.
          All interpolation in JOINT SPACE (not EE space) — avoids
            singularities. ActionChunkTransform converts EE chunks
            to joint-space via IK before storing.
                        │
┌─────────────────────────────────────────────────────────────────┐
│  InferenceLoop  (adaptive rate, isolated process)               │
│    0. Heartbeat ping to watchdog  ← every iteration            │
│    1. backend.get_obs()   + hardware timestamps                 │
│    2. ObsPipeline.process() + sensor sync (drop/window)         │
│    3. ActionAdapter.process()  ← IK, chunk transform           │
│       NaN/Inf guard: IK fail → call policy.notify_rejected(),  │
│         drop chunk, continue  ← keeps sequence model in sync   │
│    4. SafetyFilter Tier 2: collision/IK feasibility ← <10ms    │
│       Rejection also calls policy.notify_rejected()            │
│    5. policy.get_action()   ← slow VLA? chunk absorbs it        │
│    6. ActionTrajectory.put_chunk(joint_actions, t_start)        │
└─────────────────────────────────────────────────────────────────┘
```

**E-stop is not software-only**. `EStopFlag` triggers two things simultaneously:
1. `ControlLoop` clears `ActionTrajectory` and injects a pre-computed deceleration
   profile (quintic, computed at init from max joint velocities) into the buffer.
2. `ControlLoop` asserts the hardware e-stop pin via the robot driver. Software
   trajectory deceleration and hardware brakes run in parallel — whichever stops
   the robot first wins. Never rely on software alone for a physical e-stop.

**NaN guard order**: `ActionAdapter` (step 3) runs before `SafetyFilter` (step 4).
IK failures call `policy.notify_rejected(obs, action)` before dropping — sequence
models (VLAs, RNNs) update their internal state to account for the missed action.

### Action integration with variable inference latency

Policies declare `action_hz` (e.g., 5Hz for VLA, 50Hz for Diffusion, 100Hz for
scripted). `ActionChunkTransform` sizes chunks using **p99 latency** (not EMA mean)
as the floor, plus a minimum buffer depth invariant: the buffer must always hold at
least 2 full chunks before the policy is considered "warm." This absorbs sudden
latency spikes without under-running.

```python
# Chunk sizing — p99-based, not EMA-mean:
chunk_size = max(
    int(p99_inference_ms / step_ms * 1.2),   # 20% margin on worst-case latency
    min_chunk,                                # absolute floor (e.g., 5 steps)
)
# Refill trigger: start new inference when buffer_depth < 1.5 * chunk_size
# (not when empty — always stay ahead)
```

**Empty buffer behavior — action-space-aware with hard epsilon clamp:**

| Action space | Buffer empty behavior |
|---|---|
| `JOINT_POS` (absolute) | Hold last commanded position |
| `JOINT_VEL` / `DELTA_POS` | Exponential decay × clamp to 0 when \|v\| < ε |
| `JOINT_TORQUE` | Ramp to pre-computed gravity-compensation torques |
| `TeleopPolicy` (any space) | Hold last commanded position — never decay |

`ε = 0.001 rad/s` per joint (below motor noise floor). Without this clamp,
exponential decay asymptotes → continuous motor whine and servo overheating.

`TeleopPolicy` overrides empty-buffer behavior to "hold" — a human pausing is
intentional, not a failure. Decay logic is for policy inference failures only.

Gravity-compensation torques are pre-computed from `RobotDescription` at init.
`ControlLoop` holds them in a pre-allocated array — no allocation in the hot path.

**All `ActionTrajectory` interpolation is in joint space.** `ActionChunkTransform`
runs IK at chunk-generation time to convert EE-space policy outputs to joint
trajectories before storing. The ControlLoop never does EE-space interpolation —
this prevents driving through kinematic singularities between valid endpoints.

`IPolicy` interface:

```python
action_hz      : float   # nominal execution frequency; seeds p99 tracking
notify_rejected(obs, action) → None   # optional; called when chunk is dropped
                                       # sequence models override to stay in sync
```

### Per-Robot Control Frequency

Robots with different control requirements (e.g., 1000Hz force-controlled arm,
20Hz mobile base) **must not be forced onto the same control loop**. Model this as
separate robots (separate `Robot` aggregates) or separate `RobotTask`s that are
stepped independently by the backend. Bundling incompatible frequencies into one
batched policy call destroys both.

## Episode Reset on Real Hardware

`reset_fn()` (teleport) works only in sim. On real hardware, `reset_routine()`
is a generator that yields a safe trajectory home, then optionally raises
`HumanInterventionRequired` to pause for the operator:

```python
# Sim reset_routine (default): empty generator, calls reset_fn (teleport)
def reset_routine(self, backend): ...  # inherited default

# Real hardware reset_routine (override in your task):
def reset_routine(self, backend: IBackend) -> Iterator[Action]:
    if not backend.is_real:
        self.reset_fn(backend)
        return

    # Move arm to home position safely
    yield from self._plan_to_home(backend.get_obs())

    # Prompt operator to reposition physical objects
    raise HumanInterventionRequired(
        "Place the red cube at the marked position, then press Enter."
    )
```

`RoboEnv.reset()` and `RoboBridge` both handle this correctly.

**Multi-agent safety**: `HumanInterventionRequired` raised by any task triggers a
**global e-stop broadcast** to all `ControlLoop` processes before prompting the
operator. A human entering the workspace must stop all robots, not just one.
`RoboBridge` owns a shared `EStopFlag` (shared-memory boolean) read by every
`ControlLoop` at the top of each tick — before any action is popped from the buffer.

---

## Robot-centric design: Robot + RobotTask (current)

The current runtime is organized around a `Robot` aggregate and `RobotTask`
bundles (see `robodeploy/core/robot.py`). This replaced the older env-wide
`RobotConfig` / `TaskConfig` model to avoid cross-cutting wiring and to keep
arbitration local to each robot.

```python
from robodeploy import RoboEnv
from robodeploy.core.robot import Robot, RobotTask

robot = Robot(
    robot_id=\"robot0\",
    description=MyRobotDescription(),
    tasks={
        \"task0\": RobotTask(
            task=MyTask(),
            policies={\"p\": MyPolicy()},
            mode=\"sequential\",  # or \"concurrent\"
        )
    },
)
env = RoboEnv(backend=my_backend, robots=[robot])
```

### Responsibility split (current)

| Concern | Where |
|---|---|
| Robot assets/joints/limits | `Robot.description` |
| Observation transforms | `Robot.obs_pipeline` |
| Action transforms | `Robot.action_adapter` |
| Robot-local sensors | `Robot.sensors` |
| Task logic (reward/success/reset) | `RobotTask.task` |
| Policy choice within a task | `RobotTask.policy_selector` (or weights) |
| Backend execution | `RoboEnv.backend` |

### Legacy appendix (historical)

Older notes may reference `RobotConfig` / `TaskConfig`. Those names do not exist
in the current codebase; treat them as historical design docs.

---

## Package Structure

```
robodeploy/
│
├── core/                          # The Contract — never import backends here
│   ├── types.py                   # Observation, Action, SensorData, ObsSpec,
│   │                              #   SceneSpec, EpisodeInfo  (all dataclasses)
│   ├── spaces.py                  # ActionSpace enum, AssetFormat enum
│   ├── robot.py                   # Robot + RobotTask aggregates
│   ├── interfaces/
│   │   ├── backend.py             # IBackend (ABC)
│   │   ├── policy.py              # IPolicy  (ABC)
│   │   ├── task.py                # ITask    (ABC)
│   │   └── sensor.py              # ISensor  (ABC)
│   ├── registry.py                # @register_backend/robot/policy/task/sensor
│   └── interop.py                 # JAX ↔ NumPy ↔ Torch zero-copy
│
├── description/                   # Static robot definitions — no runtime state
│   ├── base.py                    # RobotDescription (ABC)
│   │                              #   → asset_path(fmt), get_kinematics_solver(),
│   │                              #     get_safety_filter()
│   └── franka/
│       ├── description.py         # FrankaDescription  @register_robot("franka")
│       ├── assets/mjcf/panda.xml
│       ├── assets/urdf/panda.urdf
│       └── assets/usd/panda.usd   (auto-generated)
│
├── backends/                      # One adapter per execution environment
│   ├── base.py                    # BackendBase(IBackend)
│   │                              #   Adds: lifecycle guards, episode/step counters
│   │                              #   Subclasses implement: _load, _reset_impl,
│   │                              #     _step_impl, _get_obs_impl, _close_impl
│   ├── sim/
│   │   ├── mujoco/backend.py      # MuJoCoBackend   @register_backend("mujoco")
│   │   ├── isaacsim/backend.py    # IsaacSimBackend @register_backend("isaacsim")
│   │   └── gazebo/backend.py      # GazeboBackend   @register_backend("ros2_gazebo")
│   └── real/
│       ├── ros2/backend.py        # ROS2Backend     @register_backend("ros2")
│       └── ...                    # real hardware adapters
│
├── policies/                      # Brains — no physics, no hardware awareness
│   ├── base.py                    # PolicyBase(IPolicy)
│   │                              #   Adds: episode counter, set_instruction,
│   │                              #     get_action_batch() (VecEnv extension point)
│   │                              #   Subclasses implement: get_action, _reset_impl
│   ├── learned/
│   │   ├── robomimic.py           # RobomimicPolicy  @register_policy("robomimic")
│   │   ├── diffusion.py           # DiffusionPolicy  @register_policy("diffusion")
│   │   └── vla.py                 # VLAPolicy        @register_policy("vla")
│   ├── scripted/
│   │   ├── waypoint.py            # WaypointPolicy   @register_policy("waypoint")
│   │   └── joint_pd.py            # JointPDPolicy    @register_policy("joint_pd")
│   └── remote/
│       ├── remote_policy.py       # RemotePolicy(PolicyBase) — IPolicy over network
│       │                          #   Drop-in for any local policy. One constructor change.
│       ├── transport.py           # IPolicyTransport (ABC)
│       │                          #   ZmqTransport  — dev/research, pip install pyzmq
│       │                          #   GrpcTransport — production, requires proto stubs
│       └── server.py              # PolicyServer — hosts any IPolicy on the GPU machine
│                                  #   serve(policy, host, port, transport="zmq")
│
├── sensors/                       # Modular perception, always in sim/real pairs
│   ├── base.py                    # SensorBase(ISensor)
│   │                              #   Adds: lifecycle guards, name/is_real props
│   │                              #   Subclasses implement: _init_impl, _read_impl,
│   │                              #     _close_impl
│   ├── camera/
│   │   ├── sim/mujoco_camera.py   # @register_sensor("wrist_camera_sim")
│   │   └── real/realsense.py      # @register_sensor("wrist_camera_real")
│   └── ft_sensor/
│       ├── sim/mujoco_ft.py       # @register_sensor("ft_sensor_sim")
│       └── real/ati_ft.py         # @register_sensor("ft_sensor_real")
│
├── tasks/                         # Scene + goal — backend-agnostic
│   ├── base.py                    # TaskBase(ITask)
│   │                              #   Adds: step/episode counters, default failure_fn
│   │                              #   Subclasses implement: obs_spec, scene_spec,
│   │                              #     language_instruction, reset_fn, reward_fn,
│   │                              #     success_fn
│   ├── randomization.py           # DomainRandomizer + DomainRandomizerConfig
│   │                              #   RandomLevel: NONE / LIGHT / FULL
│   └── manipulation/
│       ├── pick_place.py          # PickPlaceTask   @register_task("pick_place")
│       ├── pour.py                # PourTask        @register_task("pour")
│       └── peg_insertion.py       # PegTask         @register_task("peg_insertion")
│
├── kinematics/                    # Pure math — no backend dependency
│   ├── solver.py                  # KinematicsSolver: fk(), ik(), jacobian()
│   │                              #   Backed by Pinocchio. Reads URDF directly.
│   └── safety.py                  # SafetyFilter: filter(), trigger_estop()
│                                  #   Clamps joints/velocity/torque. Always active.
│
├── core/
│   └── transforms.py              # ITransform (ABC) + built-in transforms:
│                                  #   GaussianNoiseTransform — add encoder noise in sim
│                                  #   NormalizeTransform     — zero-mean/unit-var, fit from data
│                                  #   IdentityTransform      — no-op default
│
├── obs_pipeline.py                # ObsPipeline([ITransform, ...]): ordered transform chain
│                                  #   No is_real flag. Pipeline config IS the sim/real difference.
│                                  #   process(obs) → obs  |  fit(dataset)  |  append(transform)
│
├── action_adapter.py              # ActionAdapter([IActionTransform, ...]): mirrors ObsPipeline
│                                  #   Sits between policy and SafetyFilter.
│                                  #   Built-ins: IdentityActionTransform, ScaleActionTransform,
│                                  #     DeltaEEToJointPosTransform (IK), ActionChunkTransform
│
├── env.py                         # RoboEnv: gym-compatible orchestrator
│                                  #   RoboEnv(backend=..., robots=[Robot(...)], shared_sensors=[])
│                                  #   RoboEnv.make(robot=, backend=, task=, policy=)
│                                  #   RoboEnv.from_preset(name)
│                                  #   RoboEnv.from_config(cfg)
│                                  #   step() / reset() return scalar when 1 task,
│                                  #     dict-like obs when multi-robot (see `core/types.py`)
│                                  #   Handles reset_routine + HumanInterventionRequired
│
└── bridge.py                      # RoboBridge: decoupled real-time hardware deployment
                                   #   ActionBuffer    — one per agent, thread-safe
                                   #   ControlLoop     — one per agent, 100Hz, SafetyFilter here
                                   #   InferenceLoop   — one per agent, async, any rate
                                   #   async with RoboBridge(env) as b: await b.run()
                                   #   Single-agent: identical behavior as before
                                   #   Multi-agent: N ControlLoops + N InferenceLoops,
                                   #     each isolated; one agent fault does not stop others
```

---

## Legacy / historical notes (may be stale)

The sections below are retained as historical design notes. They are not guaranteed
to match the current codebase.

## Interface Contracts

### IBackend

Backend owns the physics world / hardware fleet. It knows robots and scene props
— not tasks, not policies. `SceneSpec` is passed directly (extracted by `RoboEnv`
from each task's `scene_spec()` and merged) so the backend can load props without
any dependency on `ITask`. This breaks the circular init: backend does not import
or call tasks.

```python
initialize(robots: list[RobotConfig],
           scene:  SceneSpec,           # merged from all tasks' scene_spec()
           shared_sensors: list[ISensor]) → None
reset(robot_ids: list[str] | None = None) → list[Observation]
  # robot_ids=None resets all robots
step(actions: list[Action])            → list[Observation]
get_obs()                              → list[Observation]
close()                                → None
is_real              : bool            # property
robot_count          : int             # property
supported_action_spaces: list[ActionSpace]
control_hz           : float
# scene methods:
get_prop_pose(name)                    → np.ndarray
set_prop_pose(name, pose)              → None   # sim only; no-op real
set_prop_mass(name, mass)              → None   # sim only; no-op real
get_prop_names()                       → list[str]
set_payload(robot_id, mass, com)       → None   # BOTH sim and real hardware
  # On Franka: calls FCI set_payload(). Updates gravity compensation model.
  # Must be called whenever robot picks up or releases an object.
# optional:
set_physics_params(robot_id, **kwargs) → None   # physics randomisation
```

`set_payload()` is distinct from `set_prop_mass()`. Props are scene objects.
Payload is the robot's end-effector load — affects gravity compensation in the
hardware driver. Never updating this on real hardware causes tracking drift when
the robot is holding a heavy object.

`BackendBase` provides single-robot shims (`step_single`, `reset_single`,
`get_obs_single`) that unwrap lists. Existing `MuJoCoBackend` / `ROS2Backend`
subclasses need no changes for the single-robot path.

### IPolicy

```python
reset()                     → None      # clear buffers, called each episode
get_action(obs)             → Action
action_space                : ActionSpace  # property
action_hz                   : float        # property — nominal execution frequency;
                                           # seeds adaptive chunk EMA in ActionChunkTransform
# optional:
get_action_batch(obs_list)  → list[Action]  # vectorized path for shared multi-robot policy
                                             # default impl: [get_action(o) for o in obs_list]
                                             # override for GPU-batched inference (PyTorch/JAX)
set_instruction(text)       → None      # for VLAs
warmup(obs)                 → None      # JIT compile trigger
```

`get_action_batch()` is the correct path for a single shared policy controlling N
robots (e.g., `TaskConfig(robot_ids=["r1","r2","r3"], policy=shared_policy)`).
`RoboEnv` detects multi-robot tasks and calls `get_action_batch()` instead of N
sequential `get_action()` calls. PyTorch and JAX are not thread-safe; batching
through a single call avoids concurrent inference on shared model weights.

### TeleopPolicy

`WaypointPolicy` is open-loop trajectory replay — not teleoperation.
True teleoperation requires async event-driven input from a human device.
`TeleopPolicy` is event-driven: `IInputDevice` pushes deltas to a queue;
`get_action()` drains the queue. At 100Hz the queue depth is typically 1-2 events.

```python
# policies/scripted/teleop.py
class TeleopPolicy(PolicyBase):
    """Event-driven: IInputDevice pushes to queue; get_action() drains it."""
    action_hz = 100.0    # matches control loop — no chunking needed

    def __init__(self, device: IInputDevice): ...
    def get_action(self, obs: Observation) -> Action:
        # drain device event queue (non-blocking) → delta EE pose
        # IK handled by ActionAdapter, not here
        ...

# sensors/input/spacemouse.py   @register_sensor("spacemouse")
# sensors/input/vr_controller.py @register_sensor("vr_controller")
```

`IInputDevice` is a specialization of `ISensor` — same lifecycle, same `read()`
contract. `read()` returns a `SensorData` with a `delta_pose` field. The device
driver runs in a dedicated thread (or uses OS HID callbacks) and pushes to a
`queue.SimpleQueue` — `get_action()` reads from the queue, never from the device
directly. This decouples OS USB/Bluetooth polling from the 100Hz control rate.

### ITask

```python
obs_spec()                  → ObsSpec
scene_spec()                → SceneSpec   # declares props; backend loads them
language_instruction()      → str
reset_fn(backend)           → None        # randomise prop poses each episode
reward_fn(obs, action)      → float
success_fn(obs)             → bool
failure_fn(obs)             → bool
max_steps()                 → int
# optional:
on_activate()               → None        # called by Arbitrator when task becomes active
on_deactivate()             → None        # called by Arbitrator when task yields control
```

### SceneConfig and PropConfig

The backend owns the full scene state — robots **and** props. Props (cubes, mugs,
pegs) are declared in `SceneSpec` returned by `ITask.scene_spec()`. The backend
loads them at `initialize()` time. Tasks manipulate them via `IBackend` scene methods.

```python
# core/types.py  (additions)
@dataclass
class PropConfig:
    name:     str
    asset:    Path            # MJCF / URDF / USD for this prop
    pose:     np.ndarray      # initial pose [x, y, z, qw, qx, qy, qz]
    mass:     float = 0.1
    is_fixed: bool  = False   # fixed in world vs. free-floating

@dataclass
class SceneSpec:
    props: list[PropConfig] = field(default_factory=list)
    floor: bool             = True
    lighting: str           = "default"
```

`IBackend` scene methods (optional, used by `DomainRandomizer` and `reset_fn`):

```python
# IBackend additions:
get_prop_pose(name: str)             → np.ndarray   # [x, y, z, qw, qx, qy, qz]
set_prop_pose(name: str, pose)       → None
get_prop_names()                     → list[str]
```

On real hardware `set_prop_pose` is a no-op — props are physical objects.
`reset_fn` raises `HumanInterventionRequired` to prompt the operator instead.

**Dynamic scene changes** (e.g., liquid mass increasing as a cup is filled) are
supported via backend scene methods called from within `reward_fn` or `success_fn`:

```python
# IBackend additions:
set_prop_mass(name: str, mass: float) → None   # sim only; no-op on real
get_prop_pose(name: str)              → np.ndarray
set_prop_pose(name: str, pose)        → None   # sim only; no-op on real
```

`PropConfig.mass` is the *initial* mass. Tasks may mutate it during an episode by
calling `backend.set_prop_mass()` in their step logic. The real-hardware no-op means
tasks that rely on dynamic mass must derive progress only from observable state
(e.g., FT sensor reading increasing), never from the programmed mass value.

### ISensor

```python
initialize(backend)         → None      # attach to sim renderer or open hardware
read()                      → SensorData
close()                     → None
name                        : str       # property
is_real                     : bool      # property
# optional:
warmup()                    → None      # discard unstable initial frames
```

`SensorData.timestamp` is mandatory. Timestamp sourcing priority:

1. **Hardware timestamp** (preferred): device driver timestamp — ROS2
   `sensor_msgs` header stamp, camera SDK frame timestamp (e.g., RealSense
   `frame.timestamp`), robot FCI receive timestamp. Reflects actual measurement
   time, not Python receipt time.
2. **OS monotonic fallback** (acceptable for cheap sensors): `time.monotonic_ns()`
   at the point of blocking `read()` return. Incurs OS scheduler jitter (±2–5ms).
   Sensor implementation **must** set `SensorData.timestamp_source = "software"`
   so downstream sync policies can apply appropriate jitter tolerance.
3. **Sim**: `mujoco.MjData.time` — exact, no jitter.

Many USB webcams lack hardware timestamps. The rule is not "ban OS time" —
it is "never lie about which you used." `TIME_WINDOW` sync policy automatically
widens its window when `timestamp_source == "software"`.

Camera sensors implement an additional optional interface for calibration:

```python
# optional, on camera ISensor subclasses:
intrinsics()  → CameraIntrinsics   # fx, fy, cx, cy, distortion coefficients
extrinsics()  → np.ndarray         # 4x4 camera-to-robot-base transform
```

`MuJoCoCameraRenderer.intrinsics()` reads from the MJCF `<camera>` element.
`RealSenseCamera.intrinsics()` reads from the RealSense SDK calibration API.
`ObsPipeline` can include an `UndistortTransform` that applies lens distortion
correction using these intrinsics — bridging sim (perfect pinhole) and real
(distorted lens).

`ObsPipeline` sync policy (set per-pipeline, not per-sensor):

```python
class SyncPolicy(Enum):
    DROP_LATEST   = "drop_latest"   # use most recent frame per sensor; accept skew
    TIME_WINDOW   = "time_window"   # only fuse frames within ±N ms; else use cached
    # HARDWARE_TRIGGER: no software needed; sensors already synchronized

# ObsPipeline gains:
sync_policy:      SyncPolicy = SyncPolicy.DROP_LATEST
sync_window_ms:   float      = 15.0   # used only for TIME_WINDOW
```

For state-based policies, `DROP_LATEST` is always correct.
For vision-fusing policies on real hardware, use `TIME_WINDOW` with the camera's
frame period as the window (e.g., 33ms for 30Hz cameras).

### RobotDescription

```python
dof                         : int
joint_names                 : list[str]
joint_position_limits       : ndarray [dof, 2]
joint_velocity_limits       : ndarray [dof]
joint_torque_limits         : ndarray [dof]
home_qpos                   : ndarray [dof]
ee_link_name                : str
asset_path(fmt)             → Path      # URDF / MJCF / USD; auto-converts
get_kinematics_solver()     → KinematicsSolver
get_safety_filter()         → SafetyFilter
```

---

## Three Ways to Create a RoboEnv

Choose the level that fits your project. All three produce the same `RoboEnv` object.

### Level 1 — Direct Injection (recommended for most users)

Pass Python objects directly. Most readable, full IDE autocomplete, no registration
step, supports runtime-constructed objects. Best for custom robots, tasks, and policies.
Never requires touching library source folders.

**Single agent** (unchanged API — `description`/`task`/`policy`/`sensors` kwargs still work):

```python
from robodeploy import RoboEnv
from robodeploy.backends.sim.mujoco.backend     import MuJoCoBackend
from robodeploy.description.franka              import FrankaDescription
from robodeploy.tasks.manipulation.pick_place   import PickPlaceTask
from my_project.policy                          import MyPolicy
from my_project.camera                          import MySimCamera

env = RoboEnv(
    description = FrankaDescription(),
    backend     = MuJoCoBackend(config={"enable_viewer": True}),
    task        = PickPlaceTask(),
    policy      = MyPolicy(checkpoint="ckpt.pt"),
    sensors     = [MySimCamera()],
)
```

**Multi-robot / multi-task** (new path — pass `robots` + `tasks` lists):

```python
from robodeploy.core.robot_config import RobotConfig
from robodeploy.core.task_config  import TaskConfig

env = RoboEnv(
    robots = [
        RobotConfig(FrankaDescription(), sensors=[MySimCamera()], robot_id="franka"),
        RobotConfig(KukaDescription(),   sensors=[AnotherSimCamera()], robot_id="kuka"),
    ],
    tasks = [
        TaskConfig(PickPlaceTask(), robot_ids=["franka"],
                   policy=MyPolicy(checkpoint="franka.pt")),
        TaskConfig(PourTask(),      robot_ids=["kuka"],
                   policy=OtherPolicy(checkpoint="kuka.pt")),
    ],
    backend        = MuJoCoBackend(config={"enable_viewer": True}),
    shared_sensors = [OverheadCamera()],
)
```

Swap to real hardware by changing one line — all other code is identical:

```python
from robodeploy.backends.real.ros2.backend import ROS2Backend
from my_project.camera                     import MyRealCamera

env = RoboEnv(
    description = FrankaDescription(),
    backend     = ROS2Backend(),           # ← only change
    task        = PickPlaceTask(),
    policy      = MyPolicy(checkpoint="ckpt.pt"),
    sensors     = [MyRealCamera()],        # ← only change
)
```

### Level 2 — `use()` + `make()` (for config-driven swapping)

Register your components once with `use()`, then swap them by string name.
Best for RL experiments where you want to change components via YAML or CLI
without editing code.

```python
from robodeploy import use, RoboEnv

# One line per module — imports trigger @register_* decorators in your code.
# Your files live in your own project, never inside the robodeploy source tree.
use("my_project.robots")     # contains @register_robot("myrobot")
use("my_project.tasks")      # contains @register_task("my_task")
use("my_project.policies")   # contains @register_policy("my_policy")

env = RoboEnv.make(
    robot   = "myrobot",
    backend = "mujoco",      # swap to "ros2" for real hardware
    task    = "my_task",
    policy  = "my_policy",
    policy_kwargs = {"checkpoint": "ckpt.pt"},
    sensors = ["my_camera"], # auto-resolves to my_camera_sim or my_camera_real
)
```

### Level 3 — `from_config()` (for YAML / Hydra pipelines)

Load everything from a config dict. Works with plain dicts, OmegaConf DictConfig,
and Hydra — no Hydra dependency required in RoboDeploy itself.

```python
# Plain dict
from robodeploy import RoboEnv

env = RoboEnv.from_config({
    "robot":          "franka",
    "backend":        "mujoco",
    "task":           "pick_place",
    "policy":         "my_policy",
    "policy_kwargs":  {"checkpoint": "ckpt.pt"},
    "custom_modules": ["my_project.policies"],   # use() called automatically
})

# With Hydra (from conf/experiment.yaml)
@hydra.main(config_path="conf", config_name="experiment")
def main(cfg: DictConfig):
    env = RoboEnv.from_config(cfg)   # OmegaConf DictConfig works transparently
```

### Level 3 + Entry Points (for pip-installable robot/task packages)

Third-party packages declare their components in `pyproject.toml`. Users install
the package and call `discover()` — no `use()` calls needed.

```python
from robodeploy import discover, RoboEnv

discover()   # scans entry points from all installed packages
env = RoboEnv.make(robot="community_robot", backend="mujoco", task="community_task")
```

```toml
# third_party_package/pyproject.toml
[project.entry-points."robodeploy.robots"]
community_robot = "third_party_package.robots:CommunityRobotDescription"
```

---

## Use Cases

### 1. RL Training (sim, sync loop)

```python
env = RoboEnv(
    description=FrankaDescription(), backend=MuJoCoBackend(), task=PickPlaceTask()
)
obs, info = env.reset()
while not done:
    action = policy.get_action(obs)
    obs, reward, done, info = env.step(action)
```

### 2. Policy Evaluation (compare across simulators)

```python
for BackendClass in [MuJoCoBackend, IsaacLabBackend, GenesisBackend]:
    env = RoboEnv(
        description = FrankaDescription(),
        backend     = BackendClass(),
        task        = PickPlaceTask(),
        policy      = RobomimicPolicy(checkpoint_path="ckpt.pth"),
    )
    results[BackendClass.__name__] = evaluate(env, n_episodes=50)
```

### 3. Sim-to-Real Transfer

```python
# Shared setup — identical for sim and real
description = FrankaDescription()
task        = PickPlaceTask()
policy      = RobomimicPolicy(checkpoint_path="ckpt.pth")
pipeline    = ObsPipeline(config=ObsPipelineConfig(add_joint_noise=True))

# Sim (training)
env = RoboEnv(description=description, backend=MuJoCoBackend(),
              task=task, policy=policy, obs_pipeline=pipeline)

# Real (deployment) — swap backend and sensors, everything else identical
env = RoboEnv(description=description, backend=ROS2Backend(),
              task=task, policy=policy, obs_pipeline=pipeline,
              sensors=[RealSenseCamera()])
bridge = RoboBridge(env, control_hz=100.0)
asyncio.run(bridge.run())
```

### 4. N Robots, 1 Task, 1 Shared Policy

Two arms collaborate on one task. One policy observes both, outputs two actions.

```python
r_franka = RobotConfig(description=FrankaDescription(), robot_id="franka")
r_kuka   = RobotConfig(description=KukaDescription(),   robot_id="kuka")

shared_task   = CoopPickPlaceTask()          # task.obs_spec covers both robots
shared_policy = BimanualPolicy(checkpoint="bimanual.pt")

env = RoboEnv(
    robots  = [r_franka, r_kuka],
    tasks   = [TaskConfig(task=shared_task, robot_ids=["franka", "kuka"],
                          policy=shared_policy)],
    backend = MuJoCoBackend(),
)
```

### 5. N Robots, N Tasks, N Policies (Heterogeneous Fleet)

Each arm has its own independent task and policy. One shared physics world.

```python
env = RoboEnv(
    robots = [
        RobotConfig(FrankaDescription(), robot_id="franka"),
        RobotConfig(KukaDescription(),   robot_id="kuka"),
    ],
    tasks = [
        TaskConfig(PickPlaceTask(), robot_ids=["franka"],
                   policy=RobomimicPolicy("franka.pt")),
        TaskConfig(PourTask(),      robot_ids=["kuka"],
                   policy=DiffusionPolicy("kuka.pt")),
    ],
    backend = MuJoCoBackend(),
)
```

### 6. 1 Robot, N Tasks, 1 Shared Policy (Multi-Task Policy)

One arm, two tasks, one policy handles both (e.g., a VLA with language conditioning).

```python
robot  = RobotConfig(FrankaDescription(), robot_id="franka")
policy = VLAPolicy(checkpoint="vla.pt")    # same object — shared weights

env = RoboEnv(
    robots = [robot],
    tasks  = [
        TaskConfig(PickPlaceTask(), robot_ids=["franka"], policy=policy,
                   task_id="pick"),
        TaskConfig(PourTask(),      robot_ids=["franka"], policy=policy,
                   task_id="pour"),
    ],
    backend = MuJoCoBackend(),
)
# RoboEnv calls policy.set_instruction(task.language_instruction()) per task
```

### 7. 1 Robot, N Tasks, N Policies

One arm, two tasks, different specialist policies per task.

```python
env = RoboEnv(
    robots = [RobotConfig(FrankaDescription(), robot_id="franka")],
    tasks  = [
        TaskConfig(PickPlaceTask(), robot_ids=["franka"],
                   policy=RobomimicPolicy("pick.pt")),
        TaskConfig(PourTask(),      robot_ids=["franka"],
                   policy=DiffusionPolicy("pour.pt")),
    ],
    backend = MuJoCoBackend(),
)
```

### 8. Sim-to-Real Transfer (Any Configuration)

Swap backend and per-robot sensors. Robots and tasks unchanged.

```python
env = RoboEnv(
    robots  = [r_franka, r_kuka],   # same RobotConfigs
    tasks   = [t1, t2],             # same TaskConfigs
    backend = ROS2Backend(),        # ← only change
)
bridge = RoboBridge(env)
asyncio.run(bridge.run())
```

`RoboBridge` spawns one `ControlLoop` + `InferenceLoop` per **task**.
Each task's `ActionBuffer` is isolated — one task fault does not stop others.

### 9. Data Collection / Teleoperation

```python
env = RoboEnv(
    description = FrankaDescription(),
    backend     = MuJoCoBackend(),
    task        = PickPlaceTask(),
    policy      = WaypointPolicy(waypoints=[...]),
)
obs, info = env.reset()
while not done:
    obs, reward, done, info = env.step()
    log_episode(obs, info)
```

---

## Extension Points (no core changes required)

### Adding a new simulator (built-in, lives inside robodeploy)

```python
# robodeploy/backends/sim/genesis/backend.py
from robodeploy.core.registry import register_backend

@register_backend("genesis")
class GenesisBackend(BackendBase):
    is_real = False
    control_hz = 100.0
    supported_action_spaces = [ActionSpace.JOINT_POS, ActionSpace.JOINT_TORQUE]

    def _load(self, description, task, sensors): ...
    def _reset_impl(self): ...
    def _step_impl(self, action): ...
    def _get_obs_impl(self): ...
    def _close_impl(self): ...
```

### Adding your own robot (lives in YOUR project, not inside robodeploy)

```python
# my_project/robots/my_arm.py
import numpy as np
from robodeploy.core.registry    import register_robot
from robodeploy.core.spaces      import AssetFormat
from robodeploy.description.base import RobotDescription

@register_robot("my_arm")
class MyArmDescription(RobotDescription):
    dof = 6
    joint_names = ["j1", "j2", "j3", "j4", "j5", "j6"]
    joint_position_limits = np.array([[-3.14, 3.14]] * 6)
    joint_velocity_limits = np.array([3.0] * 6)
    joint_torque_limits   = np.array([50.0] * 6)
    home_qpos             = np.array([0, -1.57, 1.57, 0, 1.57, 0])
    ee_link_name          = "tool_flange"
    display_name          = "My Robot Arm"

    def asset_path(self, fmt: AssetFormat):
        from pathlib import Path
        assets = Path(__file__).parent / "assets"
        paths = {AssetFormat.URDF: assets / "urdf/my_arm.urdf",
                 AssetFormat.MJCF: assets / "mjcf/my_arm.xml"}
        return paths[fmt]

# Usage — Level 1 (no registration needed):
from my_project.robots.my_arm import MyArmDescription
env = RoboEnv(description=MyArmDescription(), backend=MuJoCoBackend(), ...)

# Usage — Level 2 (register once, use by string):
from robodeploy import use, RoboEnv
use("my_project.robots.my_arm")
env = RoboEnv.make(robot="my_arm", backend="mujoco", ...)
```

### Adding your own task (lives in YOUR project)

```python
# my_project/tasks/my_task.py
from robodeploy.core.registry import register_task
from robodeploy.tasks.base    import TaskBase

@register_task("my_task")
class MyTask(TaskBase):
    def obs_spec(self):             return ObsSpec(rgb=True)
    def scene_spec(self):           return SceneSpec(objects=[...])
    def language_instruction(self): return "Do the thing."
    def reset_fn(self, backend):    self.randomizer.randomize(backend)
    def reward_fn(self, obs, act):  return -float(np.linalg.norm(...))
    def success_fn(self, obs):      return bool(...)
```

### Distributed Inference (local ↔ remote, one constructor change)

The swap between local and remote inference is a single constructor change.
Everything else — `RoboBridge`, `ControlLoop`, `ActionAdapter`, `SafetyFilter` — is unchanged.

```
┌──────────────────────────────────┐  ZMQ / gRPC  ┌──────────────────────────────────┐
│  Robot machine (local, 100Hz)    │ ─────────────►│  GPU inference server            │
│                                  │              │                                  │
│  RoboBridge                      │              │  from robodeploy.policies.remote  │
│    ControlLoop   (100Hz thread)  │              │  import PolicyServer, serve      │
│    InferenceLoop                 │              │                                  │
│      └── RemotePolicy            │              │  serve(                          │
│           └── ZmqTransport ──────►─────────────►│    DiffusionPolicy("ckpt.pt"),   │
│                                  │              │    host="0.0.0.0", port=5555     │
│                                  │◄─────────────│  )                               │
└──────────────────────────────────┘  Action      └──────────────────────────────────┘
```

```python
# LOCAL inference (single machine)
from robodeploy.policies.learned.diffusion import DiffusionPolicy

policy = DiffusionPolicy(checkpoint="ckpt.pt")   # runs on this machine

# DISTRIBUTED inference — identical env/bridge code, one constructor change
from robodeploy.policies.remote import RemotePolicy, ZmqTransport

policy = RemotePolicy(
    transport    = ZmqTransport(host="gpu-server.local", port=5555),
    action_space = ActionSpace.JOINT_POS,
)

# Either policy drops into RoboBridge unchanged:
env    = RoboEnv(description=FrankaDescription(), backend=ROS2Backend(),
                 task=PickPlaceTask(), policy=policy)
bridge = RoboBridge(env)
asyncio.run(bridge.run())
```

```python
# On the GPU server machine — one line
from robodeploy.policies.remote.server import serve
from my_project.policies.diffusion     import DiffusionPolicy

serve(DiffusionPolicy(checkpoint="ckpt.pt"), host="0.0.0.0", port=5555)
```

Transport options:

| Transport | Use case | Setup |
|---|---|---|
| `ZmqTransport` | Development, same machine or LAN | `pip install pyzmq` |
| `GrpcTransport` | Production, TLS, typed schema | Generate proto stubs from `policies/remote/proto/policy.proto` |

Add a custom transport by implementing `IPolicyTransport` (3 methods: `connect`, `send_obs_recv_action`, `send_reset`, `close`). No other changes.

### Adding parallel RL (VecEnv — subclass, no core changes)

VecEnv and multi-agent solve different problems:

| | VecEnv | Multi-agent `RoboEnv` |
|---|---|---|
| Use case | N copies of same task (RL training) | N robots, different tasks/policies |
| Backend | N separate physics worlds | 1 shared physics world |
| Policy | Same policy, batched | Different policy per agent |
| Scene interaction | Independent | Agents share scene (can interact) |

```python
# VecEnv: wraps N independent RoboEnv instances for RL throughput.
class VecEnv:
    def __init__(self, envs: list[RoboEnv]):
        self.envs = envs

    def reset(self):
        return [e.reset() for e in self.envs]

    def step(self, actions):
        return [e.step(a) for e, a in zip(self.envs, actions)]
```

### Adding Hydra config (no changes to RoboEnv needed)

```yaml
# conf/experiment.yaml
robot:   franka
backend: mujoco
task:    pick_place
policy:  robomimic
policy_kwargs:
  checkpoint: outputs/ckpt.pth
custom_modules:           # use() called automatically by from_config()
  - my_project.policies
```

```python
@hydra.main(config_path="conf", config_name="experiment")
def main(cfg: DictConfig):
    env = RoboEnv.from_config(cfg)   # works directly with OmegaConf
```

---

## Inheritance Hierarchy

```
IBackend   ← BackendBase  ← MuJoCoBackend
                          ← IsaacLabBackend
                          ← ROS2Backend

IPolicy    ← PolicyBase   ← RobomimicPolicy
                          ← DiffusionPolicy
                          ← WaypointPolicy
                          ← TeleopPolicy

ITask      ← TaskBase     ← PickPlaceTask
                          ← PourTask

ISensor    ← SensorBase   ← MuJoCoCameraRenderer
                          ← RealSenseCamera
                          ← SpaceMouseInputDevice    (IInputDevice ← ISensor)
                          ← VRControllerInputDevice

RobotDescription          ← FrankaDescription
                          ← KukaDescription
```

---

## File Count Summary

| Layer | Files | Role |
|---|---|---|
| `core/` | 9 | types, interfaces, registry, `RobotConfig`, `TaskConfig`, `Arbitrator` |
| `description/` | 3 per robot | Static robot metadata + asset paths |
| `backends/` | 1 per backend | Physics/hardware adapters (robots + props) |
| `policies/` | 1 per policy | obs → action inference; each declares `action_hz` |
| `sensors/` | 2 per sensor type | sim + real pairs; includes input devices |
| `tasks/` | 1 per task | scene + props + reward + success |
| `kinematics/` | 2 | FK/IK (Tier 2 safety) + SafetyFilter (Tier 1) |
| Top-level | 4 | `obs_pipeline`, `env`, `bridge`, `action_trajectory` |

New components always add files, never modify existing ones.
