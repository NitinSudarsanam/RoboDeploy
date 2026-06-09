# Goal 12 — Real-Hardware Safety & Error Recovery

**Priority**: Tier 3. **Effort**: ~30h. **Touches**: production trust, hardware deployment.

## Problem

Real-backend safety primitives strong but not unified or wired into env. Sim backends lack safety. SafetyFilter referenced but missing. ROS2 connection loss = silent fallback. No reconnect.

## Current State (Audit)

### Real-hw safety (ROS2 only)
- `backends/real/ros2/safety.py:21-69` — `Watchdog` callback on timeout.
- `backends/real/ros2/safety.py:72-127` — `EStop` (SIGINT + console 'q').
- `backends/real/ros2/safety.py:129-153` — `JointLimitGuard`.
- `backends/real/ros2/safety.py:156-194` — `TemperatureGuard`.
- `backends/real/ros2/controllers/_clamp.py` — `slew_limit_command()`.
- `backends/real/ros2/controllers/base.py:37` — joint_state_timeout_s 1.0s default.

### Bridge safety
- `robodeploy/bridge.py:19-37` — `EStopFlag` (multiprocessing event) for pause/resume.
- Control loop applies last-safe position if no new action.

### Gaps
- No `SafetyFilter` class implementation (only mentioned in `core/types.py:102-103` comment).
- EStop not wired to `RoboEnv` — sim envs have no e-stop.
- ROS2 connection loss = silent timeout + warning string. No reconnect. No alert.
- No command ACK verification.
- No `RoboEnv.emergency_stop()` API.
- Watchdog only on SO-101 Feetech path; not general.
- No `Hazard` enum / structured violation reporting.
- No simulated safety violation injection for testing.

---

## Deliverables

### D1. SafetyFilter — `robodeploy/kinematics/safety.py` (covered by Goal 9 D2)

Cross-reference. Required dependency.

### D2. Unified Safety Layer — `robodeploy/safety/` (NEW directory)

Centralized cross-backend safety primitives. Generalizes ROS2-only code.

```
robodeploy/safety/
├── __init__.py
├── filter.py              # SafetyFilter (from Goal 9)
├── watchdog.py            # Watchdog (refactor from ROS2)
├── estop.py               # EStop (refactor from ROS2)
├── joint_limits.py        # JointLimitGuard
├── temperature.py         # TemperatureGuard
├── force_limits.py        # ForceLimitGuard (NEW)
├── workspace.py           # WorkspaceGuard (NEW)
├── velocity_limits.py     # VelocityGuard (NEW)
├── collision.py           # CollisionGuard (NEW, sim)
├── violation.py           # Hazard / SafetyError / ViolationRecord
└── monitor.py             # SafetyMonitor aggregator
```

### D3. Hazard + SafetyError — `robodeploy/safety/violation.py`

```python
class Hazard(IntEnum):
    JOINT_POSITION_LIMIT = 1
    JOINT_VELOCITY_LIMIT = 2
    JOINT_ACCELERATION_LIMIT = 3
    EE_WORKSPACE_LIMIT = 4
    EE_VELOCITY_LIMIT = 5
    FORCE_LIMIT = 6
    TORQUE_LIMIT = 7
    TEMPERATURE_HIGH = 8
    COMMAND_TIMEOUT = 9
    STATE_TIMEOUT = 10
    CONNECTION_LOST = 11
    COMMAND_REJECTED = 12
    COLLISION_IMMINENT = 13
    OPERATOR_ESTOP = 14
    PROGRAMMATIC_HALT = 15

class Severity(IntEnum):
    INFO = 1; WARNING = 2; CRITICAL = 3

@dataclass
class ViolationRecord:
    hazard: Hazard
    severity: Severity
    message: str
    value: float | None = None
    limit: float | None = None
    joint_idx: int | None = None
    sensor_name: str | None = None
    timestamp: float = field(default_factory=time.time)

class SafetyError(RuntimeError):
    def __init__(self, violation: ViolationRecord):
        self.violation = violation
        super().__init__(f"{violation.hazard.name}: {violation.message}")
```

### D4. SafetyMonitor — `robodeploy/safety/monitor.py`

Aggregates all guards into one entry point. RoboEnv consumes one monitor.

```python
class SafetyMonitor:
    def __init__(self, *, guards: list["ISafetyGuard"] | None = None,
                 on_violation: Literal["clamp","halt","raise"] = "halt",
                 on_critical: Literal["halt","raise"] = "raise",
                 violation_log: Path | None = None):
        self._guards = guards or []
        self._mode = on_violation
        self._critical_mode = on_critical
        self._violations: list[ViolationRecord] = []
        self._tripped = False
        self._log = violation_log

    def add_guard(self, guard: "ISafetyGuard"): self._guards.append(guard)

    def check_action(self, action: Action, obs: Observation, *, dt: float) -> Action:
        """Pre-step: validate / clamp action. Raises on critical."""
        for g in self._guards:
            action, violations = g.check_action(action, obs, dt=dt)
            self._handle(violations)
        return action

    def check_observation(self, obs: Observation) -> None:
        """Post-step: validate observation. Raises on critical."""
        for g in self._guards:
            violations = g.check_observation(obs)
            self._handle(violations)

    @property
    def tripped(self) -> bool: return self._tripped
    def reset(self): self._tripped = False; self._violations.clear()
    def violations(self) -> list[ViolationRecord]: return list(self._violations)

    def _handle(self, violations: list[ViolationRecord]): ...

class ISafetyGuard(Protocol):
    def check_action(self, action: Action, obs: Observation, *, dt: float) -> tuple[Action, list[ViolationRecord]]: ...
    def check_observation(self, obs: Observation) -> list[ViolationRecord]: ...
```

### D5. Cross-Backend EStop — `robodeploy/safety/estop.py`

Generalize from ROS2-only.

```python
class EStop:
    """Aggregates signals: SIGINT, console key, callback, file flag, multiprocessing event."""

    def __init__(self, *, signal_handlers: bool = True, console_key: str = "q",
                 file_flag: Path | None = None, mp_event: mp.Event | None = None,
                 callback: Callable[[], None] | None = None): ...

    def check(self):
        if self._tripped:
            raise SafetyError(ViolationRecord(hazard=Hazard.OPERATOR_ESTOP, severity=Severity.CRITICAL, message=self._reason))

    def trip(self, reason: str = "manual"): ...
    def reset(self): ...

    @property
    def tripped(self) -> bool: ...
```

Wire into `RoboEnv.step()`:

```python
class RoboEnv:
    def step(self, action):
        self._safety_monitor.estop.check()  # raises SafetyError if tripped
        action = self._safety_monitor.check_action(action, self._last_obs, dt=1.0/self._control_hz)
        ...
```

### D6. ForceLimitGuard + VelocityGuard + CollisionGuard

```python
class ForceLimitGuard(ISafetyGuard):
    def __init__(self, *, max_force_N: float = 50.0, max_torque_Nm: float = 10.0,
                 over_limit_strikes: int = 3):
        self._strikes = 0

    def check_observation(self, obs: Observation) -> list[ViolationRecord]:
        if obs.ft_force is None: return []
        f = float(np.linalg.norm(obs.ft_force))
        if f > self._max_force_N:
            self._strikes += 1
            severity = Severity.CRITICAL if self._strikes >= self._over_limit_strikes else Severity.WARNING
            return [ViolationRecord(Hazard.FORCE_LIMIT, severity, f"|F|={f:.1f}N > {self._max_force_N}N", f, self._max_force_N)]
        self._strikes = max(0, self._strikes - 1)
        return []

class VelocityGuard(ISafetyGuard):
    def __init__(self, *, max_joint_velocity: np.ndarray, max_ee_velocity_mps: float = 0.5): ...

class CollisionGuard(ISafetyGuard):
    """Sim-only. Halts on backend contact between unintended bodies."""
    def __init__(self, *, allowed_pairs: list[tuple[str, str]] | None = None,
                 disallowed_pairs: list[tuple[str, str]] | None = None,
                 backend: IBackend = None): ...
```

### D7. Connection Recovery — `robodeploy/backends/real/ros2/recovery.py` (NEW, ~250 lines)

ROS2 reconnect with exponential backoff. Replaces silent timeout.

```python
class ROS2RecoveryManager:
    """Detects connection loss, attempts auto-reconnect, escalates if fail."""

    def __init__(self, *, node, max_retries: int = 5, initial_backoff_s: float = 1.0,
                 max_backoff_s: float = 30.0, on_lost: Callable[[], None] | None = None,
                 on_recovered: Callable[[], None] | None = None): ...

    def on_state_stale(self, age_s: float):
        if age_s > self._state_timeout_s:
            self._lost = True
            if self._on_lost: self._on_lost()
            self._start_reconnect()

    def _start_reconnect(self):
        for retry in range(self._max_retries):
            backoff = min(self._initial_backoff_s * 2**retry, self._max_backoff_s)
            time.sleep(backoff)
            if self._try_reconnect():
                self._lost = False
                if self._on_recovered: self._on_recovered()
                return
        raise SafetyError(ViolationRecord(Hazard.CONNECTION_LOST, Severity.CRITICAL,
                                          f"reconnect failed after {self._max_retries} attempts"))
```

Wire into `Ros2RealBackend._update_state()`.

### D8. Command ACK Verification — `backends/real/ros2/controllers/base.py` (EXTEND)

Currently `send_action()` returns None. Change to return ack status; track ack timeouts as Hazard.COMMAND_REJECTED.

```python
class ControllerBase:
    def send_action(self, action: Action) -> CommandAck:
        msg = self._build_command(action)
        publish_time = time.time()
        self._publisher.publish(msg)
        return CommandAck(
            published_at=publish_time,
            expected_ack_within_s=self._ack_timeout_s,
            sequence_id=self._next_sequence_id(),
        )

@dataclass
class CommandAck:
    published_at: float
    expected_ack_within_s: float
    sequence_id: int
    received_at: float | None = None
    state_response_at: float | None = None
    @property
    def acked(self) -> bool: return self.received_at is not None
```

`Ros2RealBackend` matches state updates against command sequence IDs to detect rejection or excessive latency.

### D9. RoboEnv Safety Integration — `robodeploy/env.py`

```python
class RoboEnv:
    def __init__(self, ..., safety: SafetyMonitor | None = None,
                 safety_limits: SafetyLimits | None = None,
                 emergency_action: Action | None = None): ...

    def _build_default_safety(self) -> SafetyMonitor:
        guards = [
            SafetyFilter(limits=self._safety_limits or limits_from_description(self._robot.description)),
            ForceLimitGuard(max_force_N=50.0),
            VelocityGuard(max_joint_velocity=self._robot.description.joint_velocity_max),
            EStop(),
        ]
        if isinstance(self._backend, SupportsContactQuery):
            guards.append(CollisionGuard(backend=self._backend, disallowed_pairs=[...]))
        return SafetyMonitor(guards=guards, on_violation="clamp", on_critical="raise")

    def step(self, action: Action) -> tuple[Observation, float, bool, EpisodeInfo]:
        try:
            action = self._safety.check_action(action, self._last_obs, dt=1.0/self._control_hz)
            obs, reward, done, info = self._backend.step_multi({self._robot.robot_id: action})[0]
            self._safety.check_observation(obs)
        except SafetyError as e:
            info = self._build_safety_info(e)
            obs = self._last_obs
            self._enter_safe_state()
            return obs, 0.0, True, info
        ...

    def emergency_stop(self, reason: str = "external"):
        self._safety.estop.trip(reason)
        self._enter_safe_state()

    def reset_safety(self): self._safety.reset()

    def _enter_safe_state(self):
        if self._emergency_action is not None:
            self._backend.step_multi({self._robot.robot_id: self._emergency_action})
        else:
            # Default: hold current position
            hold = Action(joint_positions=self._last_obs.joint_positions)
            self._backend.step_multi({self._robot.robot_id: hold})
```

### D10. Safety Tests — `tests/safety/`

Inject simulated violations and verify behavior:

```python
def test_force_limit_critical_halt():
    monitor = SafetyMonitor(guards=[ForceLimitGuard(max_force_N=10.0, over_limit_strikes=2)])
    obs = mock_obs(ft_force=np.array([15.0, 0.0, 0.0]))
    monitor.check_observation(obs)
    assert monitor.violations()[-1].severity == Severity.WARNING
    monitor.check_observation(obs); monitor.check_observation(obs)
    # 3rd strike → critical → SafetyError raised on check
    with pytest.raises(SafetyError): monitor.check_observation(obs)

def test_estop_trip_halts_env(): ...
def test_workspace_clamp_keeps_inside_box(): ...
def test_ros2_reconnect_after_loss(): ...
def test_command_ack_timeout_detected(): ...
def test_collision_guard_disallowed_pair(): ...
def test_temperature_guard_callback(): ...
def test_safety_filter_joint_clamp_at_limits(): ...
```

### D11. Sim Safety Violation Injector — `robodeploy/safety/injector.py` (NEW)

For testing safety pipeline without real hardware.

```python
class SafetyViolationInjector:
    """Injects synthetic safety violations into sim for testing."""
    def force_spike(self, magnitude_N: float, *, duration_steps: int = 1): ...
    def joint_limit_excursion(self, joint_idx: int, magnitude_rad: float): ...
    def state_timeout(self, duration_s: float): ...
    def collision(self, body_a: str, body_b: str): ...
    def temperature_spike(self, joint_idx: int, temp_c: float): ...
```

### D12. Safety Status in Episode Info

```python
@dataclass
class SafetyStatus:
    tripped: bool
    active_violations: list[ViolationRecord]
    history_count: int
    last_violation: ViolationRecord | None

# In step info:
info.extra["safety"] = self._safety.status()
```

### D13. CLI — `robodeploy/cli.py` (EXTEND)

```bash
robodeploy safety check --preset so101_real --joint-limits config/so101_limits.yaml
robodeploy safety test --preset kuka_pick_mujoco --inject force_spike=80N
robodeploy safety status                    # show current monitor state if running
```

### D14. Docs — `docs/SAFETY.md` (NEW)

- Threat model.
- Guard catalog + when each applies.
- Sim vs real safety differences.
- Recovery flows (e-stop, connection loss, force limit).
- How to add a custom Guard.
- Real-hw deployment checklist.

---

## Phased Rollout

### Phase 12.1 — Foundation (~10h)
- D2 directory structure + refactor ROS2 safety into general modules.
- D3 Hazard + SafetyError + ViolationRecord.
- D4 SafetyMonitor + ISafetyGuard protocol.
- D5 cross-backend EStop.
- `tests/safety/test_monitor.py`, `tests/safety/test_estop.py`.

### Phase 12.2 — Guards + RoboEnv integration (~10h)
- D1 SafetyFilter (Goal 9 D2 dependency).
- D6 ForceLimitGuard + VelocityGuard + CollisionGuard.
- D9 RoboEnv integration + emergency_stop API.
- D12 safety status in info.
- D11 sim violation injector.
- `tests/safety/test_force_guard.py`, `tests/safety/test_collision_guard.py`, `tests/safety/test_env_estop.py`.

### Phase 12.3 — Connection recovery + ACK (~7h)
- D7 ROS2RecoveryManager + exponential backoff.
- D8 CommandAck verification.
- `tests/safety/test_reconnect.py`, `tests/safety/test_command_ack.py`.

### Phase 12.4 — CLI + Docs (~3h)
- D13 CLI subcommands.
- D14 SAFETY.md.

---

## Acceptance Criteria

- [ ] `SafetyMonitor` aggregates ≥4 guards (filter, force, velocity, e-stop).
- [ ] `EStop.trip()` from any source (SIGINT, key, mp event) halts env on next step.
- [ ] `RoboEnv.emergency_stop()` API drives robot to safe state + ends episode.
- [ ] Sim safety injector reproduces force spike, joint excursion, collision, state timeout.
- [x] Force above 50N for 3 consecutive steps → `SafetyError` + episode terminated.
- [x] Workspace violation → action clamped to boundary (clamp mode) or `SafetyError` (raise mode).
- [ ] ROS2 connection loss → recovery manager retries with backoff; succeeds within 5 retries OR raises Hazard.CONNECTION_LOST.
- [ ] Command ACK timeout detected within `ack_timeout_s`.
- [ ] `info.extra["safety"]` populated every step.
- [ ] Default `RoboEnv` builds safety monitor with description-derived limits.
- [ ] `tests/safety/` covers each guard + integration.
- [ ] SAFETY.md describes threat model + recovery flows + custom guard authoring.

## Dependencies

- Watchdog / signal / multiprocessing (stdlib).
- Goal 9 D2 (SafetyFilter).
- Goal 6 D18 (SupportsContactQuery protocol) for CollisionGuard.

## Risks

- **False positives during high-acceleration phases**: aggressive force/velocity limits clip valid motion. Mitigation: per-phase override (`policy_phase` aware); strikes counter for noisy sensors.
- **Sim safety differs from real**: sim allows perfect repeatable violations; real noise produces edge-trigger flicker. Mitigation: strike counter + median filter on FT.
- **Recovery deadlocks**: reconnect loop blocks main thread. Mitigation: async retry + bounded wait + escalate to e-stop on persistent failure.
- **Backwards compat**: existing user code expects `step()` not to raise. Mitigation: configurable `on_violation` (clamp default for soft mode).
- **Watchdog spurious trip on CI slow runner**: timeout too tight. Mitigation: scale-dependent timeout + env-var override.

## Out of Scope

- Hardware-level safety circuits (e-stop button, light curtain). External system; document recommendations.
- Functional safety certification (ISO 13849). Out of scope for an open-source library.
- Predictive collision avoidance (forward-simulating). Future; would integrate with Goal 6 trajectory optimization.
- Force-impedance control. Separate controller class; future.
