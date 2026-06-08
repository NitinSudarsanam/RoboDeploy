# Safety

RoboDeploy layers safety checks so every action passes through joint-limit clamping and optional guards before reaching any backend. Simulation and real hardware share the same `SafetyMonitor` API; only the guard set differs.

## Threat model

| Threat | Mitigation |
|--------|------------|
| Command exceeds joint limits | `SafetyFilter` (Goal 9) clamps every action; `SafetyFilterGuard` reports clamp events |
| Excessive end-effector / joint velocity | `VelocityGuard`, `SingularityGuard` |
| Force overload during contact | `ForceLimitGuard` with strike counter |
| Unintended collision (sim) | `CollisionGuard` via `SupportsContactQuery` |
| Operator e-stop (keyboard, SIGINT, file flag) | `EStop` + teleop `Esc` → `RoboEnv.emergency_stop()` |
| Human too close to arm | `HumanProximityGuard` (object poses or `metadata.proximity_m`) |
| ROS2 state staleness | `ROS2RecoveryManager` with exponential backoff |
| Command not reflected in state | `CommandAck` timeout → `Hazard.COMMAND_REJECTED` |

**Out of scope:** certified functional safety (ISO 13849), hardware e-stop wiring, light curtains. Document and test those on the deployment site.

## Guard catalog

| Guard | When active | Mode |
|-------|-------------|------|
| `SafetyFilterGuard` | Always (default `RoboEnv`) | Clamp |
| `EStopGuard` | Always | Halt |
| `ForceLimitGuard` | FT sensor present | Warning → critical after N strikes |
| `VelocityGuard` | Always | Warning / critical |
| `CollisionGuard` | Sim backends with contact query | Critical |
| `HumanProximityGuard` | Shared workspace / proximity sensor | Critical |
| `SingularityGuard` | Teleop / cartesian control | Warning → critical |
| `WorkspaceGuard` | Cartesian actions with bounds | Clamp or raise |
| `JointLimitGuard` | ROS2 hardware controllers | Raise |
| `TemperatureGuard` | Hardware with motor temps | Warning / critical |
| `Watchdog` | Real-time command loops | Critical on timeout |

Configure violation handling on `SafetyMonitor`:

- `on_violation="clamp"` — soft limits log warnings and continue
- `on_violation="raise"` — warnings halt the episode
- `on_critical="raise"` — critical hazards always raise `SafetyError`

## Sim vs real

**Simulation** can inject repeatable violations via `SafetyViolationInjector` and `robodeploy safety test --inject …`. Collision and force spikes are deterministic.

**Real hardware** adds:

- `JointLimitGuard` + slew limiting in ROS2 controllers
- `EStop` console key and SIGINT in controller threads
- `ROS2RecoveryManager` on connection loss (replaces silent timeout)
- Noisier FT readings — use `over_limit_strikes` on `ForceLimitGuard`

## Recovery flows

### Operator e-stop

1. Teleop `Esc` → `TeleopPolicy` raises `TeleopSafetyError`
2. `InteractiveDemoSession` calls `RoboEnv.emergency_stop()`
3. `SafetyFilter.trigger_estop()` freezes joint commands
4. `EStop.trip()` marks monitor; next `step()` returns `done=True`, `info.extra["safety"]` populated
5. Resume with `env.reset_safety()` only when the workspace is clear

### Force limit

Three consecutive steps above `max_force_N` (default 50 N) → `SafetyError`, episode ends, robot holds last position.

### Connection loss (ROS2)

`ROS2RecoveryManager.on_state_stale()` starts backoff reconnect (default 5 retries). Success clears the fault; failure raises `Hazard.CONNECTION_LOST`.

## Teleop integration (Goal 4 + Goal 9)

- `TeleopPolicy` clamps cartesian deltas and raises on `e_stop`
- `RoboEnv` applies `SafetyFilter.filter()` before `SafetyMonitor.check_action()`
- E-stop trips both the monitor and the per-robot `SafetyFilter`

## CLI

```bash
# Validate limits for a preset / robot
robodeploy safety check --preset kuka_pick_mujoco
robodeploy safety check --robot franka --joint-limits my_limits.yaml

# Inject synthetic violations (dummy or preset env)
robodeploy safety test --inject force_spike=80N --steps 5
robodeploy safety test --preset kuka_pick_mujoco --inject human_proximity=0.1m

# Inspect monitor registered by a live RoboEnv process
robodeploy safety status --json
```

## Custom guard

Implement `ISafetyGuard`:

```python
class MyGuard:
    def check_action(self, action, obs, *, dt) -> tuple[Action, list[ViolationRecord]]:
        return action, []

    def check_observation(self, obs) -> list[ViolationRecord]:
        return []
```

Add to the monitor:

```python
monitor = SafetyMonitor(guards=[MyGuard(), ...])
env = RoboEnv(..., safety=monitor)
```

Return `Severity.WARNING` for soft events and `Severity.CRITICAL` for halts.

## Real-hardware deployment checklist

- [ ] Verify joint limits YAML matches calibrated soft limits
- [ ] Run `robodeploy safety check --preset <real_preset>`
- [ ] Test hardware e-stop button independently of software
- [ ] Confirm `enable_console_estop` behavior on operator station
- [ ] Set `max_force_N` from tool/contact study
- [ ] Enable `HumanProximityGuard` when humans share the workspace
- [ ] Log violations to disk: `SafetyMonitor(violation_log=Path("safety.jsonl"))`
- [ ] Document `reset_safety()` procedure after any trip

## Episode info

Every `step()` and `reset()` includes:

```python
info.extra["safety"]  # tripped, history_count, last_violation, hazard on error
```

## Tests

```bash
python -m pytest tests/safety/ -q
```

Coverage: monitor aggregation, e-stop, force/collision guards, env integration, ROS2 reconnect/ACK (mocked), teleop e-stop, real-controller e-stop (mocked).
