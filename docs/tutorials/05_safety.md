# Tutorial 5 — Safety on Sim and Real Hardware

**Time:** ~30 minutes  
**Goal:** Understand RoboDeploy safety guards, configure limits, and handle e-stop.

## Built-in safety stack

`RoboEnv` attaches a `SafetyMonitor` by default (`safety_enabled=True`). Guards include:

| Guard | Purpose |
|-------|---------|
| `JointLimitGuard` | Clamp or reject out-of-range joint commands |
| `VelocityLimitGuard` | Cap joint velocity per step |
| `ForceLimitGuard` | React to excessive FT readings |
| `WorkspaceGuard` | Keep end-effector inside a bounding box |
| `EstopGuard` | Latch halt on operator e-stop |

Violations can **clamp** actions or **raise** `SafetyError` depending on `on_violation` / `on_critical` policy.

## E-stop during teleop

Press **Esc** in keyboard teleop (`examples/teleop_keyboard_kuka.py`). This calls `env.trip_estop()` and halts the episode.

Reset after clearing the hazard:

```python
env.reset_safety()
env.reset()
```

## Configure safety in code

```python
from robodeploy.env import RoboEnv
from robodeploy.safety import SafetyMonitor, JointLimitGuard, WorkspaceGuard

monitor = SafetyMonitor(
    guards=[JointLimitGuard(), WorkspaceGuard(bounds=((0.2, -0.5, 0.0), (0.9, 0.5, 0.8)))],
    on_violation="clamp",
    on_critical="raise",
)
env = RoboEnv.from_config({...}, safety=monitor)
```

Disable safety for pure sim debugging only:

```python
env = RoboEnv.from_config({...}, safety_enabled=False)
```

## Force-torque limits

FT-driven grasp policies should respect force limits. See `robodeploy.safety.force_limits` and the FT sensor examples:

```bash
python -m examples.kuka_ft_imu_pick_mujoco.run_mujoco
```

## Real-hardware checklist

Run before connecting a physical arm:

```bash
robodeploy doctor
```

Verify:

- `[OK]` ROS 2 (for `backend: ros2` presets)
- `[OK]` `~/.robodeploy/calibration/` writable
- Serial device readable (Linux) or COM port configured (Windows)
- Joint limits in robot description match hardware

## Episode safety metadata

Each step records safety status in `EpisodeInfo.extra["safety"]`. Inspect with:

```bash
robodeploy run-episode --dummy --steps 5 --json
```

## Remote policy serving

When serving policies over ZMQ/gRPC, safety filtering still runs on the robot side:

```bash
robodeploy serve-policy --policy example_sensor_reach_pick --custom-module examples.policies --port 5555
```

Always keep e-stop hardware accessible when testing on real robots.

## Next steps

- [Getting started](01_getting_started.md) — revisit install and concepts.
- [BACKEND_SETUP.md](../BACKEND_SETUP.md) — ROS 2 and hardware setup.
