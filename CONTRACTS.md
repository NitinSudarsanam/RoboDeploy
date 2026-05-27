# RoboDeploy contracts

This document defines the **public contracts** RoboDeploy aims to keep stable.
If code and docs disagree, this file should be treated as the canonical
reference for user-facing behavior.

## Core types contract

Core data structures live in `robodeploy/core/types.py`. These are the shared
interfaces between backends, tasks, policies, and sensors.

- **Observation**: `robodeploy.core.types.Observation`
  - **Required fields**: `joint_positions`, `joint_velocities`, `joint_torques`,
    `ee_position`, `ee_orientation`, `ee_velocity`, `ee_angular_velocity`,
    plus timestamps (`timestamp`, `timestamp_hw`, `timestamp_recv`).
  - **Optional fields**: `rgb`, `depth`, per-camera maps (`images`, `depths`),
    force/torque (`ft_force`, `ft_torque`), IMU (`imu_acceleration`,
    `imu_angular_velocity`), and `gripper_state`.

- **Action**: `robodeploy.core.types.Action`
  - Policies should set the fields consistent with `action_space`.
  - Backends and safety layers must treat missing fields as “not provided” (not
    as zeros).

- **SensorData**: `robodeploy.core.types.SensorData`
  - Raw per-sensor payloads merged by backends/pipelines into an `Observation`.

- **EpisodeInfo**: `robodeploy.core.types.EpisodeInfo`
  - Returned from stepping APIs and surfaced via CLI summaries.
  - `extra` is reserved for structured extension payloads (e.g. diagnostics).

## Construction contract

RoboDeploy supports both string-based and object-based construction.

- **`RoboEnv.make(...)`**: string wiring through the registry.
  - Use this for quick experiments and CLI-style “run something by name.”

- **`RoboEnv.from_preset(name)`**: construct from YAML preset.
  - Use this as the default “reproducible config” path.

- **`RoboEnv.from_config(cfg)`**: construct from an explicit config object/dict.
  - Use this when you already have a fully resolved config in code.

- **Direct injection**: `RoboEnv(backend=..., robots=[Robot(...), ...])`.
  - Use this when you’re composing custom robot/task/policy objects, testing, or
    integrating non-registered components.

## CLI output contract

The CLI entry point is `robodeploy.cli:main` (console script: `robodeploy`).
When `--json` is provided, commands print **machine-parseable JSON** to stdout.

### `list-presets`

- **`robodeploy list-presets`**: prints one preset name per line.
- **`robodeploy list-presets --json`**: prints JSON array:

```json
["preset_a", "preset_b"]
```

### `list-registry`

- **`robodeploy list-registry`**: human-readable grouped listing.
- **`robodeploy list-registry --json`**: JSON object keyed by registry group:

```json
{
  "backends": ["mujoco", "isaacsim"],
  "robots": ["franka"],
  "tasks": ["pick_place"],
  "policies": ["robomimic"],
  "sensors": ["wrist_camera_sim"],
  "sensor_pairs": ["wrist_camera"]
}
```

### `run-episode`

- **Without `--json`**: prints a compact JSON object representing an EpisodeInfo
  summary (parseable).
- **With `--json`**: prints a wrapper object that includes command inputs plus
  `info`:

```json
{
  "preset": "my_preset",
  "dummy": false,
  "steps": 50,
  "action": "none",
  "info": {
    "episode_id": 1,
    "step": 50,
    "reward": 0.0,
    "success": false,
    "failure": false,
    "extra": {}
  }
}
```

### `export-episode`

- **Without `--json`**: prints the output path as a string.
- **With `--json`**: prints a wrapper object describing what was written:

```json
{
  "out": "path/to/file.jsonl",
  "format": "jsonl",
  "steps": 50,
  "dummy": true,
  "preset": "",
  "action": "sinusoid"
}
```

