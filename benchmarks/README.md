# RoboDeploy Benchmarks

Standardized evaluation tasks for comparing policies and backends.

## Layout

- `manipulation_v1/` — tiered manipulation suite (reach → pick → stack → …)
- `sim2real/` — reality-gap benchmarks (imports `manipulation_v1` tasks; sim/real transfer targets)
- `leaderboard/` — submission schema and user score JSON

## Quick start

```bash
# Single task (dummy backend — no simulator required)
robodeploy eval --benchmark manipulation_v1/reach_target --policy scripted --episodes 20 --backend dummy --output report.json

# Full suite
robodeploy eval --benchmark manipulation_v1 --policy scripted --episodes 10 --backend dummy --output suite_report.json

# MuJoCo preset (requires mujoco extra)
robodeploy eval --benchmark manipulation_v1/pick_place_cube --backend mujoco --policy example_sensor_reach_pick --episodes 5 --output pick_report.json
```

Benchmark presets are canonical — do not edit them for valid leaderboard submissions.

## Environment

Set `ROBODEPLOY_BENCHMARKS_ROOT` to override the default `benchmarks/` discovery path.
