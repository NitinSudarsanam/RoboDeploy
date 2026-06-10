# RoboDeploy Benchmarks

Standardized evaluation tasks for comparing policies and backends.

> **Placeholder tasks (tier 2+):** Several `manipulation_v1` tasks are not yet dedicated benchmark environments.
> Dummy presets often wire `benchmark_reach_scripted` on `benchmark_reach_target`; MuJoCo/Gazebo/Isaac presets for
> `stacking_3blocks` and `cloth_fold` still use `showcase_scene`. Those tasks are marked
> `"task_status": "placeholder"` in `manipulation_v1/spec.json`. Success rates and `reference_scores.json` baselines
> for placeholders measure harness wiring only — **not** real pick/place, stack, pour, or fold performance.
> Only `reach_target` (tier 1) is a live benchmark task today. Do not submit placeholder scores to the leaderboard
> as tier-appropriate results.

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
