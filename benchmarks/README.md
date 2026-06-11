# RoboDeploy Benchmarks

Standardized evaluation tasks for comparing policies and backends.

> **Placeholder tasks (tier 2+):** Several `manipulation_v1` tasks are not yet dedicated benchmark environments.
> Dummy presets often wire `benchmark_reach_scripted` on `benchmark_reach_target`; MuJoCo/Gazebo/Isaac presets for
> `stacking_3blocks` and `cloth_fold` still use `showcase_scene`. Those tasks are marked
> `"task_status": "placeholder"` in `manipulation_v1/spec.json`. Success rates and `reference_scores.json` baselines
> for placeholders measure harness wiring only — **not** real pick/place, stack, pour, or fold performance.
> Only `reach_target` (tier 1) is a live benchmark task today. Do not submit placeholder scores to the leaderboard
> as tier-appropriate results.

**Related:** [docs/PROJECT_GUIDE.md](../docs/PROJECT_GUIDE.md#12-evaluation-and-benchmarks), [plans/INTEGRATION_STATUS.md](../plans/INTEGRATION_STATUS.md), [docs/TRAINING.md](../docs/TRAINING.md).

---

## Layout

```text
benchmarks/
  manipulation_v1/     Tiered manipulation suite (reach → pick → stack → …)
    spec.json          Suite metadata, tier definitions, task_status flags
    reach_target/      Tier 1 — primary live task
    pick_place_cube/   Tier 2 — harness wired; policy placeholder on dummy
    …                  stacking_3blocks, cloth_fold, pour_liquid, …
  sim2real/            Reality-gap benchmarks (registry + reference scores)
  leaderboard/         submission.schema.json, submissions/
```

`pip install robodeploy` bundles this tree as the `benchmarks` package (presets + `task.py` modules). Discovery order: explicit `--benchmarks-root` → `ROBODEPLOY_BENCHMARKS_ROOT` → repo checkout `benchmarks/` → installed package.

---

## Quick start

```bash
# List suites and tasks
robodeploy list-benchmarks

# Single task (dummy backend — no simulator)
robodeploy eval \
  --benchmark manipulation_v1/reach_target \
  --policy scripted \
  --episodes 20 \
  --backend dummy \
  --output report.json

# Full suite (dummy)
robodeploy eval \
  --benchmark manipulation_v1 \
  --policy scripted \
  --episodes 10 \
  --backend dummy \
  --output suite_report.json

# MuJoCo preset (requires [sim])
robodeploy eval \
  --benchmark manipulation_v1/pick_place_cube \
  --backend mujoco \
  --policy example_sensor_reach_pick \
  --episodes 5 \
  --output pick_report.json

# Compare two runs
robodeploy eval-compare --baseline a.json --candidate b.json --output compare.html
```

### Preset files per task

Each task directory contains:

| File | Purpose |
|------|---------|
| `spec.json` | Task metadata, metrics, episode limits |
| `preset_dummy.yaml` | Dummy backend — CI default |
| `preset_mujoco.yaml` | MuJoCo physics eval |
| `preset_gazebo.yaml` | Gazebo eval (Linux) |
| `reference_scores.json` | Documented baselines (honest placeholders where noted) |

Benchmark presets are canonical — do not edit them for valid leaderboard submissions.

---

## Leaderboard

```bash
robodeploy leaderboard submit \
  --suite manipulation_v1/reach_target \
  --report report.json \
  --output benchmarks/leaderboard/submissions/manipulation_v1_reach_target/my_run.json

robodeploy leaderboard show --suite manipulation_v1/reach_target
```

Schema: `benchmarks/leaderboard/submission.schema.json`. Nightly CI validates schemas and runs dummy suite at **N=5** episodes (`benchmark.yml`).

---

## CI coverage

| What runs | Where |
|-----------|-------|
| Full dummy `manipulation_v1`, N=5 | `benchmark.yml` nightly |
| `reach_target` MuJoCo, 3 episodes | `test.yml` → `eval-mujoco-smoke` |
| Preset env build for all tiers | `test_benchmark_preset_builds_env` |
| HTML + video embed | `test_html_report_embeds_recorded_video` |
| Reproducibility | `test_benchmark_reproducibility.py` |

**Not nightly:** 100-episode MuJoCo baselines, full Gazebo manipulation suite, Isaac GPU eval.

---

## Per-task READMEs

- [manipulation_v1/reach_target/README.md](manipulation_v1/reach_target/README.md) — tier 1 task spec
- [manipulation_v1/pick_place_cube/README.md](manipulation_v1/pick_place_cube/README.md) — tier 2 harness notes
