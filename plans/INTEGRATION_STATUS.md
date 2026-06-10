# Integration status (honest)

Last updated: 2026-06-09 on branch `feat/plans-2-3-integration-core`.

What presets and benchmarks claim vs what CI actually exercises. **606 tests passed** locally with `pytest -m "not hardware"` (16 skipped).

## CI jobs → what they prove

| Workflow / job | Command or test scope | Status |
|----------------|----------------------|--------|
| `test.yml` → `unittest` | `pytest tests/ -m "not hardware"` (matrix: Py 3.10–3.12 × Linux/macOS/Windows) | **Pass** — core unit + integration |
| `test.yml` → `sensor-e2e-linux` | MuJoCo sensor/scene e2e, `kuka_pick_mujoco` 50-step CLI smoke, benchmark preset env build | **Pass** — MuJoCo smoke |
| `test.yml` → `eval-mujoco-smoke` | `robodeploy eval manipulation_v1/reach_target --backend mujoco --episodes 3` | **Pass** — MuJoCo eval harness (not full suite) |
| `test.yml` → `sensor-live-ros2` | `tests/test_live_ros2_sensors.py` (ROS2 graph) | **Pass** — live ROS2 sensors |
| `test.yml` → `sensor-live-gazebo` | `tests/test_live_gazebo_sensors.py -m live_gazebo` | **Pass** — multimodal obs + relaxed pick-place (≥1/3 seeds) |
| `test.yml` → `isaacsim-smoke` | Mocked IsaacSim parity tests (`continue-on-error: true`) | **Smoke** — no GPU Kit runtime |
| `test.yml` → `multirobot-mujoco` | `tests/test_multirobot_mujoco.py` | **Pass** — two-arm MuJoCo |
| `test.yml` → `docker-smoke` | `docker build` + `robodeploy --help` + `examples.cli list-presets` | **Pass** |
| `test.yml` → `conda-recipe` | Recipe metadata + pip-source import smoke | **Pass** — not conda-forge publish |
| `test.yml` → `package-build` | `tests/test_package_build.py` — sdist/wheel + `twine check` + wheel CLI smoke | **Pass** — no PyPI upload |
| `benchmark.yml` → `validate-schemas` | `spec.json` + leaderboard schema + PR submission validation | **Pass** |
| `benchmark.yml` → `manipulation-v1-dummy` | Full `manipulation_v1` suite, dummy backend, **N=5** episodes | **Pass** — reduced N nightly gate |
| `benchmark.yml` → `publish-pages` | Deploys dummy-suite HTML + JSON to GitHub Pages | **Pass** — dummy-only scores |
| `docs.yml` | `mkdocs build` + Pages deploy on `main` | **Pass** |
| `publish.yml` | PyPI upload on `v*` tag (`workflow_dispatch` dry-run available) | **Ready** — no PyPI release yet |

## Presets and benchmarks

| Preset / benchmark | Runnable command | CI job | Status |
|--------------------|------------------|--------|--------|
| `kuka_pick_mujoco` | `python -m examples.cli run-episode --preset kuka_pick_mujoco --steps 50` | `sensor-e2e-linux` | **Smoke** — builds env, steps MuJoCo |
| `my_kitchen_pick_mujoco` | `python -m examples.cli run-episode --preset my_kitchen_pick_mujoco` | — | **Doc only** — requires tutorial task file |
| `kuka_ft_imu_pick_gazebo` | `python -m examples.kuka_ft_imu_pick_gazebo.run_gazebo` | `sensor-live-gazebo` (`live_gazebo` marker) | **Live smoke** — multimodal obs + relaxed pick-place (≥1/3 seeds; not 80% MuJoCo parity) |
| `kuka_sensor_gazebo` | `python -m examples.kuka_sensor_gazebo.run_gazebo` | `sensor-live-gazebo` | **Sensor smoke** |
| `manipulation_v1/*/preset_dummy` | `robodeploy eval --benchmark manipulation_v1/<task> --backend dummy` | `unittest`, `benchmark-nightly` (N=5) | **Pass** — eval harness + nightly |
| `manipulation_v1/reach_target/preset_mujoco` | `robodeploy eval ... --backend mujoco --episodes 3` | `eval-mujoco-smoke` | **Short eval** — not full 100-ep suite |
| `manipulation_v1/pick_place_cube` dummy 100 ep | `robodeploy eval ... --episodes 100` | `unittest` (`test_pick_place_cube_*`; CI uses N=5) | **Pass** — JSON output; scripted dummy SR≈0% (preset uses reach policy) |
| `manipulation_v1/*/preset_gazebo` | `get_backend("gazebo")` + preset load | `test_gazebo_presets_load`, live Gazebo sensors | **YAML + sensor smoke** — full manipulation eval not nightly |
| `manipulation_v1/stacking_3blocks`, `cloth_fold` | showcase_scene + joint_track placeholder | `test_benchmark_preset_builds_env` | **Tier placeholder** — not real fold/stack tasks |
| `sim2real/*` | registry + reference scores | `test_sim2real_benchmarks.py` | **Schema/registry** — transfer eval not automated |
| `two_franka_pick_mujoco` | multi-robot MuJoCo tests | `multirobot-mujoco` | **Pass** — independent reach targets |
| `gym.make("robodeploy/kuka_pick_mujoco-v0")` | gym register | `tests/training/test_gym_register.py` | **Pass** (requires mujoco) |
| PyPI `pip install robodeploy` | — | `publish.yml` (on tag) | **Not published** — workflow ready |
| Nightly `manipulation_v1` MuJoCo/Gazebo | — | — | **Not run** — by design; MuJoCo subset on PR only |

## Goal 11 benchmark acceptance (this branch)

| Criterion | Evidence |
|-----------|----------|
| `pick_place_cube` eval 100 ep JSON | `test_pick_place_cube_eval_outputs_aggregate_json` |
| Reproducibility | `tests/test_benchmark_reproducibility.py` |
| HTML + video embed | `test_html_report_embeds_recorded_video` |
| Nightly dummy suite + Pages | `benchmark.yml` |
| FailureClassifier ≥80% | `test_fixture_audit_accuracy_at_least_80_percent` (11 fixtures) |
| Parallel ≡ sequential | `test_parallel_matches_sequential` |
| Leaderboard schema CI | `benchmark.yml` `validate-schemas` |

## Backend naming

| Context | Name |
|---------|------|
| Presets / benchmarks / `backend_for_simulator("gazebo")` | `gazebo` |
| Registered class | `ros2_gazebo` |
| Resolution | `get_backend("gazebo")` → `ROS2GazeboBackend` |

## Known gaps (not claimed)

- `pick_place_cube` dummy preset uses `benchmark_reach_scripted` — tier-2 success rate not meaningful on dummy until pick-place policy wired.
- Isaac Sim live GPU eval requires self-hosted runner (`isaacsim-gpu-live` documents blocker).
- Real-hardware benchmarks (`sim2real` real presets) need lab access.
