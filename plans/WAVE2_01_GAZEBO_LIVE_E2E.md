# Wave 2.01 — Gazebo Live Pick E2E

**Wave**: 2 | **Effort**: ~20h | **Maps to**: GOAL 06 (Gazebo), GOAL 03 (FT pick), GOAL 11 (benchmark presets)

## Honest current state

Per `plans/INTEGRATION_STATUS.md` (2026-06-09):

| What works | Evidence |
|------------|----------|
| Gazebo preset YAML loads, env builds offline | `tests/test_live_gazebo_sensors.py::GazeboSensorOfflineTests` |
| Live sensor rig populates `obs.images`, FT, IMU | `sensor-live-gazebo` CI job |
| Contact query, grasp follow, mesh/capsule/terrain | GOAL 06 acceptance items marked `[x]` |
| MuJoCo FT pick ≥80% trials | `tests/test_sensor_mujoco_integration.py` |

| What does **not** work / is not claimed | Gap |
|----------------------------------------|-----|
| `kuka_ft_imu_pick_gazebo` pick-place **success** on live GZ | CI only checks obs keys after reset |
| `manipulation_v1/*/preset_gazebo` full eval nightly | Intentionally omitted from `benchmark.yml` |
| Contact-driven grasp success on live Harmonic | `has_prop_contact` tested offline + injected contacts; live EE-touch not gated in CI |
| Controller tuning for reliable grasp under gz physics | `run_gazebo.py` steps blindly; no success-rate harness |

The `sensor-live-gazebo` job proves sensors publish; it does **not** prove the scripted FT pick policy completes a successful place.

## Problem

Gazebo backend parity for scene building and contacts is largely done, but the **end-to-end pick story** is unproven on Linux CI. Users reading `kuka_ft_imu_pick_gazebo` or benchmark Gazebo presets may assume manipulation works because sensor smoke passes.

## Scope

**In scope**

- Live Linux CI test: run `kuka_ft_imu_pick_gazebo` (or slim fixture world) to episode completion with success assertion across seeds.
- Tune FT grasp thresholds, controller gains, and episode length for Harmonic physics variance.
- Wire live contact assertions into pick success path (complement FT threshold).
- Document flakiness budget and retry policy for CI.

**Out of scope**

- Isaac Sim parity (Wave 2.05).
- `manipulation_v1` full 100-episode Gazebo nightly (defer unless pick E2E is stable at N≥10).
- Real-hardware ROS2 pick (Wave 2.03).

## Acceptance criteria

- [ ] New test `tests/test_live_gazebo_pick_e2e.py` runs in `sensor-live-gazebo` job when `ROBODEPLOY_LIVE_GAZEBO=1`.
- [ ] `kuka_ft_imu_pick_gazebo` achieves ≥70% success over 10 seeds on Ubuntu CI (document threshold; tune from MuJoCo 80% baseline).
- [ ] Live test asserts `info.success` and post-place prop pose within task tolerance (not only obs keys).
- [ ] `Ros2GazeboBackend.has_prop_contact` observed true during grasp phase on at least one successful live episode (log capture).
- [ ] `plans/INTEGRATION_STATUS.md` row for `kuka_ft_imu_pick_gazebo` upgraded from **Sensor smoke** to **Pick E2E** with evidence link.
- [ ] Flake policy documented: max 2 retries, quarantine tag if <50% over 7 days.

## Tasks

### Phase 1 — Harness (~6h)

1. Add `tests/fixtures/gazebo_pick_minimal.sdf` (smaller world, faster spawn than full kitchen).
2. Create `run_pick_episode(preset, seeds, max_steps)` helper shared with `examples/kuka_ft_imu_pick_gazebo/run_gazebo.py`.
3. Add `tests/test_live_gazebo_pick_e2e.py` with `@unittest.skipUnless(LIVE)` mirroring sensor test guards (`gz`, `rclpy`, Jazzy).

### Phase 2 — Tuning (~8h)

4. Profile failure modes: missed grasp (FT never crosses threshold), slip after grasp, place miss.
5. Expose per-preset overrides in `presets.yaml`: `grasp_success_force_min`, `max_episode_steps`, joint trajectory timeouts.
6. Align `gz_ros2_control` controller params with MuJoCo reach timing (document deltas in `docs/BACKEND_SETUP.md`).

### Phase 3 — Contacts + CI (~6h)

7. Log contact events during grasp; assert `has_prop_contact("source", ee_link)` in successful trajectories.
8. Extend `.github/workflows/test.yml` `sensor-live-gazebo` to run pick E2E after sensor tests (sequential; shared GZ process).
9. Add offline regression: injected-contact pick success using `GazeboContactMonitor.inject_contacts` (fast PR gate).

## Self-critique

| Risk | Mitigation |
|------|------------|
| **Flaky CI** — Gazebo spawn timing, ROS graph race | Single shared launch fixture; `pytest-rerunfailures` max 2; seed-0 smoke on PR, full 10-seed on `main` nightly |
| **Overfitting thresholds** to one CI world | Keep minimal fixture + full preset test on weekly schedule |
| **False success** via kinematic teleport / grasp follow | Assert FT force + contact + final prop distance jointly |
| **Scope creep** into full `manipulation_v1` Gazebo suite | Gate Wave 2.01 on single preset E2E only |
| **Harmonic vs Garden** contact topic drift | Version-detect in `GazeboContactMonitor` (already started in `test_gazebo_contact_live.py`) |

**Honest limitation**: 70% CI success is a pragmatic bar; real labs may need per-cell calibration. Do not market Gazebo as tier-2 benchmark backend until N≥30 stable runs.

## Test gates

| Gate | Command / job | Required |
|------|---------------|----------|
| PR offline | `pytest tests/test_gazebo_contact_live.py tests/test_live_gazebo_sensors.py::GazeboSensorOfflineTests -q` | Pass |
| PR fast pick regression | `pytest tests/test_live_gazebo_pick_e2e.py -k offline -q` | Pass |
| Linux live (CI) | `ROBODEPLOY_LIVE_GAZEBO=1 pytest tests/test_live_gazebo_pick_e2e.py -q` | ≥70% seeds |
| Manual pre-merge | `python -m examples.kuka_ft_imu_pick_gazebo.run_gazebo` on Jazzy+Harmonic host | Visual confirm |
| Status doc | Update `INTEGRATION_STATUS.md` preset table | Required for close |
