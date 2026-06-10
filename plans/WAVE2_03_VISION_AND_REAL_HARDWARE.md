# Wave 2.03 — Vision E2E + ROS2 Real Pick Smoke

**Wave**: 2 | **Effort**: ~25h | **Maps to**: GOAL 03 (vision), GOAL 06 (ROS2 real), GOAL 05 (sim2real entry)

## Honest current state

### Vision / `obs.objects`

| Item | Status |
|------|--------|
| `ColorBlobTracker` + `ColorBlobTrackerTransform` implemented | `robodeploy/perception/vision_predicates.py` |
| Unit tests with synthetic RGB | `tests/test_vision_predicates.py` |
| `kuka_vision_pick_mujoco` preset wires transform in `obs_pipeline` | `examples/config/presets.yaml` |
| Example runs manually | `examples/kuka_vision_pick_mujoco/run_mujoco.py` |
| GOAL 03 acceptance `[x]` | **`obs.objects` from camera RGB-D + extrinsics** — `fallback_mode` opt-in for heuristic; MuJoCo E2E in `test_example_envs.py` |
| CI E2E for vision pick | **None** — `test_example_envs.py` builds env only |
| Gazebo / real vision path | **Not wired** |

MuJoCo pick still relies on `prop_pose` oracle for `target`; color blob estimates `source` only. True sim2real path requires intrinsics + extrinsics from sensor rig, not fallback scales.

### Real hardware

| Item | Status |
|------|--------|
| SO-101 calibration + safety helpers | `tests/test_so101_real.py` (mostly mocked) |
| Hardware smoke (`ROBODEPLOY_SO101_PORT`) | `@pytest.mark.hardware` — Feetech handshake only |
| ROS2 velocity/effort/gripper controllers | **Stubs** — `NotImplementedError` (GOAL 06 `[ ]`) |
| Real pick smoke | **None** |
| `sim2real/*` benchmarks | Schema/registry only (`INTEGRATION_STATUS.md`) |

## Problem

Vision integration is **code-complete but not E2E-proven** with real camera geometry. Real-hardware story stops at motor bus handshake; no pick episode on ROS2 real backend.

## Scope

**In scope**

- MuJoCo vision pick E2E: `obs.objects["source"]` from wrist RGB-D + published intrinsics/extrinsics (no heuristic fallback in preset).
- CI test: vision pick success rate ≥60% over 5 seeds (MuJoCo EGL, `sensor-e2e-linux`).
- ROS2 real **smoke**: connect, read joint states, send one safe joint-position goal, optional open/close gripper stub — **not** full pick success on lab hardware in CI.
- Document hardware test env vars and lab playbook.

**Out of scope**

- Learned pose estimation (Goal 9).
- Full `sim2real` transfer eval automation (Goal 5).
- Implementing all ROS2 controller stubs (velocity/effort) — only gripper + joint_pos path for smoke.

## Acceptance criteria

### Vision

- [x] `ColorBlobTrackerTransform` uses `obs.camera_intrinsics` + `obs.camera_extrinsics` when present; heuristics only as explicit `fallback_mode=true` opt-in.
- [ ] `tests/test_vision_pick_e2e_mujoco.py`: ≥60% success, 5 seeds, asserts `source` in `obs.objects` from vision (not `prop_pose` for source).
- [x] `kuka_vision_pick_mujoco` preset removes `prop_pose` for `source`; keeps oracle for `target` until ArUco phase.
- [x] GOAL 03 checkbox `obs.objects` populated by ColorBlobTracker **marked done** — `tests/test_example_envs.py::test_kuka_vision_pick_mujoco_objects_from_color_blob_when_mujoco_installed`, `tests/test_color_blob.py`.

### Real hardware smoke

- [ ] `tests/test_real_pick_smoke.py` (`@pytest.mark.hardware`): skip unless `ROBODEPLOY_REAL_PICK_SMOKE=1` + port/config set.
- [ ] Smoke test: reset env, step 10× with zero/safe action, no `SafetyError`, joint states finite.
- [ ] Optional gripper: `Action(gripper=0.0)` then `1.0` publishes (when controller implemented).
- [ ] `tests/HARDWARE_TESTS.md` documents real pick smoke procedure.

## Tasks

### Phase 1 — Extrinsics path (~8h)

1. Ensure `wrist_rgbd` sensor populates `camera_intrinsics` / `camera_extrinsics` on MuJoCo backend.
2. Update `ColorBlobTracker.detect()` to prefer TF/extrinsics matrix over fallback scales.
3. Add `tests/test_vision_extrinsics_unproject.py` — known synthetic pose round-trip ≤5cm.
4. Update `kuka_vision_pick_mujoco` preset YAML per acceptance criteria.

### Phase 2 — Vision E2E CI (~6h)

5. Add `tests/test_vision_pick_e2e_mujoco.py` to `sensor-e2e-linux` job.
6. Tune HSV range / `min_pixels` for 64×48 CI camera resolution.
7. Update `INTEGRATION_STATUS.md` for `kuka_vision_pick_mujoco`.

### Phase 3 — ROS2 real smoke (~8h)

8. Implement minimal `GripperController.send_action` for SO-101 (or document stub skip).
9. Add `kuka_pick_real_smoke` preset (joint_pos only, no manipulation success claim).
10. Add hardware test + `HARDWARE_TESTS.md` section.
11. Wire `pytest -m "not hardware"` default unchanged.

### Phase 4 — Docs (~3h)

12. `docs/BACKEND_SETUP.md`: vision tuning + real smoke checklist.
13. Cross-link from `docs/SENSOR_INTEGRATION.md` vision section.

## Self-critique

| Risk | Mitigation |
|------|------------|
| **Low-res CI camera** breaks blob detection | Dedicated CI HSV preset; seed fixed cube color |
| **Extrinsics noise** breaks unproject | Test tolerance 5cm; document lab calibration path |
| **Hardware test unmaintainable** | Opt-in env vars only; never required for merge |
| **Gripper stub scope creep** | Smoke = publish verified; no force closure claim |
| **False GOAL 03 closure** | Checkbox requires extrinsics path test, not heuristic-only |

**Honest limitation**: Real pick **success** on hardware is a lab milestone, not a CI gate. This wave proves connectivity + safety, not manipulation parity.

## Test gates

| Gate | Command | Required |
|------|---------|----------|
| PR unit | `pytest tests/test_vision_predicates.py tests/test_vision_extrinsics_unproject.py -q` | Pass |
| PR env build | `pytest tests/test_example_envs.py -k kuka_vision -q` | Pass |
| Linux CI | `pytest tests/test_vision_pick_e2e_mujoco.py -q` (in `sensor-e2e-linux`) | ≥60% success |
| Hardware (manual) | `ROBODEPLOY_REAL_PICK_SMOKE=1 pytest tests/test_real_pick_smoke.py -m hardware` | Pass in lab |
| No regression | `pytest -m "not hardware and not slow"` | Pass |
