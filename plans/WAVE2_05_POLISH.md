# Wave 2.05 — Polish (Learned Policy LOC, Dashboard Defer, Isaac GPU Live)

**Wave**: 2 | **Effort**: ~22h | **Maps to**: GOAL 09, GOAL 10, GOAL 06 (IsaacSim)

## Honest current state

### Learned policy line counts (GOAL 09)

| File | Lines | Target |
|------|-------|--------|
| `robodeploy/policies/learned/robomimic.py` | 55 | ≤50 |
| `robodeploy/policies/learned/diffusion.py` | 85 | ≤50 |
| `robodeploy/policies/learned/vla.py` | 68 | ≤50 |

Supporting modules already extracted: `loader.py`, `helpers.py`, `factory.py`, `adapter.py`, `negotiation.py`. Remaining bloat is inline queueing (diffusion), camera heuristics (VLA), and smoothing (robomimic). GOAL 09 checkbox remains `[ ]`.

### Dashboard (GOAL 10)

- **Explicitly deferred** per `robodeploy/observability/DASHBOARD_DEFERRAL.md`.
- Workaround: JSONL + `robodeploy logs tail/summary` + W&B/TensorBoard/MLflow sinks.
- GOAL 10 D9 checkbox `[ ]` — **intentional defer**, not a bug.

### Isaac Sim live GPU (GOAL 06)

| Item | Status |
|------|--------|
| Mocked smoke tests | `isaacsim-smoke` job, `continue-on-error: true` |
| `isaacsim-gpu-live` job | Documents blocker only; docker pull without GPU |
| Isaac acceptance items | 6/7 unchecked (EE pos, torques, IMU, capsule, USD, multi-robot) |
| SceneIR round-trip ≤1mm | Unchecked across MuJoCo/Isaac/Gazebo |

`INTEGRATION_STATUS.md`: "Isaac Sim live GPU eval requires self-hosted runner."

## Problem

Three **non-blocking but credibility** gaps remain after wave 1 integration:

1. Learned policy shims exceed stated LOC budget despite helper extraction.
2. Dashboard deferral needs a formal wave-2 decision (reaffirm or scope minimal read-only viewer).
3. Isaac Sim claims in docs exceed what mocked CI proves; GPU live path undocumented for self-hosted adopters.

## Scope

**In scope**

- Refactor `robomimic.py`, `diffusion.py`, `vla.py` to ≤50 lines each by moving logic to existing helpers.
- **Dashboard**: reaffirm deferral in wave-2 closeout OR implement read-only static HTML viewer from JSONL (no WebSocket) — **default: reaffirm defer**.
- Isaac GPU live playbook: self-hosted runner spec, one headless smoke script, optional org runner enablement.
- Isaac **mocked** parity: fix `obs.ee_position` + `obs.joint_torques` in backend (enables tests without GPU).

**Out of scope**

- Full Isaac feature parity (USD import, multi-robot, capsule) — track under GOAL 06 phases.
- FastAPI live dashboard (deferred).
- Purchasing GPU runners for upstream CI.

## Acceptance criteria

### Learned policies

- [ ] `wc -l` on `robomimic.py`, `diffusion.py`, `vla.py` each ≤50 (exclude blank/comment-only lines per `scripts/count_policy_loc.py` if added).
- [ ] Existing learned policy tests pass unchanged.
- [ ] GOAL 09 LOC checkbox marked `[x]`.

### Dashboard

- [ ] Decision recorded in `plans/WAVE2_05_POLISH.md` closeout + `DASHBOARD_DEFERRAL.md` "Revisited wave 2" note.
- [ ] If defer reaffirmed: GOAL 10 D9 stays `[ ]` with wave-2 sign-off date.
- [ ] If minimal viewer chosen: `robodeploy logs summary --html out.html` generates static report (optional stretch).

### Isaac Sim

- [ ] `tests/test_isaacsim_obs.py`: `obs.ee_position` non-zero on mocked reach episode.
- [ ] `tests/test_isaacsim_obs.py`: `obs.joint_torques` non-zero when articulation moves (mocked physics).
- [ ] `docs/BACKEND_SETUP.md#isaac-sim-self-hosted-ci` documents GPU runner requirements mirroring `isaacsim-gpu-live` job comments.
- [ ] `scripts/isaacsim_headless_smoke.py` runs on GPU host: import `SimulationApp`, one reset/step, exit 0.
- [ ] At least one org-maintainer manual run log captured in `docs/isaacsim_gpu_smoke_log.example.txt`.

## Tasks

### Phase 1 — Learned policy LOC (~8h)

1. Move diffusion plan queue to `helpers.py` (`DiffusionActionQueue` class).
2. Move VLA camera selection to `helpers.py` (`select_vla_images(obs, config)`).
3. Move robomimic EMA smoothing to `helpers.py` or `base.py` mixin.
4. Add `scripts/count_policy_loc.py` for CI guard (non-blank lines).
5. Add `tests/test_learned_policy_loc.py` asserting ≤50 per file.

### Phase 2 — Dashboard decision (~2h)

6. Review JSONL volume from benchmark nightly; confirm external sinks suffice.
7. Update `DASHBOARD_DEFERRAL.md` with wave-2 reaffirmation + revisit trigger (e.g. "when eval HTML needs live step stream").
8. **Do not** add `fastapi` dependency unless viewer stretch approved.

### Phase 3 — Isaac mocked obs (~8h)

9. Fix `IsaacSimBackend` EE kinematics readout (replace zero hardcode at `backend.py` ~603).
10. Wire joint effort/torque from articulation view (~601).
11. Extend `tests/test_isaacsim_obs.py` assertions; keep `isaacsim-smoke` job green.

### Phase 4 — Isaac GPU playbook (~4h)

12. Add `scripts/isaacsim_headless_smoke.py` + `docs/BACKEND_SETUP.md` section.
13. Document converting `isaacsim-gpu-live` to `runs-on: [self-hosted, gpu, isaacsim]` label.
14. Template issue for adopters: "Enable Isaac GPU smoke on your runner."

## Self-critique

| Risk | Mitigation |
|------|------------|
| **LOC refactor breaks behavior** | No test changes allowed; only moves |
| **50-line limit too arbitrary** | Use shared helpers; count script excludes docstrings |
| **Dashboard scope creep** | Default = defer reaffirmation; static HTML is stretch only |
| **Isaac API churn** | Pin Isaac 4.1 in docs; version-guard imports |
| **GPU runner not available to OSS CI** | Mocked tests are merge gate; GPU is manual evidence |

**Honest limitation**: Wave 2.05 does **not** complete GOAL 06 Isaac items (USD, multi-robot, IMU sensor). It unblocks credibility on obs + documents GPU path.

## Test gates

| Gate | Command | Required |
|------|---------|----------|
| PR learned | `pytest tests/test_learned_policy*.py tests/policies/ -q` | Pass |
| PR LOC | `python scripts/count_policy_loc.py --max 50` | Pass |
| PR Isaac mock | `pytest tests/test_isaacsim_smoke.py tests/test_isaacsim_obs.py -q` | Pass |
| PR parity | `pytest tests/test_backend_parity.py -q` | Pass |
| GPU manual | `python scripts/isaacsim_headless_smoke.py` on self-hosted runner | Pass (documented) |
| Dashboard | N/A if defer reaffirmed | Sign-off in deferral doc |

## Wave 2 closeout (2026-06-09)

**Decision**: Defer learned-policy LOC refactor; reaffirm dashboard deferral; document Isaac GPU playbook (stub). No behavior-changing refactors in this wave.

### Learned policies (GOAL 09)

- LOC ≤50 per shim **deferred** — current 56 / 86 / 69 lines (`robomimic.py` / `diffusion.py` / `vla.py`).
- Supporting modules (`loader.py`, `helpers.py`, `factory.py`, `adapter.py`, `negotiation.py`) already extracted.
- GOAL 09 LOC checkbox stays `[ ]`; revisit when helper extraction can move queueing/camera/heuristic paths without test changes.

### Dashboard (GOAL 10)

- D9 **reaffirmed deferred** 2026-06-09 (`robodeploy/observability/DASHBOARD_DEFERRAL.md`).
- JSONL + `robodeploy logs tail/summary` + W&B/TensorBoard/MLflow sinks suffice for wave 2.
- GOAL 10 D9 checkbox stays `[ ]` with wave-2 sign-off date.

### Isaac Sim (GOAL 06)

- Mocked CI remains merge gate (`isaacsim-smoke`).
- GPU live path documented: `docs/BACKEND_SETUP.md#isaac-sim-self-hosted-ci`.
- `scripts/isaacsim_headless_smoke.py` + `docs/isaacsim_gpu_smoke_log.example.txt` added for self-hosted adopters.
- EE/torque obs fixes and full GOAL 06 parity (USD, multi-robot, IMU) **deferred** to follow-up.
