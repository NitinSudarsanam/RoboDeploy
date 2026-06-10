# Wave 2.04 — Training Production (500k PPO + kuka_pick_mujoco)

**Wave**: 2 | **Effort**: ~18h | **Maps to**: GOAL 02 (training loop), GOAL 11 (benchmark baselines)

## Honest current state

| Item | Status | Evidence |
|------|--------|----------|
| Gym register `robodeploy/kuka_pick_mujoco-v0` | **Pass** | `tests/training/test_gym_register.py` |
| PPO on toy `reach_target` dummy | **Pass** 10k steps, ≥80% SR | `tests/training/test_ppo_reach_target.py` (`@slow`) |
| SB3 smoke on `robodeploy/Tiny-v0` | **Pass** | `tests/training/test_sb3_smoke.py` |
| `robodeploy train ppo` CLI | **Pass** throughput test | `tests/training/test_subproc_vec_env.py` |
| `examples/train_ppo_reach.py` (500k) | **Missing** — referenced in GOAL 02 but file absent | Plan/doc drift |
| Nightly 500k PPO job | **Not configured** | `benchmark.yml` dummy-only |
| PPO on `kuka_pick_mujoco` | **Not implemented** | No example, no test, no checkpoint |
| BC pick-place example | Documented in GOAL 02 | `examples/train_bc_dummy.py` only (dummy) |

Training loop acceptance criteria in GOAL 02 are marked `[x]` based on **toy tasks**. Production-scale training and robot-specific PPO are aspirational, not shipped.

## Problem

The repo claims a training platform, but CI only validates short smoke runs. There is no reproducible 500k-step baseline, no optional nightly training gate, and no `kuka_pick_mujoco` PPO path for benchmark leaderboard entries (`ppo_500k` in GOAL 11 fixture JSON is illustrative only).

## Scope

**In scope**

- Add `examples/train_ppo_reach.py`: 16-env SubprocVecEnv, 500k steps, `reach_target` dummy or `manipulation_v1/reach_target` MuJoCo, target ≥80% eval SR.
- Optional nightly workflow `training-nightly.yml` (`workflow_dispatch` + cron): run 500k PPO, upload checkpoint artifact, fail if eval SR <70%.
- Add `examples/train_ppo_kuka_pick.py` + minimal reward wrapper for `kuka_pick_mujoco` (reach phase only or simplified pick).
- Document training time / GPU expectations; CPU-only path for CI with reduced steps variant.

**Out of scope**

- Full pick-place PPO with vision (depends on Wave 2.03).
- Hyperparameter search / W&B sweeps.
- Publishing trained weights to Hugging Face (optional follow-up).

## Acceptance criteria

- [ ] `examples/train_ppo_reach.py` exists and runs locally: 500k steps, logs eval SR ≥80% on `reach_target` (dummy or MuJoCo per flag).
- [ ] `tests/training/test_ppo_reach_500k.py` — `@pytest.mark.slow`, `@pytest.mark.optional_nightly`: 50k steps proxy with small net, asserts loss decreasing + SR ≥50% (full 500k in nightly only).
- [ ] `examples/train_ppo_kuka_pick.py` trains without error for 100k steps; eval SR ≥40% on reach-to-cube (document baseline).
- [ ] `.github/workflows/training-nightly.yml` runs weekly; `continue-on-error: true` until stable; artifacts retained 7 days.
- [ ] `docs/TRAINING.md` (or GOAL 02 section) documents commands + expected wall time.
- [ ] GOAL 02 reference to `train_ppo_reach.py` matches filesystem.

## Tasks

### Phase 1 — reach_target 500k (~6h)

1. Create `examples/train_ppo_reach.py` using `robodeploy.training.ppo.PPOTrainer` + `SubprocVecEnv`.
2. CLI flags: `--backend dummy|mujoco`, `--total-steps`, `--n-envs`, `--checkpoint-out`, `--seed`.
3. Add eval hook every 50k steps; save best checkpoint by success rate.
4. Add `tests/training/test_ppo_reach_500k.py` (short proxy for PR).

### Phase 2 — kuka_pick_mujoco PPO (~8h)

5. Define `KukaPickReachEnv` gym wrapper: obs = proprio + EE pos + cube relative pos; action = joint deltas.
6. Create `examples/train_ppo_kuka_pick.py` with sane defaults (16 envs, 100k–500k steps).
7. Add `tests/training/test_ppo_kuka_pick_smoke.py`: 2k steps, no crash, action in bounds.
8. Register optional `robodeploy/kuka_pick_reach-v0` if wrapper is reusable.

### Phase 3 — Nightly CI (~4h)

9. Add `.github/workflows/training-nightly.yml`:
   - Cron: `0 6 * * 0` (weekly).
   - `pip install -e ".[dev,sim]"`.
   - Run `python examples/train_ppo_reach.py --total-steps 500000 --backend dummy` (or MuJoCo if EGL reliable).
   - Upload `checkpoints/` artifact; JSON summary with final SR.
10. Document opt-in for maintainers to enable required check after 4 green weeks.

## Self-critique

| Risk | Mitigation |
|------|------------|
| **Nightly cost** — 500k × 16 envs is slow | Default nightly on dummy backend; MuJoCo monthly |
| **Flaky SR thresholds** | Nightly uses 70% bar; PR proxy uses 50k steps |
| **Checkpoint bloat** | Artifact retention 7d; don't commit `.pt` to repo |
| **kuka_pick too hard for vanilla PPO** | Scope to reach phase first; document SR ceiling |
| **Doc drift again** | Acceptance includes file-exists test in `test_train_cli.py` |

**Honest limitation**: `kuka_pick_mujoco` PPO will underperform scripted FT policy. Goal is **runnable baseline**, not SOTA manipulation.

## Test gates

| Gate | Command | Required |
|------|---------|----------|
| PR fast | `pytest tests/training/ -m "not slow" -q` | Pass |
| PR slow (optional) | `pytest tests/training/test_ppo_reach_target.py -q` | Pass on Linux CI |
| PR kuka smoke | `pytest tests/training/test_ppo_kuka_pick_smoke.py -q` | Pass |
| Local full | `python examples/train_ppo_reach.py --total-steps 500000` | SR ≥80% |
| Nightly | `training-nightly.yml` artifact + summary JSON | ≥70% SR when enabled |
| No default slowdown | `pytest -m "not hardware and not slow"` unchanged | Pass |
