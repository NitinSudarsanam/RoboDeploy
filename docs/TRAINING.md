# Production training paths

RoboDeploy ships toy-task CI smoke tests plus optional production-scale PPO scripts for benchmark baselines (GOAL 02 / WAVE2_04).

## Quick reference

| Path | Steps | Backend | Wall time (CPU, rough) |
|------|-------|---------|------------------------|
| `pytest tests/training/ -m "not slow"` | — | dummy | &lt;2 min |
| `examples/train_ppo_reach.py` (default) | 500k | dummy | ~30–60 min |
| `examples/train_ppo_reach.py --backend mujoco` | 500k | MuJoCo reach_target | ~1–3 h (CPU) |
| `examples/train_ppo_kuka_pick.py` | 100k–500k | kuka_pick_mujoco | ~2–6 h (CPU) |
| `ppo-nightly.yml` (weekly CI) | 50k proxy | dummy | ~10–20 min |

GPU (`cuda` when available) speeds PPO roughly 2–5× depending on env count and rollout size.

## reach_target PPO (500k baseline)

Train on the tier-1 `manipulation_v1/reach_target` benchmark:

```bash
pip install -e ".[training,sim]"

# Dummy backend — reproducible, CPU-friendly (target ≥80% eval SR)
python examples/train_ppo_reach.py \
  --backend dummy \
  --total-steps 500000 \
  --n-envs 16 \
  --eval-every 50000 \
  --checkpoint-out runs/ppo_reach_best.pt

# MuJoCo backend — same task on physics sim
python examples/train_ppo_reach.py \
  --backend mujoco \
  --total-steps 500000 \
  --n-envs 8 \
  --eval-every 50000
```

Outputs under a temp or `--log-dir` directory:

- `ppo_reach_best.pt` — best checkpoint by eval success rate
- `ppo_reach_final.pt` — weights after full run
- `training_summary.json` — final metrics and paths

### PR / nightly gates

- **PR fast:** `pytest tests/training/ -m "not slow" -q`
- **PR slow (optional):** `pytest tests/training/test_ppo_reach_target.py -q`
- **50k proxy:** `pytest tests/training/test_ppo_reach_500k.py -q` (`@slow`, `@optional_nightly`)
- **Weekly CI:** `.github/workflows/ppo-nightly.yml` runs 50k dummy PPO (`continue-on-error: true` until stable)

## kuka_pick_mujoco PPO (reach phase)

Vanilla PPO on the full pick preset is a **runnable baseline**, not SOTA manipulation. Expect lower success rates than scripted reach DSL policies.

```bash
python examples/train_ppo_kuka_pick.py --total-steps 100000 --n-envs 8
# equivalent:
python examples/train_ppo_reach.py --preset kuka_pick_mujoco --total-steps 100000
```

Documented baseline: ≥40% reach-to-cube success after 100k steps with default hyperparameters (varies by seed).

## CLI shortcuts

```bash
# Short smoke (dummy)
robodeploy train ppo --dummy --n-envs 2 --total-steps 256

# Preset path (requires examples on PYTHONPATH + MuJoCo)
robodeploy train ppo --preset kuka_pick_mujoco --n-envs 4 --total-steps 10000
```

## Evaluating checkpoints

**BC checkpoints** on dummy env:

```bash
robodeploy train eval --checkpoint runs/bc_final.pt --dummy --episodes 10 --json
```

**Benchmark eval** (scripted or learned policy on tier-1 tasks):

```bash
robodeploy eval \
  --benchmark manipulation_v1/reach_target \
  --policy runs/ppo_reach_best.pt \
  --backend dummy \
  --episodes 20 \
  --output reports/reach_eval.json
```

For MuJoCo reach_target smoke after training, see `tests/training/test_train_eval_benchmark_e2e.py` (`test_train_ppo_checkpoint_eval_mujoco_reach_target`).

## Related docs

- [Tutorial 3 — BC training](tutorials/03_training.md)
- [Policy creation](POLICY_CREATION.md) — wrap checkpoints in `TrainablePolicyBase`
- [GOAL 02 plan](../plans/GOAL_02_TRAINING_LOOP.md) — full training-loop acceptance criteria
