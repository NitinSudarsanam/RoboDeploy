# Tutorial 3 — Training a Behavior-Cloning Policy

**Time:** ~45 minutes  
**Goal:** Train BC on teleop demos and evaluate the checkpoint.

## Prerequisites

```bash
python -m pip install -e ".[training,dev]"
```

Recorded demos from [Tutorial 2](02_teleop.md) or exported episodes:

```bash
robodeploy export-episode --dummy --steps 100 --action sinusoid --out demos/dummy.jsonl
```

## Train BC

```bash
robodeploy train bc \
  --dataset demos/kuka_teleop.jsonl \
  --obs proprio \
  --epochs 50 \
  --batch-size 32 \
  --log-dir ./runs/bc_kuka \
  --out ./runs/bc_kuka/bc_final.pt
```

Use `--dummy` to synthesize a tiny dataset when testing the pipeline without real demos:

```bash
robodeploy train bc --dataset /tmp/missing.jsonl --dummy --epochs 2 --json
```

### Key flags

| Flag | Description |
|------|-------------|
| `--obs` | Comma-separated observation keys (`proprio`, `joint_pos`, …) |
| `--action-dim` | Override action vector size |
| `--lr` | Learning rate (default `1e-4`) |
| `--out` | Final checkpoint path |

## Evaluate checkpoint

```bash
robodeploy train eval --checkpoint ./runs/bc_kuka/bc_final.pt --dummy --episodes 5 --json
```

For preset-based evaluation, deploy via `examples/learned_policy_deploy/run.py`:

```bash
python -m examples.learned_policy_deploy.run \
  --checkpoint ./runs/bc_kuka/bc_final.pt \
  --preset kuka_pick_mujoco
```

## Scripted baseline (no learning)

Compare against the reach DSL policy:

```bash
python -m examples.cli run-episode --preset kuka_pick_mujoco --steps 200
```

## Benchmark evaluation

```bash
robodeploy list-benchmarks --json
robodeploy eval --benchmark manipulation_v1/reach_target --backend dummy --episodes 10 --json
```

## Next steps

- [Tutorial 4 — Sim2Real](04_sim2real.md) — domain randomization and transfer evaluation.
- [Policy creation guide](../POLICY_CREATION.md) — wrap learned checkpoints in `TrainablePolicyBase`.
