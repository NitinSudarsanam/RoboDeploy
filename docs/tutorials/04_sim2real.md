# Tutorial 4 — Sim2Real Transfer

**Time:** ~60 minutes  
**Goal:** Measure sim-to-real gap, tune domain randomization, and deploy via ROS 2.

## Prerequisites

```bash
python -m pip install -e ".[sim,real,dev]"
robodeploy doctor   # verify ROS 2 and RealSense if using hardware
```

Read [BACKEND_SETUP.md](../BACKEND_SETUP.md) for MuJoCo, Gazebo, and ROS 2 installation.

## Compare sim presets

Use `robodeploy config diff` to see what changes between sim and real presets:

```bash
robodeploy config diff kuka_pick_mujoco kuka_sensor_ros2_rviz
```

Resolve full configs:

```bash
robodeploy config resolve --preset kuka_pick_mujoco --json
robodeploy config show --preset kuka_sensor_ros2_rviz
```

## Domain-randomization sweep (dummy proxy)

Before hardware, run a DR sensitivity sweep on the dummy backend:

```bash
robodeploy dr-sweep --dummy --output ./runs/dr_sweep --seeds 3 --episodes 2 --json
```

Full sim DR sweep with examples helper:

```bash
python -m examples.sim2real.run_dr_sweep --preset kuka_pick_mujoco --output ./runs/dr
```

## Transfer evaluation

Compare matched rollouts (sim vs noisy-sim as a gap proxy):

```bash
robodeploy transfer-eval --dummy --output ./runs/transfer --episodes 5 --json
```

On real hardware, use `RoboBridge` and `TransferEvaluator` (see `robodeploy.sim2real` and `examples/sim2real/`).

## Deploy to ROS 2 / RViz

```bash
python -m examples.kuka_sensor_ros2_rviz.run_ros2_rviz
```

Or the sinusoid smoke test:

```bash
python -m examples.user_kuka_sinusoid.run_ros2_rviz
```

## Calibrate real hardware

For SO-101 and similar arms, follow [SO101_REAL.md](../SO101_REAL.md). Calibration files live under `~/.robodeploy/calibration/` (doctor verifies writability).

## Tune DR when gap is large

1. Run policy in sim with DR enabled (`task_kwargs.domain_randomization`).
2. Measure success rate vs baseline.
3. Adjust noise scales in `robodeploy.tasks.randomization`.
4. Re-run `transfer-eval` until proxy metrics stabilize.

## Preset templates for real robots

Copy anchors from `examples/presets/base_real.yaml` into `presets.yaml`:

```yaml
my_kuka_real:
  <<: *base_real
  robot: kuka
  task: pick_place
  policy: example_sensor_reach_pick
```

## Next steps

- [Tutorial 5 — Safety](05_safety.md) — force limits, workspace bounds, e-stop.
- [Scene definition](../SCENE_DEFINITION.md) — cross-backend scene validation.
