# Franka + Kuka collaborative (MuJoCo)

Two heterogeneous arms share one pick-place scene. Franka reaches toward the source
prop; Kuka holds a tray pose — **sequential** coordination (each arm runs its task
in order per `RobotTask.mode`).

## Run

```bash
pip install -e ".[dev,sim]"
python examples/multirobot/franka_kuka_collaborative_mujoco/run.py
```

## Pattern

| Robot | Type | Base pose | Role |
|-------|------|-----------|------|
| `franka_placer` | Franka MJCF | x = -0.5 m | reach + pick |
| `kuka_tray` | Kuka MJCF | x = +0.55 m | tray hold |

Both use `PickPlaceTask` with independent `JointTrackPolicy` targets.
