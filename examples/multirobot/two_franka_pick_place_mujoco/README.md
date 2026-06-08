# Two Franka pick-place (MuJoCo)

Runnable multi-robot demo: two `example_franka_mujoco` arms share one MuJoCo world and
reach **independent** joint targets (coordination pattern: **independent**).

## Run

```bash
pip install -e ".[dev,sim]"
python examples/multirobot/two_franka_pick_place_mujoco/run.py
```

Or from a preset:

```bash
python -m examples.cli run-episode --preset two_franka_pick_mujoco --steps 80
```

## Pattern

| Robot | Base pose | Policy target |
|-------|-----------|---------------|
| `franka_left` | x = -0.55 m | home + Δq₁ |
| `franka_right` | x = +0.55 m | home − Δq₁ |

Both robots use the same `PickPlaceTask` scene (shared props). `MuJoCoBackend.initialize_multi`
namespaces each arm (`franka_left/joint*`, `franka_right/joint*`) in one MJCF.
