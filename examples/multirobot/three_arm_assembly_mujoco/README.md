# Three-arm assembly (MuJoCo)

Three arms (two Franka + one Kuka) cooperate in one MJCF world. Coordination pattern:
**independent** per-arm joint targets with a shared `PickPlaceTask` scene.

## Run

```bash
pip install -e ".[dev,sim]"
python examples/multirobot/three_arm_assembly_mujoco/run.py
```

## Layout

| Robot | Arm | Base pose |
|-------|-----|-----------|
| `arm_center` | Franka | origin |
| `arm_left` | Franka | x = -0.7 m |
| `arm_right` | Kuka | x = +0.7 m |

`MuJoCoBackend.initialize_multi` namespaces joints (`arm_center/joint*`, etc.).
