# Tutorial 2b — Teleoperation and Demo Collection

> For defining a custom task with `SceneBuilder`, see [02_your_first_task.md](02_your_first_task.md).

**Time:** ~20 minutes  
**Goal:** Control a Kuka arm in MuJoCo with the keyboard and record demonstrations.

## Prerequisites

```bash
python -m pip install -e ".[sim,teleop,dev]"
```

## Keyboard teleop

The `examples/teleop_keyboard_kuka.py` script wraps `run_teleop_session` on the `kuka_pick_mujoco` preset:

```bash
python -m examples.teleop_keyboard_kuka --preset kuka_pick_mujoco --max-steps 500
```

### Default bindings

| Key | Action |
|-----|--------|
| WASD / QE | Translate end-effector |
| IJKL / UO | Rotate orientation |
| Space | Toggle gripper |
| Tab | Toggle recording |
| R | Reset episode |
| Esc | E-stop (trips safety monitor) |

## Record demonstrations

```bash
mkdir -p demos
python -m examples.teleop_keyboard_kuka \
  --record demos/kuka_teleop.jsonl \
  --max-steps 500
```

Press **Tab** during the session to start/stop recording. Each saved file is JSONL with observations and actions suitable for BC training.

## Replay a recording

```bash
python -m examples.replay_demo demos/kuka_teleop.jsonl --preset kuka_pick_mujoco
```

Or use the observability CLI:

```bash
robodeploy replay --input demos/kuka_teleop.jsonl --preset kuka_pick_mujoco
```

## Headless record → BC (no GUI)

For CI or servers without a display, use the stub teleop recorder (scripted policy steps, stamped like human demos):

```bash
python -m pytest tests/test_teleop_record_stub.py -q
```

The test exercises `record_stub_episode` → `DemoDataset` → BC train on MuJoCo. See `CONTRACTS.md` for the `TeleopCommand` / recording metadata contract.

## Other teleop devices

Install extras for SpaceMouse or gamepad:

```bash
python -m pip install -e ".[teleop]"
```

Pass `device="spacemouse"` or `device="gamepad"` to `run_teleop_session` in your own script (see `robodeploy.teleop.session`).

## Tips

- Start with `enable_viewer: true` in `backend_kwargs.config` if you want the MuJoCo viewer.
- Record at least 15–20 short successful episodes before training BC.
- Use `robodeploy lint preset examples/config/presets.yaml --check kuka_pick_mujoco` to verify preset keys.

## Next steps

- [Tutorial 3 — Training](03_training.md) — train BC on recorded demos.
- [Tutorial 5 — Safety](05_safety.md) — e-stop and force limits on real hardware.
