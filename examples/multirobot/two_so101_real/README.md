# Two SO-101 arms (ROS2)

Multi-robot ROS2 example with per-arm namespaces (`/so101_left`, `/so101_right`).
Uses embedded `dev_fake_sim` for local runs without hardware.

## Run (fake sim)

```bash
pip install -e ".[dev,real]"
python examples/multirobot/two_so101_real/run.py
```

## Real hardware

1. Calibrate each arm: `python -m examples.so101.calibrate_so101 --port /dev/ttyACM0`
2. Set `so101_left.controller` / `so101_right.controller` to `so101_feetech`
3. Pass ports via `robot0.port` keys or `ROBODEPLOY_SO101_PORT`

Coordination: **independent** hold policies per arm.
