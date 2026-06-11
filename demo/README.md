# Kuka Pick & Place Demo

Pick-and-place with the Kuka arm. Edit `SIMULATOR` at the top of `demo/run_pick.py`,
then run:

```bash
pip install -e ".[sim]"
python demo/run_pick.py
```

| `SIMULATOR` | Requires |
|-------------|----------|
| `mujoco` | `pip install -e ".[sim]"` (default) |
| `rviz` | Linux/WSL + ROS 2 Jazzy, `pip install -e ".[sim,ros2]"` |
| `gazebo` | Linux/WSL + ROS 2 Jazzy, `pip install -e ".[sim,ros2]"` |

Config (task, policy, sensors) lives in `demo/config/kuka_pick.yaml`.
Robot and backends live in `robodeploy/`.
