#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
source /opt/ros/jazzy/setup.bash
source .venv-wsl/bin/activate
python -m examples.cli run-episode --preset kuka_ft_imu_pick_ros2_rviz_headless --headless --seed 0 --steps 1500 --json > /tmp/rviz_pick.log 2>&1
python3 -c "
import json
lines = [ln for ln in open('/tmp/rviz_pick.log') if ln.startswith('{')]
if not lines:
    raise SystemExit('no JSON in /tmp/rviz_pick.log')
d = json.loads(lines[-1])
print('success', d['info']['success'], 'step', d['info']['step'])
"
