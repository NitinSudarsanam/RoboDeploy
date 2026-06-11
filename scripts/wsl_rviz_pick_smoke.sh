#!/usr/bin/env bash
# Headless RViz fake-sim pick smoke (Ubuntu 24.04 + Jazzy). Run inside WSL from repo root.
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Clean stale Gazebo bridges that publish /clock and break isolated fake-sim (CONNECTION_LOST).
pkill -f ros_gz_bridge 2>/dev/null || true
pkill -f 'gz sim' 2>/dev/null || true
pkill -f run_gazebo 2>/dev/null || true
sleep 1
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
if [[ -f .venv-wsl/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv-wsl/bin/activate
else
  echo "Run scripts/wsl24-bootstrap.sh first (creates .venv-wsl)." >&2
  exit 1
fi
exec python -m examples.cli run-episode \
  --preset kuka_ft_imu_pick_ros2_rviz_headless \
  --seed 0 \
  --steps 1500 \
  --json
