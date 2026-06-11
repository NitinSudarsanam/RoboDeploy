#!/usr/bin/env bash
# Headless Gazebo pick smoke (Ubuntu 24.04 + Jazzy). Run inside WSL from repo root.
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# Kill stale Gazebo/bridge processes (stray /clock → joint_states stale → CONNECTION_LOST).
pkill -f ros_gz_bridge 2>/dev/null || true
pkill -f 'gz sim' 2>/dev/null || true
pkill -f run_gazebo 2>/dev/null || true
sleep 2
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
if [[ -f .venv-wsl/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv-wsl/bin/activate
else
  echo "Run scripts/wsl24-bootstrap.sh first (creates .venv-wsl)." >&2
  exit 1
fi
export ROBODEPLOY_GAZEBO_HEADLESS=1
export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/lib
export LD_LIBRARY_PATH=/opt/ros/jazzy/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec python -m examples.kuka_ft_imu_pick_gazebo.run_gazebo
