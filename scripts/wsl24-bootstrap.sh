#!/usr/bin/env bash
# Bootstrap Ubuntu 24.04 WSL for RoboDeploy RViz/Gazebo demos (ROS 2 Jazzy).
# Run inside WSL: bash scripts/wsl24-bootstrap.sh
set -euo pipefail

if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
else
  echo "ERROR: /etc/os-release not found — run inside WSL/Linux." >&2
  exit 1
fi

if [[ "${VERSION_ID:-}" != "24.04" ]]; then
  echo "WARN: Ubuntu ${VERSION_ID:-unknown} detected; Jazzy apt targets noble (24.04)." >&2
  echo "      On 22.04 use Docker: docker compose -f docker/docker-compose.yml --profile ros2 run --rm demo-gazebo-pick" >&2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "==> ROS 2 Jazzy base packages"
sudo apt-get update
sudo apt-get install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu ${UBUNTU_CODENAME} main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt-get update
sudo apt-get install -y \
  ros-jazzy-desktop \
  ros-jazzy-tf2-ros \
  ros-jazzy-ros-gz \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-ros2-controllers \
  python3-venv python3-pip

echo "==> Python venv (.venv-wsl)"
python3 -m venv .venv-wsl
# shellcheck disable=SC1091
source .venv-wsl/bin/activate
pip install --upgrade pip
pip install -e ".[sim,kinematics,dev]"  # rclpy from apt; no separate [ros2] extra

echo "==> robodeploy doctor"
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
robodeploy doctor

cat <<EOF

Bootstrap complete.

Next steps:
  source /opt/ros/jazzy/setup.bash && source .venv-wsl/bin/activate
  python -m examples.cli run-episode --preset kuka_ft_imu_pick_ros2_rviz_headless --seed 0 --steps 1500 --json
  python -m examples.kuka_ft_imu_pick_gazebo.run_gazebo   # Gazebo pick (Linux/WSL)

See docs/DEMO_RUNBOOK.md for success signals and Docker fallback on Ubuntu 22.04.
EOF
