#!/usr/bin/env bash
# Full Gazebo pick stack for demo-gazebo-pick compose service.
# GazeboLauncher (via RoboEnv) starts gz sim, ros_gz_bridge, URDF spawn, and controllers.
set -euo pipefail

source /opt/ros/jazzy/setup.bash

export ROBODEPLOY_GAZEBO_HEADLESS="${ROBODEPLOY_GAZEBO_HEADLESS:-1}"
export ROBODEPLOY_GAZEBO_READINESS_TIMEOUT="${ROBODEPLOY_GAZEBO_READINESS_TIMEOUT:-120}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
# Headless Gazebo in containers; unset DISPLAY to avoid X11 probes.
unset DISPLAY

echo "demo-gazebo-pick: headless=${ROBODEPLOY_GAZEBO_HEADLESS} gz=$(command -v gz || echo missing)"

exec python3 -m examples.kuka_ft_imu_pick_gazebo.run_gazebo
