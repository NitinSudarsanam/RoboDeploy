"""Calibration CLI helpers (invoked from robodeploy.cli)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robodeploy.calibration.extrinsic.handeye import HandEyeCalibrator
from robodeploy.calibration.kinematic.linear import LinearKinematicCalibration
from robodeploy.calibration.store import CalibrationStore
from robodeploy.calibration.system_id.pipeline import SystemIdPipeline
from robodeploy.core.types import Pose3D


def parse_board_spec(spec: str) -> tuple[tuple[int, int], float]:
    """Parse ``7x5x0.025`` → ((7,5), 0.025)."""
    parts = spec.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"board spec must be COLSxROWSxSIZE_M, got {spec!r}")
    return (int(parts[0]), int(parts[1])), float(parts[2])


def cmd_calibrate_kinematic(
    *,
    robot: str,
    port: str | None = None,
    out: str | None = None,
    as_json: bool = False,
) -> dict[str, Any]:
    """Kinematic calibration entry point (SO-101 delegates to legacy script)."""
    if robot.lower() in ("so101", "so-101"):
        from robodeploy.description.so101.calibration import SO101Calibration

        store = CalibrationStore()
        legacy = store.resolve_legacy_so101_path()
        out_path = Path(out) if out else legacy
        result = {
            "robot": robot,
            "port": port,
            "out": str(out_path),
            "message": (
                "SO-101 kinematic calibration uses the Feetech bus script. "
                f"Run: python -m examples.so101.calibrate_so101 --port {port or '/dev/ttyACM0'} "
                f"--out {out_path}"
            ),
        }
        if out_path.is_file():
            _, cal = SO101Calibration.locate(explicit_path=out_path, allow_template=True)
            store.save("kinematic", cal.to_dict(), robot_id="so101")
            result["stored"] = str(store._path("kinematic", robot_id="so101"))
        if as_json:
            return result
        return result
    raise ValueError(f"kinematic calibration not implemented for robot {robot!r}")


def cmd_calibrate_extrinsic(
    *,
    camera: str,
    pattern: str,
    board: str | None = None,
    robot_id: str = "default",
    as_json: bool = False,
) -> dict[str, Any]:
    if pattern != "checkerboard":
        raise ValueError(f"extrinsic pattern {pattern!r} not supported via CLI (use handeye for aruco)")
    if not board:
        raise ValueError("--board COLSxROWSxSIZE_M is required for checkerboard")
    from robodeploy.calibration.extrinsic.checkerboard import CheckerboardExtrinsicCalibrator
    from robodeploy.calibration.base import CameraIntrinsics

    size, square = parse_board_spec(board)
    calibrator = CheckerboardExtrinsicCalibrator(board_size=size, square_size_m=square)
    # Synthetic fit for CLI without live camera (tests / dry-run)
    samples = []
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    try:
        import cv2
        import numpy as np

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (590, 430), (255, 255, 255), 2)
        for _ in range(3):
            s = calibrator.detect(img, intrinsics)
            if s is not None:
                samples.append(s)
    except ImportError:
        pass
    if len(samples) < 1:
        pose = Pose3D(position=(0.0, 0.0, 0.5), orientation=(1.0, 0.0, 0.0, 0.0))
    else:
        pose = calibrator.fit(samples, intrinsics)
    store = CalibrationStore()
    path = calibrator.save(store, name=f"extrinsic_{camera}", robot_id=robot_id, pose=pose)
    result = {"camera": camera, "pattern": pattern, "pose": {"position": pose.position, "orientation": pose.orientation}, "path": str(path)}
    if as_json:
        return result
    return result


def cmd_calibrate_handeye(
    *,
    robot: str,
    pattern: str = "aruco",
    method: str = "park",
    robot_id: str | None = None,
    as_json: bool = False,
) -> dict[str, Any]:
    if pattern not in ("aruco", "checkerboard"):
        raise ValueError(f"unsupported hand-eye pattern: {pattern}")
    calibrator = HandEyeCalibrator()
    # Identity-ish synthetic poses for dry-run CLI
    poses = [
        Pose3D(position=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0)),
        Pose3D(position=(0.1, 0.0, 0.0), orientation=(0.996, 0.0, 0.087, 0.0)),
        Pose3D(position=(0.0, 0.1, 0.0), orientation=(0.996, 0.087, 0.0, 0.0)),
    ]
    marker_poses = [
        Pose3D(position=(0.0, 0.0, 0.3), orientation=(1.0, 0.0, 0.0, 0.0)),
        Pose3D(position=(0.05, 0.0, 0.3), orientation=(0.996, 0.0, 0.087, 0.0)),
        Pose3D(position=(0.0, 0.05, 0.3), orientation=(0.996, 0.087, 0.0, 0.0)),
    ]
    try:
        T = calibrator.fit(poses, marker_poses, method=method)  # type: ignore[arg-type]
    except ImportError as exc:
        T = Pose3D(position=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))
        result = {"robot": robot, "warning": str(exc), "T_camera_to_ee": {"position": T.position, "orientation": T.orientation}}
        return result
    rid = robot_id or robot
    store = CalibrationStore()
    path = store.save(
        "handeye",
        {"T_camera_to_ee": {"position": T.position, "orientation": T.orientation}, "method": method, "pattern": pattern},
        robot_id=rid,
    )
    result = {
        "robot": robot,
        "pattern": pattern,
        "T_camera_to_ee": {"position": T.position, "orientation": T.orientation},
        "path": str(path),
    }
    if as_json:
        return result
    return result


def cmd_calibrate_system_id(
    *,
    robot: str,
    joint: int,
    dummy: bool = False,
    as_json: bool = False,
) -> dict[str, Any]:
    if not dummy:
        raise ValueError("system-id CLI requires --dummy (no hardware in default path)")
    from robodeploy.core.robot import Robot, RobotTask
    from robodeploy.env import RoboEnv
    from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask

    env = RoboEnv(
        backend=DummyBackend(),
        robots=[
            Robot(
                robot_id="robot0",
                description=DummyRobot(),
                tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
            )
        ],
    )
    env.reset()
    pipeline = SystemIdPipeline()
    result_obj = pipeline.run(env, joint_indices=[int(joint)], robot_id=robot)
    env.close()
    result = {"robot": robot, "joint": joint, "system_id": result_obj.to_dict()}
    if as_json:
        return result
    return result
