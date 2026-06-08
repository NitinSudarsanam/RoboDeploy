"""Generic calibration framework (kinematic, extrinsic, system-ID)."""

from robodeploy.calibration.base import (
    SCHEMA_VERSION,
    CameraIntrinsics,
    IKinematicCalibration,
    IExtrinsicCalibration,
)
from robodeploy.calibration.store import CalibrationStore
from robodeploy.calibration.kinematic.linear import JointLinearMap, LinearKinematicCalibration
from robodeploy.calibration.kinematic.motor_bus import MotorBusCalibration
from robodeploy.calibration.extrinsic.checkerboard import CheckerboardExtrinsicCalibrator
from robodeploy.calibration.extrinsic.handeye import HandEyeCalibrator
from robodeploy.calibration.system_id.pipeline import SystemIdPipeline

__all__ = [
    "SCHEMA_VERSION",
    "CalibrationStore",
    "CameraIntrinsics",
    "IKinematicCalibration",
    "IExtrinsicCalibration",
    "JointLinearMap",
    "LinearKinematicCalibration",
    "MotorBusCalibration",
    "CheckerboardExtrinsicCalibrator",
    "HandEyeCalibrator",
    "SystemIdPipeline",
]
