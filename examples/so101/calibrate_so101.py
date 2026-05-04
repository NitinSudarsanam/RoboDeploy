"""Generate ``~/.robodeploy/so101_calibration.json`` for the SO-101 Feetech bus (lerobot).

Two-pose linear fit per joint: tick ≈ zero_ticks + q_rad * ticks_per_rad.

Usage::

    python -m examples.so101.calibrate_so101 --port /dev/ttyACM0 --out ~/.robodeploy/so101_calibration.json

Requires: ``pip install "lerobot[feetech]"`` and a ROS-free terminal (no ``rclpy`` needed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_path() -> None:
    r = _repo_root()
    if str(r) not in sys.path:
        sys.path.insert(0, str(r))


def _import_bus():
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        return FeetechMotorsBus, Motor, MotorNormMode
    except ImportError as e:
        raise SystemExit(
            'Install lerobot with Feetech extras:  pip install "lerobot[feetech]"\n' f"Original error: {e}"
        ) from e


def _motors(Motor, MotorNormMode):
    norm = MotorNormMode.DEGREES
    return {str(i): Motor(id=i, model="sts3215", norm_mode=norm) for i in range(1, 7)}


def _read_ticks(bus, names: list[str]) -> dict[str, int]:
    out = bus.sync_read("Present_Position", names, normalize=False)
    return {k: int(v) for k, v in out.items()}


def _parse_floats(line: str, default: list[float]) -> list[float]:
    line = line.strip()
    if not line:
        return list(default)
    parts = line.split()
    if len(parts) != 6:
        raise ValueError("expected 6 space-separated floats")
    return [float(x) for x in parts]


def main() -> None:
    _ensure_path()
    ap = argparse.ArgumentParser(description="SO-101 Feetech calibration (two poses)")
    ap.add_argument("--port", required=True, help="Serial device, e.g. /dev/ttyACM0")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".robodeploy" / "so101_calibration.json",
        help="Output JSON path",
    )
    args = ap.parse_args()

    from robodeploy.description.so101 import SO101Description
    from robodeploy.description.so101.calibration import JointCalibration, SO101Calibration

    FeetechMotorsBus, Motor, MotorNormMode = _import_bus()
    bus = FeetechMotorsBus(str(args.port), _motors(Motor, MotorNormMode))
    bus.connect(handshake=True)
    bus.disable_torque()
    names = [str(i) for i in range(1, 7)]

    desc = SO101Description()
    lim = desc.joint_position_limits

    print("Move the arm to the FIRST calibration pose (torque off). Default joint radians = all zeros.")
    input("Press Enter when ready… ")
    t_a = _read_ticks(bus, names)
    print("Enter 6 joint positions in radians for this pose [default: 0 0 0 0 0 0]:")
    q_a = _parse_floats(input(), [0.0] * 6)

    print("Move the arm to a SECOND pose (different from the first).")
    input("Press Enter when ready… ")
    t_b = _read_ticks(bus, names)
    print("Enter 6 joint positions in radians for this pose:")
    q_b = _parse_floats(input(), [0.0] * 6)

    joints: list[JointCalibration] = []
    for i, name in enumerate(names):
        da = q_a[i]
        db = q_b[i]
        t0 = float(t_a[name])
        t1 = float(t_b[name])
        denom = db - da
        if abs(denom) < 1e-6:
            raise SystemExit(f"joint {name}: second pose must differ in radians from first (dq≈0).")
        ticks_per_rad = (t1 - t0) / denom
        zero_ticks = int(round(t0 - da * ticks_per_rad))
        joints.append(
            JointCalibration(
                name=name,
                motor_id=i + 1,
                zero_ticks=zero_ticks,
                ticks_per_rad=float(ticks_per_rad),
                soft_min_rad=float(lim[i, 0]),
                soft_max_rad=float(lim[i, 1]),
            )
        )

    cal = SO101Calibration(joints=tuple(joints))
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cal.to_dict(), indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print("Set ROBODEPLOY_SO101_CALIBRATION to this path or pass robot0.calibration_path=... in config_overrides.")

    bus.disconnect(disable_torque=True)


if __name__ == "__main__":
    main()
