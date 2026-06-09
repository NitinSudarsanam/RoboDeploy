"""Environment health check for RoboDeploy installations."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from robodeploy import __version__

Status = Literal["ok", "warn", "fail", "skip"]

_STATUS_LABEL = {"ok": "OK", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}


@dataclass
class DoctorCheck:
    status: Status
    label: str
    detail: str = ""
    fix: str = ""


def _try_import(module: str, attr: str = "__version__") -> tuple[bool, str]:
    try:
        mod = __import__(module)
    except ImportError:
        return False, "not installed"
    if attr and hasattr(mod, attr):
        return True, str(getattr(mod, attr))
    return True, "installed"


def _git_head(repo: Path) -> str | None:
    git = shutil.which("git")
    if not git:
        return None
    try:
        proc = subprocess.run(
            [git, "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _calibration_dir() -> Path:
    return Path.home() / ".robodeploy" / "calibration"


def _serial_device_checks() -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    if platform.system() == "Windows":
        checks.append(
            DoctorCheck(
                "skip",
                "Serial device access",
                "Skipped on Windows (check COM port permissions manually).",
            )
        )
        return checks

    device = os.environ.get("ROBODEPLOY_SERIAL_DEVICE", "/dev/ttyACM0")
    if os.path.exists(device):
        readable = os.access(device, os.R_OK)
        writable = os.access(device, os.W_OK)
        if readable and writable:
            checks.append(DoctorCheck("ok", f"Serial device {device}", "readable and writable"))
        else:
            checks.append(
                DoctorCheck(
                    "fail",
                    f"Serial device {device}",
                    "not readable/writable",
                    fix="Add user to dialout group: sudo usermod -aG dialout $USER",
                )
            )
    else:
        checks.append(
            DoctorCheck(
                "warn",
                f"Serial device {device}",
                "not present (OK if no hardware connected)",
            )
        )
    return checks


def run_doctor_checks(*, repo_root: Path | None = None) -> list[DoctorCheck]:
    """Run all doctor checks and return structured results."""
    checks: list[DoctorCheck] = []
    repo = repo_root or Path(__file__).resolve().parents[1]

    ok, ver = _try_import("numpy")
    checks.append(
        DoctorCheck("ok" if ok else "fail", "numpy", ver if ok else ver, fix="" if ok else "pip install numpy")
    )

    ok, ver = _try_import("mujoco")
    checks.append(
        DoctorCheck(
            "ok" if ok else "warn",
            "MuJoCo",
            ver if ok else "not installed",
            fix="" if ok else "pip install -e \".[sim]\"",
        )
    )

    ok, ver = _try_import("torch")
    if ok:
        cuda = "CUDA available" if getattr(__import__("torch"), "cuda").is_available() else "CPU only"
        checks.append(DoctorCheck("ok", "torch", f"{ver} ({cuda})"))
    else:
        checks.append(
            DoctorCheck(
                "warn",
                "torch",
                "not installed",
                fix="pip install -e \".[training]\" for BC / learned policies",
            )
        )

    ok, _ = _try_import("pyrealsense2", attr="")
    checks.append(
        DoctorCheck(
            "ok" if ok else "warn",
            "pyrealsense2",
            "installed" if ok else "not installed",
            fix="" if ok else "pip install -e \".[real]\" for RealSense cameras",
        )
    )

    try:
        importlib.metadata.distribution("isaacsim")
        checks.append(DoctorCheck("ok", "Isaac Sim extra", "isaacsim marker present"))
    except importlib.metadata.PackageNotFoundError:
        checks.append(
            DoctorCheck(
                "warn",
                "Isaac Sim",
                "extras [isaacsim] not installed",
                fix="Use NVIDIA Isaac Sim Python env; see docs/BACKEND_SETUP.md",
            )
        )

    ros_ok = shutil.which("ros2") is not None
    checks.append(
        DoctorCheck(
            "ok" if ros_ok else "warn",
            "ROS 2",
            "ros2 CLI found" if ros_ok else "not detected",
            fix="" if ros_ok else "Install ROS 2 and source setup.bash; see docs/BACKEND_SETUP.md",
        )
    )

    gz_ok = shutil.which("gz") is not None
    checks.append(
        DoctorCheck(
            "ok" if gz_ok else "warn",
            "Gazebo Harmonic (gz)",
            "gz CLI found" if gz_ok else "not detected",
            fix="" if gz_ok else "Install Gazebo Harmonic; see docs/BACKEND_SETUP.md",
        )
    )

    pin_ok, pin_detail = _try_import("pinocchio")
    checks.append(
        DoctorCheck(
            "ok" if pin_ok else "warn",
            "Pinocchio (kinematics)",
            pin_detail if pin_ok else "not installed",
            fix="" if pin_ok else 'pip install -e ".[kinematics]" for URDF reach IK',
        )
    )

    cal = _calibration_dir()
    if cal.exists() and os.access(cal, os.W_OK):
        checks.append(DoctorCheck("ok", f"{cal}", "writable"))
    else:
        try:
            cal.mkdir(parents=True, exist_ok=True)
            checks.append(DoctorCheck("ok", f"{cal}", "created and writable"))
        except OSError:
            checks.append(
                DoctorCheck(
                    "fail",
                    f"{cal}",
                    "not writable",
                    fix=f"mkdir -p {cal} && chmod u+rwx {cal}",
                )
            )

    checks.extend(_serial_device_checks())

    try:
        dist_ver = importlib.metadata.version("robodeploy")
    except importlib.metadata.PackageNotFoundError:
        dist_ver = None

    head = _git_head(repo)
    if dist_ver:
        if head and dist_ver == __version__:
            checks.append(
                DoctorCheck(
                    "ok",
                    "RoboDeploy package",
                    f"pip {dist_ver} matches package __version__ ({head} git head)",
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    "ok" if dist_ver == __version__ else "warn",
                    "RoboDeploy package",
                    f"pip {dist_ver}" + (f", git head {head}" if head else ""),
                    fix="" if dist_ver == __version__ else "pip install -e . to sync editable install",
                )
            )
    else:
        checks.append(
            DoctorCheck(
                "warn",
                "RoboDeploy package",
                "not installed via pip (run from source?)",
                fix="pip install -e . from repo root",
            )
        )

    checks.append(DoctorCheck("ok", "Python", f"{sys.version.split()[0]} on {platform.system()}"))
    return checks


def format_doctor_report(checks: list[DoctorCheck]) -> str:
    lines = [f"RoboDeploy Doctor v{__version__}", "=" * 24]
    fixes: list[str] = []
    for chk in checks:
        tag = _STATUS_LABEL.get(chk.status, chk.status.upper())
        suffix = f" — {chk.detail}" if chk.detail else ""
        lines.append(f"[{tag:4}] {chk.label}{suffix}")
        if chk.fix:
            fixes.append(chk.fix)
    if fixes:
        lines.append("")
        lines.append("Suggested fixes:")
        for fix in fixes:
            lines.append(f"  - {fix}")
    return "\n".join(lines)


def cmd_doctor(*, as_json: bool = False, pretty: bool = False) -> int:
    from robodeploy.cli_helpers import print_json

    checks = run_doctor_checks()
    if as_json:
        payload = {
            "version": __version__,
            "checks": [
                {
                    "status": c.status,
                    "label": c.label,
                    "detail": c.detail,
                    "fix": c.fix,
                }
                for c in checks
            ],
            "ok": not any(c.status == "fail" for c in checks),
        }
        print_json(payload, pretty=pretty)
    else:
        print(format_doctor_report(checks))
    return 1 if any(c.status == "fail" for c in checks) else 0
