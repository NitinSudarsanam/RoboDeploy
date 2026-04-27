"""URDF spawner for Gazebo via ros_gz_sim."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UrdfSpawnConfig:
    urdf_path: str
    name: str = "robot0"
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


class UrdfSpawner:
    def __init__(self, cfg: UrdfSpawnConfig) -> None:
        self._cfg = cfg

    def spawn(self) -> None:
        ros2 = shutil.which("ros2")
        if not ros2:
            return
        p = Path(self._cfg.urdf_path)
        if not p.exists():
            raise FileNotFoundError(f"URDF not found: {p}")
        x, y, z = self._cfg.xyz
        rr, rp, ry = self._cfg.rpy
        subprocess.run(
            [
                ros2,
                "run",
                "ros_gz_sim",
                "create",
                "-name",
                str(self._cfg.name),
                "-file",
                str(p),
                "-x",
                str(float(x)),
                "-y",
                str(float(y)),
                "-z",
                str(float(z)),
                "-R",
                str(float(rr)),
                "-P",
                str(float(rp)),
                "-Y",
                str(float(ry)),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

