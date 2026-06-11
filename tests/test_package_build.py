from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


class PackageBuildTests(unittest.TestCase):
    def test_sdist_and_wheel_build_twine_check_and_wheel_install(self):
        """Local sdist/wheel smoke: build, twine check, install wheel in isolated venv."""
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "build", "twine", "-q"],
            cwd=str(REPO),
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "dist"
            out_dir.mkdir()
            subprocess.check_call(
                [sys.executable, "-m", "build", "--outdir", str(out_dir)],
                cwd=str(REPO),
            )

            artifacts = sorted(out_dir.iterdir())
            self.assertTrue(any(p.suffix == ".whl" for p in artifacts), "wheel missing")
            self.assertTrue(any(p.suffix == ".gz" and p.name.endswith(".tar.gz") for p in artifacts), "sdist missing")

            subprocess.check_call(
                [sys.executable, "-m", "twine", "check", *[str(p) for p in artifacts]],
                cwd=str(REPO),
            )

            wheel = next(p for p in artifacts if p.suffix == ".whl")
            venv_dir = Path(tmp) / "venv"
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
            py = _venv_python(venv_dir)
            subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip", "-q"])
            subprocess.check_call([str(py), "-m", "pip", "install", str(wheel), "-q"])
            subprocess.check_call(
                [
                    str(py),
                    "-c",
                    "import robodeploy; assert robodeploy.__version__",
                ],
            )
            subprocess.check_call([str(py), "-m", "robodeploy.cli", "--help"], cwd=str(REPO))

    def test_wheel_benchmarks_discovery_and_preset_resolve(self):
        """Installed wheel must ship benchmarks/ and resolve presets without repo checkout."""
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "build", "pyyaml", "-q"],
            cwd=str(REPO),
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "dist"
            out_dir.mkdir()
            subprocess.check_call(
                [sys.executable, "-m", "build", "--outdir", str(out_dir), "--wheel"],
                cwd=str(REPO),
            )
            wheel = next(p for p in out_dir.iterdir() if p.suffix == ".whl")
            venv_dir = Path(tmp) / "venv"
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
            py = _venv_python(venv_dir)
            subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip", "-q"])
            subprocess.check_call([str(py), "-m", "pip", "install", str(wheel), "pyyaml", "-q"])

            subprocess.check_call(
                [
                    str(py),
                    "-c",
                    "from robodeploy.evaluation.registry import BenchmarkRegistry; "
                    "r = BenchmarkRegistry(); "
                    "assert 'manipulation_v1' in r.list_suites(); "
                    "task = r.load_suite('manipulation_v1').get_task('reach_target'); "
                    "assert 'dummy' in task.available_backends(); "
                    "task.import_task_module()",
                ],
            )
            subprocess.check_call(
                [
                    str(py),
                    "-m",
                    "robodeploy.cli",
                    "list-benchmarks",
                    "--json",
                ],
            )


if __name__ == "__main__":
    unittest.main()
