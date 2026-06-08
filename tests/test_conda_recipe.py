from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / "conda-recipe" / "meta.yaml"


def _conda_on_path() -> bool:
    return shutil.which("conda") is not None


def _conda_build_ready() -> bool:
    """True when conda-build is already installed (do not auto-install in tests)."""
    if not _conda_on_path():
        return False
    try:
        subprocess.run(
            ["conda", "build", "--version"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


class CondaRecipeTests(unittest.TestCase):
    def test_meta_yaml_exists_and_lists_robodeploy(self):
        self.assertTrue(RECIPE.is_file(), f"Missing conda recipe at {RECIPE}")
        text = RECIPE.read_text(encoding="utf-8")
        self.assertIn("name: robodeploy", text)
        self.assertIn("robodeploy", text)
        try:
            import yaml
        except ImportError:
            raise unittest.SkipTest("pyyaml not installed")
        meta = yaml.safe_load(text)
        self.assertEqual(meta["package"]["name"], "robodeploy")
        self.assertIn("robodeploy", meta["test"]["imports"])
        self.assertIn("python", str(meta["requirements"]["host"]))
        self.assertIn("numpy", meta["requirements"]["run"])

    def test_recipe_source_install_imports(self):
        """Fallback when conda-build is unavailable: pip install from recipe source path."""
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", str(REPO), "--no-deps", "-q"],
            cwd=str(REPO),
        )
        subprocess.check_call(
            [
                sys.executable,
                "-c",
                "import robodeploy; assert robodeploy.__version__",
            ],
            cwd=str(REPO),
        )

    def test_conda_build_when_available(self):
        if not _conda_build_ready():
            raise unittest.SkipTest("conda-build not available (install via conda-forge to enable)")
        if platform.system() == "Windows":
            raise unittest.SkipTest(
                "conda path builds copy local .venv symlinks on Windows; run on Linux/macOS CI"
            )
        if os.environ.get("ROBODEPLOY_SKIP_CONDA_BUILD", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            raise unittest.SkipTest("ROBODEPLOY_SKIP_CONDA_BUILD is set")

        out_dir = REPO / "dist-conda"
        out_dir.mkdir(exist_ok=True)
        subprocess.check_call(
            [
                "conda",
                "build",
                str(REPO / "conda-recipe"),
                "--output-folder",
                str(out_dir),
            ],
            cwd=str(REPO),
            timeout=900,
        )
        pkgs = list(out_dir.glob("robodeploy-*.tar.bz2")) + list(
            out_dir.glob("robodeploy-*.conda")
        )
        self.assertTrue(pkgs, f"Expected conda package under {out_dir}")


if __name__ == "__main__":
    unittest.main()
