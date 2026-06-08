"""Multi-sensor showcase — delegates to MuJoCo Universe (mujoco_showcase_kuka preset)."""

from __future__ import annotations

import sys


def main() -> int:
    if sys.platform == "win32":
        print("Skipping EGL/GLFW showcase on Windows (run on Linux CI or with viewer).")
        return 0
    from examples.mujoco_universe.run import main as universe_main

    return universe_main(["--preset", "mujoco_showcase_kuka", "--steps", "80", "--log-every", "20"])


if __name__ == "__main__":
    raise SystemExit(main())
