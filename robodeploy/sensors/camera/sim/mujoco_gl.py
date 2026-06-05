"""MuJoCo GL backend selection for headless/offscreen rendering."""

from __future__ import annotations

import os
import sys


def ensure_mujoco_gl_backend() -> str:
    """Pick a stable MuJoCo GL backend when ``MUJOCO_GL`` is unset."""
    current = os.environ.get("MUJOCO_GL", "").strip().lower()
    if current:
        return current
    if sys.platform == "win32":
        os.environ["MUJOCO_GL"] = "wgl"
        return "wgl"
    if sys.platform == "darwin":
        os.environ["MUJOCO_GL"] = "glfw"
        return "glfw"
    os.environ["MUJOCO_GL"] = "egl"
    return "egl"
