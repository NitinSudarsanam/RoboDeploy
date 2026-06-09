"""Episode video recording during benchmark evaluation."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from robodeploy.core.types import Observation
from robodeploy.env import RoboEnv


def _frame_from_obs(obs: Observation, camera_name: str) -> Any | None:
    images = getattr(obs, "images", None) or {}
    if camera_name in images and images[camera_name] is not None:
        return images[camera_name]
    rgb = getattr(obs, "rgb", None)
    if rgb is not None:
        return rgb
    if isinstance(getattr(obs, "extra", None), dict):
        extra_rgb = obs.extra.get("rgb") or obs.extra.get(camera_name)
        if extra_rgb is not None:
            return extra_rgb
    return None


def _to_uint8_rgb(frame: Any) -> Any | None:
    try:
        import numpy as np
    except ImportError:
        return None
    arr = np.asarray(frame)
    if arr.ndim < 2:
        return None
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 255.0)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None
    return arr


class EpisodeVideoRecorder:
    """Collect RGB frames from observations and write MP4 on finish."""

    def __init__(
        self,
        *,
        env: RoboEnv,
        camera_name: str = "overhead_camera",
        out_dir: Path | str,
        fps: int = 30,
    ) -> None:
        self._env = env
        self._camera_name = str(camera_name)
        self._out_dir = Path(out_dir)
        self._fps = max(1, int(fps))
        self._frames: list[Any] = []
        self._episode_id = ""

    def start(self, episode_id: str) -> None:
        self._episode_id = str(episode_id)
        self._frames = []

    def observe(self, obs: Observation) -> None:
        frame = _frame_from_obs(obs, self._camera_name)
        if frame is None:
            return
        rgb = _to_uint8_rgb(frame)
        if rgb is not None:
            self._frames.append(rgb)

    def finish(self) -> Path | None:
        if not self._frames:
            return None
        self._out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._out_dir / f"{self._episode_id}.mp4"
        try:
            import imageio.v2 as imageio
        except ImportError:
            png_dir = self._out_dir / self._episode_id
            png_dir.mkdir(parents=True, exist_ok=True)
            try:
                import imageio.v2 as imageio
            except ImportError:
                return None
            for idx, frame in enumerate(self._frames):
                imageio.imwrite(png_dir / f"frame_{idx:04d}.png", frame)
            return png_dir
        imageio.mimsave(str(out_path), self._frames, fps=self._fps)
        return out_path

    _EMBED_MAX_BYTES = 2 * 1024 * 1024

    @classmethod
    def embed_path(cls, path: Path | None) -> str | None:
        if path is None or not path.is_file():
            return None
        if path.stat().st_size > cls._EMBED_MAX_BYTES:
            return None
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:video/mp4;base64,{data}"
