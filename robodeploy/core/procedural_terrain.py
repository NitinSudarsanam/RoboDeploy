"""Procedural terrain generators shared across simulation backends."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from robodeploy.core.types import TerrainSpec


class ProceduralTerrainGenerator:
    """Heightfield generators for MuJoCo / Gazebo / IsaacSim parity."""

    GENERATORS = ("perlin", "ridge", "stairs")

    @staticmethod
    def perlin(
        size_m: tuple[float, float] = (4.0, 4.0),
        *,
        resolution: int = 128,
        octaves: int = 4,
        persistence: float = 0.5,
        seed: int = 0,
    ) -> np.ndarray:
        """Return normalized height grid in [0, 1] with shape (resolution, resolution)."""
        rng = np.random.default_rng(int(seed))
        grid = np.zeros((resolution, resolution), dtype=np.float64)
        amplitude = 1.0
        total_amp = 0.0
        for octave in range(max(1, int(octaves))):
            freq = 2**octave
            phase_x = rng.uniform(0.0, 2.0 * np.pi)
            phase_y = rng.uniform(0.0, 2.0 * np.pi)
            xs = np.linspace(0.0, freq * float(size_m[0]), resolution, endpoint=False)
            ys = np.linspace(0.0, freq * float(size_m[1]), resolution, endpoint=False)
            xx, yy = np.meshgrid(xs, ys, indexing="ij")
            layer = np.sin(xx + phase_x) * np.cos(yy + phase_y)
            grid += amplitude * layer
            total_amp += amplitude
            amplitude *= float(persistence)
        if total_amp > 0:
            grid /= total_amp
        grid = (grid - grid.min()) / max(float(grid.max() - grid.min()), 1e-9)
        return grid.astype(np.float32)

    @staticmethod
    def ridge(
        size_m: tuple[float, float] = (4.0, 4.0),
        *,
        resolution: int = 128,
        ridges: int = 5,
        seed: int = 0,
    ) -> np.ndarray:
        """Ridged terrain: parallel sine ridges along the X axis."""
        rng = np.random.default_rng(int(seed))
        xs = np.linspace(0.0, float(size_m[0]), resolution, endpoint=False)
        ys = np.linspace(0.0, float(size_m[1]), resolution, endpoint=False)
        xx, _ = np.meshgrid(xs, ys, indexing="ij")
        phase = rng.uniform(0.0, 2.0 * np.pi)
        freq = max(1, int(ridges)) * np.pi / max(float(size_m[0]), 1e-6)
        grid = 0.5 + 0.5 * np.sin(freq * xx + phase)
        return grid.astype(np.float32)

    @staticmethod
    def stairs(
        size_m: tuple[float, float] = (4.0, 4.0),
        *,
        resolution: int = 128,
        num_steps: int = 8,
    ) -> np.ndarray:
        """Stepped terrain rising along the Y axis."""
        steps = max(2, int(num_steps))
        ys = np.linspace(0.0, 1.0, resolution, endpoint=False)
        step_idx = np.floor(ys * steps).astype(np.int32)
        profile = step_idx.astype(np.float32) / float(steps - 1)
        grid = np.tile(profile, (resolution, 1))
        return grid.astype(np.float32)

    @classmethod
    def generate(
        cls,
        generator: str,
        *,
        size_m: tuple[float, float] = (4.0, 4.0),
        resolution: int = 128,
        seed: int = 0,
        **kwargs: object,
    ) -> np.ndarray:
        key = str(generator).lower()
        if key == "ridge":
            return cls.ridge(size_m=size_m, resolution=resolution, seed=seed, ridges=int(kwargs.get("ridges", 5)))
        if key == "stairs":
            return cls.stairs(size_m=size_m, resolution=resolution, num_steps=int(kwargs.get("num_steps", 8)))
        return cls.perlin(
            size_m=size_m,
            resolution=resolution,
            seed=seed,
            octaves=int(kwargs.get("octaves", 4)),
            persistence=float(kwargs.get("persistence", 0.5)),
        )

    @staticmethod
    def to_png(heightfield: np.ndarray, out_path: Path, *, max_height_m: float = 0.5) -> Path:
        """Write 16-bit grayscale PNG suitable for Gazebo heightmaps."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to export procedural terrain PNGs.") from exc
        scaled = np.clip(heightfield * float(max_height_m), 0.0, 1.0)
        img16 = (scaled * 65535.0).astype(np.uint16)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img16, mode="I;16").save(out_path)
        return out_path

    @classmethod
    def to_temp_heightmap(
        cls,
        *,
        size_m: tuple[float, float] = (4.0, 4.0),
        resolution: int = 64,
        seed: int = 0,
        max_height_m: float = 0.25,
        generator: str = "perlin",
        **kwargs: object,
    ) -> Path:
        grid = cls.generate(generator, size_m=size_m, resolution=resolution, seed=seed, **kwargs)
        fd, path = tempfile.mkstemp(prefix="robodeploy_terrain_", suffix=".png")
        try:
            import os

            os.close(fd)
        except Exception:
            pass
        return cls.to_png(grid, Path(path), max_height_m=max_height_m)

    @classmethod
    def resolve_terrain(cls, terrain: TerrainSpec) -> TerrainSpec:
        """Convert procedural terrain to a heightfield-backed ``TerrainSpec``."""
        if terrain.kind != "procedural":
            return terrain
        params = dict(terrain.procedural_params or {})
        png_path = cls.to_temp_heightmap(
            size_m=tuple(terrain.size),
            resolution=int(params.get("resolution", 64)),
            seed=int(params.get("seed", 0)),
            max_height_m=float(params.get("max_height_m", 0.25)),
            generator=str(params.get("generator", "perlin")),
            ridges=int(params.get("ridges", 5)),
            num_steps=int(params.get("num_steps", 8)),
            octaves=int(params.get("octaves", 4)),
            persistence=float(params.get("persistence", 0.5)),
        )
        return replace(
            terrain,
            kind="heightfield",
            heightfield_path=str(png_path),
        )
