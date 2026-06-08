"""Deterministic seed derivation and RNG state capture for RoboEnv."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SeedSet:
    env_seed: int
    policy_seed: int
    randomizer_seed: int
    sensor_noise_seed: int
    obs_pipeline_seed: int


def derive_seeds(master_seed: int) -> SeedSet:
    """Derive stable child seeds from a single master seed."""
    rng = np.random.default_rng(int(master_seed))
    return SeedSet(
        env_seed=int(rng.integers(0, 2**31)),
        policy_seed=int(rng.integers(0, 2**31)),
        randomizer_seed=int(rng.integers(0, 2**31)),
        sensor_noise_seed=int(rng.integers(0, 2**31)),
        obs_pipeline_seed=int(rng.integers(0, 2**31)),
    )


def seed_global_rngs(seed: int) -> None:
    """Seed numpy, Python random, and torch (if installed)."""
    s = int(seed)
    np.random.seed(s)
    random.seed(s)
    try:
        import torch

        torch.manual_seed(s)
    except ImportError:
        pass


def capture_rng_state() -> dict[str, Any]:
    """Capture numpy, Python random, and optional torch RNG states."""
    state: dict[str, Any] = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    try:
        import torch

        state["torch"] = torch.get_rng_state().tolist()
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states previously captured by capture_rng_state()."""
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])
    try:
        import torch

        if "torch" in state:
            import torch as _torch

            _torch.set_rng_state(_torch.ByteTensor(state["torch"]))
    except ImportError:
        pass


def seedset_as_dict(seeds: SeedSet) -> dict[str, int]:
    return {k: int(v) for k, v in asdict(seeds).items()}
