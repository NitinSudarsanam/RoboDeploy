"""Nonlinear DH-parameter kinematic refinement (least-squares stub)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:
    least_squares = None  # type: ignore[assignment]

_FORMAT = "robodeploy-nonlinear-dh-v1"


@dataclass
class DHParams:
    """Simplified per-link DH offsets for system identification."""

    link_lengths: np.ndarray
    link_offsets: np.ndarray
    residuals: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": _FORMAT,
            "link_lengths": self.link_lengths.tolist(),
            "link_offsets": self.link_offsets.tolist(),
            "residuals": self.residuals,
        }


class NonlinearDHCalibration:
    """Fit small DH perturbations from FK/IK pose pairs."""

    def __init__(self, n_links: int = 6) -> None:
        self._n = int(n_links)
        self._params: DHParams | None = None

    def fit(self, pose_pairs: list[tuple[np.ndarray, np.ndarray]]) -> DHParams:
        if not pose_pairs:
            raise ValueError("pose_pairs required")
        if least_squares is None:
            lengths = np.ones(self._n, dtype=np.float64)
            offsets = np.zeros(self._n, dtype=np.float64)
            self._params = DHParams(link_lengths=lengths, link_offsets=offsets, residuals=0.0)
            return self._params

        x0 = np.concatenate([np.ones(self._n), np.zeros(self._n)])

        def residual(x: np.ndarray) -> np.ndarray:
            lengths = x[: self._n]
            offsets = x[self._n :]
            errs = []
            for fk, ik in pose_pairs:
                fk = np.asarray(fk, dtype=np.float64).reshape(-1)
                ik = np.asarray(ik, dtype=np.float64).reshape(-1)
                n = min(len(fk), len(ik), self._n)
                pred = fk[:n] + lengths[:n] * 0.01 + offsets[:n]
                errs.extend((pred - ik[:n]).tolist())
            return np.asarray(errs, dtype=np.float64)

        result = least_squares(residual, x0)
        lengths = result.x[: self._n]
        offsets = result.x[self._n :]
        self._params = DHParams(
            link_lengths=lengths,
            link_offsets=offsets,
            residuals=float(np.linalg.norm(result.fun)),
        )
        return self._params

    def save(self, path: str | Path) -> None:
        if self._params is None:
            raise RuntimeError("fit() before save()")
        Path(path).write_text(json.dumps(self._params.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DHParams:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return DHParams(
            link_lengths=np.asarray(data["link_lengths"], dtype=np.float64),
            link_offsets=np.asarray(data["link_offsets"], dtype=np.float64),
            residuals=float(data.get("residuals", 0.0)),
        )
