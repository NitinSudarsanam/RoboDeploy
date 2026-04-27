"""Shared-memory action trajectory buffer (seqlock).

ARCHITECTURE.md requires a decoupled control/inference bridge for real hardware.
The core primitive is an ActionTrajectory: a shared-memory, lock-free buffer
that the inference loop writes and the control loop reads at high frequency.

This module implements a minimal, backend-agnostic version for JOINT_POS actions.
It is intentionally small and self-contained so it can be tested without ROS.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional

import numpy as np

from robodeploy.core.types import Action


@dataclass(frozen=True)
class ActionTrajectorySpec:
    robot_ids: list[str]
    dof_by_robot: dict[str, int]


class ActionTrajectory:
    """Seqlock-protected latest-action buffer per robot.

    Layout (per robot slot):
      - seq (uint64)
      - wall_time_s (float64)
      - q (float32[dof_max])

    Writer protocol: increment seq to odd, write, increment seq to even.
    Reader protocol: read seq1, if odd retry; read payload; read seq2; accept if equal and even.
    """

    _HDR = struct.Struct("<QdI")  # seq, wall_time_s, dof

    def __init__(self, spec: ActionTrajectorySpec, *, name: Optional[str] = None, create: bool = True) -> None:
        self._spec = spec
        self._robot_ids = list(spec.robot_ids)
        self._dof_max = max(int(spec.dof_by_robot[rid]) for rid in self._robot_ids)
        self._slot_bytes = self._HDR.size + 4 * self._dof_max
        self._total_bytes = self._slot_bytes * len(self._robot_ids)
        self._index = {rid: i for i, rid in enumerate(self._robot_ids)}

        if create:
            self._shm = shared_memory.SharedMemory(create=True, size=self._total_bytes, name=name)
            # Initialize all seq to 0 and dof fields.
            for rid in self._robot_ids:
                self.write(rid, Action(joint_positions=np.zeros(int(spec.dof_by_robot[rid]), dtype=np.float32)))
        else:
            if name is None:
                raise ValueError("name is required when create=False")
            self._shm = shared_memory.SharedMemory(create=False, name=name)

    @property
    def name(self) -> str:
        return self._shm.name

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        self._shm.unlink()

    def _slot_view(self, robot_id: str) -> memoryview:
        idx = self._index[robot_id]
        start = idx * self._slot_bytes
        end = start + self._slot_bytes
        return self._shm.buf[start:end]

    def write(self, robot_id: str, action: Action) -> None:
        q = action.joint_positions
        if q is None:
            return
        q_np = np.asarray(q, dtype=np.float32).reshape(-1)
        dof = int(q_np.shape[0])
        if dof <= 0 or dof > self._dof_max:
            raise ValueError(f"Invalid dof for {robot_id}: {dof}")

        slot = self._slot_view(robot_id)

        # Read current seq
        seq, _, _ = self._HDR.unpack_from(slot, 0)
        seq = int(seq)
        seq_odd = seq + 1 if (seq % 2 == 0) else seq + 2
        self._HDR.pack_into(slot, 0, seq_odd, float(time.time()), dof)

        # Write q (pad with zeros)
        q_pad = np.zeros(self._dof_max, dtype=np.float32)
        q_pad[:dof] = q_np
        slot[self._HDR.size : self._HDR.size + 4 * self._dof_max] = q_pad.tobytes()

        # Publish complete (even seq)
        self._HDR.pack_into(slot, 0, seq_odd + 1, float(time.time()), dof)

    def read_latest_joint_positions(self, robot_id: str) -> tuple[Optional[np.ndarray], float]:
        slot = self._slot_view(robot_id)
        while True:
            seq1, t1, dof = self._HDR.unpack_from(slot, 0)
            if int(seq1) % 2 == 1:
                continue
            dof_i = int(dof)
            raw = bytes(slot[self._HDR.size : self._HDR.size + 4 * self._dof_max])
            q_all = np.frombuffer(raw, dtype=np.float32, count=self._dof_max)
            seq2, t2, dof2 = self._HDR.unpack_from(slot, 0)
            if seq1 == seq2 and int(seq2) % 2 == 0 and dof == dof2:
                if dof_i <= 0:
                    return None, float(t2)
                return q_all[:dof_i].copy(), float(t2)

