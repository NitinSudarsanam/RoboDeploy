"""Subprocess parallel episode evaluation (SubprocVecEnv-style worker pool)."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Sequence

from robodeploy.evaluation.metrics import EpisodeMetrics

_CMD_RUN = "run"
_CMD_CLOSE = "close"


def _episode_worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    run_fn,
) -> None:
    parent_remote.close()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == _CMD_RUN:
                remote.send(run_fn(data))
            elif cmd == _CMD_CLOSE:
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown subproc eval command: {cmd}")
    except KeyboardInterrupt:
        pass


class SubprocEvalPool:
    """Episode-level parallelism using the same pipe/process model as SubprocVecEnv."""

    def __init__(
        self,
        run_fn,
        *,
        n_workers: int,
        start_method: str = "spawn",
    ) -> None:
        self._run_fn = run_fn
        self._closed = False
        self._n_workers = max(1, int(n_workers))
        ctx = mp.get_context(start_method)
        self._remotes, work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(self._n_workers)])
        self._processes = [
            ctx.Process(
                target=_episode_worker,
                args=(work_remote, remote, run_fn),
                daemon=True,
            )
            for work_remote, remote in zip(work_remotes, self._remotes)
        ]
        for process in self._processes:
            process.start()
        for work_remote in work_remotes:
            work_remote.close()

    def map_jobs(self, jobs: Sequence[Any]) -> list[Any]:
        if not jobs:
            return []
        results: dict[int, Any] = {}
        pending: dict[int, int] = {}
        job_iter = enumerate(jobs)
        worker_idx = 0
        while len(results) < len(jobs):
            while worker_idx < self._n_workers:
                try:
                    job_id, job = next(job_iter)
                except StopIteration:
                    break
                self._remotes[worker_idx].send((_CMD_RUN, job))
                pending[worker_idx] = int(job_id)
                worker_idx += 1
            if not pending:
                break
            for idx in list(pending):
                if not self._remotes[idx].poll(timeout=0.05):
                    continue
                job_id = pending.pop(idx)
                results[job_id] = self._remotes[idx].recv()
                try:
                    next_id, next_job = next(job_iter)
                except StopIteration:
                    worker_idx = idx
                    continue
                self._remotes[idx].send((_CMD_RUN, next_job))
                pending[idx] = int(next_id)
        return [results[i] for i in range(len(jobs))]

    def map_episode_jobs(self, jobs: Sequence[dict[str, Any]]) -> list[EpisodeMetrics]:
        raw = self.map_jobs(jobs)
        out: list[EpisodeMetrics] = []
        for item in raw:
            if isinstance(item, EpisodeMetrics):
                out.append(item)
            elif isinstance(item, dict):
                out.append(EpisodeMetrics(**item))
            else:
                raise TypeError(f"Unexpected episode result type: {type(item)}")
        return out

    def close(self) -> None:
        if self._closed:
            return
        for remote in self._remotes:
            try:
                remote.send((_CMD_CLOSE, None))
            except (BrokenPipeError, OSError):
                pass
        for process in self._processes:
            if process._popen is None:
                continue
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
        self._closed = True

    def __enter__(self) -> "SubprocEvalPool":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        if not self._closed:
            self.close()
