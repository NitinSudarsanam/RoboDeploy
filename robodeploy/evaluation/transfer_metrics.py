"""Sim-vs-real transfer validation metrics and reporting hooks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from robodeploy.observability.manifest import RunManifest

EnvFactory = Callable[[int], Any]
PolicyFactory = Callable[[Any], Callable[[Any], Any | None]]


def _quat_geodesic(q1: np.ndarray, q2: np.ndarray) -> float:
    a = np.asarray(q1, dtype=np.float64)
    b = np.asarray(q2, dtype=np.float64)
    dot = float(np.clip(np.abs(np.dot(a, b)), -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


def compute_trajectory_distance(
    sim_traj: list[dict[str, np.ndarray]],
    real_traj: list[dict[str, np.ndarray]],
) -> dict[str, float]:
    """L2 / geodesic distances between aligned sim and real trajectories."""
    n = min(len(sim_traj), len(real_traj))
    if n == 0:
        return {"joint_pos_l2": 0.0, "ee_pos_l2": 0.0, "ee_quat_geodesic": 0.0}

    jp = []
    ee = []
    quat = []
    for i in range(n):
        s = sim_traj[i]
        r = real_traj[i]
        if "joint_positions" in s and "joint_positions" in r:
            jp.append(float(np.linalg.norm(np.asarray(s["joint_positions"]) - np.asarray(r["joint_positions"]))))
        if "ee_position" in s and "ee_position" in r:
            ee.append(float(np.linalg.norm(np.asarray(s["ee_position"]) - np.asarray(r["ee_position"]))))
        if "ee_orientation" in s and "ee_orientation" in r:
            quat.append(_quat_geodesic(s["ee_orientation"], r["ee_orientation"]))

    return {
        "joint_pos_l2": float(np.mean(jp)) if jp else 0.0,
        "ee_pos_l2": float(np.mean(ee)) if ee else 0.0,
        "ee_quat_geodesic": float(np.mean(quat)) if quat else 0.0,
    }


def _obs_vector(obs: Any, key: str) -> np.ndarray | None:
    arr = getattr(obs, key, None)
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _kl_gaussian(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.size == 0 or q.size == 0:
        return 0.0
    mu_p, mu_q = p.mean(), q.mean()
    var_p = max(float(p.var()), eps)
    var_q = max(float(q.var()), eps)
    return float(0.5 * (np.log(var_q / var_p) + (var_p + (mu_p - mu_q) ** 2) / var_q - 1.0))


@dataclass
class EpisodeRollout:
    """One episode of states, actions, and outcome."""

    success: bool
    reward: float
    trajectory: list[dict[str, np.ndarray]] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)


@dataclass
class TransferMetrics:
    """Aggregated sim-vs-real gap metrics."""

    sim_success_rate: float
    real_success_rate: float
    success_gap: float
    trajectory_distance: dict[str, float] = field(default_factory=dict)
    obs_distribution_kl: dict[str, float] = field(default_factory=dict)
    action_distribution_l2: float = 0.0
    per_episode_breakdown: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sim_success_rate": self.sim_success_rate,
            "real_success_rate": self.real_success_rate,
            "success_gap": self.success_gap,
            "trajectory_distance": self.trajectory_distance,
            "obs_distribution_kl": self.obs_distribution_kl,
            "action_distribution_l2": self.action_distribution_l2,
            "per_episode_breakdown": self.per_episode_breakdown,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _run_rollout(
    env: Any,
    policy_fn: Callable[[Any], Any | None],
    *,
    max_steps: int,
    seed: int,
) -> EpisodeRollout:
    reset = getattr(env, "reset", None)
    if reset is None:
        raise TypeError("env must implement reset()")
    obs, _info = reset()
    traj: list[dict[str, np.ndarray]] = []
    actions: list[np.ndarray] = []
    total_reward = 0.0
    success = False
    for _ in range(int(max_steps)):
        traj.append(
            {
                "joint_positions": np.asarray(obs.joint_positions, dtype=np.float64),
                "ee_position": np.asarray(obs.ee_position, dtype=np.float64),
                "ee_orientation": np.asarray(obs.ee_orientation, dtype=np.float64),
            }
        )
        action = policy_fn(obs)
        if action is not None and getattr(action, "joint_positions", None) is not None:
            actions.append(np.asarray(action.joint_positions, dtype=np.float64))
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        if done:
            success = bool(getattr(info, "success", False))
            break
    return EpisodeRollout(success=success, reward=total_reward, trajectory=traj, actions=actions)


class TransferEvaluator:
    """Run matched rollouts on sim and real (or proxy) envs and compute gap metrics."""

    def __init__(
        self,
        *,
        sim_env_fn: EnvFactory,
        real_env_fn: EnvFactory,
        policy_fn: PolicyFactory | Callable[[Any], Callable[[Any], Any | None]],
        n_episodes: int = 5,
        max_steps_per_episode: int = 100,
    ) -> None:
        self._sim_env_fn = sim_env_fn
        self._real_env_fn = real_env_fn
        self._policy_fn = policy_fn
        self.n_episodes = int(n_episodes)
        self.max_steps = int(max_steps_per_episode)
        self._metrics: TransferMetrics | None = None

    def run(self) -> TransferMetrics:
        sim_rollouts: list[EpisodeRollout] = []
        real_rollouts: list[EpisodeRollout] = []
        breakdown: list[dict[str, Any]] = []

        for ep in range(self.n_episodes):
            seed = ep
            sim_env = self._sim_env_fn(seed)
            real_env = self._real_env_fn(seed)
            try:
                sim_policy = self._policy_fn(sim_env)
                real_policy = self._policy_fn(real_env)
                sim_ep = _run_rollout(sim_env, sim_policy, max_steps=self.max_steps, seed=seed)
                real_ep = _run_rollout(real_env, real_policy, max_steps=self.max_steps, seed=seed)
                sim_rollouts.append(sim_ep)
                real_rollouts.append(real_ep)
                traj_dist = compute_trajectory_distance(sim_ep.trajectory, real_ep.trajectory)
                breakdown.append(
                    {
                        "episode": ep,
                        "seed": seed,
                        "sim_success": sim_ep.success,
                        "real_success": real_ep.success,
                        "sim_reward": sim_ep.reward,
                        "real_reward": real_ep.reward,
                        "trajectory_distance": traj_dist,
                    }
                )
            finally:
                for env in (sim_env, real_env):
                    close = getattr(env, "close", None)
                    if callable(close):
                        close()

        sim_rate = float(np.mean([r.success for r in sim_rollouts])) if sim_rollouts else 0.0
        real_rate = float(np.mean([r.success for r in real_rollouts])) if real_rollouts else 0.0
        traj_keys = ("joint_pos_l2", "ee_pos_l2", "ee_quat_geodesic")
        traj_avg = {
            key: float(
                np.mean([b["trajectory_distance"][key] for b in breakdown])
            )
            if breakdown
            else 0.0
            for key in traj_keys
        }

        obs_kl: dict[str, float] = {}
        for field_name in ("joint_positions", "ee_position"):
            sim_vals = []
            real_vals = []
            for rollout in sim_rollouts:
                for frame in rollout.trajectory:
                    if field_name == "joint_positions":
                        sim_vals.extend(np.asarray(frame["joint_positions"]).tolist())
                    else:
                        sim_vals.extend(np.asarray(frame["ee_position"]).tolist())
            for rollout in real_rollouts:
                for frame in rollout.trajectory:
                    if field_name == "joint_positions":
                        real_vals.extend(np.asarray(frame["joint_positions"]).tolist())
                    else:
                        real_vals.extend(np.asarray(frame["ee_position"]).tolist())
            if sim_vals and real_vals:
                obs_kl[field_name] = _kl_gaussian(np.asarray(sim_vals), np.asarray(real_vals))

        sim_actions = [a for r in sim_rollouts for a in r.actions]
        real_actions = [a for r in real_rollouts for a in r.actions]
        action_l2 = 0.0
        if sim_actions and real_actions:
            n = min(len(sim_actions), len(real_actions))
            diffs = [
                float(np.linalg.norm(sim_actions[i] - real_actions[i]))
                for i in range(n)
            ]
            action_l2 = float(np.mean(diffs))

        self._metrics = TransferMetrics(
            sim_success_rate=sim_rate,
            real_success_rate=real_rate,
            success_gap=sim_rate - real_rate,
            trajectory_distance=traj_avg,
            obs_distribution_kl=obs_kl,
            action_distribution_l2=action_l2,
            per_episode_breakdown=breakdown,
        )
        return self._metrics

    def build_run_manifest(self, *, run_name: str = "transfer-eval") -> RunManifest:
        """Build a GOAL_10 RunManifest envelope for this transfer evaluation."""
        import time

        git_hash, git_dirty = None, False
        try:
            from robodeploy.observability.manifest import _git_state

            git_hash, git_dirty = _git_state()
        except Exception:
            pass
        return RunManifest(
            run_name=run_name,
            started_at=time.time(),
            finished_at=time.time(),
            env_config={"n_episodes": self.n_episodes, "max_steps": self.max_steps},
            backend="transfer_eval",
            task="sim2real_transfer",
            policy="transfer_evaluator",
            git_hash=git_hash,
            git_dirty=git_dirty,
        )

    def render_report(self, out_dir: str | Path, *, run_name: str = "transfer-eval") -> Path:
        if self._metrics is None:
            raise RuntimeError("Call run() before render_report()")
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        json_path = root / "transfer_metrics.json"
        payload = self._metrics.to_dict()
        manifest = self.build_run_manifest(run_name=run_name)
        manifest.env_config = {**manifest.env_config, "transfer_metrics": payload}
        manifest.save(root / "manifest.json")
        json_path.write_text(self._metrics.to_json(), encoding="utf-8")
        combined = root / "transfer_report.json"
        combined.write_text(
            json.dumps({"manifest": manifest.__dict__, "metrics": payload}, indent=2),
            encoding="utf-8",
        )
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(
                ["sim", "real"],
                [self._metrics.sim_success_rate, self._metrics.real_success_rate],
            )
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("success rate")
            ax.set_title("Sim vs real transfer")
            fig.savefig(root / "success_rates.png", bbox_inches="tight")
            plt.close(fig)
        except ImportError:
            pass
        return json_path
