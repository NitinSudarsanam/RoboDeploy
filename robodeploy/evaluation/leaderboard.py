"""Leaderboard submission helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robodeploy.evaluation.registry import benchmarks_root
from robodeploy.evaluation.schema_validate import validate_leaderboard_submission


def _submissions_root(benchmarks_root_path: Path | str | None = None) -> Path:
    if benchmarks_root_path:
        root = Path(benchmarks_root_path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = benchmarks_root(None)
    return root / "leaderboard" / "submissions"


def report_to_submission(
    report_payload: dict[str, Any],
    *,
    author: str,
    policy_checkpoint: str | None = None,
    docker_image: str = "robodeploy/cpu:latest",
) -> dict[str, Any]:
    aggregate = report_payload.get("aggregate") or {}
    manifest = report_payload.get("manifest") or {}
    benchmark = str(report_payload.get("benchmark_name", ""))
    submission: dict[str, Any] = {
        "benchmark": benchmark,
        "benchmark_version": str(report_payload.get("benchmark_version", "1.0")),
        "policy_name": str(manifest.get("policy") or manifest.get("policy_name") or "unknown"),
        "policy_checkpoint": policy_checkpoint or str(manifest.get("policy_checkpoint") or ""),
        "author": str(author),
        "success_rate": float(aggregate.get("success_rate", 0.0)),
        "success_rate_ci95": list(aggregate.get("success_rate_ci95") or [0.0, 0.0]),
        "n_episodes": int(aggregate.get("n_episodes", 0)),
        "manifest": manifest,
        "reproduce": {
            "command": (
                f"robodeploy eval --benchmark {benchmark} "
                f"--policy {manifest.get('policy', 'scripted')} "
                f"--backend {manifest.get('backend', 'dummy')} "
                f"--episodes {int(aggregate.get('n_episodes', 100))}"
            ),
            "docker_image": docker_image,
        },
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    robo_score = aggregate.get("robo_score")
    if robo_score is not None:
        submission["robo_score"] = float(robo_score)
    return submission


def submit_score(
    report_path: Path | str,
    *,
    benchmark: str,
    author: str,
    benchmarks_root_path: Path | str | None = None,
    policy_checkpoint: str | None = None,
) -> Path:
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if benchmark:
        payload["benchmark_name"] = benchmark
    submission = report_to_submission(payload, author=author, policy_checkpoint=policy_checkpoint)
    errors = validate_leaderboard_submission(submission)
    if errors:
        raise ValueError("Leaderboard submission failed schema validation:\n" + "\n".join(errors))

    bench_key = submission["benchmark"].replace("/", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    safe_author = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in author)
    out_dir = _submissions_root(benchmarks_root_path) / bench_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_author}_{stamp}.json"
    out_path.write_text(json.dumps(submission, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def list_submissions(
    suite: str,
    *,
    benchmarks_root_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    root = _submissions_root(benchmarks_root_path)
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        bench = str(payload.get("benchmark", ""))
        if suite and not (bench == suite or bench.startswith(f"{suite}/")):
            continue
        payload["_path"] = str(path)
        rows.append(payload)
    rows.sort(key=lambda r: float(r.get("success_rate", 0.0)), reverse=True)
    return rows


def show_leaderboard(
    suite: str,
    *,
    benchmarks_root_path: Path | str | None = None,
    as_json: bool = False,
) -> str | list[dict[str, Any]]:
    rows = list_submissions(suite, benchmarks_root_path=benchmarks_root_path)
    if as_json:
        return rows
    lines = [f"Leaderboard: {suite}", "-" * 60]
    for idx, row in enumerate(rows, start=1):
        sr = float(row.get("success_rate", 0.0))
        author = row.get("author", "?")
        policy = row.get("policy_name", "?")
        bench = row.get("benchmark", "?")
        lines.append(f"{idx:2d}. {author:20s} {policy:20s} {bench:30s} {sr*100:5.1f}%")
    if not rows:
        lines.append("(no submissions)")
    return "\n".join(lines)
