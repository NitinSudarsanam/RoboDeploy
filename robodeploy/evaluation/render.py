"""HTML rendering for EvalReport and comparison views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robodeploy.evaluation.report import EvalReport


def _reward_histogram_svg(rewards: list[float]) -> str:
    if not rewards:
        return "<p>No reward data.</p>"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return f"<pre>rewards: {json.dumps(rewards[:20])}</pre>"

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(rewards, bins=min(20, max(5, len(rewards))))
    ax.set_title("Reward distribution")
    ax.set_xlabel("Total reward")
    ax.set_ylabel("Count")
    fig.tight_layout()
    import io
    import base64

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img alt="reward histogram" src="data:image/png;base64,{data}" />'


def _failure_pie_svg(counts: dict[str, int]) -> str:
    labels = [k for k, v in counts.items() if v > 0]
    values = [counts[k] for k in labels]
    if not values:
        return "<p>No failures recorded.</p>"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return f"<pre>failures: {json.dumps(counts)}</pre>"

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(values, labels=labels, autopct="%1.0f%%")
    ax.set_title("Failure modes")
    fig.tight_layout()
    import io
    import base64

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img alt="failure pie" src="data:image/png;base64,{data}" />'


def _baseline_table_rows(
    report: EvalReport,
    baseline: EvalReport | dict[str, Any] | None,
) -> str:
    if baseline is None:
        return ""
    if isinstance(baseline, EvalReport):
        base_agg = baseline.aggregate
        base_name = baseline.benchmark_name
    else:
        base_agg = baseline.get("aggregate", {})
        base_name = str(baseline.get("benchmark_name", "baseline"))
    rows = [
        "<tr><th>Metric</th><th>Run</th><th>Baseline</th><th>Delta</th></tr>",
        _delta_row("Success rate", report.aggregate.success_rate, _num(base_agg, "success_rate")),
        _delta_row("Mean reward", report.aggregate.mean_reward, _num(base_agg, "mean_reward")),
        _delta_row("RoboScore", report.aggregate.robo_score, _num(base_agg, "robo_score")),
    ]
    return f"<h3>vs baseline ({base_name})</h3><table>{''.join(rows)}</table>"


def _num(agg: Any, key: str) -> float | None:
    if hasattr(agg, key):
        val = getattr(agg, key)
        return float(val) if val is not None else None
    if isinstance(agg, dict):
        val = agg.get(key)
        return float(val) if val is not None else None
    return None


def _delta_row(label: str, current: float | None, baseline: float | None) -> str:
    cur = 0.0 if current is None else float(current)
    base = 0.0 if baseline is None else float(baseline)
    delta = cur - base
    sign = "+" if delta >= 0 else ""
    return (
        f"<tr><td>{label}</td><td>{cur:.4f}</td><td>{base:.4f}</td>"
        f"<td>{sign}{delta:.4f}</td></tr>"
    )


def _video_section(video_paths: list[str]) -> str:
    if not video_paths:
        return ""
    from pathlib import Path

    from robodeploy.evaluation.video import EpisodeVideoRecorder

    items: list[str] = []
    for raw in video_paths:
        src = EpisodeVideoRecorder.embed_path(Path(raw)) or raw
        items.append(f'<li><video controls width="320" src="{src}"></video></li>')
    return f"<h3>Episode videos</h3><ul>{''.join(items)}</ul>"


def render_report(
    report: EvalReport,
    baseline: EvalReport | dict[str, Any] | None = None,
    out: Path | str | None = None,
    *,
    failure_counts: dict[str, int] | None = None,
    video_paths: list[str] | None = None,
) -> str:
    try:
        from jinja2 import Template
    except ImportError as exc:
        raise ImportError("HTML reports require jinja2. pip install 'robodeploy[eval]'") from exc

    agg = report.aggregate
    rewards = [ep.reward_total for ep in report.episodes]
    failures = failure_counts or {}
    template = Template(_REPORT_TEMPLATE)
    html = template.render(
        benchmark_name=report.benchmark_name,
        benchmark_version=report.benchmark_version,
        success_rate=agg.success_rate,
        success_ci=list(agg.success_rate_ci95),
        mean_reward=agg.mean_reward,
        robo_score=agg.robo_score,
        n_episodes=agg.n_episodes,
        reward_plot=_reward_histogram_svg(rewards),
        failure_plot=_failure_pie_svg(failures),
        baseline_table=_baseline_table_rows(report, baseline),
        task_results=report.task_results,
        videos=_video_section(list(video_paths or [])),
        manifest=json.dumps(report.manifest.to_dict() if hasattr(report.manifest, "to_dict") else {}, indent=2),
    )
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    return html


def render_comparison(
    report_a: dict[str, Any] | EvalReport,
    report_b: dict[str, Any] | EvalReport,
    out: Path | str,
) -> Path:
    a = report_a.to_json() if isinstance(report_a, EvalReport) else dict(report_a)
    b = report_b.to_json() if isinstance(report_b, EvalReport) else dict(report_b)
    try:
        from jinja2 import Template
    except ImportError as exc:
        raise ImportError("HTML comparison requires jinja2.") from exc

    rows = []
    metrics = [
        ("benchmark_name", "Benchmark"),
        ("aggregate.success_rate", "Success rate"),
        ("aggregate.mean_reward", "Mean reward"),
        ("aggregate.robo_score", "RoboScore"),
        ("aggregate.n_episodes", "Episodes"),
    ]

    def _get(payload: dict, dotted: str):
        cur: Any = payload
        for part in dotted.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    for key, label in metrics:
        va = _get(a, key)
        vb = _get(b, key)
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = f"{float(va) - float(vb):+.4f}"
        rows.append((label, va, vb, delta))

    template = Template(_COMPARE_TEMPLATE)
    html = template.render(
        name_a=a.get("benchmark_name", "A"),
        name_b=b.get("benchmark_name", "B"),
        rows=rows,
    )
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{{ benchmark_name }} eval report</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; color: #111; }
    table { border-collapse: collapse; margin: 1rem 0; }
    th, td { border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
    pre { background: #f6f6f6; padding: 1rem; overflow: auto; }
  </style>
</head>
<body>
  <h1>{{ benchmark_name }} <small>v{{ benchmark_version }}</small></h1>
  <p><strong>Success rate:</strong> {{ '%.1f'|format(success_rate * 100) }}%
     (95% CI {{ '%.1f'|format(success_ci[0] * 100) }}–{{ '%.1f'|format(success_ci[1] * 100) }}%)</p>
  <p><strong>Mean reward:</strong> {{ '%.3f'|format(mean_reward) }}
     {% if robo_score is not none %}| <strong>RoboScore:</strong> {{ '%.3f'|format(robo_score) }}{% endif %}
     | <strong>Episodes:</strong> {{ n_episodes }}</p>
  <div class="grid">
    <div>{{ reward_plot|safe }}</div>
    <div>{{ failure_plot|safe }}</div>
  </div>
  {{ baseline_table|safe }}
  {% if task_results %}
  <h3>Per-task results</h3>
  <table>
    <tr><th>Task</th><th>Success</th><th>Reward</th></tr>
    {% for row in task_results %}
    <tr>
      <td>{{ row.benchmark }}</td>
      <td>{{ '%.3f'|format(row.success_rate) }}</td>
      <td>{{ '%.3f'|format(row.mean_reward) }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}
  {{ videos|safe }}
  <h3>Manifest</h3>
  <pre>{{ manifest }}</pre>
</body>
</html>
"""

_COMPARE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Eval comparison</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 0.5rem 0.8rem; }
  </style>
</head>
<body>
  <h1>Eval comparison</h1>
  <p><strong>A:</strong> {{ name_a }} | <strong>B:</strong> {{ name_b }}</p>
  <table>
    <tr><th>Metric</th><th>A</th><th>B</th><th>A − B</th></tr>
    {% for label, va, vb, delta in rows %}
    <tr><td>{{ label }}</td><td>{{ va }}</td><td>{{ vb }}</td><td>{{ delta }}</td></tr>
    {% endfor %}
  </table>
</body>
</html>
"""
