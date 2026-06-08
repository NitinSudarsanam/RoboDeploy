"""JSON Schema validation for benchmark specs and leaderboard submissions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _require_jsonschema():
    try:
        import jsonschema
    except ImportError as exc:
        raise ImportError(
            "Schema validation requires jsonschema. pip install 'robodeploy[eval]'"
        ) from exc
    return jsonschema


def load_schema(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_json(instance: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    jsonschema = _require_jsonschema()
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    return [f"{'.'.join(str(p) for p in err.path) or '<root>'}: {err.message}" for err in errors]


def validate_leaderboard_submission(payload: dict[str, Any], *, schema_path: Path | str | None = None) -> list[str]:
    root = Path(__file__).resolve().parents[2] / "benchmarks" / "leaderboard"
    path = Path(schema_path) if schema_path else root / "schema.json"
    schema = load_schema(path)
    return validate_json(payload, schema)


def validate_benchmark_spec(payload: dict[str, Any], *, schema_path: Path | str | None = None) -> list[str]:
    root = Path(__file__).resolve().parents[2] / "benchmarks" / "manipulation_v1"
    path = Path(schema_path) if schema_path else root / "spec.schema.json"
    schema = load_schema(path)
    return validate_json(payload, schema)
