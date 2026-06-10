# Release process

## Versioning

RoboDeploy follows [Semantic Versioning](https://semver.org/). Public API changes are recorded in [`CHANGELOG.md`](../CHANGELOG.md).

Pre-release tags use PEP 440 suffixes on the version in `pyproject.toml` (for example `0.2.0a1`, `0.2.0rc1`).

## Local dry-run (no PyPI upload)

Validate the sdist and wheel before tagging:

```bash
python -m pip install --upgrade build twine
python -m build
twine check dist/*
```

Optional smoke install from the wheel:

```bash
pip install dist/robodeploy-*.whl
robodeploy --help
robodeploy assets verify
```

## PyPI publish (automated)

Workflow: [`.github/workflows/publish.yml`](../.github/workflows/publish.yml)

| Trigger | Action |
|---------|--------|
| Push tag `v*` (e.g. `v0.2.0`) | Build sdist/wheel, `twine check`, upload to PyPI via trusted publishing |
| `workflow_dispatch` | Same build + `twine check` only (dry-run; no upload) |

### First release checklist

1. Bump `version` in `pyproject.toml` and `robodeploy/__init__.py`.
2. Update `CHANGELOG.md` with the release date and summary.
3. Regenerate `robodeploy/_assets/manifest.json` SHA256 entries if bundled assets changed.
4. Run `robodeploy assets verify` locally.
5. Configure [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) for this GitHub repo and workflow.
6. Tag and push: `git tag v0.2.0 && git push origin v0.2.0`

Until the first tag is pushed and trusted publishing is configured, `pip install robodeploy` from PyPI will not work; install from source with `pip install -e ".[sim]"` instead.

## Conda

Recipe: [`conda-recipe/meta.yaml`](../conda-recipe/meta.yaml). CI validates the recipe via `tests/test_conda_recipe.py`. Publishing to conda-forge is manual after the first PyPI release.

## Docker

CPU image: `docker/Dockerfile.cpu`. CI job `docker-smoke` builds the image and runs `robodeploy --help` plus `python -m examples.cli list-presets`.
