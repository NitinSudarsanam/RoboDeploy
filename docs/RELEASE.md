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

Or run the automated smoke test (build + `twine check` + wheel install in an isolated venv):

```bash
python -m pytest tests/test_package_build.py -q
```

Windows shortcut (build + `twine check` only, no upload):

```powershell
powershell -File scripts/pypi_dry_run.ps1
```

Optional manual smoke install from the wheel:

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

CI also runs `test.yml` → `package-build` on every PR (pytest wrapper around the steps above; no PyPI credentials required).

### Required CI before tagging

`publish.yml` runs on tag push and does **not** wait for the `tests` workflow on the same commit. Before pushing `v*`:

1. Confirm the [`tests` workflow](https://github.com/RahulSajnani/RoboDeploy/actions/workflows/test.yml) passed on the release commit (PR merge to `main` or the exact SHA you will tag).
2. Run `python -m pytest tests/test_package_build.py -q` locally if you changed packaging.

### First release checklist

1. Bump `version` in `pyproject.toml` and `robodeploy/__init__.py` (must match; `robodeploy --version` / `import robodeploy` assert same).
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

## Wave 2 integration (0.2.0)

Wave 2 (training, sensors, Gazebo/Isaac parity, benchmarks honesty, package-build CI) is **merged to `main`**. Version in `pyproject.toml` is `0.2.0`.

**Still open for 0.2.0 release:**

1. Configure [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) for this repo.
2. Confirm green [`tests` workflow](https://github.com/RahulSajnani/RoboDeploy/actions/workflows/test.yml) on the release commit.
3. Tag: `git tag v0.2.0 && git push origin v0.2.0` → triggers `publish.yml`.

Until the tag is pushed, install from source: `pip install -e ".[sim]"`.

### Pre-tag demo smoke (optional, local)

| Demo | Command | Pass signal |
|------|---------|-------------|
| MuJoCo pick | `python -m examples.cli run-episode --preset kuka_ft_imu_pick_mujoco --seed 0 --steps 2000 --json` | `"success": true` ~step 306 |
| RViz pick (Docker) | `docker compose -f docker/docker-compose.yml --profile ros2 run --rm demo-rviz-pick` | `success=true` ~step 950 |
| Gazebo pick (Docker) | `docker compose -f docker/docker-compose.yml --profile ros2 run --rm demo-gazebo-pick` | `success=true` ~step 950 (default place snap on) |

See [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md) for WSL2 / honest JTC (`ROBODEPLOY_GAZEBO_PLACE_SNAP=0`) caveats.

Integration audit: [plans/INTEGRATION_STATUS.md](../plans/INTEGRATION_STATUS.md).
