# Wave 2.02 — PyPI Tag Publish + Conda-Forge

**Wave**: 2 | **Effort**: ~8h (mostly ops) | **Maps to**: GOAL 08 (distribution), GOAL 07 (first-run docs)

## Honest current state

| Item | Status | Evidence |
|------|--------|----------|
| `publish.yml` workflow | **Ready** — builds sdist/wheel, `twine check`, trusted publishing on `v*` tag | `.github/workflows/publish.yml` |
| `workflow_dispatch` dry-run | **Works** — no upload | `docs/RELEASE.md` |
| First PyPI release | **Not done** — `pip install robodeploy` fails | GOAL 08 `[ ]` |
| Conda recipe | **CI smoke only** — `conda-recipe/meta.yaml` + `tests/test_conda_recipe.py` | `test.yml` `conda-recipe` job |
| conda-forge feedstock | **Not submitted** | GOAL 08 `[ ]` |
| Docker smoke | **Pass** | `docker-smoke` job |
| New-user 60s install criterion | **Unchecked** — blocked on PyPI | GOAL 07 `[ ]` |

Version in tree: `0.2.0` (pre-release). No `v0.2.0` tag pushed. Trusted publishing must be configured on PyPI before first tag.

## Problem

Distribution infrastructure is built but **unreleased**. Goal acceptance items for PyPI and conda install remain blocked. Docs and tutorials reference `pip install robodeploy` which does not work yet.

## Scope

**In scope**

- Execute first `v0.2.0` (or `v0.2.0rc1`) tag publish to PyPI.
- Verify install from PyPI on clean venv (Linux + Windows smoke).
- Prepare and submit conda-forge feedstock PR checklist.
- Close GOAL 08 distribution checkboxes with evidence.

**Out of scope**

- Homebrew, apt, or Snap packaging.
- Private PyPI mirrors.
- Automating conda-forge version bumps (manual until feedstock exists).

## Acceptance criteria

- [ ] PyPI hosts `robodeploy==0.2.0` (or approved rc) with sdist + wheel passing `twine check`.
- [ ] Clean venv: `pip install robodeploy` → `robodeploy --help` + `robodeploy assets verify` in <60s on Linux CI.
- [ ] Windows smoke: `pip install robodeploy` → `robodeploy doctor` exits 0 (no MuJoCo required for base).
- [ ] `CHANGELOG.md` dated entry matches tag; `pyproject.toml` version aligned.
- [ ] conda-forge feedstock PR opened (or merged) with recipe derived from `conda-recipe/meta.yaml`.
- [ ] `conda install -c conda-forge robodeploy` works on Linux after feedstock merge (may lag PyPI by days).
- [ ] GOAL 07 criterion: new user install path documented and verified post-release.

## Tasks

### Phase 1 — Pre-release validation (~2h)

1. Run local dry-run per `docs/RELEASE.md`: `python -m build`, `twine check dist/*`.
2. Confirm `robodeploy/_assets/manifest.json` SHA256 entries current; `robodeploy assets verify`.
3. Run `publish.yml` via `workflow_dispatch` on release branch; confirm green build artifacts.
4. Audit `pyproject.toml` optional extras (`sim`, `dev`, `eval`, `kinematics`) — document which are PyPI-installable vs source-only.

### Phase 2 — PyPI trusted publishing (~2h)

5. Create PyPI project `robodeploy` (if absent).
6. Configure [trusted publisher](https://docs.pypi.org/trusted-publishers/) for `anthropic-ai/robodeploy` → `publish.yml`.
7. Bump version if needed; commit `CHANGELOG.md` + version files on `main` or release branch.
8. Tag: `git tag v0.2.0 && git push origin v0.2.0`.
9. Verify workflow upload; download wheel from PyPI and smoke-install in fresh venv.

### Phase 3 — Post-release verification (~2h)

10. Add CI job `pypi-smoke` (optional): `pip install robodeploy==${{ github.event.release.tag }}` on tag webhook or manual.
11. Update README install section: PyPI primary, `pip install -e ".[sim]"` for dev.
12. Mark GOAL 08 PyPI checkbox with PyPI URL + CI job name.

### Phase 4 — Conda-forge submission (~2h)

13. Fork `conda-forge/staged-recipes`; add `recipes/robodeploy/meta.yaml` from `conda-recipe/meta.yaml`.
    - Fix `about.home` URL if repo path differs.
    - Pin `mujoco` run dep or make optional per conda-forge policy.
    - Use PyPI sdist as `source.url` after first release (not git HEAD).
14. Open staged-recipes PR; respond to linter bot feedback.
15. After merge, document `conda install -c conda-forge robodeploy` in `docs/RELEASE.md`.
16. Mark GOAL 08 conda checkbox with feedstock link.

## Self-critique

| Risk | Mitigation |
|------|------------|
| **Publishing broken wheel** | Mandatory `workflow_dispatch` dry-run + `twine check` before tag |
| **Trusted publishing misconfig** | Test with `v0.2.0rc1` first if nervous |
| **Conda mujoco pin conflicts** | Follow conda-forge mujoco feedstock version; consider `robodeploy-base` without mujoco |
| **Version drift** | Single source: tag → PyPI → conda recipe bump bot |
| **Over-promising extras** | Document `pip install robodeploy[sim]` separately; base package stays lean |

**Honest limitation**: conda-forge review can take 1–2 weeks; PyPI can ship first. Do not block `v0.2.0` on conda merge.

## Test gates

| Gate | Command / action | Required |
|------|------------------|----------|
| Local build | `python -m build && twine check dist/*` | Pass |
| CI dry-run | `workflow_dispatch` on `publish.yml` | Pass |
| PyPI install Linux | `pip install robodeploy && robodeploy assets verify` | Pass |
| PyPI install Windows | `pip install robodeploy && robodeploy doctor` | Pass |
| Conda recipe CI | `pytest tests/test_conda_recipe.py -q` | Pass (pre-merge) |
| Conda install | `conda install -c conda-forge robodeploy` + `python -c "import robodeploy"` | Pass (post-feedstock) |
| Docs | `mkdocs build` with updated install instructions | Pass |
