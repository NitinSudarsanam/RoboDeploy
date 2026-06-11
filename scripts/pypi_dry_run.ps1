# Local PyPI publish prep: build sdist/wheel + twine check (no upload).
# Safe to run without a release tag. Mirrors publish.yml dry-run job.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

python -m pip install --upgrade pip build twine -q
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
python -m build
python -m twine check dist/*
Write-Host "PyPI dry-run OK - dist/ ready for tag push (v*) to trigger publish.yml"
