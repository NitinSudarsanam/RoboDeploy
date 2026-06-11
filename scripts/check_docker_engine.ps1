# Quick Docker Desktop engine probe (Windows). Exit 0 = engine up, 1 = down.
$ErrorActionPreference = "Continue"
$info = docker info 2>&1 | Out-String
if ($LASTEXITCODE -ne 0 -or $info -match "cannot find the file specified|error") {
    Write-Host "Docker engine DOWN - start Docker Desktop, then retry:"
    Write-Host "  docker compose -f docker/docker-compose.yml --profile ros2 run --rm demo-gazebo-pick"
    exit 1
}
Write-Host "Docker engine UP"
exit 0
