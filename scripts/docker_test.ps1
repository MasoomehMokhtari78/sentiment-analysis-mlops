# Run the full pytest suite inside Docker (CI-equivalent).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Image = if ($env:TEST_IMAGE) { $env:TEST_IMAGE } else { "sentiment-test:latest" }

Write-Host "=== Building test image ===" -ForegroundColor Cyan
docker build -f Dockerfile.test -t $Image .

New-Item -ItemType Directory -Force -Path "$Root\htmlcov" | Out-Null

Write-Host "`n=== Lint (Ruff) ===" -ForegroundColor Cyan
docker run --rm $Image `
  ruff check src tests scripts train.py update_model.py

Write-Host "`n=== Pytest with coverage ===" -ForegroundColor Cyan
docker run --rm `
  -v "${Root}/htmlcov:/app/htmlcov" `
  $Image `
  pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

Write-Host "`n=== Data pipeline smoke ===" -ForegroundColor Cyan
docker run --rm $Image python -m src.data.ingest --help

Write-Host "`nAll Docker tests passed." -ForegroundColor Green
