# Full Docker pipeline: MLflow + data prep + ingest + multiple training runs
param(
    [int]$Runs = 3,
    [string]$ModelType = "baseline",
    [int]$DataSampleSize = 5000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "=== Building Docker images ===" -ForegroundColor Cyan
docker compose build

Write-Host "`n=== Starting MLflow server ===" -ForegroundColor Cyan
docker compose up mlflow -d

Write-Host "`n=== Downloading sample data ===" -ForegroundColor Cyan
docker compose run --rm app scripts/prepare_data.py --sample-size $DataSampleSize

Write-Host "`n=== Running data ingest pipeline ===" -ForegroundColor Cyan
docker compose run --rm app -m src.data.ingest --input-path data/raw/imdb_train.csv --sample-size $DataSampleSize

Write-Host "`n=== Running $Runs training experiments ($ModelType) ===" -ForegroundColor Cyan
docker compose run --rm app run_experiments.py --runs $Runs --model-type $ModelType

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host "MLflow UI: http://localhost:5000"
Write-Host "Models saved to: ./models/"
