# Build, start, and smoke-test the sentiment API via Docker
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$ApiPort = if ($env:API_HOST_PORT) { $env:API_HOST_PORT } else { "8081" }
$ApiBase = "http://localhost:$ApiPort"

Write-Host "=== Building images (mlflow + api) ===" -ForegroundColor Cyan
docker compose build mlflow api

Write-Host "`n=== Starting services ===" -ForegroundColor Cyan
docker compose up -d mlflow api

Write-Host "`n=== Waiting for API health (up to 3 min) ===" -ForegroundColor Cyan
$healthy = $false
for ($i = 0; $i -lt 36; $i++) {
    try {
        $resp = Invoke-RestMethod -Uri "$ApiBase/health" -TimeoutSec 5
        if ($resp.status -eq "healthy") {
            $healthy = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 5
    }
}

if (-not $healthy) {
    Write-Host "API did not become healthy. Container logs:" -ForegroundColor Red
    docker compose logs api --tail 80
    exit 1
}

Write-Host "`n=== Health ===" -ForegroundColor Green
Invoke-RestMethod -Uri "$ApiBase/health" | ConvertTo-Json

Write-Host "`n=== Predict ===" -ForegroundColor Green
$body = @{ text = "This movie was absolutely wonderful!" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$ApiBase/predict" -ContentType "application/json" -Body $body | ConvertTo-Json

Write-Host "`n=== Batch predict ===" -ForegroundColor Green
$batch = @{ texts = @("great film", "terrible acting", "it was okay") } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$ApiBase/predict/batch?variant=batched" -ContentType "application/json" -Body $batch | ConvertTo-Json -Depth 5

Write-Host "`n=== Swagger UI ===" -ForegroundColor Green
Write-Host "$ApiBase/docs"
Write-Host "`nAll Docker API smoke tests passed." -ForegroundColor Green
