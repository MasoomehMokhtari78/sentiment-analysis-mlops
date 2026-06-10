# Run as Administrator: right-click PowerShell -> "Run as administrator"
# Uninstalls Docker Desktop and removes Docker WSL distros from C:
# Optionally exports Ubuntu to E: before removal (set $ExportUbuntu = $true)

#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

$ExportUbuntu = $true
$BackupDir = "E:\WSL\backup"
$DockerLocal = "$env:LOCALAPPDATA\Docker"
$DockerRoaming = "$env:APPDATA\Docker"

Write-Host "=== Step 1: Stop Docker and WSL ===" -ForegroundColor Cyan
Get-Process "Docker Desktop" -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process "com.docker.backend" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3
wsl --shutdown
Start-Sleep -Seconds 2

Write-Host "`n=== Step 2: Export Ubuntu to E: (optional backup) ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
if ($ExportUbuntu) {
    $ubuntu = wsl --list --quiet | Where-Object { $_ -match "Ubuntu" } | Select-Object -First 1
    if ($ubuntu) {
        $ubuntu = $ubuntu.Trim()
        $tarPath = Join-Path $BackupDir "ubuntu-backup.tar"
        Write-Host "Exporting '$ubuntu' to $tarPath (may take several minutes)..."
        wsl --export $ubuntu $tarPath
        Write-Host "Ubuntu exported." -ForegroundColor Green
    } else {
        Write-Host "No Ubuntu distro found — skipping export."
    }
}

Write-Host "`n=== Step 3: Remove Docker WSL distros ===" -ForegroundColor Cyan
foreach ($distro in @("docker-desktop-data", "docker-desktop")) {
    $exists = wsl --list --quiet 2>$null | Where-Object { $_.Trim() -eq $distro }
    if ($exists) {
        Write-Host "Unregistering $distro..."
        wsl --unregister $distro
    }
}

Write-Host "`n=== Step 4: Uninstall Docker Desktop ===" -ForegroundColor Cyan
$uninstalled = $false
if (Get-Command winget -ErrorAction SilentlyContinue) {
  try {
    winget uninstall "Docker.DockerDesktop" --accept-source-agreements 2>$null
    $uninstalled = $true
  } catch { }
}
if (-not $uninstalled) {
    $dockerUninstaller = "${env:ProgramFiles}\Docker\Docker\Docker Desktop Installer.exe"
    if (Test-Path $dockerUninstaller) {
        Write-Host "Run the Docker uninstaller manually from Apps & Features if needed."
    }
    Write-Host "Open Settings -> Apps -> Docker Desktop -> Uninstall" -ForegroundColor Yellow
}

Write-Host "`n=== Step 5: Remove leftover Docker data on C: ===" -ForegroundColor Cyan
foreach ($path in @($DockerLocal, $DockerRoaming)) {
    if (Test-Path $path) {
        Write-Host "Removing $path ..."
        Remove-Item -Recurse -Force $path
    }
}

Write-Host "`n=== Step 6: Unregister Ubuntu from C: (backup is on E:) ===" -ForegroundColor Cyan
$ubuntuName = wsl --list --quiet 2>$null | Where-Object { $_ -match "Ubuntu" } | Select-Object -First 1
if ($ubuntuName) {
    $ubuntuName = $ubuntuName.Trim()
    Write-Host "Unregistering $ubuntuName from C: ..."
    wsl --unregister $ubuntuName
}

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host "Reboot recommended, then run: .\scripts\install_docker_wsl_e_drive.ps1"
Write-Host "Ubuntu backup (if exported): $BackupDir\ubuntu-backup.tar"
