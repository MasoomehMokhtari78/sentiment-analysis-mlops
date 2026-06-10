# Run as Administrator AFTER uninstall + reboot
# Installs WSL + Docker Desktop with all data on E:

#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

$DockerRoot   = "E:\Docker"
$DockerData   = "E:\Docker\wsl-data"
$WslRoot      = "E:\WSL"
$UbuntuName   = "Ubuntu-24.04"
$UbuntuPath   = "E:\WSL\Ubuntu-24.04"
$UbuntuBackup = "E:\WSL\backup\ubuntu-backup.tar"
$DockerInstaller = "E:\Docker\DockerDesktopInstaller.exe"

Write-Host "=== Step 1: Create E: directories ===" -ForegroundColor Cyan
@($DockerRoot, $DockerData, $WslRoot, "E:\WSL\backup") | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

Write-Host "`n=== Step 2: Enable WSL (Windows feature) ===" -ForegroundColor Cyan
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
Write-Host "If prompted to reboot, reboot and re-run this script." -ForegroundColor Yellow

Write-Host "`n=== Step 3: Set WSL 2 as default ===" -ForegroundColor Cyan
wsl --set-default-version 2

Write-Host "`n=== Step 4: Restore or install Ubuntu on E: ===" -ForegroundColor Cyan
$ubuntuExists = wsl --list --quiet 2>$null | Where-Object { $_.Trim() -eq $UbuntuName }
if (-not $ubuntuExists) {
    if (Test-Path $UbuntuBackup) {
        Write-Host "Importing Ubuntu from backup to $UbuntuPath ..."
        wsl --import $UbuntuName $UbuntuPath $UbuntuBackup --version 2
        Write-Host "Ubuntu restored on E:." -ForegroundColor Green
    } else {
        Write-Host "No backup found. Install Ubuntu manually after Docker setup:" -ForegroundColor Yellow
        Write-Host "  wsl --install -d Ubuntu-24.04"
        Write-Host "  OR download from Store, then move with: wsl --export / wsl --import to E:\WSL\"
    }
} else {
    Write-Host "Ubuntu already registered."
}

Write-Host "`n=== Step 5: Install Docker Desktop ===" -ForegroundColor Cyan
if (-not (Test-Path $DockerInstaller)) {
    Write-Host "Downloading Docker Desktop installer to E: ..."
    $url = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
    Invoke-WebRequest -Uri $url -OutFile $DockerInstaller -UseBasicParsing
}

Write-Host "Installing Docker Desktop (UI may appear)..."
Write-Host "IMPORTANT: After install, open Docker Desktop -> Settings -> Resources -> Advanced"
Write-Host "Set 'Disk image location' to: $DockerData"
Write-Host ""

# Install Docker; allow user to complete setup wizard
Start-Process -FilePath $DockerInstaller -ArgumentList "install", "--accept-license" -Wait

Write-Host "`n=== Step 6: Configure Docker disk location ===" -ForegroundColor Cyan
Write-Host @"

After Docker Desktop starts for the FIRST time:
  1. Settings (gear) -> Resources -> Advanced (or Disk)
  2. Disk image location -> $DockerData
  3. Apply & Restart

If Docker already created data on C:, use uninstall script first.

"@ -ForegroundColor Yellow

Write-Host "`n=== Step 7: Verify ===" -ForegroundColor Cyan
Write-Host "After Docker is running:"
Write-Host "  docker --version"
Write-Host "  cd `"$((Get-Location).Path)`""
Write-Host "  .\scripts\docker_test_api.ps1"
Write-Host "`nSetup script finished." -ForegroundColor Green
