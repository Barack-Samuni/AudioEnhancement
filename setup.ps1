# setup.ps1
$ErrorActionPreference = "Stop"

function Pause-And-Exit([int]$code) {
    Write-Host ""
    Read-Host "Press Enter to close"
    exit $code
}

try {
    Write-Host "== Project Setup (Python 3.10) ==" -ForegroundColor Cyan

    # --- Ensure py launcher + Python 3.10 exist ---
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python Launcher 'py' not found. Install Python for Windows (with the launcher)."
    }
    & py -3.10 --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.10 not found via 'py -3.10'. Install Python 3.10 and try again."
    }

    # --- Create .venv using Python 3.10 ---
    if (-not (Test-Path ".venv")) {
        Write-Host "`nCreating .venv..." -ForegroundColor Cyan
        & py -3.10 -m venv .venv
    } else {
        Write-Host "`n.venv already exists." -ForegroundColor Yellow
    }

    $venvPython = ".\.venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) { throw "Missing $venvPython. .venv creation failed." }

    # --- Verify venv is 3.10 ---
    $ver = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($ver -ne "3.10") { throw "Venv Python is $ver, expected 3.10." }

    # --- Upgrade pip + install dev tools in venv ---
    Write-Host "`nInstalling dev tooling in .venv..." -ForegroundColor Cyan
    & $venvPython -m pip install --upgrade pip

    if (-not (Test-Path "requirements-dev.txt")) {
@"
pre-commit
pip-tools
pip-chill
pytest
"@ | Out-File -Encoding utf8 "requirements-dev.txt"
    }

    & $venvPython -m pip install -r requirements-dev.txt

    # --- FIX: pre-commit refuses if core.hooksPath is set ---
    Write-Host "`nChecking git core.hooksPath..." -ForegroundColor Cyan
    $hooksPath = git config --local --get core.hooksPath
    if ($LASTEXITCODE -eq 0 -and $hooksPath) {
        Write-Host "Found core.hooksPath=$hooksPath. Unsetting it so pre-commit can manage hooks." -ForegroundColor Yellow
        git config --local --unset-all core.hooksPath
    } else {
        Write-Host "core.hooksPath not set (good)." -ForegroundColor Green
    }

    # --- Install pre-commit hooks ---
    Write-Host "`nInstalling pre-commit hooks..." -ForegroundColor Cyan
    & $venvPython -m pre_commit install
    & $venvPython -m pre_commit autoupdate


    Write-Host ""
    Write-Host "✅ Setup complete." -ForegroundColor Green
    Write-Host "To activate in an existing PowerShell window:" -ForegroundColor Green
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host ""
    Write-Host "`nRunning pre-commit on all files..." -ForegroundColor Cyan
    & $venvPython -m pre_commit run --all-files
    Pause-And-Exit 0
}
catch {
    Write-Host ""
    Write-Host "❌ SETUP FAILED:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Pause-And-Exit 1
}
