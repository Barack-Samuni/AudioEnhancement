$ErrorActionPreference = "Stop"

function Pause-And-Exit([int]$code) {
    Write-Host ""
    Read-Host "Press Enter to close"
    exit $code
}

try {
    # Detect repository root
    $RepoRoot = Split-Path -Parent $PSScriptRoot
    Set-Location $RepoRoot

    Write-Host "== Project Setup (Python 3.10) ==" -ForegroundColor Cyan
    Write-Host "Repo root: $RepoRoot" -ForegroundColor DarkGray

    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python Launcher 'py' not found. Install Python for Windows."
    }

    & py -3.10 --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.10 not found via 'py -3.10'."
    }

    $VenvPath = Join-Path $RepoRoot ".venv"
    $VenvPython = Join-Path $VenvPath "Scripts\python.exe"
    $RequirementsDev = Join-Path $RepoRoot "requirements-dev.txt"
    $RequirementsRuntime = Join-Path $RepoRoot "requirements.txt"

    # Create venv
    if (-not (Test-Path $VenvPath)) {
        Write-Host "`nCreating .venv..." -ForegroundColor Cyan
        & py -3.10 -m venv $VenvPath
    } else {
        Write-Host "`n.venv already exists." -ForegroundColor Yellow
    }

    if (-not (Test-Path $VenvPython)) {
        throw "Missing $VenvPython. .venv creation failed."
    }

    # Ensure correct python version
    $ver = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($ver -ne "3.10") {
        throw "Venv Python is $ver, expected 3.10."
    }

    Write-Host "`nUpgrading pip..." -ForegroundColor Cyan
    & $VenvPython -m pip install --upgrade pip

    # Ensure dev requirements file exists
    if (-not (Test-Path $RequirementsDev)) {
@"
pre-commit
pip-tools
pip-chill
pytest
ruff
black
"@ | Out-File -Encoding utf8 $RequirementsDev
    }

    Write-Host "`nInstalling dev dependencies..." -ForegroundColor Cyan
    & $VenvPython -m pip install -r $RequirementsDev

    # Install runtime dependencies if present
    if (Test-Path $RequirementsRuntime) {
        Write-Host "`nInstalling project dependencies..." -ForegroundColor Cyan
        & $VenvPython -m pip install -r $RequirementsRuntime
    } else {
        Write-Host "`nrequirements.txt not found yet." -ForegroundColor Yellow
    }

    # Ensure git hooks path is clean
    Write-Host "`nChecking git hooks configuration..." -ForegroundColor Cyan
    $hooksPath = git config --local --get core.hooksPath
    if ($LASTEXITCODE -eq 0 -and $hooksPath) {
        Write-Host "Removing custom hooksPath ($hooksPath)" -ForegroundColor Yellow
        git config --local --unset-all core.hooksPath
    }

    # Install pre-commit
    Write-Host "`nInstalling pre-commit hooks..." -ForegroundColor Cyan
    & $VenvPython -m pre_commit install

    # Run hooks once to clean repo
    Write-Host "`nRunning pre-commit on all files..." -ForegroundColor Cyan
    & $VenvPython -m pre_commit run --all-files

    Write-Host ""
    Write-Host "✅ Setup complete." -ForegroundColor Green
    Write-Host "Activate environment with:"
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host ""

    Pause-And-Exit 0
}
catch {
    Write-Host ""
    Write-Host "❌ SETUP FAILED:" -ForegroundColor Red
    Write-Host $_.Exception.Message
    Pause-And-Exit 1
}