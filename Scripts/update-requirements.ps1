$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$venvPython = ".\.venv\Scripts\python.exe"
$venvPipChill = ".\.venv\Scripts\pip-chill.exe"
$requirementsIn = Join-Path $RepoRoot "requirements.in"
$tempFile = Join-Path $env:TEMP "requirements_current.in"

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found at .\.venv"
}

& $venvPipChill | Out-File -Encoding utf8 $tempFile

$shouldUpdate = $true
if (Test-Path $requirementsIn) {
    $existing = Get-Content $requirementsIn | Where-Object { $_.Trim() -ne "" } | Sort-Object
    $current = Get-Content $tempFile | Where-Object { $_.Trim() -ne "" } | Sort-Object

    if (($existing -join "`n") -eq ($current -join "`n")) {
        $shouldUpdate = $false
    }
}

if (-not $shouldUpdate) {
    Write-Host "No dependency changes detected. Skipping requirements update."
    Remove-Item $tempFile -ErrorAction SilentlyContinue
    exit 0
}

Write-Host "Dependency changes detected. Updating requirements..."

Move-Item -Force $tempFile $requirementsIn
& $venvPython -m piptools compile --extra-index-url=https://download.pytorch.org/whl/cu130 --output-file=requirements.txt --strip-extras requirements.in

Write-Host "requirements.in and requirements.txt updated."