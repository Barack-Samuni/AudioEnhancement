# scripts/update_requirements.ps1
$ErrorActionPreference = "Stop"

function Fail($msg) {
    Write-Host ""
    Write-Host "❌ update_requirements FAILED: $msg" -ForegroundColor Red
    exit 1
}

# 1) Ensure .venv exists
$venvPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Fail "Expected venv at .venv. Run setup.ps1 first to create it."
}

# 2) Ensure we are using the venv Python (not global python)
# (We always call venv python directly, so this is guaranteed.)
$ver = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($ver -ne "3.10") {
    Fail "Venv Python version is $ver, expected 3.10. Recreate .venv using: py -3.10 -m venv .venv"
}

Write-Host "Using venv Python ($venvPython) version $ver" -ForegroundColor DarkGray

# 3) Ensure tools exist inside venv
& $venvPython -m pip install --upgrade pip | Out-Null
& $venvPython -m pip install pip-tools pip-chill | Out-Null

# 4) Generate top-level deps -> requirements.in
# Filter out tooling packages so they don't leak into runtime deps.
$ignore = @(
  "pip", "setuptools", "wheel",
  "pre-commit", "pip-tools", "pip-chill",
  "pytest"
)

$lines = & $venvPython -m pip_chill
$lines = $lines | Where-Object {
    $name = $_.Split('==')[0].Trim()
    -not ($ignore -contains $name)
}

$lines | Sort-Object | Out-File -Encoding utf8 "requirements.in"

# 5) Compile pinned deps -> requirements.txt
& $venvPython -m piptools compile requirements.in -o requirements.txt --quiet

# 6) Stage both files (so pre-commit includes them)
git add requirements.in requirements.txt

Write-Host "✅ Updated + staged requirements.in and requirements.txt" -ForegroundColor Green
