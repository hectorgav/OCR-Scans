$Host.UI.RawUI.WindowTitle = "OCR Pipeline Setup"
Clear-Host

# --- DIRECTORY RESOLUTION ---
$ScriptDir = $PSScriptRoot
if ($ScriptDir -like "*scripts") {
    $ProjectRoot = Split-Path -Parent $ScriptDir
} else {
    $ProjectRoot = $ScriptDir
}
Set-Location -Path $ProjectRoot

Write-Host "--- Setting up OCR Pipeline ---" -ForegroundColor Cyan

# Remove old venv
if (Test-Path ".venv") {
    Write-Host "[INFO] Removing old virtual environment..."
    Remove-Item -Recurse -Force ".venv"
}

# 1. Create new venv using standard Python
Write-Host "[INFO] Creating virtual environment (Standard Python)..."
& python -m venv .venv

$ActivateScript = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"

# Safety Check
if (-not (Test-Path $ActivateScript)) {
    Write-Host "`n[CRITICAL ERROR] The virtual environment was not created." -ForegroundColor Red
    Write-Host "Please ensure Python is installed and in your System PATH."
    Read-Host "Press Enter to exit"
    exit 1
}

# 2. Activate the environment
Write-Host "[INFO] Environment created. Activating..."
. $ActivateScript

# 3. The Bootstrapping Trick: Install 'uv' ONLY inside this isolated environment
Write-Host "[INFO] Bootstrapping 'uv' for high-speed installation..."
& python -m pip install --upgrade pip uv

# --- THE FIX: CLEAR CORRUPTED CACHES ---
Write-Host "[INFO] Clearing uv cache to prevent corrupted wheel deadlocks..."
& uv cache clean

# 4. Unleash 'uv' on the requirements file (WITH VERBOSE LOGGING)
$ReqFile = Join-Path $ProjectRoot "requirements.txt"
Write-Host "`n[INFO] Installing dependencies using uv..."

# The '-v' flag tells uv to print exactly what it is doing.
& uv pip install -r "$ReqFile" -v

Write-Host "`n[SUCCESS] Setup complete!" -ForegroundColor Green
Write-Host "Activate manually with: .venv\Scripts\Activate.ps1"
Read-Host "Press Enter to continue"