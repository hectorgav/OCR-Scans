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

# Create new venv
Write-Host "[INFO] Creating virtual environment..."
& uv venv .venv

$ActivateScript = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"

# Safety Check
if (-not (Test-Path $ActivateScript)) {
    Write-Host "`n[CRITICAL ERROR] The virtual environment was not created." -ForegroundColor Red
    Write-Host "Please ensure Python is installed and in your System PATH."
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[INFO] Environment created. Installing dependencies..."
. $ActivateScript

$ReqFile = Join-Path $ProjectRoot "requirements.txt"
& uv pip install -r "$ReqFile"

Write-Host "`n[SUCCESS] Setup complete!" -ForegroundColor Green
Write-Host "Activate manually with: .venv\Scripts\Activate.ps1"
Read-Host "Press Enter to continue"