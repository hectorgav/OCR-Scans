$Host.UI.RawUI.WindowTitle = "OCR Pipeline - PRODUCTION"
Clear-Host

# --- DIRECTORY RESOLUTION ---
$ScriptDir = $PSScriptRoot
if ($ScriptDir -like "*scripts") {
    $ProjectRoot = Split-Path -Parent $ScriptDir
} else {
    $ProjectRoot = $ScriptDir
}
Set-Location -Path $ProjectRoot

# 1. Load the .env file
$EnvFile = Join-Path $ProjectRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)\s*=\s*(.*)$') {
            $name = $Matches[1].Trim()
            $value = $Matches[2].Trim('"', "'", " ")
            Set-Item -Path "Env:\$name" -Value $value
        }
    }
} else {
    Write-Host "ERROR: .env file not found at $EnvFile" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit
}

# 2. Get Input Path from .env
$WorkEnv = $env:WORK_ENV
$ProdVar = "${WorkEnv}_INPUT_PROD"
$TargetInput = (Get-Item "Env:$ProdVar").Value
$BatchID = "Run_PROD_" + (Get-Date -Format "yyyyMMdd_HHmm")

Write-Host "Environment: $WorkEnv"
Write-Host "Target Input: $TargetInput"
Write-Host "Batch ID:    $BatchID"

# 3. Execute Pipeline
Write-Host "`n[1/2] Running main.py orchestrator..."
$MainPy = Join-Path $ProjectRoot "main.py"
& python "$MainPy" --input-dir "$TargetInput" --batch-id "$BatchID"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: main.py exited with an error code." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit $LASTEXITCODE
}

# 4. Launch Dashboard
Write-Host "`n[2/2] Starting Streamlit Server..."
$DashboardDir = Join-Path $ProjectRoot "dashboard"
Set-Location -Path $DashboardDir
& streamlit run "app.py"

Read-Host "Press Enter to close"