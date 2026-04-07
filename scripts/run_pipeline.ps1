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

# 2. Dynamically Resolve Environment and Mode
$WorkEnv = $env:WORK_ENV
$AppMode = $env:APP_MODE

# Create a clean short mode (DEV vs PROD) to match the .env file
$ShortMode = if ($AppMode -eq "DEVELOPMENT") { "DEV" } else { "PROD" }

# Update Terminal Window Title dynamically
$Host.UI.RawUI.WindowTitle = "OCR Pipeline - $AppMode"

# Construct the exact variable name defined in your .env (e.g., OFFICE_INPUT_DEV)
$InputVarName = "${WorkEnv}_INPUT_${ShortMode}"
$TargetInput = (Get-Item "Env:$InputVarName" -ErrorAction SilentlyContinue).Value

# Safety check in case the .env is missing the constructed key
if (-not $TargetInput) {
    Write-Host "`n[CRITICAL ERROR] Missing configuration in .env" -ForegroundColor Red
    Write-Host "Could not find a path for: $InputVarName"
    Read-Host "Press Enter to exit"
    exit
}

$BatchID = "Run_${ShortMode}_" + (Get-Date -Format "yyyyMMdd_HHmm")

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Environment: $WorkEnv"
Write-Host " Application: $AppMode"
Write-Host " Target Dir:  $TargetInput"
Write-Host " Batch ID:    $BatchID"
Write-Host "========================================" -ForegroundColor Cyan

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
# Build the absolute path to app.py instead of moving into its directory
$AppPath = Join-Path $ProjectRoot "dashboard\app.py"

# Run Streamlit from the root directory by pointing it directly to the app file
& streamlit run "$AppPath"