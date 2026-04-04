$Host.UI.RawUI.WindowTitle = "OCR Pipeline - DEVELOPMENT"
Clear-Host

# --- DIRECTORY RESOLUTION ---
# Find the Project Root (one level up from \scripts\)
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
    exit
}

# 2. Get Input Path from .env
$WorkEnv = $env:WORK_ENV
$BaseVar = "${WorkEnv}_INPUT_DEV"
$BaseDir = (Get-Item "Env:$BaseVar").Value

Write-Host "Environment: $WorkEnv"
Write-Host "Base Path:   $BaseDir"

# 3. Interactive Menu
$Target = $BaseDir
if (Test-Path $BaseDir) {
    $Subfolders = Get-ChildItem -Path $BaseDir -Directory
    if ($Subfolders.Count -gt 0) {
        Write-Host "`nSelect a subdirectory to process:" -ForegroundColor Cyan
        Write-Host "[0] Process ALL (Root)"
        for ($i=0; $i -lt $Subfolders.Count; $i++) {
            Write-Host "[$($i+1)] $($Subfolders[$i].Name)"
        }
        $choice = Read-Host "`nEnter selection"
        if ($choice -as [int] -and $choice -gt 0 -and $choice -le $Subfolders.Count) {
            $Target = $Subfolders[$choice-1].FullName
        }
    }
}

# 4. Execute
$BatchID = "Run_DEV_" + (Get-Date -Format "yyyyMMdd_HHmm")
$MainPy = Join-Path $ProjectRoot "main.py"

Write-Host "`nTargeting: $Target" -ForegroundColor Green
& python "$MainPy" --input-dir "$Target" --batch-id "$BatchID" --auto-verify "batch_test"

Write-Host "`nPipeline execution finished."
Read-Host "Press Enter to close"