@echo off
TITLE "OCR Production Pipeline and Dashboard [PROD]"
COLOR 0B

echo ==================================================
echo       STARTING OCR PRODUCTION PIPELINE
echo ==================================================
echo.

:: --- ENVIRONMENT INJECTION ---
:: Force Production Mode routing
set APP_MODE=PRODUCTION
set WORK_ENV=OFFICE

:: --- CONFIGURATION ---
:: Point this to your incoming scans folder
set INPUT_DIRECTORY=G:\Scans\preprocess\00-input\00-production-batch

:: Auto-generate a timestamp for the Batch ID (Format: YYYYMMDD_HHMMSS)
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set BATCH_ID=Run_%mydate%_%mytime: =0%

echo [INFO] App Mode:        %APP_MODE%
echo [INFO] Input Directory: %INPUT_DIRECTORY%
echo [INFO] Target Batch ID: %BATCH_ID%
echo.

:: --- STAGE 1: EXECUTION (OCR Extraction & Triage) ---
echo [1/2] Running main.py orchestrator...
python main.py --input-dir "%INPUT_DIRECTORY%" --batch-id "%BATCH_ID%"

echo.
echo ==================================================
echo       OCR COMPLETE. LAUNCHING DASHBOARD...
echo ==================================================
echo.

:: --- STAGE 2: MANUAL REVIEW (Streamlit UI) ---
echo [2/2] Starting Streamlit Server...
echo Please keep this window open until you are done reviewing.
echo Press Ctrl+C in this window when you are ready to shut down the dashboard.
echo.

:: Note: Streamlit will automatically read the APP_MODE=PRODUCTION variable
:: and look in the `00-output` folder instead of `00-output-dev`
streamlit run dashboard/app.py

pause