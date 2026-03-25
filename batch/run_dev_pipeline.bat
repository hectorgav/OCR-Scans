@echo off
TITLE "OCR Pipeline and Dashboard Verifier [DEV]"
COLOR 0A

echo ==================================================
echo       STARTING OCR BATCH PIPELINE (DEVELOPMENT)
echo ==================================================
echo.

:: --- ENVIRONMENT INJECTION ---
:: These override the .env file for this specific run
set APP_MODE=DEVELOPMENT
set WORK_ENV=OFFICE

:: --- CONFIGURATION ---
set INPUT_DIRECTORY=G:\Scans\preprocess\00-input\00-test
set GROUND_TRUTH_BATCH=batch_test

:: Auto-generate a timestamp for the Batch ID (Format: YYYYMMDD_HHMMSS)
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set BATCH_ID=Run_%mydate%_%mytime: =0%

echo [INFO] App Mode:        %APP_MODE%
echo [INFO] Input Directory: %INPUT_DIRECTORY%
echo [INFO] Target Batch ID: %BATCH_ID%
echo [INFO] Auto-Verify GT:  %GROUND_TRUTH_BATCH%
echo.

:: --- EXECUTION ---
echo Running main.py orchestrator...
python main.py --input-dir "%INPUT_DIRECTORY%" --batch-id "%BATCH_ID%" --auto-verify "%GROUND_TRUTH_BATCH%"

echo.
echo ==================================================
echo       PIPELINE COMPLETE
echo ==================================================
pause