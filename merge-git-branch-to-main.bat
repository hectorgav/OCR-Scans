@echo off
echo Switching to main branch...
git checkout main

echo Pulling latest changes from GitHub...
git pull origin main

echo Merging feature branch...
git merge feature/production-refactor

:: --- THE SAFETY CHECK ---
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo ⚠️ CRITICAL WARNING: Merge conflict detected!
    echo The script has stopped to protect the main branch.
    echo Please resolve the conflicts manually in your code editor.
    echo Once resolved, stage, commit, and push your changes manually.
    echo.
    pause
    exit /b %ERRORLEVEL%
)
:: ------------------------

echo Merge successful! Pushing updated main branch to GitHub...
git push origin main

echo Deleting the local feature branch...
git branch -d feature/production-refactor

echo Deleting the remote feature branch on GitHub...
git push origin --delete feature/production-refactor

echo.
echo ✅ Production update and cleanup complete!
pause