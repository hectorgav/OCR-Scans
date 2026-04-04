$Host.UI.RawUI.WindowTitle = "Git Merge - Production"
Clear-Host

Write-Host "Switching to main branch..."
git checkout main

Write-Host "Pulling latest changes from GitHub..."
git pull origin main

Write-Host "Merging feature branch..."
git merge feature/production-refactor

# --- THE SAFETY CHECK ---
if ($LASTEXITCODE -ne 0) {
    Write-Host "CRITICAL WARNING: Merge conflict detected!" -ForegroundColor Red
    Write-Host "The script has stopped to protect the main branch."
    Write-Host "Please resolve the conflicts manually in your code editor."
    Write-Host "Once resolved, stage, commit, and push your changes manually."
    Read-Host "Press Enter to exit..."
    exit $LASTEXITCODE
}
# ------------------------

Write-Host "Merge successful!"
Write-Host "Pushing updated main branch to GitHub..."
git push origin main

# Optional Cleanup (Uncomment to use)
# Write-Host "Deleting the local feature branch..."
# git branch -d feature/production-refactor
# Write-Host "Deleting the remote feature branch on GitHub..."
# git push origin --delete feature/production-refactor

Write-Host "Production update and cleanup complete!" -ForegroundColor Green
Read-Host "Press Enter to continue..."