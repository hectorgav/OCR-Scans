$message = Read-Host "Enter commit message"
git add .
git commit -m "$message"
git push
Read-Host "Press Enter to continue..."