@echo off
echo 🚀 Setting up OCR Pipeline with Python 3.10...

# Remove old venv
if exist .venv (
    echo 🗑️  Removing old virtual environment...
    rmdir /s /q .venv
)

# Create new venv with Python 3.10
echo 📦 Creating virtual environment...
uv venv --python 3.10

# Activate and install
echo 📥 Installing dependencies...
call .venv\Scripts\activate
uv pip install -r requirements.txt

echo ✅ Setup complete!
echo.
echo Activate with: .venv\Scripts\activate
pause