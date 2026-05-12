import os
import shutil
import sys
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# --- NEW CONFIGURATION FOR INITIAL TASK ---
INITIAL_PDF_SOURCE = r"G:\Scans"
INITIAL_PDF_DEST = r"G:\Scans\preprocess\00-input\00-production-batch"

def move_initial_pdfs():
    """Moves only PDF files from the root of the source to the destination."""
    print("[INFO] Starting initial task: Moving PDFs to preprocessing...")
    if not os.path.exists(INITIAL_PDF_SOURCE):
        print(f"[WARNING] Source path does not exist: {INITIAL_PDF_SOURCE}")
        return
    
    # Ensure destination directory exists
    os.makedirs(INITIAL_PDF_DEST, exist_ok=True)
    
    moved_count = 0
    try:
        # os.scandir looks ONLY at the immediate directory (no sub-folders)
        with os.scandir(INITIAL_PDF_SOURCE) as entries:
            for entry in entries:
                # Check if it's a file (not a folder) AND ends with .pdf
                if entry.is_file() and entry.name.lower().endswith('.pdf'):
                    src_path = entry.path
                    dst_path = os.path.join(INITIAL_PDF_DEST, entry.name)
                    
                    # If file already exists in destination, remove it to overwrite safely
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                        
                    shutil.move(src_path, dst_path)
                    moved_count += 1
                    
        print(f"[SUCCESS] Moved {moved_count} PDF(s) to production batch folder.\n")
    except Exception as e:
        print(f"[ERROR] Failed to move PDFs: {e}\n")
# ------------------------------------------

def run_command(command):
    """Executes a command and exits if it fails."""
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with code {e.returncode}")
        input("Press Enter to exit...")
        sys.exit(e.returncode)

def verify_environment():
    """Checks if the core dependencies are correctly installed."""
    print("[INFO] Verifying environment readiness...")
    try:
        import torch
        if not hasattr(torch, 'save'):
            raise AttributeError("module 'torch' has no attribute 'save'. Your environment might be corrupted.")
        
        # Check ultralytics
        from ultralytics import YOLO
        
        # Check numpy
        import numpy as np
        
        print("[SUCCESS] Environment verified.")
        return True
    except ImportError as e:
        print(f"\n[CRITICAL ERROR] Missing dependency: {e}")
        print("Please run 'python setup_initial.py' to repair your environment.")
        input("Press Enter to exit...")
        sys.exit(1)
    except AttributeError as e:
        print(f"\n[CRITICAL ERROR] Environment corruption detected: {e}")
        print("Please run 'python setup_initial.py' to repair your environment.")
        input("Press Enter to exit...")
        sys.exit(1)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Unexpected environment issue: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

def main():
    # --- 0. NEW INITIAL TASK ---
    # Executes immediately before anything else
    move_initial_pdfs()
    # ---------------------------

    # 1. Directory Resolution
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    os.chdir(project_root)

    # 1.5 Verify Environment
    verify_environment()

    # 2. Load .env file
    env_file = project_root / ".env"
    if not env_file.exists():
        print(f"ERROR: .env file not found at {env_file}")
        input("Press Enter to exit...")
        sys.exit(1)
    
    load_dotenv(dotenv_path=env_file)

    # 3. Resolve Environment and Mode
    work_env = os.getenv("WORK_ENV")
    app_mode = os.getenv("APP_MODE")
    
    if not work_env or not app_mode:
        print("ERROR: WORK_ENV or APP_MODE not found in .env")
        sys.exit(1)

    short_mode = "DEV" if app_mode == "DEVELOPMENT" else "PROD"
    
    # Construct key and get value
    input_var_name = f"{work_env}_INPUT_{short_mode}"
    target_input = os.getenv(input_var_name)

    if not target_input:
        print(f"\n[CRITICAL ERROR] Missing configuration in .env")
        print(f"Could not find a path for: {input_var_name}")
        input("Press Enter to exit...")
        sys.exit(1)

    batch_id = f"Run_{short_mode}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    print("=" * 40)
    print(f" Environment: {work_env}")
    print(f" Application: {app_mode}")
    print(f" Target Dir:  {target_input}")
    print(f" Batch ID:    {batch_id}")
    print("=" * 40)

    # 4. Execute Pipeline (main.py)
    print(f"\n[1/2] Running main.py orchestrator...")
    # Use sys.executable to ensure we use the same Python environment
    run_command([sys.executable, "main.py", "--input-dir", target_input, "--batch-id", batch_id])

    # 5. Launch Dashboard
    print(f"\n[2/2] Starting Streamlit Server...")
    app_path = project_root / "dashboard" / "app.py"
    
    # Note: Using 'streamlit' as a module is more robust
    run_command([sys.executable, "-m", "streamlit", "run", str(app_path)])

if __name__ == "__main__":
    main()