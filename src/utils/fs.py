# src/utils/fs.py
"""
Filesystem helpers: directory setup and routing.
"""
import shutil
import logging
import re
from pathlib import Path
from config import OUTPUT_DIR, DEBUG_BASE_DIR, DEBUG_FOLDERS, REPORTS_DIR

logger = logging.getLogger(__name__)


def _sanitize_foldername(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", str(name))
    return sanitized.strip()[:100] or "unnamed"


def setup_directory_structure(base: Path = OUTPUT_DIR):
    """Creates directory structure including new debug layout."""
    logger.info("Setting up project directories...")

    (base / "success").mkdir(parents=True, exist_ok=True)
    (base / "failed").mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create Debug Structure
    if DEBUG_BASE_DIR:
        DEBUG_BASE_DIR.mkdir(parents=True, exist_ok=True)
        # Explicitly create the preprocessed folder (now inside debug)
        DEBUG_FOLDERS["preprocessed"].mkdir(parents=True, exist_ok=True)

        for key, folder_path in DEBUG_FOLDERS.items():
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Could not create debug folder {key}: {e}")

    logger.info("✅ Directory structure verified.")


def route_to_success(pdf_path: Path, job_number: str, base: Path = OUTPUT_DIR) -> Path:
    safe_job = _sanitize_foldername(job_number)
    job_dir = base / "success" / safe_job
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / pdf_path.name
    try:
        shutil.copy2(pdf_path, dest)
        logger.info(f"Routed SUCCESS file '{pdf_path.name}' to '{job_dir}'")
    except Exception as e:
        logger.error(f"Failed to copy success file: {e}")
    return dest


def route_to_failed(
    pdf_path: Path, error: str = "unknown_error", base: Path = OUTPUT_DIR
) -> Path:
    error_bucket = _sanitize_foldername(error)
    bucket = base / "failed" / error_bucket
    bucket.mkdir(parents=True, exist_ok=True)
    dest = bucket / pdf_path.name
    try:
        shutil.copy2(pdf_path, dest)
        logger.warning(f"Routed FAILED file '{pdf_path.name}' to '{bucket}'")
    except Exception as e:
        logger.error(f"Failed to copy failed file: {e}")
    return dest


def get_report_path(filename: str = "processing_report.csv") -> Path:
    return REPORTS_DIR / filename
