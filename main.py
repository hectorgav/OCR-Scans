# =============================================================================
# OCR BATCH PIPELINE - Main Orchestrator
# =============================================================================
# 
# Usage Examples:
#   python main.py                              # Auto-generated batch ID
#   python main.py --batch-id batch_001         # Custom batch ID
#   python main.py --input-dir ./scans/         # Override input directory
#   python main.py --no-archive                 # Disable archiving
#
# Report Structure:
#   reports/
#   ├── latest/                                 → Symlink to most recent batch
#   │   ├── statistics_report.txt
#   │   └── smart_filing_summary.json
#   └── archive/
#       └── 2026-03-16_14-30-00_batch_001/      → Archived batch reports
#
# =============================================================================

import time
import os
import logging
import argparse
import shutil
from datetime import datetime

# =============================================================================
# STABILITY FIXES: Environment Configuration
# =============================================================================
# Clear any rogue handlers from libraries like paddle/torch that call basicConfig
logging.getLogger().handlers = []

# Force DLL search path for Torch before any heavy imports (Windows compatibility)
try:
    import sys
    venv_base = os.environ.get("VIRTUAL_ENV") or os.getcwd()
    torch_lib = os.path.join(venv_base, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        
    import torch
    from ultralytics import YOLO
except Exception:
    pass

# =============================================================================
# CONFIG IMPORTS (Sets environment stability flags first)
# =============================================================================
from config import (
    INPUT_DIR, PREPROCESSED_DIR, REPORTS_DIR, MAX_WORKERS,
    SUPPORTED_EXTENSIONS
)

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Set
from tqdm import tqdm
import cv2
import numpy as np

# =============================================================================
# PIPELINE IMPORTS
# =============================================================================
from src.pipeline.extractor import extract_job_number
from src.pipeline.smart_filing import smart_correct_batch, Record, FAILED_JOB_MARKERS

# =============================================================================
# UTILS IMPORTS
# =============================================================================
from src.utils.pdf_utils import pdf_to_images
from src.utils.fs import (
    setup_directory_structure,
    get_report_path,
    route_to_success,
    route_to_failed,
)
from src.utils.log import (
    get_logger,
    log_pipeline_start,
    log_error,
    log_banner,
)
from src.utils.statistics import generate_and_save_statistics, WallClockTracker

logger = get_logger("main_orchestrator")

# =============================================================================
# HELPER: User Prompt for Known Jobs
# =============================================================================
def get_known_jobs_from_user() -> Optional[Set[str]]:
    """
    Prompt user for known-valid job numbers to restrict smart corrections.
    
    Returns:
        Set of known job numbers, or None if user skips validation
    """
    print("\n" + "=" * 50)
    print("OPTIONAL: POST-OCR VALIDATION")
    print("Enter 'known good' job numbers to restrict smart corrections.")
    print("Example: 250371-01, 240216-04")
    print("Leave BLANK and press Enter to skip validation.")
    print("=" * 50)

    user_input = input("Known Jobs > ")

    if not user_input.strip():
        logger.info("No known jobs provided. Smart filing will use best-guess consensus logic.")
        return None

    jobs = {job.strip() for job in user_input.split(",") if job.strip()}
    logger.info(f"Loaded {len(jobs)} known valid job numbers for validation.")
    return jobs


# =============================================================================
# STAGE 1: FILE DISCOVERY & CONVERSION
# =============================================================================
def find_input_files(directory: Path) -> List[Path]:
    """
    Discover all valid input files in the specified directory.
    
    Args:
        directory: Path to search for input files
        
    Returns:
        Sorted list of valid file paths
    """
    if not directory.exists():
        logger.error(f"Input directory not found: {directory}")
        return []
    
    valid_exts = {ext.replace("*", "").lower() for ext in SUPPORTED_EXTENSIONS}
    return sorted([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])


def run_conversion_stage(input_dir: Path, output_dir: Path) -> int:
    """
    Convert input files (PDF/images) to standardized JPG format for OCR.
    
    Args:
        input_dir: Directory containing source files
        output_dir: Directory to save converted images
        
    Returns:
        Count of successfully converted files
    """
    logger_conversion = get_logger("conversion_stage")
    log_banner(logger_conversion, "STARTING STAGE 1: FILE CONVERSION (PDF -> IMG)")

    input_files = find_input_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for file_path in tqdm(input_files, desc="Preparing Files"):
        try:
            out_path = output_dir / f"{file_path.stem}_ready.jpg"
            if file_path.suffix.lower() == ".pdf":
                images = pdf_to_images(str(file_path), max_pages=1)
                if images is not None and len(images) > 0:
                    cv2.imwrite(str(out_path), images[0], [cv2.IMWRITE_JPEG_QUALITY, 95])
                    success_count += 1
            else:
                img = cv2.imread(str(file_path))
                if img is not None:
                    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    success_count += 1
        except Exception as e:
            logger.error(f"Failed to prepare {file_path.name}: {e}")

    return success_count


# =============================================================================
# STAGE 2: EXTRACTION
# =============================================================================
def process_single_file(image_path: str, original_file_path: str) -> Dict[str, Any]:
    """
    Process a single image file through the OCR extraction pipeline.
    
    Args:
        image_path: Path to pre-processed image
        original_file_path: Path to original source file
        
    Returns:
        Dictionary containing extraction results and metadata
    """
    start_time = time.monotonic()
    try:
        job_number, confidence, method, details = extract_job_number(
            image_path=Path(image_path), original_file_path=Path(original_file_path)
        )

        result = {
            "filename": Path(original_file_path).name,
            "original_path": str(original_file_path),
            "job_number": job_number,
            "confidence": confidence,
            "method": method,
            "meta": details,
            "processing_time_sec": round(time.monotonic() - start_time, 2),
            "page": 1,
            "error": None,
            "status": "Success",
            "raw_job_number": job_number,
        }

        if not job_number or job_number in FAILED_JOB_MARKERS:
            error_reason = details.get("error", "extraction_failed")
            result.update({
                "method": "failed",
                "error": error_reason,
                "status": f"Failure: {error_reason}",
                "raw_job_number": "failed",
            })

        return result

    except Exception as e:
        return {
            "filename": Path(original_file_path).name,
            "original_path": str(original_file_path),
            "job_number": None,
            "confidence": 0.0,
            "method": "failed",
            "processing_time_sec": round(time.monotonic() - start_time, 2),
            "page": 1,
            "error": f"critical_error: {e}",
            "status": f"Failure: Critical error - {e}",
            "raw_job_number": "failed",
        }


def run_extraction_batch(
    oriented_dir: Path, 
    original_dir: Path, 
    max_workers: int
) -> List[Dict[str, Any]]:
    """
    Run OCR extraction on all pre-processed images in batch mode.
    
    Args:
        oriented_dir: Directory containing pre-processed images
        original_dir: Directory containing original source files
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of extraction result dictionaries
    """
    logger_extraction = get_logger("extraction_stage")
    log_banner(logger_extraction, "STARTING STAGE 2: JOB NUMBER EXTRACTION")

    original_files = {p.stem: p for p in find_input_files(original_dir)}
    ready_images = sorted(list(oriented_dir.glob("*_ready.jpg")))

    if not ready_images:
        logger.warning(f"No pre-processed images found in {oriented_dir}.")
        return []

    tasks = [
        (str(img_path), str(original_files[img_path.stem.replace("_ready", "")])) 
        for img_path in ready_images 
        if img_path.stem.replace("_ready", "") in original_files
    ]

    config_summary = {
        "Mode": "Heuristic-First Pipeline Extraction",
        "Files to Process": len(tasks),
        "Max Workers": max_workers,
    }
    log_pipeline_start(logger_extraction, config_summary)

    results = []
    effective_workers = max_workers
    
    if effective_workers <= 1:
        logger.info("Running stage 2 in sequential mode (1 worker) for maximum stability.")
        for img, pdf in tqdm(tasks, desc="Extracting"):
            results.append(process_single_file(img, pdf))
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_to_task = {
                executor.submit(process_single_file, img, pdf): (img, pdf) 
                for img, pdf in tasks
            }
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Extracting"):
                try:
                    results.append(future.result())
                except Exception as exc:
                    log_error(logger_extraction, "Worker exception", error=exc)

    log_banner(logger_extraction, "COMPLETED STAGE 2: JOB NUMBER EXTRACTION")
    return results


# =============================================================================
# STAGE 3: SMART FILING & ROUTING
# =============================================================================
def run_smart_filing_and_routing(
    results: List[Dict[str, Any]], 
    known_jobs: Optional[Set[str]],
    elapsed_time: Optional[float] = None,
    batch_id: Optional[str] = None,
    report_dir: Optional[Path] = None,
):
    """
    Apply smart filing corrections and route files to success/failed folders.
    Also generates statistics reports with optional archiving.
    
    Args:
        results: List of extraction result dictionaries
        known_jobs: Optional set of known-valid job numbers for validation
        elapsed_time: Total wall-clock time for the pipeline
        batch_id: Unique batch identifier for archiving
        report_dir: Directory path for report storage (if archiving enabled)
    """
    logger_final = get_logger("final_stage")
    log_banner(logger_final, "STARTING STAGE 3: SMART FILING CONSENSUS")

    records = [
        Record(
            filename=r["filename"],
            raw_job=r.get("job_number") if r.get("job_number") else "failed",
            confidence=r.get("confidence", 0.0),
            meta=r.get("meta", {})
        )
        for r in results
    ]

    corrected_records = smart_correct_batch(records, known_jobs=known_jobs)

    corrections_count = 0
    for res, rec in zip(results, corrected_records):
        original_job = res["job_number"]

        if rec.corrected_job and rec.corrected_job not in FAILED_JOB_MARKERS:
            if rec.corrected_job != original_job:
                res["job_number"] = rec.corrected_job
                res["method"] += f" (+smart_filing: {rec.reason})"
                res["status"] = "Success"
                res["error"] = None
                corrections_count += 1
                logger_final.info(
                    f"Smart Correction: {res['filename']} | "
                    f"{original_job} -> {rec.corrected_job} ({rec.reason})"
                )
        else:
            res["status"] = "Failed"
            res["job_number"] = None
            res["error"] = rec.reason if rec.reason else "extraction_failed"

    logger_final.info(f"Smart Filing complete. {corrections_count} files corrected.")

    log_banner(logger_final, "STARTING STAGE 4: ROUTING & REPORTING")
    for res in tqdm(results, desc="Routing"):
        original_path = Path(res["original_path"]) 

        if res["status"] == "Success" and res.get("job_number") and res["job_number"] not in FAILED_JOB_MARKERS:
            route_to_success(original_path, res["job_number"])
        else:
            route_to_failed(original_path, error=str(res.get("error", "unknown")))

    # =========================================================
    # REPORT GENERATION: Use archive directory if provided
    # =========================================================
    if report_dir and batch_id:
        stats_report_path = report_dir / "statistics_report.txt"
        logger_final.info(f"📁 Reports will be archived to: {report_dir}")
    else:
        stats_report_path = get_report_path("statistics_report.txt")

    generate_and_save_statistics(
        results, 
        stats_report_path, 
        elapsed_time=elapsed_time,
        batch_id=batch_id,
    )

    try:
        if stats_report_path.exists():
            print("\n" + "=" * 80)
            with open(stats_report_path, "r", encoding="utf-8") as f:
                print(f.read())
            print("=" * 80 + "\n")
    except Exception as e:
        logger_final.error(f"Failed to print statistics to console: {e}")

    log_banner(logger_final, "PIPELINE COMPLETED")


# =============================================================================
# ARCHIVE MANAGEMENT: Report Directory Setup
# =============================================================================
def setup_report_directory(batch_id: str) -> Path:
    """
    Create timestamped report directory with 'latest' symlink.
    
    Args:
        batch_id: Unique identifier for this batch (e.g., "batch_001" or timestamp)
        
    Returns:
        Path to the archive directory for this batch
    """
    archive_dir = REPORTS_DIR / "archive" / batch_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    latest_dir = REPORTS_DIR / "latest"
    
    # Clean up existing 'latest' symlink or directory
    if latest_dir.exists() or latest_dir.is_symlink():
        try:
            if latest_dir.is_symlink():
                latest_dir.unlink()
            else:
                shutil.rmtree(latest_dir)
        except OSError as e:
            logger.warning(f"Failed to remove existing 'latest' directory: {e}")
    
    # Create new symlink (Unix/Mac) or copy (Windows)
    try:
        latest_dir.symlink_to(archive_dir)
    except OSError:
        # Windows fallback: copy instead of symlink
        try:
            if latest_dir.exists():
                shutil.rmtree(latest_dir)
            shutil.copytree(archive_dir, latest_dir)
        except OSError as e:
            logger.warning(f"Failed to create 'latest' reference: {e}")
    
    return archive_dir


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # =========================================================
    # CLI ARGUMENT PARSING
    # =========================================================
    parser = argparse.ArgumentParser(description="OCR Batch Pipeline")
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Batch identifier for archiving (e.g., 'batch_001'). Auto-generated if not provided."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override input directory"
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Disable archiving (save to reports/ only)"
    )
    args = parser.parse_args()
    
    setup_directory_structure()
    main_logger = get_logger("main_orchestrator")
    known_jobs_list = get_known_jobs_from_user()

    # =========================================================
    # BATCH IDENTIFICATION: Use CLI argument or auto-generate timestamp
    # Format: YYYY-MM-DD_HH-MM-SS or custom (e.g., "batch_001")
    # =========================================================
    if args.batch_id:
        batch_id = args.batch_id
    else:
        batch_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # =========================================================
    # ARCHIVE SETUP: Create directory structure (unless --no-archive)
    # =========================================================
    if not args.no_archive:
        report_dir = setup_report_directory(batch_id)
        main_logger.info(f"📁 Batch ID: {batch_id}")
        main_logger.info(f"📁 Reports archived to: {report_dir}")
    else:
        report_dir = None
        batch_id = None
        main_logger.info("⚠️  Archiving disabled (--no-archive)")
    
    # Override input directory if provided
    input_dir = Path(args.input_dir) if args.input_dir else INPUT_DIR

    ready_files_count = run_conversion_stage(input_dir=input_dir, output_dir=PREPROCESSED_DIR)

    if ready_files_count > 0:
        with WallClockTracker() as tracker:
            extraction_results = run_extraction_batch(
                oriented_dir=PREPROCESSED_DIR,
                original_dir=input_dir,
                max_workers=MAX_WORKERS,
            )
            
        if extraction_results:
            run_smart_filing_and_routing(
                results=extraction_results,
                known_jobs=known_jobs_list,
                elapsed_time=tracker.elapsed,
                batch_id=batch_id,
                report_dir=report_dir,
            )
        else:
            main_logger.warning("No extraction results generated.")
    else:
        main_logger.warning("No files were prepared. Halting.")