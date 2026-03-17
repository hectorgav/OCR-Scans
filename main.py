# =============================================================================
# OCR BATCH PIPELINE - Main Orchestrator (JSON-First Architecture)
# =============================================================================
# 
# Usage Examples:
#   python main.py                              # Auto-generated batch ID
#   python main.py --batch-id Run_A             # Custom batch ID
#   python main.py --input-dir ./scans/         # Override input directory
#   python main.py --auto-verify batch_001      # Run OCR and instantly verify!
#
# Data Flow:
#   1. Runs OCR on inputs.
#   2. Exports raw data to: 00-output/reports/latest_run_data.json
#   3. (Optional) Triggers verify.py to build dashboard metrics.
#
# =============================================================================

import time
import os
import sys  # <-- CRITICAL FIX: Ensures we use the active python environment
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Set

from tqdm import tqdm
import cv2

# =============================================================================
# STABILITY FIXES: Environment Configuration
# =============================================================================
logging.getLogger().handlers = []

try:
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
# CONFIG & PIPELINE IMPORTS
# =============================================================================
from config import (
    INPUT_DIR, PREPROCESSED_DIR, REPORTS_DIR, MAX_WORKERS,
    SUPPORTED_EXTENSIONS
)

from src.pipeline.extractor import extract_job_number
from src.pipeline.smart_filing import smart_correct_batch, Record, FAILED_JOB_MARKERS
from src.utils.pdf_utils import pdf_to_images
from src.utils.fs import setup_directory_structure, route_to_success, route_to_failed
from src.utils.log import get_logger, log_pipeline_start, log_error, log_banner

from src.utils.statistics import generate_pipeline_run_data, WallClockTracker

logger = get_logger("main_orchestrator")


# =============================================================================
# SESSION CONTEXT MANAGER
# =============================================================================
class BatchSession:
    def __init__(self, cli_args: argparse.Namespace):
        if cli_args.batch_id:
            self.batch_id = cli_args.batch_id
        else:
            self.batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        self.input_dir = Path(cli_args.input_dir) if cli_args.input_dir else INPUT_DIR
        
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.raw_data_json = REPORTS_DIR / "latest_run_data.json"
        
        self.auto_verify_batch = cli_args.auto_verify


def get_known_jobs_from_user() -> Optional[Set[str]]:
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


def find_input_files(directory: Path) -> List[Path]:
    if not directory.exists():
        logger.error(f"Input directory not found: {directory}")
        return []
    valid_exts = {ext.replace("*", "").lower() for ext in SUPPORTED_EXTENSIONS}
    return sorted([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])


def run_conversion_stage(session: BatchSession, output_dir: Path) -> int:
    logger_conversion = get_logger("conversion_stage")
    log_banner(logger_conversion, "STARTING STAGE 1: FILE CONVERSION (PDF -> IMG)")

    input_files = find_input_files(session.input_dir)
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


def process_single_file(image_path: str, original_file_path: str) -> Dict[str, Any]:
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


def run_extraction_batch(session: BatchSession, oriented_dir: Path, max_workers: int) -> List[Dict[str, Any]]:
    logger_extraction = get_logger("extraction_stage")
    log_banner(logger_extraction, "STARTING STAGE 2: JOB NUMBER EXTRACTION")

    original_files = {p.stem: p for p in find_input_files(session.input_dir)}
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


def run_smart_filing_and_routing(
    session: BatchSession,
    results: List[Dict[str, Any]], 
    known_jobs: Optional[Set[str]],
    elapsed_time: Optional[float] = None
):
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

    formatted_report = generate_pipeline_run_data(
        results=results, 
        output_json_path=session.raw_data_json,
        batch_id=session.batch_id,
        elapsed_time=elapsed_time
    )

    print("\n" + "=" * 80)
    print(formatted_report)
    print("=" * 80 + "\n")

    log_banner(logger_final, "PIPELINE COMPLETED")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Batch Pipeline")
    parser.add_argument(
        "--batch-id", type=str, default=None,
        help="Batch identifier for archiving. Auto-generated if not provided."
    )
    parser.add_argument("--input-dir", type=str, default=None, help="Override input directory")
    parser.add_argument(
        "--auto-verify", type=str, default=None, 
        help="Immediately run verify.py after execution using this ground truth CSV"
    )
    
    args = parser.parse_args()
    
    setup_directory_structure()
    session = BatchSession(args)
    main_logger = get_logger("main_orchestrator")
    
    main_logger.info(f"📁 Batch ID: {session.batch_id}")
    main_logger.info(f"📁 Raw Data Export: {session.raw_data_json}")

    known_jobs_list = get_known_jobs_from_user()

    ready_files_count = run_conversion_stage(session=session, output_dir=PREPROCESSED_DIR)

    if ready_files_count > 0:
        with WallClockTracker() as tracker:
            extraction_results = run_extraction_batch(
                session=session,
                oriented_dir=PREPROCESSED_DIR,
                max_workers=MAX_WORKERS,
            )
            
        if extraction_results:
            run_smart_filing_and_routing(
                session=session,
                results=extraction_results,
                known_jobs=known_jobs_list,
                elapsed_time=tracker.elapsed
            )
            
            # =========================================================
            # STAGE 5: AUTO-VERIFICATION (Bulletproofed)
            # =========================================================
            if session.auto_verify_batch:
                main_logger.info(f"🔄 Triggering Auto-Verification against: {session.auto_verify_batch}.csv")
                
                # Resolves the absolute path to verify.py so subprocess never gets lost
                verify_script_path = Path(__file__).resolve().parent / "verify.py"
                
                try:
                    # Uses sys.executable to ensure we use your active Python venv
                    subprocess.run(
                        [sys.executable, str(verify_script_path), "--batch", session.auto_verify_batch], 
                        check=True
                    )
                except subprocess.CalledProcessError as e:
                    main_logger.error(f"❌ Auto-verification script failed with error code: {e.returncode}")
                except Exception as e:
                    main_logger.error(f"❌ Failed to launch verify.py: {e}")
                    
        else:
            main_logger.warning("No extraction results generated.")
    else:
        main_logger.warning("No files were prepared. Halting.")