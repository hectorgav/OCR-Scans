# src/utils/log.py
"""
Centralized logging utility for the entire OCR pipeline.

Provides a standardized logger instance and a collection of helper functions
for structured logging of pipeline stages, events, and statistics.
This functional approach promotes reusability and consistency across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict

from config import LOG_FILENAME, LOG_LEVEL, VERBOSE_OUTPUT


def get_logger(
    name: str = "ocr_pipeline",
    log_file: Optional[Path] = None,
    level: str = LOG_LEVEL,
) -> logging.Logger:
    """
    Get or create a standardized logger for the pipeline.
    Uses a root 'ocr_pipeline' logger to manage handlers and prevents duplication.
    """
    root_name = "ocr_pipeline"
    
    # 1. Standardize name: ensure it's ocr_pipeline.something or just ocr_pipeline
    if name != root_name and not name.startswith(root_name + "."):
        full_name = f"{root_name}.{name}"
    else:
        full_name = name
        
    logger = logging.getLogger(full_name)
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # 2. Configure the ocr_pipeline base logger
    base_logger = logging.getLogger(root_name)
    base_logger.setLevel(log_level)
    
    # CRITICAL: Prevent propagation to the true root logger to avoid duplicates 
    # if libraries like Paddle or Torch have called logging.basicConfig()
    base_logger.propagate = False
    
    # 3. Only attach handlers once to the ocr_pipeline logger
    if not base_logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)
        console_level = logging.INFO if VERBOSE_OUTPUT else logging.WARNING
        console_handler.setLevel(console_level)
        base_logger.addHandler(console_handler)

        # File Handler
        if log_file is None:
            # Re-read from config if not provided
            try:
                from config import LOG_FILENAME
                target_file = LOG_FILENAME
            except ImportError:
                target_file = log_file

        if target_file:
            try:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(target_file, mode="a", encoding="utf-8")
                file_handler.setFormatter(fmt)
                file_handler.setLevel(logging.DEBUG)
                base_logger.addHandler(file_handler)
            except Exception:
                pass

    return logger


# ============================================================================
# General & High-Level Logging Helpers
# ============================================================================


def log_banner(logger: logging.Logger, message: str, level: int = logging.INFO) -> None:
    """Prints a banner-style log message."""
    border = "=" * 80
    logger.log(level, "")
    logger.log(level, border)
    logger.log(level, f" {message} ".center(80, "="))
    logger.log(level, border)


def log_pipeline_start(logger: logging.Logger, config_summary: Dict) -> None:
    """Logs the start of a batch processing job with key config settings."""
    log_banner(logger, "STARTING BATCH PROCESSING JOB")
    for key, value in config_summary.items():
        logger.info(f"{key:<20}: {value}")


def log_batch_summary(
    logger: logging.Logger,
    total_files: int,
    successful: int,
    failed: int,
    total_time: float,
    methods_used: Dict[str, int],
) -> None:
    """Logs a final summary of the entire batch processing job."""
    log_banner(logger, "BATCH PROCESSING JOB COMPLETED")
    success_rate = (successful / total_files * 100) if total_files > 0 else 0
    avg_time = (total_time / total_files) if total_files > 0 else 0

    logger.info(f"Total Files Processed: {total_files}")
    logger.info(f"  - Successful: {successful} ({success_rate:.1f}%)")
    logger.info(f"  - Failed: {failed}")
    logger.info(f"Total Time Elapsed: {total_time:.2f} seconds")
    logger.info(f"Average Time per File: {avg_time:.2f} seconds")
    if methods_used:
        logger.info("Final Method Counts:")
        for method, count in sorted(methods_used.items()):
            logger.info(f"  - {method}: {count}")


# ============================================================================
# File & Stage Progression Logging Helpers
# ============================================================================


def log_file_start(logger: logging.Logger, filename: str) -> None:
    """Logs the start of processing for a single file."""
    logger.info("-" * 80)
    logger.info(f"🚀 Processing file: {filename}")
    logger.info("-" * 80)


def log_stage_start(logger: logging.Logger, stage_name: str, details: str = "") -> None:
    """Logs the start of a specific pipeline stage."""
    msg = f"--- Starting Stage: {stage_name} ---"
    if details:
        msg += f" ({details})"
    logger.info(msg)


def log_file_summary(
    logger: logging.Logger,
    filename: str,
    status: str,
    job_number: Optional[str],
    method: str,
    elapsed_time: float,
) -> None:
    """Logs the summary for a single file's processing."""
    if status == "SUCCESS":
        logger.info(
            f"✅ SUCCESS: Found '{job_number}' in {filename} via {method} ({elapsed_time:.2f}s)"
        )
    else:
        logger.warning(
            f"❌ FAILED: Could not find job number in {filename} ({elapsed_time:.2f}s)"
        )


def log_file_saved(logger: logging.Logger, output_path: Path) -> None:
    """Logs when an output file (like a debug image) has been saved."""
    logger.debug(f"💾 Saved output to: {output_path}")


# ============================================================================
# Generic Status & Result Logging Helpers
# ============================================================================


def log_success(logger: logging.Logger, message: str, details: Dict = None) -> None:
    """Logs a generic success message with optional structured details."""
    details_str = (
        " | ".join([f"{k}: {v}" for k, v in details.items()]) if details else ""
    )
    logger.info(f"✅ {message}" + (f" | {details_str}" if details_str else ""))


def log_warning(logger: logging.Logger, message: str, details: str = "") -> None:
    """Logs a generic warning message."""
    logger.warning(f"⚠️  {message}" + (f" | {details}" if details else ""))


def log_error(logger: logging.Logger, message: str, error: Exception = None) -> None:
    """Logs an error message, including exception info if provided."""
    logger.error(f"❌ {message}", exc_info=error)


def log_debug(
    logger: logging.Logger, message: str, data: Optional[Dict] = None
) -> None:
    """Logs a debug message with optional structured data."""
    data_str = " | ".join([f"{k}={v}" for k, v in data.items()]) if data else ""
    logger.debug(f"🔬 {message}" + (f" | {data_str}" if data_str else ""))


# ============================================================================
# Task-Specific Logging Helpers (Example: Orientation Detection)
# ============================================================================


def log_orientation_scores(
    logger: logging.Logger, method: str, scores: Dict[int, float]
) -> None:
    """Logs the calculated scores for each angle for an orientation method."""
    score_list = ", ".join(
        [f"{angle}°: {score:.2f}" for angle, score in sorted(scores.items())]
    )
    logger.debug(f"  Orientation Scores [{method}]: {score_list}")


def log_orientation_decision(
    logger: logging.Logger, angle: int, method: str, applied: bool
) -> None:
    """Logs the final orientation decision for a file."""
    status = "APPLIED" if applied else "NOT APPLIED (angle below threshold)"
    logger.info(f"  Orientation Decision: {angle}° rotation ({status}) via [{method}]")
