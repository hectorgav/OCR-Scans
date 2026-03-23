# =============================================================================
# STATISTICS UTILITIES - OCR Batch Pipeline (JSON-First Architecture)
# =============================================================================
"""
Statistics utilities for the OCR batch pipeline.

This module is responsible for Phase 1 of the data flow:
It takes the raw extraction results, calculates hardware and timing metrics, 
and exports a purely structural JSON file (pipeline_run_data.json). 

It DOES NOT perform ground truth verification. It simply acts as the clean 
data handoff for the dashboard builder (verify.py).
"""
from __future__ import annotations

import time
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter

from src.utils.log import get_logger

logger = get_logger(__name__)


# =============================================================================
# TIME FORMATTING UTILITIES
# =============================================================================
def format_duration(seconds: float) -> str:
    """Return a short human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.2f}s"


def format_duration_detailed(seconds: float) -> str:
    """Return a detailed duration string (hours, minutes, seconds)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {remaining_seconds:.2f}s"
    if minutes > 0:
        return f"{minutes}m {remaining_seconds:.2f}s"
    return f"{remaining_seconds:.2f}s"


# =============================================================================
# WALL CLOCK TRACKER (CONTEXT MANAGER)
# =============================================================================
class WallClockTracker:
    """Context manager to track real wall-clock time."""
    def __init__(self) -> None:
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def __enter__(self) -> "WallClockTracker":
        self._start_time = time.perf_counter()
        self._end_time = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._end_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        if self._end_time is None:
            return time.perf_counter() - self._start_time
        return self._end_time - self._start_time


# =============================================================================
# DATA EXTRACTION HELPERS
# =============================================================================
def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default) if d else default

def _extract_ocr_method(method_str: str) -> str:
    if not method_str: return "unknown"
    
    # Catch the parenthesis version first
    if "(+smart_filing:" in method_str:
        return method_str.split("(+smart_filing:")[0].strip()
        
    if "smart_filing:" in method_str:
        return method_str.split("smart_filing:")[0].strip()
        
    return method_str.strip()

def _extract_confidence(method_str: str, fallback: Optional[float] = None) -> float:
    if fallback is not None: return float(fallback)
    match = re.search(r"conf\s*([0-9.]+)", method_str or "")
    if match:
        try: return float(match.group(1))
        except ValueError: pass
    return 0.0

def _extract_correction_reason(method_str: str) -> str:
    if not method_str: return "kept: valid"
    if "smart_filing:" in method_str:
        reason_part = method_str.split("smart_filing:")[1].strip()
        if reason_part.endswith(")"): reason_part = reason_part[:-1]
        return reason_part
    return "kept: valid"


# =============================================================================
# STATISTICS GENERATOR & RAW JSON EXPORT
# =============================================================================
def generate_pipeline_run_data(
    results: List[Dict[str, Any]],
    output_json_path: Path,
    batch_id: str,
    elapsed_time: Optional[float] = None
) -> str:
    """
    Builds the raw pipeline JSON report (pipeline_run_data.json), saves it to 
    disk, and returns the human-readable text equivalent for console output.
    
    Args:
        results: The raw list of dictionaries from the OCR pipeline.
        output_json_path: The exact file path to write the JSON to.
        batch_id: The identifier for the current run.
        elapsed_time: The wall-clock time from the tracker.
        
    Returns:
        A formatted string summary to be printed to the console by main.py.
    """
    total_files = len(results)
    if total_files == 0:
        logger.warning("No results to analyze. Skipping data generation.")
        return "No results processed."

    successful_extractions = [r for r in results if r.get("status") == "Success"]
    failed_extractions = [r for r in results if r.get("status") != "Success"]

    success_count = len(successful_extractions)
    failure_count = len(failed_extractions)
    success_rate = (success_count / total_files) * 100 if total_files > 0 else 0.0

    total_processing_time = sum(float(r.get("processing_time_sec", 0.0)) for r in results)
    avg_processing_time = total_processing_time / total_files if total_files > 0 else 0.0
    actual_wall_time = float(elapsed_time) if elapsed_time else total_processing_time
    parallel_speedup = (total_processing_time / actual_wall_time) if actual_wall_time > 0 else 1.0

    # ---------------------------------------------------------
    # 1. BUILD RAW JSON DICTIONARY (Handoff for verify.py)
    # ---------------------------------------------------------
    file_records = []
    for res in results:
        raw_job = _safe_get(res, "raw_job_number") or _safe_get(res, "job_number", "")
        corrected_job = _safe_get(res, "corrected_job") or _safe_get(res, "job_number", "")
        confidence = _safe_get(res, "confidence") or _extract_confidence(_safe_get(res, "method", ""))
        reason = _safe_get(res, "reason") or _extract_correction_reason(_safe_get(res, "method", ""))
        method = _safe_get(res, "ocr_method") or _extract_ocr_method(_safe_get(res, "method", ""))
        
        file_records.append({
            "filename": _safe_get(res, "filename", ""),
            "raw_job": str(raw_job) if raw_job else "",
            "corrected_job": str(corrected_job) if corrected_job else "",
            "confidence": float(confidence),
            "reason": str(reason),
            "status": _safe_get(res, "status", "Unknown"),
            "processing_time_sec": float(_safe_get(res, "processing_time_sec", 0.0)),
            "method": str(method),
            "error": str(_safe_get(res, "error", "")) if _safe_get(res, "error") else None
        })

    pipeline_run_data = {
        "batch_metadata": {
            "batch_id": batch_id,
            "total_files": total_files,
            "successful": success_count,
            "failed": failure_count,
            "success_rate_pct": round(success_rate, 2),
            "total_wall_time_sec": round(actual_wall_time, 2),
            "parallel_speedup": round(parallel_speedup, 2),
            "generated_at": datetime.now().isoformat()
        },
        "files": file_records
    }

    # ---------------------------------------------------------
    # 2. SAVE RAW JSON TO DISK
    # ---------------------------------------------------------
    try:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline_run_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Pipeline run data saved to: {output_json_path}")
    except OSError as exc:
        logger.error(f"Error writing pipeline run data: {exc}")

    # ---------------------------------------------------------
    # 3. BUILD TEXT REPORT FOR CONSOLE OUTPUT
    # ---------------------------------------------------------
    method_times: Dict[str, List[float]] = {}
    method_counts: Counter[str] = Counter()

    for r in successful_extractions:
        clean_method = _extract_ocr_method(r.get("method", "Unknown"))
        time_sec = float(r.get("processing_time_sec", 0.0))
        method_counts[clean_method] += 1
        method_times.setdefault(clean_method, []).append(time_sec)

    lines: List[str] = [
        "==================================================",
        "           FINAL OCR PIPELINE SUMMARY",
        "==================================================",
        f"Total Files Processed: {total_files}",
        f"Successful Extractions: {success_count} ({success_rate:.1f}%)",
        f"Failed Extractions: {failure_count}",
        f"Total Compute: {format_duration_detailed(total_processing_time)}",
        f"Total Elapsed: {format_duration_detailed(actual_wall_time)}",
        f"Parallel Speedup: {parallel_speedup:.2f}x",
        f"Avg Compute Time: {format_duration(avg_processing_time)} per file\n",
        "--- METHOD EFFICIENCY ---"
    ]

    for method in sorted(method_times.keys(), key=lambda m: sum(method_times[m]), reverse=True):
        count = method_counts[method]
        total_time = sum(method_times[method])
        avg_time = total_time / count if count > 0 else 0.0
        lines.append(f"• {method}: {count} uses | Avg: {format_duration(avg_time)}")

    lines.append("\n==================================================")
    lines.append(f"{'FILENAME': <25} | {'JOB NUMBER': <20} | {'TIME': <12} | ")
    lines.append("--------------------------------------------------")

    for r in sorted(successful_extractions, key=lambda x: x.get("filename", " ")):
        fname = Path(r.get("filename", " ")).stem[:23]
        job_num = r.get("job_number", "Unknown")
        time_sec = float(r.get("processing_time_sec", 0.0))
        lines.append(f"{fname: <25} | {job_num: <20} | {format_duration(time_sec): <12} | ")

    if failed_extractions:
        lines.append("\nFAILED FILES: ")
        for r in failed_extractions:
            fname = Path(r.get("filename", " ")).stem[:23]
            error = r.get("error", "Unknown error")
            lines.append(f"❌ {fname: <23} | {error: <20}")

    return "\n".join(lines)