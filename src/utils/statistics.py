# =============================================================================
# STATISTICS UTILITIES - OCR Batch Pipeline
# =============================================================================
"""
Statistics utilities for the OCR batch pipeline.

This module computes and saves performance statistics for the batch
processing job. It clearly separates:

- Compute time: sum of per-file processing times (CPU work)
- Wall clock time: real elapsed time for the whole pipeline

Use this to see your true parallel speedup and to identify bottlenecks.

Report Structure:
    reports/
    ├── latest/                                 → Symlink to most recent
    │   ├── statistics_report.txt
    │   └── smart_filing_summary.json
    └── archive/
        └── {batch_id}/                         → Historical batches
            ├── statistics_report.txt
            └── smart_filing_summary.json
"""
from __future__ import annotations

import os  # ← CRITICAL: Required for Windows junction check
import time
import json  # ← CRITICAL: Required for JSON export
import re  # ← CRITICAL: Required for method parsing
import shutil  # ← CRITICAL: Required for directory operations
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
# DATA EXTRACTION HELPERS (FOR JSON EXPORT)
# =============================================================================
def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get value from dict with default fallback."""
    return d.get(key, default) if d else default


def _extract_ocr_method(method_str: str) -> str:
    """Extract clean OCR method name from complex method string."""
    if not method_str:
        return "unknown"
    if "smart_filing:" in method_str:
        return method_str.split("smart_filing:")[0].strip()
    return method_str.strip()


def _extract_confidence(method_str: str, fallback: Optional[float] = None) -> float:
    """Extract confidence score from method string or use fallback."""
    if fallback is not None:
        return float(fallback)
    
    match = re.search(r"conf\s*([0-9.]+)", method_str or "")
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 0.0


def _extract_correction_reason(method_str: str) -> str:
    """Extract correction reason from method string."""
    if not method_str:
        return "kept: valid"
    if "smart_filing:" in method_str:
        reason_part = method_str.split("smart_filing:")[1].strip()
        if reason_part.endswith(")"):
            reason_part = reason_part[:-1]
        return reason_part
    return "kept: valid"


# =============================================================================
# ENHANCED JSON EXPORT (CRITICAL - WAS MISSING!)
# =============================================================================
def export_enhanced_json(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Export enhanced JSON summary with per-file metadata for verify.py.
    
    Creates smart_filing_summary.json with batch_metadata and files list.
    """
    if not results:
        logger.warning("No results provided for enhanced JSON export.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_records = []
    for res in results:
        raw_job = _safe_get(res, "raw_job") or _safe_get(res, "job_number", "")
        corrected_job = _safe_get(res, "corrected_job") or _safe_get(res, "job_number", "")
        
        confidence = _safe_get(res, "confidence")
        if confidence is None:
            confidence = _extract_confidence(_safe_get(res, "method", ""))
            
        reason = _safe_get(res, "reason")
        if not reason:
            reason = _extract_correction_reason(_safe_get(res, "method", ""))
            
        method = _safe_get(res, "ocr_method")
        if not method:
            method = _extract_ocr_method(_safe_get(res, "method", ""))
        
        file_records.append({
            "filename": _safe_get(res, "filename", ""),
            "raw_job": str(raw_job) if raw_job else "",
            "corrected_job": str(corrected_job) if corrected_job else "",
            "confidence": float(confidence),
            "reason": str(reason),
            "status": _safe_get(res, "status", "Unknown"),
            "processing_time_sec": float(_safe_get(res, "processing_time_sec", 0.0)),
            "method": str(method)
        })
    
    total_files = len(results)
    successful = sum(1 for r in results if _safe_get(r, "status") == "Success")
    failed = total_files - successful
    
    batch_metadata = {
        "total_files": total_files,
        "successful": successful,
        "failed": failed,
        "generated_at": datetime.now().isoformat()
    }
    
    enhanced_data = {
        "batch_metadata": batch_metadata,
        "files": file_records
    }
    
    json_path = output_dir / "smart_filing_summary.json"
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        logger.info("✅ Enhanced JSON summary saved to: %s", json_path)
    except OSError as exc:
        logger.error("Error writing enhanced JSON summary: %s", exc)
        raise


# =============================================================================
# BATCH ARCHIVE MANAGEMENT (CRITICAL - WAS MISSING!)
# =============================================================================
def create_batch_archive(output_path: Path, batch_id: Optional[str] = None) -> Path:
    """
    Create archived directory structure for batch reports.
    
    Creates reports/archive/{batch_id}/ and updates reports/latest/ symlink.
    """
    if batch_id is None:
        batch_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    archive_dir = output_path.parent / "archive" / batch_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    latest_dir = output_path.parent / "latest"
    
    # Remove existing latest
    if latest_dir.exists() or latest_dir.is_symlink():
        try:
            if latest_dir.is_symlink():
                latest_dir.unlink()
            else:
                shutil.rmtree(latest_dir)
        except OSError as e:
            logger.warning(f"Failed to remove existing 'latest': {e}")
    
    # Windows: Create directory junction instead of symlink (more reliable)
    try:
        if os.name == 'nt':  # Windows
            import subprocess
            subprocess.run(
                ['mklink', '/J', str(latest_dir), str(archive_dir)],
                check=True,
                shell=True
            )
            logger.info(f"📁 Created Windows junction: latest -> {archive_dir.name}")
        else:
            latest_dir.symlink_to(archive_dir)
    except Exception as e:
        # Fallback: Just copy files to latest/
        logger.warning(f"Symlink/junction failed, copying instead: {e}")
        try:
            if latest_dir.exists():
                shutil.rmtree(latest_dir)
            shutil.copytree(archive_dir, latest_dir)
            logger.info(f"📁 Copied reports to latest/")
        except Exception as copy_err:
            logger.error(f"Failed to create 'latest' reference: {copy_err}")
    
    return archive_dir


# =============================================================================
# STATISTICS GENERATOR
# =============================================================================
def generate_and_save_statistics(
    results: List[Dict[str, Any]],
    output_path: Path,
    elapsed_time: Optional[float] = None,
    batch_id: Optional[str] = None,
) -> None:
    """
    Analyze results and write a detailed statistics report to disk.
    Also exports a structured JSON summary for verification tools.
    """
    # Create archive directory if batch_id provided
    if batch_id:
        archive_dir = create_batch_archive(output_path, batch_id)
        output_path = archive_dir / output_path.name
        logger.info(f"📁 Archiving reports to: {archive_dir}")
    
    total_files = len(results)
    if total_files == 0:
        logger.warning("No results to analyze. Skipping statistics generation.")
        return

    successful_extractions = [r for r in results if r.get("status") == "Success"]
    failed_extractions = [r for r in results if r.get("status") != "Success"]

    success_count = len(successful_extractions)
    failure_count = len(failed_extractions)
    success_rate = (success_count / total_files) * 100 if total_files > 0 else 0.0

    total_processing_time = sum(
        float(r.get("processing_time_sec", 0.0)) for r in results
    )
    avg_processing_time = total_processing_time / total_files if total_files > 0 else 0.0

    if elapsed_time is not None:
        actual_wall_time = float(elapsed_time)
    else:
        actual_wall_time = float(total_processing_time)
        logger.warning(
            "No elapsed_time provided; using total_processing_time as proxy. "
            "This hides parallel speedup. Pass `elapsed_time` to fix this."
        )

    method_times: Dict[str, List[float]] = {}
    method_counts: Counter[str] = Counter()

    for r in successful_extractions:
        method = r.get("method", "Unknown")
        clean_method = _extract_ocr_method(method)
        time_sec = float(r.get("processing_time_sec", 0.0))
        method_counts[clean_method] += 1
        method_times.setdefault(clean_method, []).append(time_sec)

    slowest_files = sorted(
        results,
        key=lambda x: float(x.get("processing_time_sec", 0.0)),
        reverse=True,
    )[:5]

    lines: List[str] = []

    lines.append("==================================================")
    lines.append("           FINAL OCR PIPELINE SUMMARY")
    lines.append("==================================================")
    lines.append(f"Total Files Processed: {total_files}")
    lines.append(f"Successful Extractions: {success_count} ({success_rate:.1f}%)")
    lines.append(f"Failed Extractions: {failure_count}")

    lines.append(
        "Total Compute (Cumulative):  "
        f"{format_duration_detailed(total_processing_time)}  "
        f"({total_processing_time:.2f} seconds)"
    )
    lines.append(
        "Total Elapsed (Wall Clock):  "
        f"{format_duration_detailed(actual_wall_time)}  "
        f"({actual_wall_time:.2f} seconds)"
    )

    if actual_wall_time > 0:
        parallel_speedup = total_processing_time / actual_wall_time
    else:
        parallel_speedup = 1.0

    lines.append(f"Parallel Speedup Factor: {parallel_speedup:.2f}x")
    lines.append(
        f"Average Compute Time: {format_duration(avg_processing_time)} per file  "
        f"({avg_processing_time:.2f} seconds/file)"
    )

    lines.append("\n--- METHOD EFFICIENCY & BOTTLENECKS ---")
    if not method_counts:
        lines.append("No successful methods utilized.")
    else:
        sorted_methods = sorted(
            method_times.keys(),
            key=lambda m: sum(method_times[m]),
            reverse=True,
        )

        for method in sorted_methods:
            count = method_counts[method]
            total_method_time = sum(method_times[method])
            avg_time = total_method_time / count if count > 0 else 0.0
            pct_of_total_time = (
                (total_method_time / total_processing_time) * 100.0
                if total_processing_time > 0
                else 0.0
            )

            lines.append(f"• {method}")
            lines.append(f"   Usage: {count} times")
            lines.append(
                "   Avg Time:  "
                f"{format_duration(avg_time)} | Total Cost:  "
                f"{format_duration_detailed(total_method_time)}  "
                f"({pct_of_total_time:.1f}% of run)"
            )

    lines.append("\n--- TOP 5 SLOWEST FILES (ANOMALY DETECTOR) ---")
    for r in slowest_files:
        fname = Path(r.get("filename", " ")).stem
        time_sec = float(r.get("processing_time_sec", 0.0))
        status = "SUCCESS" if r.get("status") == "Success" else "FAILED"
        lines.append(f"• {fname} -> {format_duration_detailed(time_sec)} [{status}]")

    lines.append("\n==================================================")
    lines.append(f"{'FILENAME': <25} | {'JOB NUMBER': <20} | {'TIME': <12} | ")
    lines.append("--------------------------------------------------")

    sorted_success = sorted(
        successful_extractions,
        key=lambda x: x.get("filename", " "),
    )
    for r in sorted_success:
        fname = Path(r.get("filename", " ")).stem[:23]
        job_num = r.get("job_number", "Unknown")
        time_sec = float(r.get("processing_time_sec", 0.0))
        lines.append(
            f"{fname: <25} | {job_num: <20} | {format_duration(time_sec): <12} | "
        )

    lines.append("--------------------------------------------------")

    if failed_extractions:
        lines.append("\nFAILED FILES: ")
        for r in failed_extractions:
            fname = Path(r.get("filename", " ")).stem[:23]
            error = r.get("error", "Unknown error")
            time_sec = float(r.get("processing_time_sec", 0.0))
            lines.append(
                f"❌ {fname: <23} | {error: <20} | {format_duration(time_sec)}"
            )

    # ----------------------------------------------------------------------
    # HITL AUDIT FLAG (Human-in-the-loop)
    # ----------------------------------------------------------------------
    audit_files = []
    for r in sorted_success:
        method_str = r.get("method", " ")
        if "+smart_filing: Tier 3" in method_str or "+smart_filing: Tier 4" in method_str:
            try:
                reason = method_str.split("+smart_filing: ")[1].strip(") ")
                if " (" in reason:
                    reason = reason.split(" (")[0]

                fname = Path(r.get("filename", " ")).stem
                job_num = r.get("job_number", "Unknown")
                audit_files.append((fname, reason, job_num))
            except IndexError:
                pass
            
    if audit_files:
        lines.append("\n🚨 Audit Flag: Manual Review Recommended for Sequence Interpolation")
        for fname, reason, job_num in audit_files:
            lines.append(f"  • {fname}: {reason} ({job_num})")

    lines.append("\n==================================================")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("✅ Statistics report saved to: %s", output_path)
        
        # CRITICAL: Generate enhanced JSON summary for verify.py
        output_dir = output_path.parent
        export_enhanced_json(results, output_dir)
        
    except OSError as exc:
        logger.error("Error writing statistics report: %s", exc)
        raise