# =============================================================================
# OCR DASHBOARD BUILDER (verify.py)
# =============================================================================
#
# Usage Examples:
#   python verify.py                              # Verify latest run against default GT
#   python verify.py --batch batch_001            # Verify against ground_truth/batch_001.csv
#   python verify.py --trends                     # Analyze historical dashboard metrics
#
# Data Flow:
#   Input:  00-output/reports/latest_run_data.json (From statistics.py)
#   Output: 00-output/dashboard_data/{batch_id}_metrics.json
#
# =============================================================================

import re
import json
import argparse
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Any

# =============================================================================
# CONFIG IMPORTS & PATHS
# =============================================================================
from config import OUTPUT_DIR, REPORTS_DIR

DASHBOARD_DIR = OUTPUT_DIR / "dashboard_data"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

# Try to import validator for base comparison (fallback if not available)
try:
    from src.validation.validation import UnifiedValidator
    _validator = UnifiedValidator()
except ImportError:
    _validator = None

# =============================================================================
# GROUND TRUTH CONFIGURATION
# =============================================================================
DEFAULT_GROUND_TRUTH: Dict[str, str] = {
    "scan226": "250240-11", "scan227": "250240-11", "scan228": "250240-11",
    "scan229": "250240-11", "scan230": "250405-03", "scan231": "250405-03",
    "scan234": "250513-05", "scan235": "250513-05", "scan236": "250405-06",
    "scan237": "250405-06", "scan238": "250405-06", "scan239": "250405-06",
    # Note: Keep the rest of your default dictionary here!
}
GROUND_TRUTH_DIR = Path(__file__).resolve().parent / "ground_truth"


def normalize_text(text: str) -> str:
    """Remove spaces, parentheses, and asterisks for fair comparison."""
    if not text: return ""
    clean = re.sub(r"\(corrected:.*?\)", "", text)
    return re.sub(r"[)*\s]", "", clean).strip()


def load_ground_truth(batch_name: Optional[str] = None, explicit_path: Optional[str] = None) -> Dict[str, str]:
    """Loads ground truth from a specified CSV, batch name, or default."""
    if explicit_path:
        csv_path = Path(explicit_path)
    elif batch_name:
        csv_path = GROUND_TRUTH_DIR / f"{batch_name}.csv"
    else:
        return DEFAULT_GROUND_TRUTH.copy()
        
    gt = {}
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                sid, jnum = row.get('scan_id', '').strip(), row.get('job_number', '').strip()
                if sid and jnum:
                    gt[sid.replace('.pdf', '').replace('.PDF', '')] = jnum
        print(f"✅ Loaded {len(gt)} ground truth entries from {csv_path.name}")
        return gt
        
    print(f"⚠️ CSV not found: {csv_path}. Using default ground truth.")
    return DEFAULT_GROUND_TRUTH.copy()


# =============================================================================
# ERROR CATEGORIZATION
# =============================================================================
def _hamming_distance(s1: str, s2: str) -> int:
    if len(s1) != len(s2): return float('inf')
    return sum(a != b for a, b in zip(s1, s2))

def _categorize_error(expected: str, actual: str) -> str:
    if not actual or actual in {"failed", "None"}:
        return "extraction_failure"
    
    exp_norm, act_norm = normalize_text(expected), normalize_text(actual)
    if exp_norm == act_norm: return "match"
    
    exp_base = exp_norm.split("-")[0] if "-" in exp_norm else exp_norm
    act_base = act_norm.split("-")[0] if "-" in act_norm else act_norm
    
    if _validator:
        exp_base = _validator.get_base(expected) or exp_base
        act_base = _validator.get_base(actual) or act_base
    
    if _hamming_distance(exp_base, act_base) == 1: return "single_char_typo"
    if exp_base != act_base: return "base_mismatch"
    return "suffix_mismatch"


# =============================================================================
# DASHBOARD METRICS GENERATOR
# =============================================================================
def build_dashboard_metrics(ground_truth: Dict[str, str]) -> None:
    """Reads raw pipeline data, verifies it, and outputs dashboard metrics."""
    raw_data_path = REPORTS_DIR / "latest_run_data.json"
    
    if not raw_data_path.exists():
        print(f"❌ Error: Raw data not found at {raw_data_path}. Run main.py first.")
        return

    print(f"📂 Reading raw pipeline data: {raw_data_path.name}")
    
    with open(raw_data_path, 'r', encoding="utf-8") as f:
        raw_data = json.load(f)

    batch_meta = raw_data.get("batch_metadata", {})
    batch_id = batch_meta.get("batch_id", "unknown_batch")
    files_data = raw_data.get("files", [])

    # Tracking Variables
    passed_count = 0
    verification_errors = 0
    extraction_failures = 0
    total_verified = 0

    confidences = []
    processing_times = []
    tier_counts = Counter()
    error_types = Counter()
    
    failed_files_list = []
    extraction_failures_list = []
    per_file_results = []

    print("-" * 75)
    print(f"{'FILE':<10} | {'EXPECTED':<15} | {'ACTUAL':<20} | {'STATUS'}")
    print("-" * 75)

    # Process each file from the run
    for record in files_data:
        filename = record.get("filename", "")
        # Extract core scan ID (e.g., 'scan226_ready.jpg' -> 'scan226')
        scan_match = re.search(r"(scan\d+)", filename, re.IGNORECASE)
        scan_id = scan_match.group(1).lower() if scan_match else filename.split('.')[0]

        actual = record.get("corrected_job") or record.get("raw_job", "")
        status = record.get("status", "")
        conf = float(record.get("confidence", 0.0))
        tier = record.get("reason", "unknown")
        proc_time = float(record.get("processing_time_sec", 0.0))
        
        confidences.append(conf)
        processing_times.append(proc_time)

        # Handle Extraction Failures First
        if status != "Success" or actual == "failed":
            extraction_failures += 1
            extraction_failures_list.append({"filename": scan_id, "reason": record.get("error", "no_correction_found")})
            print(f"{scan_id:<10} | {'N/A':<15} | {'FAILED':<20} | ❌ EXTR_FAIL")
            continue

        # If it succeeded, verify it against ground truth
        if scan_id in ground_truth:
            total_verified += 1
            expected = ground_truth[scan_id]
            norm_actual = normalize_text(actual)
            norm_expected = normalize_text(expected)
            
            is_match = (norm_actual == norm_expected)
            tier_counts[tier] += 1
            
            file_result = {
                "filename": scan_id,
                "expected": expected,
                "actual": actual,
                "passed": is_match,
                "confidence": conf,
                "tier": tier,
                "processing_time_sec": proc_time
            }

            if is_match:
                passed_count += 1
                print(f"{scan_id:<10} | {expected:<15} | {actual:<20} | ✅ PASS")
            else:
                verification_errors += 1
                err_type = _categorize_error(expected, actual)
                error_types[err_type] += 1
                
                failed_files_list.append({
                    "filename": scan_id,
                    "expected": expected,
                    "actual": actual,
                    "error_type": err_type
                })
                print(f"{scan_id:<10} | {expected:<15} | {actual:<20} | ❌ FAIL")

            per_file_results.append(file_result)

    print("-" * 75)

    # Calculate final accuracy securely
    accuracy = (passed_count / total_verified) if total_verified > 0 else 0.0
    total_errors = verification_errors + extraction_failures

    print(f"Total Processed: {len(files_data)} | Verified: {total_verified}")
    print(f"Correct: {passed_count} | Verification Errors: {verification_errors} | Extr Fails: {extraction_failures}")
    if total_verified > 0:
        print(f"Accuracy (Verified Set): {accuracy * 100:.2f}%\n")

    # ---------------------------------------------------------
    # CONSTRUCT DASHBOARD JSON
    # ---------------------------------------------------------
    dashboard_json = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_in_batch": len(files_data),
            "total_verified": total_verified,
            "passed": passed_count,
            "verification_errors": verification_errors,
            "extraction_failures": extraction_failures,
            "total_errors": total_errors,
            "accuracy": round(accuracy, 4)
        },
        "tier_breakdown": dict(tier_counts),
        "error_analysis": dict(error_types),
        "confidence_stats": {
            "mean": round(sum(confidences)/len(confidences), 3) if confidences else 0,
            "min": round(min(confidences), 3) if confidences else 0,
            "max": round(max(confidences), 3) if confidences else 0,
            "low_confidence_count": sum(1 for c in confidences if c < 0.70)
        },
        "batch_info": {
            "batch_id": batch_id,
            "ground_truth_count": len(ground_truth),
            "avg_processing_time_sec": round(sum(processing_times)/len(processing_times), 2) if processing_times else 0.0,
            "extraction_failure_count": extraction_failures,
            "extraction_failure_files": [f["filename"] for f in extraction_failures_list]
        },
        "data_source": {
            "raw_data_path": str(raw_data_path)
        },
        "failed_files": failed_files_list,
        "extraction_failures": extraction_failures_list,
        "per_file_results": per_file_results
    }

    # ---------------------------------------------------------
    # EXPORT TO DASHBOARD DATA LAKE
    # ---------------------------------------------------------
    output_path = DASHBOARD_DIR / f"{batch_id}_metrics.json"
    latest_path = DASHBOARD_DIR / "latest_metrics.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard_json, f, indent=2, ensure_ascii=False)
    
    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard_json, f, indent=2, ensure_ascii=False)

    print(f"✅ Dashboard metrics saved to: {output_path.name}")


# =============================================================================
# TREND ANALYSIS (Scans the Dashboard Data Lake)
# =============================================================================
def analyze_trends() -> None:
    """Analyze accuracy trends from historical dashboard metrics."""
    json_files = sorted(DASHBOARD_DIR.glob("*_metrics.json"))
    # Filter out 'latest_metrics.json' to avoid duplicate counting
    json_files = [f for f in json_files if "latest" not in f.name]
    
    if not json_files:
        print(f"⚠️  No history files found in {DASHBOARD_DIR}")
        return
        
    print("=" * 75)
    print("📈 DASHBOARD TRENDS ANALYSIS".center(75))
    print("=" * 75)
    
    print(f"{'REPORT (BATCH ID)':<30} | {'ACCURACY':<10} | {'PASSED':<8} | {'TOTAL VERIFIED'}")
    print("-" * 75)
    
    accuracies = []
    for j_file in json_files:
        try:
            with open(j_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            summary = data.get("summary", {})
            acc = summary.get("accuracy", 0.0)
            passed = summary.get("passed", 0)
            verified = summary.get("total_verified", 0)
            
            accuracies.append(acc)
            
            acc_str = f"{acc * 100:.2f}%"
            print(f"{j_file.stem.replace('_metrics', ''):<30} | {acc_str:<10} | {passed:<8} | {verified:<8}")
        except Exception:
            pass
            
    print("-" * 75)
    if accuracies:
        avg = sum(accuracies) / len(accuracies)
        print(f"📊 Average System Accuracy (All History): {avg * 100:.2f}%")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Dashboard Metrics Builder")
    parser.add_argument("--trends", action="store_true", help="Analyze accuracy trends from dashboard data")
    parser.add_argument("--batch", type=str, default=None, help="Batch name for ground truth CSV (e.g., 'batch_001')")
    parser.add_argument("--ground-truth", type=str, default=None, help="Direct path to custom ground truth CSV")
    
    args = parser.parse_args()
    
    if args.trends:
        analyze_trends()
    else:
        gt = load_ground_truth(batch_name=args.batch, explicit_path=args.ground_truth)
        build_dashboard_metrics(gt)