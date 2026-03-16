# =============================================================================
# OCR VERIFICATION SCRIPT - With JSON Export & Trend Tracking
# =============================================================================
#
# Usage Examples:
#   python verify.py                              # Basic verification
#   python verify.py --enhanced                   # Advanced metrics
#   python verify.py --enhanced --export-auto     # Auto-export JSON report
#   python verify.py --trends                     # Analyze historical trends
#   python verify.py --batch batch_001            # Use specific ground truth CSV
#
# Report Structure:
#   reports/
#   ├── latest/                                   → Symlink to most recent batch
#   │   ├── statistics_report.txt
#   │   └── smart_filing_summary.json
#   └── archive/
#       └── 2026-03-16_14-30-00_batch_001/        → Archived batch reports
#
# =============================================================================

import re
import sys
import json
import argparse
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Set, Any

from config import REPORTS_DIR

# =============================================================================
# VALIDATOR IMPORT (For Hamming Distance Analysis)
# =============================================================================
try:
    from src.validation.validation import UnifiedValidator
    _validator = UnifiedValidator()
except ImportError:
    _validator = None

# =============================================================================
# GROUND TRUTH CONFIGURATION
# =============================================================================
# Default fallback ground truth (backward compatible if no CSV provided)
DEFAULT_GROUND_TRUTH: Dict[str, str] = {
    "scan226": "250240-11",
    "scan227": "250240-11",
    "scan228": "250240-11",
    "scan229": "250240-11",
    "scan230": "250405-03",
    "scan231": "250405-03",
    "scan232": "250405-03",
    "scan233": "250405-03",
    "scan234": "250513-05",
    "scan235": "250513-05",
    "scan236": "250405-06",
    "scan237": "250405-06",
    "scan238": "250405-06",
    "scan239": "250405-06",
    "scan240": "250405-04",
    "scan241": "250405-04",
    "scan242": "250405-04",
    "scan243": "250405-07",
    "scan244": "250405-07",
    "scan245": "250405-07",
    "scan246": "250405-07",
    "scan247": "250405-07",
    "scan248": "250405-05",
    "scan249": "250405-05",
    "scan250": "250405-05",
    "scan251": "250405-05",
    "scan252": "250405-05",
    "scan253": "250292-03",
    "scan254": "250292-03",
    "scan255": "240216-21",
    "scan256": "250312-01",
    "scan257": "250312-01",
    "scan258": "250312-01",
    "scan259": "250312-01",
    "scan260": "250312-01",
    "scan261": "250312-01",
    "scan262": "250240-14",
    "scan263": "250240-14",
    "scan264": "250240-14",
    "scan265": "250240-14",
    "scan266": "250240-14",
    "scan267": "250312-03",
    "scan268": "250312-03",
    "scan269": "250312-03",
    "scan270": "250312-03",
    "scan271": "250312-03",
    "scan272": "250312-03",
    "scan273": "250555-01",
    "scan274": "250555-01",
    "scan275": "260064-01",
    "scan276": "260064-01",
    "scan277": "260064-01",
    "scan278": "260064-01",
    "scan279": "260064-01",
    "scan280": "260064-01",
    "scan281": "260064-01",
    "scan282": "260064-01",
    "scan283": "260064-01",
    "scan284": "250294-08",
    "scan285": "250294-08",
    "scan286": "250436-05",
    "scan287": "250436-05",
    "scan288": "250436-05",
    "scan289": "250436-05",
    "scan290": "250408-04",
    "scan291": "250408-04",
    "scan292": "250408-04",
    "scan293": "250408-04",
    "scan294": "250481-01",
    "scan295": "250481-01",
    "scan296": "250481-01",
    "scan297": "250481-01",
    "scan298": "250408-14",
    "scan299": "250408-14",
    "scan300": "250411-18",
    "scan301": "250411-18",
    "scan302": "250190-02",
    "scan303": "250190-02",
    "scan304": "250190-02",
    "scan305": "250408-12",
    "scan306": "250408-12",
    "scan307": "250408-12",
    "scan308": "250408-12",
    "scan309": "250408-13",
    "scan310": "250408-13",
    "scan311": "250408-13",
    "scan312": "250408-01",
    "scan313": "250408-01",
    "scan314": "250408-01",
    "scan315": "250408-01",
    "scan316": "250408-01",
    "scan317": "250408-01",
    "scan318": "250408-01",
    "scan319": "250408-01",
    "scan320": "250408-01",
    "scan321": "250408-01",
    "scan322": "250408-01",
    "scan323": "250408-01",
    "scan324": "250408-01",
    "scan325": "250513-06",
    "scan326": "250513-06",
    "scan327": "250411-09",
    "scan328": "250411-09",
    "scan329": "250408-02",
    "scan330": "250408-02",
    "scan331": "250408-02",
    "scan332": "250408-02",
    "scan333": "250408-02",
    "scan334": "250408-05",
    "scan335": "250408-05",
    "scan336": "250408-05",
    "scan337": "250408-05",
    "scan338": "250408-05",
    "scan339": "250241-02",
    "scan340": "250241-02",
    "scan341": "250241-02",
    "scan342": "250408-05",
    "scan343": "250408-05",
    "scan344": "250408-05",
    "scan345": "250408-05",
    "scan346": "250408-05",
    "scan347": "250408-05",
    "scan348": "250310-05",
    "scan349": "250310-05",
    "scan350": "250310-05",
    "scan351": "250310-08",
    "scan352": "250310-08",
    "scan353": "250310-08",
    "scan354": "250241-01",
    "scan355": "250241-01",
    "scan356": "250585-01",
    "scan357": "250585-01",
    "scan358": "250585-01",
    "scan359": "250585-01",
    "scan360": "250585-01",
    "scan361": "250585-01",
    "scan362": "240297-10",
    "scan363": "240297-10",
    "scan364": "250295-06",
    "scan365": "250295-06",
    "scan366": "250408-03",
    "scan367": "250408-03",
    "scan368": "250408-03",
    "scan369": "250408-03",
    "scan370": "250408-03",
    "scan371": "250292-01",
    "scan372": "250292-01",
    "scan373": "250292-01",
    "scan374": "250292-01",
    "scan375": "250292-01",
    "scan376": "250292-01",
    "scan377": "250513-04",
    "scan378": "250513-04",
    "scan379": "250513-04",
    "scan380": "250513-04",
    "scan381": "250513-04",
    "scan382": "250513-04",
    "scan383": "250513-04",
    "scan384": "250513-04",
    "scan385": "250190-06",
    "scan386": "250190-06",
    "scan387": "250190-06",
    "scan388": "250513-03",
    "scan389": "250513-03",
    "scan390": "250513-03",
    "scan391": "250513-03",
    "scan392": "250513-03",
    "scan393": "250513-03",
    "scan394": "250513-03",
    "scan395": "250513-03",
    "scan396": "250513-03",
    "scan397": "250513-03",
    "scan398": "250513-03",
    "scan399": "250513-03",
    "scan400": "250335-31",
    "scan401": "250335-31",
    "scan402": "250335-31",
    "scan403": "250335-31",
    "scan404": "250335-31",
    "scan405": "250335-31"
}

# Ground truth CSV directory
GROUND_TRUTH_DIR = Path("./ground_truth")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def normalize_text(text: str) -> str:
    """
    Remove spaces, parentheses, and asterisks for fair comparison.
    
    Args:
        text: Raw text to normalize
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    clean = re.sub(r"\(corrected:.*?\)", "", text)
    clean = re.sub(r"[)*\s]", "", clean)
    return clean.strip()


def load_ground_truth_from_csv(csv_path: Path) -> Dict[str, str]:
    """
    Load ground truth data from CSV file.
    
    CSV Format:
        scan_id,job_number,notes
        scan226,250240-11,Optional notes
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Dict mapping scan_id → job_number
    """
    ground_truth: Dict[str, str] = {}
    
    if not csv_path.exists():
        print(f"⚠️  Ground truth CSV not found: {csv_path}")
        return {}
    
    print(f"📂 Loading ground truth from: {csv_path}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                scan_id = row.get('scan_id', '').strip()
                job_number = row.get('job_number', '').strip()
                
                if scan_id and job_number:
                    # Normalize scan_id (remove .pdf extension if present)
                    scan_id = scan_id.replace('.pdf', '').replace('.PDF', '')
                    ground_truth[scan_id] = job_number
        
        print(f"✅ Loaded {len(ground_truth)} ground truth entries")
        return ground_truth
    
    except Exception as e:
        print(f"❌ Error loading ground truth CSV: {e}")
        return {}


def load_ground_truth(batch_name: Optional[str] = None) -> Dict[str, str]:
    """
    Load ground truth from CSV or use default fallback.
    
    Args:
        batch_name: Optional batch name (e.g., "batch_001")
                   If provided, loads from ground_truth/{batch_name}.csv
                   If None, uses DEFAULT_GROUND_TRUTH
    
    Returns:
        Dict mapping scan_id → job_number
    """
    if batch_name:
        csv_path = GROUND_TRUTH_DIR / f"{batch_name}.csv"
        gt = load_ground_truth_from_csv(csv_path)
        
        if gt:
            return gt
        else:
            print(f"⚠️  Falling back to default ground truth")
    
    return DEFAULT_GROUND_TRUTH.copy()


def parse_report_file(report_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Parse the statistics report text file.
    Extracts filename → job_number mapping from the table.
    
    Args:
        report_path: Path to statistics_report.txt (auto-finds latest if None)
        
    Returns:
        Dict mapping filename (without .pdf) → job number string
    """
    if report_path is None:
        report_path = find_latest_report()
    
    extracted_data: Dict[str, str] = {}
    
    if not Path(report_path).exists():
        print(f"❌ Error: Source file '{report_path}' not found.")
        return {}

    print(f"📂 Reading source: {report_path}")
    
    with open(report_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    # Find the start index of the LATEST table to ignore old appended runs
    table_start_idx = 0
    for i, line in enumerate(lines):
        if "FILENAME" in line and "JOB NUMBER" in line:
            table_start_idx = i

    # Parse only from the latest table header downwards
    for line in lines[table_start_idx + 1:]:
        # HARD STOP: Break the loop the moment the table formatting ends
        if "===" in line or "FAILED FILES" in line:
            break
            
        if "|" in line and "---" not in line:
            parts = line.split("|")
            if len(parts) >= 2:
                fname = parts[0].strip()
                job_num = parts[1].strip()
                if fname:
                    key = fname.split('_')[0]
                    extracted_data[key] = job_num

    return extracted_data


def parse_smart_filing_json(json_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Parse the enhanced JSON summary from smart_filing.
    
    Search order:
    1. reports/latest/smart_filing_summary.json
    2. reports/archive/{latest_batch}/smart_filing_summary.json
    """
    if json_path is None:
        # Try 1: latest/ directory
        json_path = REPORTS_DIR / "latest" / "smart_filing_summary.json"
        
        # Try 2: archive/ (most recent batch with JSON)
        if not json_path.exists():
            archive_dir = REPORTS_DIR / "archive"
            if archive_dir.exists():
                batch_folders = sorted(
                    [d for d in archive_dir.iterdir() if d.is_dir()],
                    key=lambda x: x.name,
                    reverse=True
                )
                for batch_folder in batch_folders:
                    json_path = batch_folder / "smart_filing_summary.json"
                    if json_path.exists():
                        break
    
    if not Path(json_path).exists():
        print(f"⚠️  Enhanced JSON not found: {json_path}")
        print("   Tier/Confidence metrics will use fallback values.")
        return None
    
    print(f"📂 Reading enhanced metadata: {json_path}")
    
    try:
        with open(json_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        
        file_map: Dict[str, Dict[str, Any]] = {}
        for record in data.get("files", []):
            fname = record.get("filename", "")
            key_match = re.search(r"scan(\d+)", fname, re.IGNORECASE)
            if key_match:
                key = f"scan{key_match.group(1)}"
                file_map[key] = record
        
        return {
            "batch_metadata": data.get("batch_metadata", {}),
            "files": data.get("files", []),
            "file_map": file_map
        }
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️  Warning: Could not parse JSON: {e}")
        return None


def find_latest_report() -> Path:
    """
    Find the most recent archived statistics report.
    
    Search order:
    1. reports/archive/{latest_batch}/statistics_report.txt
    2. reports/latest/statistics_report.txt
    
    Returns:
        Path to the most recent statistics report
    """
    # Try 1: Check latest/ symlink/junction first
    latest_path = REPORTS_DIR / "latest" / "statistics_report.txt"
    if latest_path.exists():
        print(f"📂 Found report in latest/: {latest_path}")
        return latest_path
    
    # Try 2: Search archive/ for most recent batch with report
    archive_dir = REPORTS_DIR / "archive"
    if archive_dir.exists():
        # Get all batch folders, sorted by name (timestamp), newest first
        batch_folders = sorted(
            [d for d in archive_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True
        )
        
        for batch_folder in batch_folders:
            report_path = batch_folder / "statistics_report.txt"
            if report_path.exists():
                print(f"📂 Found report in archive: {batch_folder.name}")
                return report_path
    
    # Fallback: Return latest path even if it doesn't exist (will show error)
    return REPORTS_DIR / "latest" / "statistics_report.txt"


def _hamming_distance(s1: str, s2: str) -> int:
    """
    Calculate Hamming distance between two strings of equal length.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Number of character differences, or infinity if lengths differ
    """
    if len(s1) != len(s2):
        return float('inf')
    return sum(a != b for a, b in zip(s1, s2))


def _categorize_error(expected: str, actual: str) -> str:
    """
    Categorize error type using physics-based OCR error model.
    
    Error categories:
    - match: Strings are identical
    - extraction_failure: No result or failure marker
    - single_char_typo: One character difference (faded ink, sensor noise)
    - base_mismatch: Different base number (batch change or major failure)
    - suffix_mismatch: Same base, different suffix
    
    Args:
        expected: Expected job number
        actual: Actual extracted job number
        
    Returns:
        Error category string
    """
    if not actual or actual in {"", "failed", "all_methods_failed", "None", "no_correction_found"}:
        return "extraction_failure"
    
    # Normalize for comparison
    exp_norm = normalize_text(expected)
    act_norm = normalize_text(actual)
    
    if exp_norm == act_norm:
        return "match"
    
    # Extract bases for comparison
    exp_base = exp_norm.split("-")[0] if "-" in exp_norm else exp_norm
    act_base = act_norm.split("-")[0] if "-" in act_norm else act_norm
    
    if _validator:
        exp_base = _validator.get_base(expected) or exp_base
        act_base = _validator.get_base(actual) or act_base
    
    # Single-char mutation = likely OCR typo (faded ink, sensor noise)
    if _hamming_distance(exp_base, act_base) == 1:
        return "single_char_typo"
    
    # Different base = likely batch change or major OCR failure
    if exp_base != act_base:
        return "base_mismatch"
    
    # Same base, different suffix = suffix parsing issue
    return "suffix_mismatch"


# =============================================================================
# CORE VERIFICATION FUNCTIONS
# =============================================================================
def run_check(ground_truth_override: Optional[Dict[str, str]] = None) -> None:
    """
    Run basic verification against ground truth.
    
    Args:
        ground_truth_override: Optional dict to override default ground truth
    """
    gt = ground_truth_override if ground_truth_override is not None else DEFAULT_GROUND_TRUTH
    actual_results = parse_report_file()
    
    if not actual_results:
        print("⚠️  No data found in report. Exiting.")
        return

    print("-" * 75)
    print(f"{'FILE':<10} | {'EXPECTED':<15} | {'ACTUAL (FROM REPORT)':<25} | {'STATUS':<10}")
    print("-" * 75)

    correct_count = 0
    total_checks = 0
    errors = []

    for key, expected_raw in gt.items():
        if key not in actual_results:
            continue

        total_checks += 1
        expected = normalize_text(expected_raw)
        actual_raw = actual_results[key]
        actual = normalize_text(actual_raw)

        is_match = (expected == actual)
        
        status = "✅ PASS" if is_match else "❌ FAIL"
        if is_match:
            correct_count += 1
        else:
            errors.append((key, expected, actual_raw))

        print(f"{key:<10} | {expected:<15} | {actual_raw:<25} | {status}")

    print("-" * 75)
    print("VERIFICATION COMPLETE")
    print(f"Total Verified: {total_checks}")
    print(f"Correct:        {correct_count}")
    print(f"Errors:         {len(errors)}")
    
    if total_checks > 0:
        acc = (correct_count / total_checks) * 100
        print(f"Accuracy:       {acc:.2f}%")

    if errors:
        print("\n🚨 DETAILED ERRORS:")
        for key, exp, act in errors:
            print(f"  • {key}: Expected '{exp}' but got '{act}'")


def run_check_enhanced(
    report_path: Optional[Path] = None,
    export_json: Optional[str] = None,
    ground_truth_override: Optional[Dict[str, str]] = None,
    batch_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Enhanced verification with advanced metrics and JSON export.
    
    Features:
    - Loads confidence/tier data from smart_filing_summary.json
    - Supports CSV-based ground truth
    - Exports detailed JSON report for CI/CD
    
    Args:
        report_path: Path to statistics_report.txt (auto-finds latest if None)
        export_json: Optional path to export detailed JSON report
        ground_truth_override: Optional dict to override ground truth
        batch_name: Optional batch name for CSV ground truth loading
        
    Returns:
        Dict with verification results, or None if failed
    """
    # Load ground truth
    if ground_truth_override:
        gt = ground_truth_override
    elif batch_name:
        gt = load_ground_truth(batch_name)
    else:
        gt = DEFAULT_GROUND_TRUTH.copy()
    
    # Auto-find latest report if not specified
    if report_path is None:
        report_path = find_latest_report()
    
    actual_results = parse_report_file(report_path)
    
    # Load enhanced JSON metadata for Tier/Confidence
    smart_filing_data = parse_smart_filing_json()
    
    if not actual_results:
        print("⚠️  No data found in report. Exiting.")
        return None

    # Data collection for metrics
    results: List[Dict[str, Any]] = []
    tier_counts: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    confidences: List[float] = []
    
    # Track whether we're using enhanced data
    using_enhanced_data = smart_filing_data is not None

    print("-" * 75)
    print(f"{'FILE':<10} | {'EXPECTED':<15} | {'ACTUAL (FROM REPORT)':<25} | {'STATUS':<10}")
    print("-" * 75)

    correct_count = 0
    total_checks = 0
    errors = []

    for key, expected_raw in gt.items():
        if key not in actual_results:
            continue

        total_checks += 1
        expected = normalize_text(expected_raw)
        actual_raw = actual_results[key]
        actual = normalize_text(actual_raw)

        is_match = (expected == actual)
        
        # Get confidence and tier from enhanced JSON if available
        confidence = 0.0
        tier = "unknown (check smart_filing logs)"
        
        if using_enhanced_data and smart_filing_data:
            file_record = smart_filing_data["file_map"].get(key)
            if file_record:
                confidence = float(file_record.get("confidence", 0.0))
                tier = file_record.get("reason", "kept: valid")
        
        confidences.append(confidence)
        tier_counts[tier] += 1
        
        status = "✅ PASS" if is_match else "❌ FAIL"
        if is_match:
            correct_count += 1
        else:
            errors.append((key, expected, actual_raw))
            error_types[_categorize_error(expected_raw, actual_raw)] += 1

        results.append({
            "filename": key,
            "expected": expected,
            "actual": actual_raw,
            "passed": is_match,
            "confidence": confidence,
            "tier": tier
        })
        
        print(f"{key:<10} | {expected:<15} | {actual_raw:<25} | {status}")

    print("-" * 75)
    print("VERIFICATION COMPLETE")
    print(f"Total Verified: {total_checks}")
    print(f"Correct:        {correct_count}")
    print(f"Errors:         {len(errors)}")
    
    if total_checks > 0:
        acc = (correct_count / total_checks) * 100
        print(f"Accuracy:       {acc:.2f}%")

    # =========================================================
    # Advanced Metrics Section
    # =========================================================
    if total_checks > 0:
        print("\n📊 ADVANCED METRICS:")
        
        # Tier breakdown
        print("\n🔧 Tier Breakdown:")
        for tier, count in sorted(tier_counts.items(), key=lambda x: -x[1]):
            pct = count / total_checks * 100
            print(f"   • {tier}: {count} files ({pct:.1f}%)")
        
        # Error analysis
        if error_types:
            print("\n🚨 Error Analysis:")
            for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
                pct = count / len(errors) * 100 if errors else 0
                print(f"   • {error_type}: {count} files ({pct:.1f}%)")
        
        # Confidence stats
        print(f"\n📈 Confidence Stats:")
        if confidences:
            mean_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            low_conf = sum(1 for c in confidences if c < 0.70)
            print(f"   • Mean: {mean_conf:.3f}")
            print(f"   • Range: [{min_conf:.3f}, {max_conf:.3f}]")
            print(f"   • Low confidence (<0.70): {low_conf} files ({low_conf/len(confidences)*100:.1f}%)")
        else:
            print(f"   • No confidence data available")
        
        # Data source indicator
        if using_enhanced_data:
            print(f"\n✅ Using enhanced metadata from smart_filing_summary.json")
        else:
            print(f"\n⚠️  Enhanced metadata not available (using fallback values)")
    
    # Detailed errors
    if errors:
        print("\n🚨 DETAILED ERRORS:")
        for key, exp, act in errors:
            error_cat = _categorize_error(exp, act)
            print(f"  • {key}: Expected '{exp}' but got '{act}' [{error_cat}]")

    # Optional JSON export
    if export_json:
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total_checks,
                "passed": correct_count,
                "failed": len(errors),
                "accuracy": round(correct_count / total_checks, 4) if total_checks > 0 else 0
            },
            "tier_breakdown": dict(tier_counts),
            "error_analysis": dict(error_types),
            "confidence_stats": {
                "mean": round(sum(confidences)/len(confidences), 3) if confidences else 0,
                "min": round(min(confidences), 3) if confidences else 0,
                "max": round(max(confidences), 3) if confidences else 0,
                "low_confidence_count": sum(1 for c in confidences if c < 0.70) if confidences else 0
            },
            "data_source": {
                "enhanced_metadata": using_enhanced_data,
                "json_path": str(REPORTS_DIR / "latest" / "smart_filing_summary.json") if using_enhanced_data else None
            },
            "failed_files": [
                {"filename": k, "expected": e, "actual": a, "error_type": _categorize_error(e, a)}
                for k, e, a in errors
            ],
            "per_file_results": results
        }
        
        Path(export_json).parent.mkdir(parents=True, exist_ok=True)
        with open(export_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Report exported to: {export_json}")

    return {
        "accuracy": correct_count / total_checks if total_checks > 0 else 0,
        "total": total_checks,
        "passed": correct_count,
        "errors": errors
    }


# =============================================================================
# TREND ANALYSIS (Track Accuracy Over Time)
# =============================================================================
def analyze_trends(reports_dir: Optional[str] = None, pattern: str = "*_batch_metrics.json") -> Optional[List[Dict[str, Any]]]:
    """
    Analyze accuracy trends from all historical JSON reports.
    
    Uses sequence mathematics to identify patterns:
    - Accuracy trend (improving/declining/stable)
    - Error type distribution over time
    - Confidence correlation with accuracy
    
    Args:
        reports_dir: Directory containing JSON reports (default: ./reports/)
        pattern: Glob pattern for JSON files
        
    Returns:
        List of trend data dictionaries, or None if no reports found
    """
    if reports_dir is None:
        reports_dir = "./reports"
    
    reports_path = Path(reports_dir)
    
    if not reports_path.exists():
        print(f"❌ Reports directory not found: {reports_path}")
        return None
    
    # Find all JSON reports using pattern matching
    json_files = sorted(reports_path.glob(pattern))
    
    if not json_files:
        print(f"⚠️  No JSON reports found matching '{pattern}' in {reports_path}")
        return None
    
    print("=" * 75)
    print("📈 ACCURACY TRENDS ANALYSIS".center(75))
    print("=" * 75)
    print(f"Reports analyzed: {len(json_files)}")
    print("-" * 75)
    
    # Collect data from all reports
    trends: List[Dict[str, Any]] = []
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            trends.append({
                "file": json_file.name,
                "timestamp": data.get("timestamp", "unknown"),
                "accuracy": data.get("summary", {}).get("accuracy", 0),
                "total": data.get("summary", {}).get("total", 0),
                "passed": data.get("summary", {}).get("passed", 0),
                "failed": data.get("summary", {}).get("failed", 0),
                "error_analysis": data.get("error_analysis", {}),
                "confidence_stats": data.get("confidence_stats", {}),
                "tier_breakdown": data.get("tier_breakdown", {})
            })
        except Exception as e:
            print(f"⚠️  Warning: Could not parse {json_file.name}: {e}")
    
    if not trends:
        return None
    
    # Display trend table
    print(f"{'REPORT':<35} | {'ACCURACY':<10} | {'PASSED':<8} | {'TOTAL':<8}")
    print("-" * 75)
    
    for t in trends:
        acc_pct = f"{t['accuracy'] * 100:.2f}%"
        print(f"{t['file']:<35} | {acc_pct:<10} | {t['passed']:<8} | {t['total']:<8}")
    
    print("-" * 75)
    
    # Calculate trend statistics using sequence mathematics
    accuracies = [t["accuracy"] for t in trends]
    avg_accuracy = sum(accuracies) / len(accuracies)
    min_accuracy = min(accuracies)
    max_accuracy = max(accuracies)
    
    # Determine trend direction (simple linear regression slope)
    if len(accuracies) >= 2:
        n = len(accuracies)
        x_mean = (n - 1) / 2
        y_mean = avg_accuracy
        
        numerator = sum((i - x_mean) * (acc - y_mean) for i, acc in enumerate(accuracies))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        if slope > 0.01:
            trend_direction = "📈 IMPROVING"
        elif slope < -0.01:
            trend_direction = "📉 DECLINING"
        else:
            trend_direction = "➡️ STABLE"
    else:
        trend_direction = "⚠️ INSUFFICIENT DATA"
        slope = 0
    
    # Summary statistics
    print("\n📊 TREND SUMMARY:")
    print(f"   • Average Accuracy: {avg_accuracy * 100:.2f}%")
    print(f"   • Range: [{min_accuracy * 100:.2f}%, {max_accuracy * 100:.2f}%]")
    print(f"   • Trend Direction: {trend_direction}")
    print(f"   • Reports Analyzed: {len(trends)}")
    
    # Error type evolution (if multiple reports)
    if len(trends) >= 2:
        print("\n🚨 ERROR TYPE EVOLUTION:")
        all_error_types = set()
        for t in trends:
            all_error_types.update(t["error_analysis"].keys())
        
        for error_type in sorted(all_error_types):
            counts = [t["error_analysis"].get(error_type, 0) for t in trends]
            print(f"   • {error_type}: [{', '.join(map(str, counts))}]")
    
    # Tier breakdown evolution (if multiple reports)
    if len(trends) >= 2:
        print("\n🔧 TIER USAGE EVOLUTION:")
        all_tiers = set()
        for t in trends:
            all_tiers.update(t["tier_breakdown"].keys())
        
        for tier in sorted(all_tiers):
            counts = [t["tier_breakdown"].get(tier, 0) for t in trends]
            if any(counts):
                print(f"   • {tier}: [{', '.join(map(str, counts))}]")
    
    # Confidence correlation (if available)
    confidences = [t["confidence_stats"].get("mean", 0) for t in trends if t["confidence_stats"]]
    if confidences and len(confidences) >= 2:
        print(f"\n📈 CONFIDENCE TREND:")
        avg_conf = sum(confidences) / len(confidences)
        print(f"   • Average Mean Confidence: {avg_conf:.3f}")
        print(f"   • Range: [{min(confidences):.3f}, {max(confidences):.3f}]")
    
    print("=" * 75)
    
    return trends


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OCR Verification Script for Job Number Extraction Pipeline"
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Enable advanced metrics (tier breakdown, confidence stats)"
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to statistics_report.txt (auto-finds latest if not specified)"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export detailed report to JSON file"
    )
    parser.add_argument(
        "--export-auto",
        action="store_true",
        help="Auto-export JSON with timestamp in filename"
    )
    parser.add_argument(
        "--trends",
        action="store_true",
        help="Analyze accuracy trends from all historical JSON reports"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="./reports",
        help="Directory containing JSON reports for trend analysis"
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Batch name for ground truth CSV (e.g., 'batch_001' loads ground_truth/batch_001.csv)"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Direct path to ground truth CSV file (overrides --batch)"
    )
    
    args = parser.parse_args()
    
    # Load ground truth based on arguments
    ground_truth = None
    if args.ground_truth:
        # Direct CSV path specified
        ground_truth = load_ground_truth_from_csv(Path(args.ground_truth))
        if not ground_truth:
            print("⚠️  No ground truth loaded, using default")
            ground_truth = DEFAULT_GROUND_TRUTH.copy()
    elif args.batch:
        # Batch name specified
        ground_truth = load_ground_truth(args.batch)
    
    # Handle trend analysis mode
    if args.trends:
        analyze_trends(reports_dir=args.reports_dir)
    
    # Handle enhanced mode with auto-timestamp export
    elif args.enhanced or args.export or args.export_auto:
        export_path = args.export
        
        # Auto-generate timestamp filename if --export-auto is used
        if args.export_auto and not export_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"./reports/{timestamp}_batch_metrics.json"
        elif args.export_auto and export_path:
            # Insert timestamp into provided path
            path = Path(export_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = str(path.parent / f"{timestamp}_{path.name}")
        
        run_check_enhanced(
            report_path=Path(args.report) if args.report else None,
            export_json=export_path,
            ground_truth_override=ground_truth,
            batch_name=args.batch
        )
    
    # Default: run original verification
    else:
        run_check(ground_truth_override=ground_truth)