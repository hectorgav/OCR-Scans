# =============================================================================
# OCR VERIFICATION SCRIPT - With Extraction Failure Tracking
# ==============================================================================
# 
# Export JSON for future CI/CD:
#   python verify.py --enhanced --export ./reports/batch_001_metrics.json
#
# Save each batch's JSON report with a timestamp (Windows):
#   python verify.py --enhanced --export-auto
#
# Analyze trends from all historical JSON reports:
#   python verify.py --trends
#
# ==============================================================================

import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

from config import REPORTS_DIR

# Import validator for Hamming distance analysis (adjust path as needed)
try:
    from src.validation.validation import UnifiedValidator
    _validator = UnifiedValidator()
except ImportError:
    _validator = None

# ==============================================================================
# 1. THE TRUTH TABLE (Difficult Cases Only)
# ==============================================================================
GROUND_TRUTH = {
    "scan326": "250209-01",
    "scan358": "1108490",
    "scan360": "240283-16",
    "scan366": "240275-02",
    "scan367": "240275-02",
    "scan380": "230531-13",
    "scan382": "230531-13",
    "scan383": "230531-13",
    "scan395": "240291-09",
    "scan410": "240517-06",
    "scan411": "240517-06",
    "scan412": "240517-06",
    "scan413": "250306-04",
    "scan414": "250306-04",
    "scan415": "250306-04",
    "scan416": "240384-03",
    "scan417": "240384-03",
    "scan418": "240384-03",
    "scan419": "240384-03",
    "scan420": "240384-03",
    "scan979": "250341-01",
    "scan981": "250341-02",
    "scan983": "250286-03",
    "scan989": "250284-04",
    "scan990": "250287-06",
    "scan991": "250287-06",
    "scan997": "240292-22"
}

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def normalize_text(text):
    """Removes spaces, parentheses, and asterisks for fair comparison."""
    if not text: 
        return ""
    clean = re.sub(r"\(corrected:.*?\)", "", text) 
    clean = re.sub(r"[)*\s]", "", clean)
    return clean.strip()


def parse_report_file(report_path=REPORTS_DIR/"statistics_report.txt"):
    """
    Parses strictly the LAST text table from the report file.
    Returns dict mapping filename (without .pdf) → job number string.
    """
    extracted_data = {}
    
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
    for line in lines[table_start_idx+1:]:
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


def parse_total_files_processed(report_path=REPORTS_DIR/"statistics_report.txt"):
    """
    Extract 'Total Files Processed' count from statistics_report.txt.
    """
    if not Path(report_path).exists():
        return None
    
    with open(report_path, 'r', encoding="utf-8") as f:
        for line in f:
            match = re.search(r"Total Files Processed:\s*(\d+)", line)
            if match:
                return int(match.group(1))
    
    return None


def parse_failed_files(report_path=REPORTS_DIR/"statistics_report.txt"):
    """
    Extract list of failed files from the FAILED FILES section.
    """
    failed_files = []
    
    if not Path(report_path).exists():
        return failed_files
    
    with open(report_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    
    in_failed_section = False
    for line in lines:
        if "FAILED FILES:" in line:
            in_failed_section = True
            continue
        
        if in_failed_section and ("===" in line or line.strip() == ""):
            break
        
        if in_failed_section and "❌" in line and "|" in line:
            parts = line.split("|")
            if len(parts) >= 2:
                fname = parts[0].strip().replace("❌", "").strip()
                if fname:
                    key = fname.split('_')[0]
                    failed_files.append(key)
    
    return failed_files


def parse_smart_filing_json(json_path=REPORTS_DIR/"smart_filing_summary.json"):
    """
    Parse the enhanced JSON summary from smart_filing.
    
    Returns:
        dict with:
            - 'batch_metadata': summary stats
            - 'files': list of per-file records with confidence, reason, etc.
            - 'file_map': dict mapping filename → record for quick lookup
    """
    if not Path(json_path).exists():
        print(f"⚠️  Enhanced JSON not found: {json_path}")
        print("   Tier/Confidence metrics will use fallback values.")
        return None
    
    print(f"📂 Reading enhanced metadata: {json_path}")
    
    try:
        with open(json_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        
        # Build quick lookup map
        file_map = {}
        for record in data.get("files", []):
            fname = record.get("filename", "")
            # Extract scan number for consistent key matching
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


def _hamming_distance(s1: str, s2: str) -> int:
    """Calculate Hamming distance between two strings of equal length."""
    if len(s1) != len(s2):
        return float('inf')
    return sum(a != b for a, b in zip(s1, s2))


def _categorize_error(expected: str, actual: str) -> str:
    """Categorize error type using physics-based OCR error model."""
    if not actual or actual in {"", "failed", "all_methods_failed", "None", "no_correction_found"}:
        return "extraction_failure"
    
    exp_norm = normalize_text(expected)
    act_norm = normalize_text(actual)
    
    if exp_norm == act_norm:
        return "match"
    
    exp_base = exp_norm.split("-")[0] if "-" in exp_norm else exp_norm
    act_base = act_norm.split("-")[0] if "-" in act_norm else act_norm
    
    if _validator:
        exp_base = _validator.get_base(expected) or exp_base
        act_base = _validator.get_base(actual) or act_base
    
    if _hamming_distance(exp_base, act_base) == 1:
        return "single_char_typo"
    
    if exp_base != act_base:
        return "base_mismatch"
    
    return "suffix_mismatch"


# ==============================================================================
# 3. CORE VERIFICATION FUNCTIONS
# ==============================================================================

def run_check():
    """Original verification function (backward compatible)."""
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

    for key, expected_raw in GROUND_TRUTH.items():
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
    report_path=None,
    export_json=None,
    ground_truth_override=None
):
    """
    Enhanced version with advanced metrics, extraction failure tracking, and JSON export.
    """
    if report_path is None:
        report_path = REPORTS_DIR / "statistics_report.txt"
    
    gt = ground_truth_override if ground_truth_override is not None else GROUND_TRUTH
    actual_results = parse_report_file(report_path)
    
    # Load enhanced JSON metadata for Tier/Confidence
    smart_filing_data = parse_smart_filing_json(REPORTS_DIR / "smart_filing_summary.json")
    
    if not actual_results:
        print("⚠️  No data found in report. Exiting.")
        return None

    total_files_in_batch = parse_total_files_processed(report_path)
    failed_extraction_files = parse_failed_files(report_path)
    
    if total_files_in_batch:
        print(f"📊 Total files in batch: {total_files_in_batch}")
    if failed_extraction_files:
        print(f"🚨 Extraction failures: {len(failed_extraction_files)} files")

    results = []
    tier_counts = Counter()
    error_types = Counter()
    confidences = []
    
    using_enhanced_data = smart_filing_data is not None

    print("-" * 75)
    print(f"{'FILE':<10} | {'EXPECTED':<15} | {'ACTUAL (FROM REPORT)':<25} | {'STATUS':<10}")
    print("-" * 75)

    correct_count = 0
    total_checks = 0
    verification_errors = []

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
            verification_errors.append((key, expected, actual_raw))
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

    extraction_failure_count = len(failed_extraction_files)
    total_errors = len(verification_errors) + extraction_failure_count
    
    if total_files_in_batch and total_files_in_batch > 0:
        accuracy = (total_files_in_batch - total_errors) / total_files_in_batch
    else:
        accuracy = correct_count / total_checks if total_checks > 0 else 0

    print("-" * 75)
    print("VERIFICATION COMPLETE")
    if total_files_in_batch:
        print(f"Total files in batch: {total_files_in_batch}")
    print(f"Total Verified: {total_checks}")
    print(f"Correct:        {correct_count}")
    print(f"Verification Errors: {len(verification_errors)}")
    if extraction_failure_count > 0:
        print(f"Extraction Failures: {extraction_failure_count}")
    print(f"Total Errors:   {total_errors}")
    
    if total_files_in_batch and total_files_in_batch > 0:
        print(f"Accuracy:       {accuracy * 100:.2f}% (based on total batch size)")
    elif total_checks > 0:
        print(f"Accuracy:       {accuracy * 100:.2f}% (based on verified files)")

    if total_checks > 0 or extraction_failure_count > 0:
        print("\n📊 ADVANCED METRICS:")
        
        print("\n🔧 Tier Breakdown:")
        for tier, count in sorted(tier_counts.items(), key=lambda x: -x[1]):
            pct = count / total_checks * 100 if total_checks > 0 else 0
            print(f"   • {tier}: {count} files ({pct:.1f}%)")
        
        if error_types or extraction_failure_count > 0:
            print("\n🚨 Error Analysis:")
            for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
                pct = count / total_errors * 100 if total_errors > 0 else 0
                print(f"   • {error_type}: {count} files ({pct:.1f}%)")
            if extraction_failure_count > 0:
                pct = extraction_failure_count / total_errors * 100 if total_errors > 0 else 0
                print(f"   • extraction_failure: {extraction_failure_count} files ({pct:.1f}%)")
        
        print(f"\n📈 Confidence Stats:")
        if confidences:
            mean_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            low_conf = sum(1 for c in confidences if c < 0.70)
            print(f"   • Mean: {mean_conf:.3f}")
            print(f"   • Range: [{min_conf:.3f}, {max_conf:.3f}]")
            print(f"   • Low confidence (<0.70): {low_conf} files")
        else:
            print(f"   • No confidence data available")
        
        if using_enhanced_data:
            print(f"\n✅ Using enhanced metadata from smart_filing_summary.json")
        else:
            print(f"\n⚠️  Enhanced metadata not available (using fallback values)")
        
        if total_files_in_batch:
            print(f"\n📦 Batch Statistics:")
            print(f"   • Total files in batch: {total_files_in_batch}")
            print(f"   • Files in ground truth: {len(gt)}")
            print(f"   • Verified: {total_checks} ({total_checks/len(gt)*100:.1f}% of ground truth)")
            print(f"   • Extraction failures: {extraction_failure_count} ({extraction_failure_count/total_files_in_batch*100:.1f}% of batch)")
    
    if verification_errors:
        print("\n🚨 VERIFICATION ERRORS:")
        for key, exp, act in verification_errors:
            error_cat = _categorize_error(exp, act)
            print(f"  • {key}: Expected '{exp}' but got '{act}' [{error_cat}]")
    
    if failed_extraction_files:
        print("\n🚨 EXTRACTION FAILURES:")
        for fname in failed_extraction_files:
            print(f"  • {fname}: no_correction_found")

    if export_json:
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_in_batch": total_files_in_batch,
                "total_verified": total_checks,
                "passed": correct_count,
                "verification_errors": len(verification_errors),
                "extraction_failures": extraction_failure_count,
                "total_errors": total_errors,
                "accuracy": round(accuracy, 4)
            },
            "tier_breakdown": dict(tier_counts),
            "error_analysis": dict(error_types),
            "confidence_stats": {
                "mean": round(sum(confidences)/len(confidences), 3) if confidences else 0,
                "min": round(min(confidences), 3) if confidences else 0,
                "max": round(max(confidences), 3) if confidences else 0,
                "low_confidence_count": sum(1 for c in confidences if c < 0.70) if confidences else 0
            },
            "batch_info": {
                "total_files_processed": total_files_in_batch,
                "ground_truth_count": len(gt),
                "verified_count": total_checks,
                "extraction_failure_count": extraction_failure_count,
                "extraction_failure_files": failed_extraction_files
            },
            "data_source": {
                "enhanced_metadata": using_enhanced_data,
                "json_path": str(REPORTS_DIR / "smart_filing_summary.json") if using_enhanced_data else None
            },
            "failed_files": [
                {"filename": k, "expected": e, "actual": a, "error_type": _categorize_error(e, a)}
                for k, e, a in verification_errors
            ],
            "extraction_failures": [
                {"filename": f, "reason": "no_correction_found"}
                for f in failed_extraction_files
            ],
            "per_file_results": results
        }
        
        Path(export_json).parent.mkdir(parents=True, exist_ok=True)
        with open(export_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Report exported to: {export_json}")

    return {
        "accuracy": accuracy,
        "total_in_batch": total_files_in_batch,
        "total_verified": total_checks,
        "passed": correct_count,
        "verification_errors": len(verification_errors),
        "extraction_failures": extraction_failure_count,
        "total_errors": total_errors
    }


def analyze_trends(reports_dir=None, pattern="*_batch_metrics.json"):
    """Analyze accuracy trends from all historical JSON reports."""
    if reports_dir is None:
        reports_dir = Path("./reports")
    else:
        reports_dir = Path(reports_dir)
    
    if not reports_dir.exists():
        print(f"❌ Reports directory not found: {reports_dir}")
        return
    
    json_files = sorted(reports_dir.glob(pattern))
    
    if not json_files:
        print(f"⚠️  No JSON reports found matching '{pattern}' in {reports_dir}")
        return
    
    print("=" * 75)
    print("📈 ACCURACY TRENDS ANALYSIS".center(75))
    print("=" * 75)
    print(f"Reports analyzed: {len(json_files)}")
    print("-" * 75)
    
    trends = []
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            trends.append({
                "file": json_file.name,
                "timestamp": data.get("timestamp", "unknown"),
                "accuracy": data.get("summary", {}).get("accuracy", 0),
                "total_in_batch": data.get("summary", {}).get("total_in_batch"),
                "total_verified": data.get("summary", {}).get("total_verified", 0),
                "passed": data.get("summary", {}).get("passed", 0),
                "verification_errors": data.get("summary", {}).get("verification_errors", 0),
                "extraction_failures": data.get("summary", {}).get("extraction_failures", 0),
                "total_errors": data.get("summary", {}).get("total_errors", 0),
                "error_analysis": data.get("error_analysis", {}),
                "confidence_stats": data.get("confidence_stats", {}),
                "batch_info": data.get("batch_info", {}),
                "tier_breakdown": data.get("tier_breakdown", {})
            })
        except Exception as e:
            print(f"⚠️  Warning: Could not parse {json_file.name}: {e}")
    
    if not trends:
        return
    
    print(f"{'REPORT':<35} | {'ACCURACY':<10} | {'BATCH':<8} | {'ERRORS':<10}")
    print("-" * 75)
    
    for t in trends:
        acc_pct = f"{t['accuracy'] * 100:.2f}%"
        batch_size = t.get('total_in_batch') or t.get('batch_info', {}).get('total_files_processed') or 'N/A'
        total_errors = t.get('total_errors', 0)
        print(f"{t['file']:<35} | {acc_pct:<10} | {str(batch_size):<8} | {total_errors:<10}")
    
    print("-" * 75)
    
    accuracies = [t["accuracy"] for t in trends]
    avg_accuracy = sum(accuracies) / len(accuracies)
    min_accuracy = min(accuracies)
    max_accuracy = max(accuracies)
    
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
    
    print("\n📊 TREND SUMMARY:")
    print(f"   • Average Accuracy: {avg_accuracy * 100:.2f}%")
    print(f"   • Range: [{min_accuracy * 100:.2f}%, {max_accuracy * 100:.2f}%]")
    print(f"   • Trend Direction: {trend_direction}")
    print(f"   • Reports Analyzed: {len(trends)}")
    
    batch_sizes = [t.get('total_in_batch') or t.get('batch_info', {}).get('total_files_processed') for t in trends if t.get('total_in_batch') or t.get('batch_info', {}).get('total_files_processed')]
    if batch_sizes:
        print(f"\n📦 BATCH SIZE STATS:")
        print(f"   • Average batch size: {sum(batch_sizes)/len(batch_sizes):.1f} files")
        print(f"   • Range: [{min(batch_sizes)}, {max(batch_sizes)}] files")
    
    if len(trends) >= 2:
        print("\n🚨 ERROR TYPE EVOLUTION:")
        all_error_types = set()
        for t in trends:
            all_error_types.update(t["error_analysis"].keys())
        
        for error_type in sorted(all_error_types):
            counts = [t["error_analysis"].get(error_type, 0) for t in trends]
            print(f"   • {error_type}: [{', '.join(map(str, counts))}]")
    
    if len(trends) >= 2:
        print("\n🔧 TIER USAGE EVOLUTION:")
        all_tiers = set()
        for t in trends:
            all_tiers.update(t["tier_breakdown"].keys())
        
        for tier in sorted(all_tiers):
            counts = [t["tier_breakdown"].get(tier, 0) for t in trends]
            if any(counts):
                print(f"   • {tier}: [{', '.join(map(str, counts))}]")
    
    if len(trends) >= 2:
        print("\n📈 CONFIDENCE TREND:")
        conf_means = [t.get("confidence_stats", {}).get("mean", 0) for t in trends]
        if any(conf_means):
            avg_conf = sum(conf_means) / len(conf_means)
            print(f"   • Average Mean Confidence: {avg_conf:.3f}")
            print(f"   • Range: [{min(conf_means):.3f}, {max(conf_means):.3f}]")
    
    print("=" * 75)
    
    return trends


# ==============================================================================
# 5. CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Verification Script for Difficult Cases")
    parser.add_argument("--enhanced", action="store_true", help="Enable advanced metrics")
    parser.add_argument("--report", type=str, default=None, help="Path to statistics_report.txt")
    parser.add_argument("--export", type=str, default=None, help="Export detailed report to JSON")
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
    
    args = parser.parse_args()
    
    if args.trends:
        analyze_trends(reports_dir=args.reports_dir)
    
    elif args.enhanced or args.export or args.export_auto:
        export_path = args.export
        
        if args.export_auto and not export_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"./reports/{timestamp}_batch_metrics.json"
        elif args.export_auto and export_path:
            path = Path(export_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = str(path.parent / f"{timestamp}_{path.name}")
        
        run_check_enhanced(
            report_path=args.report,
            export_json=export_path
        )
    
    else:
        run_check()