# =============================================================================
# src/pipeline/smart_filing.py
# =============================================================================
# SMART FILING ENGINE - Tiered Correction System
# =============================================================================

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Iterable, Set
from collections import Counter
from src.validation.validation import UnifiedValidator
from config import SMART_FILING_CONFIG
import re
import logging

from src.utils.log import get_logger
logger = get_logger("smart_filing")

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

FAILED_JOB_MARKERS = {"all_methods_failed", "failed", "", None}
SCAN_NUM_RE = re.compile(r"scan(\d+)", re.IGNORECASE)
TRUST_OCR_CONFIDENCE_THRESHOLD = SMART_FILING_CONFIG.get("trust_ocr_threshold", 0.70)
DEBUG_SMART_FILING = SMART_FILING_CONFIG.get("debug_mode", False)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Record:
    """Represents a single scan file with its extraction results."""
    filename: str
    raw_job: str
    confidence: float = 0.0
    corrected_job: Optional[str] = None
    reason: str = ""
    meta: Dict = field(default_factory=dict)
    candidates: List[Dict] = field(default_factory=list)

# =============================================================================
# MODULE-LEVEL VALIDATOR
# =============================================================================

validator = UnifiedValidator()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def scan_index(filename: str, fallback_idx: int = 0) -> int:
    """Extract numeric scan index from filename."""
    match = SCAN_NUM_RE.search(filename)
    if match:
        return int(match.group(1))
    fallback_match = re.findall(r"\d+", filename)
    return int(fallback_match[-1]) if fallback_match else fallback_idx


def is_valid_format(job: str, confidence: float = 0.0) -> bool:
    """
    Check if job ID passes full validation (format + business logic).
    
    CRITICAL: Passes confidence to validator for ocr_confidence_override support.
    """
    if not job:
        return False
    return validator.is_valid(job, confidence)


def get_base(job: str) -> Optional[str]:
    """Extract base portion of job ID (before suffix)."""
    if not job:
        return None
    return validator.get_base(job)


def _is_single_char_mutation(s1: str, s2: str) -> bool:
    """
    Check if two strings differ by exactly one character (Hamming distance = 1).
    
    Physics-based OCR error model: faded ink, sensor noise typically cause
    single-character mutations, not multi-character changes.
    """
    if not s1 or not s2 or len(s1) != len(s2):
        return False
    return sum(a != b for a, b in zip(s1, s2)) == 1


def _is_failed_job(job: str) -> bool:
    """
    Check if job is a failure marker using set membership.
    
    Prevents Sandwich Rule from propagating failures through the pipeline.
    """
    return job in FAILED_JOB_MARKERS or job is None


def _debug_log(msg: str, record: Optional[Record] = None) -> None:
    """Conditional debug logging for troubleshooting."""
    if DEBUG_SMART_FILING:
        prefix = f"[{record.filename}] " if record else ""
        logger.debug(f"🔍 DEBUG: {prefix}{msg}")


# =============================================================================
# TIER 1: SANDWICH RULE (UPDATED)
# =============================================================================

def _apply_sandwich_rule(records: List[Record], known_jobs: Set[str]) -> None:
    """
    Tier 1: Adopt neighbor job when bracketed by identical valid jobs.
    
    IMPROVEMENTS:
    1. Check suffix mutation (03→04 is typo, protect 01→10 is batch change)
    2. Require physical proximity (consecutive scans)
    3. Never propagate failure markers
    """
    sorted_recs = sorted(records, key=lambda r: scan_index(r.filename))
    
    for i in range(1, len(sorted_recs) - 1):
        prev_r, curr_r, next_r = sorted_recs[i - 1], sorted_recs[i], sorted_recs[i + 1]
        
        prev_job = prev_r.corrected_job or prev_r.raw_job
        curr_job = curr_r.corrected_job or curr_r.raw_job
        next_job = next_r.corrected_job or next_r.raw_job
        
        # =========================================================
        # CRITICAL FIX: Skip if neighbors are failed/None
        # Prevents failure propagation through the pipeline
        # =========================================================
        if _is_failed_job(prev_job) or _is_failed_job(next_job):
            _debug_log(f"Sandwich skipped: neighbors are failed (prev={prev_job}, next={next_job})", curr_r)
            continue
        
        # Check sandwich condition (prev == next, curr differs)
        if not (prev_job and next_job and prev_job == next_job and curr_job != prev_job):
            continue
        
        # Skip known jobs
        if curr_job in known_jobs:
            continue
        
        # =========================================================
        # ENHANCED SUFFIX SHIELD (v2.0)
        # Protects legitimate unique suffixes, corrects typos
        # =========================================================
        if is_valid_format(curr_job, curr_r.confidence) and get_base(curr_job) == get_base(prev_job):
            
            # Extract suffixes for comparison
            curr_suffix = curr_job.split('-')[1] if '-' in curr_job else None
            prev_suffix = prev_job.split('-')[1] if '-' in prev_job else None
            
            # If both have suffixes, check if it's a single-char mutation
            if curr_suffix and prev_suffix:
                # Case A: Single-char suffix mutation (03→04, 01→07) = TYPO, correct it
                if _is_single_char_mutation(curr_suffix, prev_suffix):
                    _debug_log(f"Suffix mutation detected: {curr_suffix} → {prev_suffix} (typo, will correct)", curr_r)
                    # Don't protect - let sandwich correction apply
                # Case B: Multi-char suffix difference (01→10, 03→30) = BATCH CHANGE, protect it
                elif curr_r.confidence >= 0.75:
                    _debug_log(f"Unique suffix protected: {curr_suffix} (conf={curr_r.confidence:.2f})", curr_r)
                    curr_r.reason = f"kept: valid unique suffix (conf {curr_r.confidence:.2f})"
                    continue
            # If confidence is high and no suffix to compare, protect
            elif curr_r.confidence >= 0.75:
                curr_r.reason = f"kept: valid unique suffix (conf {curr_r.confidence:.2f})"
                continue
        
        # =========================================================
        # Check physical proximity (must be consecutive scans)
        # Uses topographical layout: consecutive scans = same batch
        # =========================================================
        dist_prev = abs(scan_index(curr_r.filename, i) - scan_index(prev_r.filename, i - 1))
        dist_next = abs(scan_index(next_r.filename, i + 1) - scan_index(curr_r.filename, i))
        
        if dist_prev <= 1 and dist_next <= 1:
            curr_r.corrected_job = prev_job
            curr_r.reason = f"Tier 1: Sandwich Rule (bracketed by {prev_job})"
            logger.info(f"✅ Sandwich Rule: {curr_r.filename} | {curr_job} → {prev_job}")


# =============================================================================
# TIER 2: SUFFIX INHERITANCE
# =============================================================================

def _apply_suffix_inheritance(records: List[Record], known_jobs: Set[str]) -> None:
    """Tier 2: Inherit suffix from neighbor when base matches."""
    sorted_recs = sorted(records, key=lambda r: scan_index(r.filename))
    
    for i, curr_r in enumerate(sorted_recs):
        curr_job = curr_r.corrected_job or curr_r.raw_job
        if curr_job in known_jobs:
            continue
        if not curr_job or "-" in str(curr_job) or not is_valid_format(curr_job, curr_r.confidence):
            continue

        curr_base = get_base(curr_job)
        neighbors = [sorted_recs[n] for n in [i-1, i+1] if 0 <= n < len(sorted_recs)]
        
        for n_r in neighbors:
            n_job = n_r.corrected_job or n_r.raw_job
            if n_job and is_valid_format(n_job, n_r.confidence) and "-" in n_job and get_base(n_job) == curr_base:
                curr_r.corrected_job = n_job
                curr_r.reason = f"Tier 2: Suffix Inheritance (from {n_job})"
                break


# =============================================================================
# TIER 3: WEIGHTED CONSENSUS
# =============================================================================

def _apply_weighted_consensus(records: List[Record], window: int, max_dist: int, known_jobs: Set[str]) -> None:
    """Tier 3: Weighted neighbor consensus with outlier protection."""
    sorted_recs = sorted(records, key=lambda r: scan_index(r.filename))
    
    for i, curr_r in enumerate(sorted_recs):
        curr_job = curr_r.corrected_job or curr_r.raw_job
        if is_valid_format(curr_job, curr_r.confidence) or curr_job in known_jobs:
            continue

        neigh_counts = Counter()
        curr_idx = scan_index(curr_r.filename, i)
        lo, hi = max(0, i - window), min(len(sorted_recs), i + window + 1)
        
        # SEQUENCE CONTINUITY PROFILING: Find the local majority base
        window_bases = [
            get_base(sorted_recs[x].corrected_job or sorted_recs[x].raw_job) 
            for x in range(lo, hi) 
            if is_valid_format(sorted_recs[x].corrected_job or sorted_recs[x].raw_job,
                              sorted_recs[x].confidence)
        ]
        majority_base = Counter(window_bases).most_common(1)[0][0] if window_bases else None

        for n in range(lo, hi):
            if n == i:
                continue
            n_r = sorted_recs[n]
            physical_dist = abs(scan_index(n_r.filename, n) - curr_idx)
            
            if physical_dist > max_dist:
                continue

            n_job = n_r.corrected_job or n_r.raw_job
            
            if n_job and is_valid_format(n_job, n_r.confidence):
                # CONFIDENCE GATE
                if n_r.confidence >= 0.70 or "Tier" in n_r.reason:
                    base_weight = (window + 1) - abs(i - n)
                    weight = base_weight if physical_dist <= 1 else base_weight * 0.3
                    
                    # THE SEQUENCE OUTLIER PENALTY
                    if majority_base and get_base(n_job) != majority_base:
                        weight *= 0.1
                        
                    neigh_counts[n_job] += weight

        if neigh_counts:
            best_job = neigh_counts.most_common(1)[0][0]
            curr_r.corrected_job = best_job
            curr_r.reason = f"Tier 3: Weighted Consensus (adopted {best_job})"


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def smart_correct_batch(records: List[Record], known_jobs: Optional[Iterable[str]] = None) -> List[Record]:
    """Apply all smart filing correction tiers to a batch of records."""
    known_jobs_set = set(known_jobs or [])
    window = SMART_FILING_CONFIG.get("neighbor_window", 2)
    max_dist = SMART_FILING_CONFIG.get("max_physical_distance", 5)

    # Initialize corrected_job from raw_job
    for r in records: 
        r.corrected_job = r.raw_job if r.raw_job not in FAILED_JOB_MARKERS else None

    # Apply correction tiers in order
    _apply_sandwich_rule(records, known_jobs_set)
    _apply_suffix_inheritance(records, known_jobs_set)
    _apply_weighted_consensus(records, window, max_dist, known_jobs_set)

    # Finalize all records
    for r in records:
        if not r.corrected_job:
            r.corrected_job, r.reason = r.raw_job, "no_correction_found"
        elif not r.reason:
            r.reason = "kept: valid"
    
    return records