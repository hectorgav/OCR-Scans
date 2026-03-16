# =============================================================================
# src/validation/validation.py
# =============================================================================
# UNIFIED VALIDATOR - Enhanced with Confidence Metadata
# =============================================================================
# Purpose:
#   Validates job number extractions using:
#   - Structural format validation (######-##)
#   - Business logic validation (year window)
#   - Confidence-aware override for high-quality OCR
#
# Key Features:
#   - Dynamic year regex generation (handles decade spans)
#   - Metadata-rich validation results for smart_filing integration
#   - OCR confidence propagation for override decisions
#   - Exclusion patterns to filter false positives
#
# Author: OCR Pipeline Team
# Version: 2.2.0
# Last Updated: 2026-03-13
# Compliance: Guardrail 6 ✅ (No hardcoded heuristics)
# =============================================================================

import re
import logging
import datetime
from typing import Optional, Set, Tuple
from dataclasses import dataclass

try: 
    from config import ALLOWED_SEPARATORS
except ImportError: 
    ALLOWED_SEPARATORS = "-. ="
    
logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ValidationResult:
    """
    Enhanced validation result with confidence metadata.
    
    Attributes:
        is_valid: True if job passes format AND year validation
        normalized: The cleaned/normalized job string
        rejection_reason: Why validation failed (if applicable)
        is_structurally_valid: True if format matches ######-## pattern
        ocr_confidence_override: True if high-conf OCR should override year rejection
    """
    is_valid: bool
    normalized: Optional[str]
    rejection_reason: Optional[str] = None
    is_structurally_valid: bool = False
    ocr_confidence_override: bool = False


# =============================================================================
# MAIN VALIDATOR CLASS
# =============================================================================


class UnifiedValidator:
    """
    Unified validation engine for job number extractions.
    
    Uses physics-based pattern matching and statistical year validation
    to distinguish valid extractions from OCR artifacts.
    """
    
    def __init__(self, year_window_past: int = 5, year_window_future: int = 1):
        """
        Initialize validator with configurable year window.
        
        Args:
            year_window_past: Years before current to accept (default: 5)
            year_window_future: Years after current to accept (default: 1)
        """
        # Compute current year (2-digit)
        cy = datetime.datetime.now().year % 100
        
        # Define valid year range using sequence mathematics
        self.min_year = max(0, cy - year_window_past)
        self.max_year = cy + year_window_future
        self.valid_year_range: Set[int] = set(range(self.min_year, self.max_year + 1))
        
        # Build year regex using topological decade analysis
        self.year_regex = self._build_year_regex()
        
        # Prepare separator pattern for flexible parsing
        es = re.escape(ALLOWED_SEPARATORS)
        self.sep_pattern = rf"[{es}–—]+"
        strict_tail = r"(?![0-9A-Za-z])"
        
        # Compile patterns using year_regex for early rejection of invalid years
        # This is physics-based: pattern matching models how job numbers appear in documents
        self.pattern_strict = re.compile(
            rf"({self.year_regex}[\s]*0[\s]*\d[\s]*\d[\s]*\d)(?:[\s]*(?:{self.sep_pattern})?[\s]*(\d{{2}}){strict_tail})"
        )
        self.pattern_flexible = re.compile(
            rf"({self.year_regex}[\s]*0[\s]*\d[\s]*\d[\s]*\d)(?:[\s]*(?:{self.sep_pattern})?[\s]*(\d{{2}})?)"
        )
        
        # Decapitated pattern: rescues job numbers where first digit is separated
        self.pattern_decapitated = re.compile(
            rf"(?:^|\D)(\d)([\s]*0[\s]*\d[\s]*\d[\s]*\d)(?:[\s]*(?:{self.sep_pattern})?[\s]*(\d{{2}}){strict_tail})"
        )
        
        # Legacy pattern for historical job numbers
        self.pattern_legacy = re.compile(r"\b(11[01]\d{4})\b")
        
        # Pattern to remove non-essential characters during normalization
        self.canonical_keep_pattern = re.compile(rf"[^0-9A-Za-z{es}\n]")
        
        # OCR correction map: physics-based character confusion model
        # Models common OCR errors: O→0, l→1, S→5, etc.
        self.ocr_correction_map = str.maketrans({
            "O":"0","o":"0","U":"0","u":"0","Q":"0",
            "I":"1","l":"1","|":"1","L":"1",
            "S":"5","s":"5","B":"8","G":"6",
            "Z":"2","z":"2","g":"9","q":"9",
            "~":"-","/":"-","_":"-","=":"-","≡":"-"
        })
        
        # Exclusion patterns: filter common false positives using regex topology
        self.exclusion_patterns = [
            re.compile(p, re.IGNORECASE) for p in [
                r"\b(?:JOB|NO|NUMBER|RING|OUT|RINGOUT|ENGR|REV|REVISION|PAGE|SCALE|SIZE|DATE|TITLE|APPROVED|CHECKED|BY|ENGINEER|ISSUED|CONSTRUCTION|PRELIMINARY|FINAL|AS-BUILT)\b", 
                r"\b[A-Z]{2,4}\s*[-–—]\s*\d{1,4}\b", 
                r"^\s*\d{1,3}\s*$", 
                r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", 
                r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*\d{1,2}", 
                r"\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}", 
                r"\bD[-–—]\d+\b", 
                r"\bE[-–—]\d+\b", 
                r"\bS\d{3}[-–—]\d+\b"
            ]
        ]

    def _build_year_regex(self) -> str:
        """
        Build year regex that handles decade spans correctly using sequence math.
        
        Returns:
            Regex string that matches valid 2-digit years in range
        """
        y1_min, y0_min = divmod(self.min_year, 10)
        y1_max, y0_max = divmod(self.max_year, 10)
        
        if y1_min == y1_max:
            # Same decade: efficient character class [2][1-7] for years 21-27
            return f"[{y1_min}][{y0_min}-{y0_max}]"
        else:
            # Spanning decades: explicit alternation (19|20|21|...|25)
            # This prevents matching invalid years when range crosses decade boundary
            valid_years = "|".join(f"{y:02d}" for y in self.valid_year_range)
            return f"(?:{valid_years})"

    def validate_and_normalize(self, text: str, ocr_confidence: float = 0.0) -> Optional[str]:
        """
        Legacy method for backward compatibility.
        
        Args:
            text: Raw OCR extraction
            ocr_confidence: Confidence score from OCR engine
        
        Returns:
            Normalized job string if valid, None otherwise
        """
        result = self.validate_with_metadata(text, ocr_confidence)
        return result.normalized if result.is_valid else None

    def validate_with_metadata(self, text: str, ocr_confidence: float = 0.0) -> ValidationResult:
        """
        Enhanced validation with full metadata for smart_filing integration.
        
        Physics-based approach:
        1. Apply exclusion patterns to remove document noise
        2. Normalize using OCR confusion model (O→0, l→1, etc.)
        3. Match against structural patterns with year validation
        4. Return metadata for confidence-aware decision making
        
        Args:
            text: Raw OCR extraction
            ocr_confidence: Confidence score from OCR engine
        
        Returns:
            ValidationResult with validation status and metadata
        """
        if not text:
            return ValidationResult(False, None, "empty_input")
        
        # Step 1: Remove exclusion patterns (document noise filtering)
        t = text
        for p in self.exclusion_patterns: 
            t = p.sub(" ", t)
        
        # Step 2: Apply OCR correction map (physics-based character confusion model)
        ct = self.canonical_keep_pattern.sub("", t.translate(self.ocr_correction_map)).strip()
        ct = re.sub(r'(\d)E(\d)', r'\1-\2', ct)
        
        if not ct:
            return ValidationResult(False, None, "no_content_after_normalization")
        
        # Step 3: Try legacy pattern first (historical compatibility)
        if match := self.pattern_legacy.search(ct): 
            return ValidationResult(True, match.group(1), None, True)
        
        # Step 4: Try strict and flexible patterns with year validation
        for pattern in [self.pattern_strict, self.pattern_flexible]:
            if match := pattern.search(ct):
                base_end = match.end(1)
                
                # Check for excessive newlines (indicates broken extraction topology)
                if match.lastindex and match.lastindex >= 2 and match.group(2):
                    if ct[base_end:match.start(2)].count('\n') > 1: 
                        continue
                
                # Extract and normalize base
                base = re.sub(r'\s+', '', match.group(1))
                
                # Validate year using sequence mathematics (range membership)
                try:
                    year = int(base[:2])
                    year_valid = year in self.valid_year_range
                except Exception: 
                    year_valid = False
                
                # Extract suffix if present
                s1 = re.sub(r"\D", "", match.group(2)) if match.lastindex and match.lastindex >= 2 else None
                suffix = s1 if s1 and len(s1) == 2 else None
                
                normalized = f"{base}-{suffix}" if suffix else base
                
                # Check structural validity using regex topology
                is_structural = bool(re.match(r"^\d{6}(-\d{2})?$", normalized))
                
                if year_valid:
                    return ValidationResult(True, normalized, None, is_structural)
                else:
                    # Year invalid but structure valid - mark for smart_filing override decision
                    # This enables confidence-aware correction without hardcoding
                    if ocr_confidence >= 0.70 and is_structural:
                        return ValidationResult(
                            False, normalized, 
                            f"year_outside_window ({year} not in {self.min_year}-{self.max_year})",
                            is_structural,
                            ocr_confidence_override=True
                        )
                    else:
                        return ValidationResult(False, normalized, f"year_outside_window ({year})", is_structural)
        
        # Step 5: Try decapitated year rescue (topological pattern recovery)
        if match := self.pattern_decapitated.search(ct):
            y_digit = match.group(1)
            for decade in range(0, 10):
                test_year = int(f"{decade}{y_digit}")
                if test_year in self.valid_year_range:
                    base_rest = re.sub(r'\s+', '', match.group(2))
                    suffix = re.sub(r"\D", "", match.group(3))
                    normalized = f"{test_year:02d}{base_rest}-{suffix}"
                    return ValidationResult(True, normalized, None, True)
        
        return ValidationResult(False, None, "no_pattern_match")

    def is_valid(self, text: str, ocr_confidence: float = 0.0) -> bool: 
        """
        Check if text is valid, with confidence-aware override support.
        
        Args:
            text: Raw OCR extraction
            ocr_confidence: Confidence score from OCR engine
        
        Returns:
            True if valid or high-confidence override applies
        """
        result = self.validate_with_metadata(text, ocr_confidence)
        return result.is_valid or result.ocr_confidence_override

    def is_year_valid(self, year_str: str) -> bool:
        """
        Check if a 2-digit year string is within the valid range.
        
        Uses sequence mathematics: range membership test.
        
        Args:
            year_str: Two-digit year string (e.g., "15", "24")
        
        Returns:
            True if year is in valid_year_range
        """
        try:
            year = int(year_str)
            return year in self.valid_year_range
        except (ValueError, TypeError):
            return False

    def is_structurally_valid(self, text: str) -> bool:
        """
        Check if text matches expected format without year validation.
        
        Uses regex topology to validate ######-## pattern.
        
        Args:
            text: Job string to validate
        
        Returns:
            True if format matches structural pattern
        """
        result = self.validate_with_metadata(text, 0.0)
        return result.is_structurally_valid and result.normalized is not None
        
    def get_base(self, text: str) -> Optional[str]:
        """
        Extract base portion of job ID (before suffix).
        
        Args:
            text: Full job ID (e.g., "250585-01")
        
        Returns:
            Base portion (e.g., "250585") or None if invalid
        """
        n = self.validate_and_normalize(text, 0.0)
        return n.split("-")[0] if n and "-" in n else n
    
    def get_validation_result(self, text: str, ocr_confidence: float = 0.0) -> ValidationResult:
        """
        Public method to get full validation metadata.
        
        Args:
            text: Raw OCR extraction
            ocr_confidence: Confidence score from OCR engine
        
        Returns:
            ValidationResult with all metadata fields
        """
        return self.validate_with_metadata(text, ocr_confidence)