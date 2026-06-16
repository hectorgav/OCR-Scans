# src/pipeline/confidence_scoring.py

"""
Macro/Micro Candidate Scoring Utility.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================
# --- COMBINED CONFIDENCE RANKER ---
# ============================================================================

def combine_confidences(
    det_conf: Optional[float], 
    ocr_reliability: float, 
    method: str = ""
) -> float:
    
    # Default detection confidence to neutral if YOLO didn't provide it 
    detection_score = det_conf if det_conf is not None else 0.70

    # Base Weighted Average: 60% detection confidence, 40% OCR reliability
    base_score = (0.6 * detection_score) + (0.4 * ocr_reliability)

    method_lower = method.lower()

    # --- MICRO-ORIENTATION PENALTIES ---
    if "90deg" in method_lower:
        base_score -= 0.05
    elif "180deg" in method_lower:
        base_score -= 0.02

    # CRITICAL FIX: Penalize dilated stages for solid black text (Scan 203)
    # If the method includes "dilated" but the original image was grayscale (no color mask),
    # it's more likely to have mutation artifacts
    if "dilated" in method_lower and "color" not in method_lower:
        base_score -= 0.10  

    # --- OCR ENGINE CAPS ---
    if "easyocr" in method_lower:
        base_score = min(base_score, 0.85)

    # Ensure final score remains strictly bounded between 0 and 1
    return float(max(0.0, min(1.0, base_score)))