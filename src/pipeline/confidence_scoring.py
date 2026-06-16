# =============================================================================
# src/pipeline/confidence_scoring.py
# =============================================================================
# CONFIDENCE SCORING ENGINE - Calibrated Weighted Fusion
# =============================================================================

"""
Macro/Micro Candidate Scoring Utility.
Combines object detection confidence with text recognition reliability
using calibrated, configuration-driven weights and structural penalties.
"""

import logging
from typing import Optional

# Safe import from config.py to preserve decoupling
try:
    from config import YOLO_WEIGHT, OCR_WEIGHT
except ImportError:
    # Safe fallbacks if config is not fully initialized
    YOLO_WEIGHT = 0.35
    OCR_WEIGHT = 0.65

logger = logging.getLogger(__name__)

# ============================================================================
# --- CALIBRATED WEIGHTED FUSION ---
# ============================================================================

def weighted_confidence_fusion(
    yolo_conf: Optional[float], 
    ocr_conf: float, 
    yolo_weight: float = None, 
    ocr_weight: float = None
) -> float:
    """
    Performs calibrated weighted confidence fusion between YOLO and OCR.
    
    Formula:
        Fused = (YOLO_Weight * YOLO_Conf) + (OCR_Weight * OCR_Conf)
    """
    # Use config weights if not explicitly overridden in call
    w_yolo = yolo_weight if yolo_weight is not None else YOLO_WEIGHT
    w_ocr = ocr_weight if ocr_weight is not None else OCR_WEIGHT

    # Default detection confidence to neutral (0.70) if YOLO didn't run (e.g. Sniper falls)
    detection_score = yolo_conf if yolo_conf is not None else 0.70

    # Calculate weighted average
    fused_score = (w_yolo * detection_score) + (w_ocr * ocr_conf)
    
    return float(max(0.0, min(1.0, fused_score)))


# ============================================================================
# --- COMBINED CONFIDENCE RANKER (LEGACY COMPATIBILITY) ---
# ============================================================================

def combine_confidences(
    det_conf: Optional[float], 
    ocr_reliability: float, 
    method: str = ""
) -> float:
    """
    Legacy entry point. Integrates calibrated weights while preserving 
    historical orientation and dilation penalties.
    """
    # Use our new calibrated weighted fusion
    base_score = weighted_confidence_fusion(det_conf, ocr_reliability)

    method_lower = method.lower()

    # --- MICRO-ORIENTATION PENALTIES ---
    if "90deg" in method_lower or "rot90" in method_lower:
        base_score -= 0.05
    elif "180deg" in method_lower or "rot180" in method_lower:
        base_score -= 0.02

    # --- HISTORICAL RECOVERY SAFEGUARDS ---
    # Penalize dilated stages for solid black text (Scan 203)
    if "dilated" in method_lower and "color" not in method_lower:
        base_score -= 0.10  

    # --- OCR ENGINE CAPS ---
    if "easyocr" in method_lower:
        base_score = min(base_score, 0.85)

    # Ensure final score remains strictly bounded between 0.0 and 1.0
    return float(max(0.0, min(1.0, base_score)))

# =============================================================================
# END OF FILE
# =============================================================================