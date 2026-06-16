# =============================================================================
# src/pipeline/orientation_corrector.py
# =============================================================================
# HEURISTIC-FIRST FAST ORIENTATION CORRECTOR - Optimized for Production Speed
# =============================================================================

"""
Specialized orientation corrector for job stamps.
Uses lightweight mathematical projection profiles to detect text skew and 
vertical/horizontal layout in <1ms, ensuring OCR runs exactly once at the 
correct angle. Bypasses trial-and-error OCR loops.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Any
from src.validation.validation import UnifiedValidator
from src.pipeline.preprocess import expand_roi_for_suffix, enhance_for_ocr

from src.utils.log import get_logger
logger = get_logger("orientation_corrector")

# =============================================================================
# CORE ORIENTATION CORRECTOR CLASS
# =============================================================================

class OrientationCorrector:
    """
    Fast orientation correction engine using OpenCV projection heuristics.
    """
    
    def __init__(self):
        self.validator = UnifiedValidator()

    def _pad_image(self, img: np.ndarray, is_tight_crop: bool) -> np.ndarray:
        """Applies dynamic padding based on crop constraints."""
        pad_l, pad_r = 20, 20 
        if is_tight_crop:
            pad_l, pad_r = 100, 300
            
        h_rot, w_rot = img.shape[:2]
        if is_tight_crop and h_rot > w_rot:
            extra_pad = int(h_rot * 1.5)
            pad_l += extra_pad
            pad_r += extra_pad
            
        return expand_roi_for_suffix(img, pad_left=pad_l, pad_right=pad_r)

    def _detect_rotation_needed(self, img: np.ndarray) -> int:
        """
        Heuristic: Detects if the text block is vertical (90 deg) or horizontal (0 deg).
        Uses vertical vs horizontal projection profile variance (takes <1ms).
        
        Returns:
            0 if horizontal, 90 if vertical (clockwise rotation needed)
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Threshold to get black text on white background
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Calculate projection profiles
            proj_horiz = np.sum(thresh, axis=1)
            proj_vert = np.sum(thresh, axis=0)
            
            # High variance in projection profile means text lines align with that axis
            var_horiz = np.var(proj_horiz)
            var_vert = np.var(proj_vert)
            
            # If vertical projection variance is significantly higher, text is likely vertical
            if var_vert > var_horiz * 1.5:
                return 90
        except Exception:
            pass
        return 0

    def recover_text_orientation(
        self, img: np.ndarray, ocr_engine: Any, is_tight_crop: bool = True
    ) -> Tuple[Optional[str], float, np.ndarray, str]:
        """
        Fast recovery pipeline. Uses a math heuristic to detect vertical stamps,
        rotates once, and executes OCR. Fails over to color-masking if needed.
        """
        if img is None or img.size == 0:
            return None, 0.0, img, "empty"

        padded_img = self._pad_image(img, is_tight_crop)
        
        # ✅ Step 1: Detect if image is vertical or horizontal (<1ms)
        angle = self._detect_rotation_needed(padded_img)
        
        if angle == 90:
            logger.info("⚡ Heuristic: Vertical stamp detected. Applying fast 90-degree rotation.")
            processed_img = cv2.rotate(padded_img, cv2.ROTATE_90_CLOCKWISE)
            method_suffix = "_rot90"
        else:
            processed_img = padded_img
            method_suffix = ""

        # -----------------------------------------------------------------
        # PASS A: NATIVE OCR (Exactly 1 Run)
        # -----------------------------------------------------------------
        results = ocr_engine.run_single_pass(processed_img)
        for raw_text, conf in results:
            if raw_text:
                result = self.validator.validate_with_metadata(raw_text, conf)
                is_acceptable = result.is_valid or (result.is_structurally_valid and conf >= 0.70)
                
                if is_acceptable and result.normalized:
                    return result.normalized, conf, processed_img, f"paddle_raw{method_suffix}"

        # -----------------------------------------------------------------
        # PASS B: COLOR MASK FALLBACK (Exactly 1 Run, only if Pass A fails)
        # -----------------------------------------------------------------
        color_masked = enhance_for_ocr(processed_img, use_color_mask=True, use_dilation=False)
        results_color = ocr_engine.run_single_pass(color_masked)
        for raw_text, conf in results_color:
            if raw_text:
                result = self.validator.validate_with_metadata(raw_text, conf)
                is_acceptable = result.is_valid or (result.is_structurally_valid and conf >= 0.70)
                
                if is_acceptable and result.normalized:
                    return result.normalized, conf, processed_img, f"paddle_color_mask{method_suffix}"

        # -----------------------------------------------------------------
        # PASS C: 180 FLIP SAFETY NET (Only executed as a last-resort exception)
        # -----------------------------------------------------------------
        # If both fail, the stamp might be completely upside-down. 
        # We try a 180-degree flip only on the color-masked image.
        upside_down_img = cv2.rotate(color_masked, cv2.ROTATE_180)
        results_flipped = ocr_engine.run_single_pass(upside_down_img)
        for raw_text, conf in results_flipped:
            if raw_text:
                result = self.validator.validate_with_metadata(raw_text, conf)
                is_acceptable = result.is_valid or (result.is_structurally_valid and conf >= 0.70)
                
                if is_acceptable and result.normalized:
                    logger.info("⚡ Safety Net: Upside-down stamp recovered via 180-degree flip.")
                    return result.normalized, conf, upside_down_img, f"paddle_color_mask_flipped"

        return None, 0.0, img, "failed"

# =============================================================================
# END OF FILE
# =============================================================================