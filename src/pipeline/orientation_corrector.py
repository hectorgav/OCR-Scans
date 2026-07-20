# =============================================================================
# src/pipeline/orientation_corrector.py
# =============================================================================
# NATIVE ANGLE-CLASSIFIER ORIENTATION CORRECTOR
# =============================================================================

"""
Smart orientation corrector that relies entirely on PaddleOCR's native angle
classifier (use_angle_cls=True).

Because PaddleOCR evaluates the rotation of individual text lines using a
neural network, we no longer need brittle image-level mathematical heuristics.
This perfectly handles scenarios where the drawing and the stamp are at
different angles.
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
    Smart orientation corrector. Relies on PaddleOCR's native angle classifier
    to handle 0°, 90°, -90°, and 180° text lines automatically.
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

    def recover_text_orientation(
        self, img: np.ndarray, ocr_engine: Any, is_tight_crop: bool = True
    ) -> Tuple[Optional[str], float, np.ndarray, str]:
        """
        Smart recovery pipeline. Passes the image directly to PaddleOCR.
        PaddleOCR's internal angle classifier will automatically detect and
        rotate any vertical or upside-down text lines before reading them.
        """
        if img is None or img.size == 0:
            return None, 0.0, img, "empty"

        padded_img = self._pad_image(img, is_tight_crop)

        # -----------------------------------------------------------------
        # PASS A: NATIVE OCR (PaddleOCR handles rotation internally)
        # -----------------------------------------------------------------
        results = ocr_engine.run_single_pass(padded_img)
        for raw_text, conf in results:
            if raw_text:
                result = self.validator.validate_with_metadata(raw_text, conf)
                is_acceptable = result.is_valid or (
                    result.is_structurally_valid and conf >= 0.70
                )

                if is_acceptable and result.normalized:
                    return (
                        result.normalized,
                        conf,
                        padded_img,
                        "paddle_raw_native_angle_cls",
                    )

        # -----------------------------------------------------------------
        # PASS B: COLOR MASK FALLBACK
        # -----------------------------------------------------------------
        # If native fails, we isolate the blue ink and try again.
        # PaddleOCR will still handle the rotation internally on the masked image.
        color_masked = enhance_for_ocr(
            padded_img, use_color_mask=True, use_dilation=False
        )
        results_color = ocr_engine.run_single_pass(color_masked)
        for raw_text, conf in results_color:
            if raw_text:
                result = self.validator.validate_with_metadata(raw_text, conf)
                is_acceptable = result.is_valid or (
                    result.is_structurally_valid and conf >= 0.70
                )

                if is_acceptable and result.normalized:
                    return (
                        result.normalized,
                        conf,
                        color_masked,
                        "paddle_color_mask_native_angle_cls",
                    )

        # Note: The 180° safety net and manual rotation loops have been removed.
        # PaddleOCR's use_angle_cls=True natively handles 0°, 90°, -90°, and 180°.
        # Removing the manual loops cuts worst-case execution time in half!

        return None, 0.0, img, "failed"


# =============================================================================
# END OF FILE
# =============================================================================
