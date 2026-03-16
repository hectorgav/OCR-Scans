# src/pipeline/orientation_corrector.py

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Any
from src.validation.validation import UnifiedValidator
from src.pipeline.preprocess import expand_roi_for_suffix, enhance_for_ocr

from src.utils.log import get_logger
logger = get_logger("orientation_corrector")

class OrientationCorrector:
    def __init__(self):
        self.validator = UnifiedValidator()

    def _pad_image(self, img: np.ndarray, is_tight_crop: bool) -> np.ndarray:
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
        
        if img is None or img.size == 0:
            return None, 0.0, img, "empty"

        padded_img = self._pad_image(img, is_tight_crop)
        
        # ---------------------------------------------------------------------
        # PASS 1: NATIVE PADDLEOCR
        # ---------------------------------------------------------------------
        results = ocr_engine.run_single_pass(padded_img)
        for raw_text, conf in results:
            if raw_text:
                jn = self.validator.validate_and_normalize(raw_text)
                if jn:
                    return jn, conf, padded_img, "paddle_raw"

        # ---------------------------------------------------------------------
        # PASS 2: COLOR MASK FALLBACK
        # Strips black CAD lines to rescue overlapping red/blue ink.
        # ---------------------------------------------------------------------
        color_masked = enhance_for_ocr(padded_img, use_color_mask=True, use_dilation=False)
        results = ocr_engine.run_single_pass(color_masked)
        for raw_text, conf in results:
            if raw_text:
                jn = self.validator.validate_and_normalize(raw_text)
                if jn:
                    return jn, conf, padded_img, "paddle_color_mask"

        return None, 0.0, img, "failed"