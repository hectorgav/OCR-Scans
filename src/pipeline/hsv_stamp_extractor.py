# =============================================================================
# FILE: src/pipeline/hsv_stamp_extractor.py
# =============================================================================
# HSV Stamp Extractor - Fixed to preserve high-confidence structural matches
# =============================================================================
import logging
from typing import Optional, Tuple, List
import cv2
import numpy as np

from src.validation.validation import UnifiedValidator

from src.utils.log import get_logger
logger = get_logger("hsv_stamp_extractor")

LOWER_BLUE = np.array([85, 45, 20], dtype=np.uint8)
UPPER_BLUE = np.array([140, 255, 255], dtype=np.uint8)
LOWER_GREEN = np.array([35, 45, 30], dtype=np.uint8)
UPPER_GREEN = np.array([85, 255, 255], dtype=np.uint8)
# --- ASYMMETRIC PADDING ---
CROP_PAD_Y_PX = 20      # Tight vertical padding
CROP_PAD_LEFT_PX = 50   # Short padding to the left
CROP_PAD_RIGHT_PX = 250 # Massive dragnet to the right for distant suffixes

def _build_multi_spectrum_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    return cv2.bitwise_or(mask_blue, mask_green)

def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    return cv2.dilate(mask, kernel, iterations=2)

def _find_stamp_boxes(mask: np.ndarray, img_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    img_h, img_w = img_shape[:2]
    min_area = (img_h * img_w) * 0.001
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > min_area and w > h:
            boxes.append((x, y, w, h))
    return boxes

def _crop_padded(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    img_h, img_w = image.shape[:2]

    # Apply right-heavy asymmetric padding
    x1 = max(0, x - CROP_PAD_LEFT_PX)
    y1 = max(0, y - CROP_PAD_Y_PX)
    x2 = min(img_w, x + w + CROP_PAD_RIGHT_PX)
    y2 = min(img_h, y + h + CROP_PAD_Y_PX)

    return image[y1:y2, x1:x2]

def extract_blue_stamp(image: np.ndarray, ocr_engine) -> Optional[Tuple[str, float, np.ndarray]]:
    """
    Locates blue/green stamps and extracts text using the unified PaddleOCR engine.
    Returns the validated Job Number, its OCR Confidence, and the exact cropped ROI image.
    
    FIX: Preserves structurally valid extractions and passes the ROI back to the dashboard.
    """
    if image is None or image.size == 0:
        return None

    try:
        mask = _build_multi_spectrum_mask(image)
        dilated = _dilate_mask(mask)
        boxes = _find_stamp_boxes(dilated, image.shape)

        if not boxes:
            return None

        best_text = None
        best_conf = 0.0
        best_crop = None  # Track the winning image slice
        validator = UnifiedValidator()

        for box in boxes:
            crop = _crop_padded(image, box)
            results = ocr_engine.run_single_pass(crop)
            
            for raw_text, conf in results:
                # Use metadata validation to preserve high-conf OCR
                result = validator.validate_with_metadata(raw_text, conf)
                
                # Keep if: (1) structurally valid format, AND (2) higher confidence
                if result.normalized and conf > best_conf:
                    best_text = result.normalized
                    best_conf = conf
                    best_crop = crop.copy()  # <--- NEW: Save the exact image slice
                    logger.debug(f"HSV: Kept '{result.normalized}' (conf={conf:.3f}, valid={result.is_valid}, structural={result.is_structurally_valid})")

        # <--- NEW: Return all 3 variables 
        if best_text and best_crop is not None:
            return best_text, best_conf, best_crop

    except Exception as e:
        logger.error(f"HSV Stamp Extractor failed: {e}")

    return None