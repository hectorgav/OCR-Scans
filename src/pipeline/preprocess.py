# src/pipeline/preprocess.py

import logging
import cv2
import numpy as np
from typing import Tuple, Optional
from config import SUPPORTED_EXTENSIONS # Dummy import to keep config reference if needed, or just remove

from src.utils.log import get_logger
logger = get_logger("preprocess")
_CLAHE_CACHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def expand_roi_for_suffix(image: np.ndarray, pad_left: int = 0, pad_right: int = 60) -> np.ndarray:
    if image is None or image.size == 0: return image
    return cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

def enhance_for_ocr(
    image: np.ndarray, use_color_mask: bool = False, use_dilation: bool = False,
    dilation_kernel: Tuple[int, int] = (2, 2), use_upscale: bool = True
) -> np.ndarray:
    if image is None or image.size == 0: return image

    found_colored_ink = False
    
    if use_color_mask and len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 10, 30]), np.array([30, 255, 255]))     # Orange
        mask |= cv2.inRange(hsv, np.array([150, 10, 30]), np.array([180, 255, 255])) # Red Wrap
        mask |= cv2.inRange(hsv, np.array([35, 15, 30]), np.array([85, 255, 255]))   # Green
        mask |= cv2.inRange(hsv, np.array([80, 5, 30]), np.array([140, 255, 255]))   # Cyan/Blue (expanded)
        mask |= cv2.inRange(hsv, np.array([135, 10, 30]), np.array([160, 255, 255])) # Magenta/Purple
        
        if cv2.countNonZero(mask) > 0:
            found_colored_ink = True
            if use_dilation:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel)
                mask = cv2.dilate(mask, kernel, iterations=1)
            processed = np.full(image.shape[:2], 255, dtype=np.uint8)
            processed[mask > 0] = 0  
        else:
            processed = image
    else:
        processed = image
        
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed.copy()
    enhanced = _CLAHE_CACHE.apply(gray)
    
    # RESTORED FIX: Lock at 65. Preserves clean text geometry, preventing 5 -> 3 bleeding.
    if use_upscale:
        if enhanced.shape[0] < 65: 
            enhanced = cv2.resize(enhanced, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if use_dilation:
        if use_color_mask and not found_colored_ink:
            pass 
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel)
            thresh = cv2.erode(thresh, kernel, iterations=1) 
            
    return thresh

def recover_faint_ink(image: np.ndarray) -> Optional[np.ndarray]:
    if image is None or len(image.shape) != 3: return None
    h_img, w_img = image.shape[:2]
    if h_img < 5 or w_img < 10: return None
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2]  
    l_channel = lab[:, :, 0]  
    
    b_cleaned = b_channel.copy()
    b_cleaned[l_channel < 30] = 128  
    b_cleaned[b_channel > 118] = 128
    
    inverted = cv2.bitwise_not(b_cleaned)
    blurred = cv2.GaussianBlur(inverted, (3, 3), sigmaX=1.0)
    _, binary = cv2.threshold(blurred, 132, 255, cv2.THRESH_BINARY)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 10: cleaned[labels == i] = 255
            
    density = cv2.countNonZero(cleaned) / max(1, cleaned.size)
    # HARDENED: Density threshold raised to reject sparse noise blocks
    if not (0.002 <= density <= 0.50): return None
        
    recovered = cv2.resize(cleaned, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    return cv2.bitwise_not(recovered)