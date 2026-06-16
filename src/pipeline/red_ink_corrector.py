import cv2
import numpy as np
import logging
from src.utils.log import get_logger

logger = get_logger("red_ink_corrector")

# Define Red Color Ranges in HSV
# Red wraps around 0/180, so we need two masks
def get_red_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Mask 1: 0-10
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    
    # Mask 2: 170-180
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), 
                          cv2.inRange(hsv, lower_red2, upper_red2))
    return mask

def detect_correction_mark(crop: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Detects if there is a significant red ink correction mark 
    inside the YOLO job number box.
    """
    mask = get_red_mask(crop)
    
    # Calculate percentage of the box covered by red ink
    red_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    coverage = red_pixels / total_pixels
    
    logger.debug(f"Red ink coverage in box: {coverage:.4f}")
    
    # Return True if red ink occupies more than threshold of the box
    return coverage > threshold