# =============================================================================
# src/preprocessing/blue_stamp_enhancer.py
# =============================================================================
# BLUE STAMP ENHANCER - Specialized preprocessing for faded blue ink stamps
# =============================================================================
# Purpose:
#   Enhances faded blue/green stamps to improve OCR accuracy by:
#   - Boosting blue channel saturation and contrast
#   - Applying adaptive histogram equalization
#   - Morphological operations to restore faded strokes
#   - Multi-spectrum enhancement for various blue shades
#
# Author: OCR Pipeline Team
# Version: 1.0.0
# Last Updated: 2026-03-13
# =============================================================================

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List
from config import SMART_FILING_CONFIG

from src.utils.log import get_logger
logger = get_logger("blue_stamp_enhancer")


class BlueStampEnhancer:
    """
    Specialized image enhancer for faded blue/green ink stamps.
    
    Applies multiple enhancement techniques to restore faded text:
    1. HSV color space manipulation (saturation/value boost)
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    3. Blue channel isolation and enhancement
    4. Morphological operations (dilation to restore broken strokes)
    5. Gamma correction for brightness adjustment
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize enhancer with configurable parameters.
        
        Args:
            config: Optional dictionary with enhancement parameters
        """
        self.config = config or {}
        
        # HSV Enhancement parameters
        self.hue_range = self.config.get("hue_range", (85, 130))  # Blue-green range
        self.saturation_boost = self.config.get("saturation_boost", 1.5)
        self.value_boost = self.config.get("value_boost", 1.2)
        self.min_saturation = self.config.get("min_saturation", 20)
        
        # CLAHE parameters
        self.clahe_clip_limit = self.config.get("clahe_clip_limit", 3.0)
        self.clahe_tile_size = self.config.get("clahe_tile_size", (8, 8))
        
        # Morphological operations
        self.dilation_iterations = self.config.get("dilation_iterations", 1)
        self.kernel_size = self.config.get("kernel_size", 3)
        
        # Gamma correction
        self.gamma = self.config.get("gamma", 0.8)  # <1.0 brightens mid-tones
        
        # Debug mode
        self.debug_mode = self.config.get("debug_mode", False)
    
    def enhance(self, image: np.ndarray, method: str = "combined") -> np.ndarray:
        """
        Apply enhancement to faded blue stamp image.
        
        Args:
            image: Input BGR image (OpenCV format)
            method: Enhancement method to use:
                   - "hsv": HSV color space enhancement
                   - "clahe": Adaptive histogram equalization
                   - "blue_channel": Blue channel isolation
                   - "morphological": Stroke restoration
                   - "combined": All methods combined (recommended)
        
        Returns:
            Enhanced image in BGR format
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to enhancer")
            return image
        
        if method == "hsv":
            return self._enhance_hsv(image)
        elif method == "clahe":
            return self._enhance_clahe(image)
        elif method == "blue_channel":
            return self._enhance_blue_channel(image)
        elif method == "morphological":
            return self._enhance_morphological(image)
        elif method == "combined":
            return self._enhance_combined(image)
        else:
            logger.warning(f"Unknown enhancement method: {method}, using combined")
            return self._enhance_combined(image)
    
    def _enhance_hsv(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance blue stamp using HSV color space manipulation.
        
        Boosts saturation and value of blue/green hues while preserving other colors.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        h, s, v = cv2.split(hsv)
        
        # Create mask for blue-green hues
        lower_blue = np.array(self.hue_range[0], dtype=np.uint8)
        upper_blue = np.array(self.hue_range[1], dtype=np.uint8)
        mask = cv2.inRange(hsv[:,:,0].astype(np.uint8), lower_blue, upper_blue)
        
        # Boost saturation for blue regions
        s_boosted = s * self.saturation_boost
        s_boosted = np.clip(s_boosted, 0, 255)
        
        # Apply boost only to blue regions
        s_final = np.where(mask > 0, s_boosted, s)
        
        # Boost value (brightness) for blue regions
        v_boosted = v * self.value_boost
        v_boosted = np.clip(v_boosted, 0, 255)
        v_final = np.where(mask > 0, v_boosted, v)
        
        # Merge channels back
        hsv_enhanced = cv2.merge([h.astype(np.uint8), s_final.astype(np.uint8), v_final.astype(np.uint8)])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        if self.debug_mode:
            logger.debug(f"HSV enhancement: saturation_boost={self.saturation_boost}, value_boost={self.value_boost}")
        
        return enhanced
    
    def _enhance_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Improves local contrast, especially useful for faded text.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel (lightness)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size
        )
        l_enhanced = clahe.apply(l)
        
        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        if self.debug_mode:
            logger.debug(f"CLAHE enhancement: clip_limit={self.clahe_clip_limit}, tile_size={self.clahe_tile_size}")
        
        return enhanced
    
    def _enhance_blue_channel(self, image: np.ndarray) -> np.ndarray:
        """
        Isolate and enhance blue channel.
        
        Blue ink stamps appear dark in the blue channel, making text more visible.
        """
        # Split BGR channels
        b, g, r = cv2.split(image)
        
        # Apply gamma correction to blue channel
        inv_gamma = 1.0 / self.gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        b_enhanced = cv2.LUT(b, table)
        
        # Apply CLAHE to blue channel
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size
        )
        b_clahe = clahe.apply(b_enhanced)
        
        # Merge channels back (enhanced blue, original green/red)
        enhanced = cv2.merge([b_clahe, g, r])
        
        if self.debug_mode:
            logger.debug(f"Blue channel enhancement: gamma={self.gamma}")
        
        return enhanced
    
    def _enhance_morphological(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to restore faded strokes.
        
        Dilation thickens thin/faded lines, helping OCR recognize broken characters.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold to isolate text
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        # Create morphological kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.kernel_size, self.kernel_size)
        )
        
        # Apply dilation to thicken strokes
        if self.dilation_iterations > 0:
            dilated = cv2.dilate(thresh, kernel, iterations=self.dilation_iterations)
            # Apply slight erosion to restore original thickness
            eroded = cv2.erode(dilated, kernel, iterations=max(0, self.dilation_iterations - 1))
        else:
            eroded = thresh
        
        # Create mask from morphological result
        mask = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
        
        # Apply mask to original image
        enhanced = cv2.bitwise_and(image, mask)
        
        # Add white background where mask is black
        enhanced = cv2.add(enhanced, cv2.bitwise_not(mask))
        
        if self.debug_mode:
            logger.debug(f"Morphological enhancement: kernel={self.kernel_size}, iterations={self.dilation_iterations}")
        
        return enhanced
    
    def _enhance_combined(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all enhancement methods in optimal sequence.
        
        Sequence:
        1. HSV enhancement (boost blue color)
        2. CLAHE (improve contrast)
        3. Morphological (restore faded strokes)
        """
        if self.debug_mode:
            logger.debug("Applying combined enhancement pipeline")
        
        # Step 1: HSV enhancement
        enhanced = self._enhance_hsv(image)
        
        # Step 2: CLAHE for contrast
        enhanced = self._enhance_clahe(enhanced)
        
        # Step 3: Morphological operations
        enhanced = self._enhance_morphological(enhanced)
        
        return enhanced
    
    def enhance_multiple_methods(
        self, 
        image: np.ndarray,
        methods: List[str] = None
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Apply multiple enhancement methods and return all results.
        
        Useful for testing which method works best for a specific image.
        
        Args:
            image: Input image
            methods: List of methods to try (default: all methods)
        
        Returns:
            List of tuples: (method_name, enhanced_image)
        """
        if methods is None:
            methods = ["hsv", "clahe", "blue_channel", "morphological", "combined"]
        
        results = []
        for method in methods:
            try:
                enhanced = self.enhance(image, method=method)
                results.append((method, enhanced))
            except Exception as e:
                logger.error(f"Enhancement method '{method}' failed: {e}")
        
        return results


def enhance_blue_stamp(
    image: np.ndarray,
    config: Optional[dict] = None
) -> np.ndarray:
    """
    Convenience function to enhance faded blue stamp.
    
    Args:
        image: Input BGR image
        config: Optional configuration dictionary
    
    Returns:
        Enhanced BGR image
    """
    enhancer = BlueStampEnhancer(config)
    return enhancer.enhance(image, method="combined")


# =============================================================================
# END OF FILE
# =============================================================================