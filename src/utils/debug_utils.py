# src/utils/debug_utils.py

import logging
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional

try:
    from config import DEBUG_FOLDERS
except ImportError:
    DEBUG_FOLDERS = {}

logger = logging.getLogger(__name__)

class DebugVisualizer:
    """
    Handles the generation of debug artifacts with consolidated routing:
    - 2_macro_vision: YOLO detections, Anchor Zones, Title Block bounds.
    - 3_micro_vision: Tesseract ROI crops, OCR validation cards.
    """

    def __init__(self, output_dir: Path = None, enabled: bool = True):
        self.enabled = enabled
        
        if self.enabled:
            for folder in DEBUG_FOLDERS.values():
                try:
                    folder.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

    def _get_folder(self, key: str) -> Optional[Path]:
        if not self.enabled: return None
        return DEBUG_FOLDERS.get(key)

    # ============================================================================
    # MACRO-VISION (Where are we looking?)
    # ============================================================================

    def save_yolo_debug(self, image_stem: str, image: np.ndarray, detections: list) -> None:
        folder = self._get_folder('macro_vision')
        if not folder: return

        debug_img = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['box']
            conf = det.get('confidence', 0.0)
            
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"JobNum {conf:.2f}"
            cv2.putText(debug_img, label, (x1, max(0, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite(str(folder / f"{image_stem}_01_yolo_detections.jpg"), debug_img)

    def save_title_block(self, image_stem: str, crop_type: str, crop: np.ndarray) -> None:
        folder = self._get_folder('macro_vision')
        if not folder: return
        cv2.imwrite(str(folder / f"{image_stem}_02_{crop_type}.jpg"), crop)

    def save_debug_roi(self, image_stem: str, roi: np.ndarray, idx: int, label: str) -> None:
        """Saves Macro-regions like the Bottom-Right Quadrant or Anchor Search Zones."""
        folder = self._get_folder('macro_vision')
        if not folder or roi is None or roi.size == 0: return

        roi_to_save = roi
        if roi.dtype != np.uint8:
            roi_to_save = np.clip(roi, 0, 255).astype(np.uint8)

        cv2.imwrite(str(folder / f"{image_stem}_03_macro_{label}.jpg"), roi_to_save)

    # ============================================================================
    # MICRO-VISION (What are we reading?)
    # ============================================================================

    def save_ocr_debug(self, image_stem: str, stage: str, text: str, confidence: float, roi_img: np.ndarray = None) -> None:
        """
        Creates a visual card with the exact ROI text and the pipeline method overlaid.
        """
        folder = self._get_folder('micro_vision')
        if not folder or roi_img is None or roi_img.size == 0: return
        
        try:
            # Handle grayscale vs BGR
            if len(roi_img.shape) == 2:
                roi_pil = Image.fromarray(roi_img, mode='L').convert('RGB')
            elif roi_img.shape[2] == 3:
                roi_pil = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
            else:
                return
                
            padding = 10
            header_height = 100
            canvas_w = max(roi_pil.width + 2 * padding, 450)
            canvas_h = roi_pil.height + header_height + 2 * padding
            
            canvas = Image.new('RGB', (canvas_w, canvas_h), (245, 245, 245))
            draw = ImageDraw.Draw(canvas)
            
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                title_font = ImageFont.truetype("arial.ttf", 22)
            except IOError:
                font = ImageFont.load_default()
                title_font = ImageFont.load_default()
            
            # Header Texts
            display_text = text.strip() if text else "FAILED"
            draw.text((padding, padding), f"OCR: {display_text}", font=title_font, fill=(0, 100, 0) if text else (200, 0, 0))
            
            draw.text((padding, 50), f"Method: {stage}", font=font, fill=(50, 50, 50))
            draw.text((padding, 70), f"Confidence: {confidence:.3f}", font=font, fill=(0, 0, 200))
            
            canvas.paste(roi_pil, (padding, header_height + padding))
            
            # Save the card
            clean_stage = stage.replace(" ", "_").replace(":", "")
            canvas.save(str(folder / f"{image_stem}_micro_{clean_stage}.jpg"))
            
        except Exception as e:
            logger.error(f"Failed to generate micro-vision card for {image_stem}: {e}")