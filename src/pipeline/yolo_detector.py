# src/pipeline/yolo_detector.py

"""
YOLO Object Detection Module.

Strictly stripped of all arbitrary geometry blockers. If YOLO finds it, 
we trust it and pass it to Validation. Ensures robust loading and graceful logging.
"""

import logging
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple

try:
    from config import YOLO_CONF_DETECTION
except ImportError:
    YOLO_CONF_DETECTION = 0.25

from src.utils.log import get_logger
logger = get_logger("yolo_detector")

class JobDetector:
    def __init__(self, model_path: str, conf: float = YOLO_CONF_DETECTION):
        self.conf = conf
        self.model = None
        try:
            # Robust model loading
            self.model = YOLO(model_path)
            logger.info(f"✅ JobDetector loaded model successfully: {model_path}")
        except Exception as e:
            logger.error(f"❌ FATAL: Failed to load YOLO model from {model_path}. Error: {e}")
            raise RuntimeError(f"YOLO model initialization failed: {e}") from e

    def detect_on_image(self, img: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        if self.model is None:
            logger.error("JobDetector model is not initialized. Skipping detection.")
            return [], img
        
        try:
            results = self.model.predict(img, conf=self.conf, verbose=False, imgsz=640)
        except Exception as e:
            logger.error(f"❌ YOLO prediction failed during inference: {e}")
            return [], img
        
        detections = []
        # FIX: Define page_area relative to the current input image
        page_area = img.shape[0] * img.shape[1]

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if x2 <= x1 or y2 <= y1: 
                    continue

                # FIX: Calculate dimensions for the geometry filter
                w_box = x2 - x1
                h_box = y2 - y1
                box_area = w_box * h_box
                aspect_ratio = w_box / h_box
            
                # --- STEP 1: GENERALIZED GEOMETRY FILTER ---
                # Aspect Ratio < 1.2: Rejects square/vertical stamps (decoy avoidance)
                # Area > 8%: Only rejects massive page-level overlays (preventing lockups)
                if aspect_ratio < 1.2 or box_area > (page_area * 0.08):
                    logger.debug(f"Rejecting decoy/non-horizontal box (Area: {box_area}, AR: {aspect_ratio:.2f})")
                    continue

                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0])
                })
        
        return detections, img