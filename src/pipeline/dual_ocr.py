import logging
from typing import Tuple, List
import cv2
import numpy as np
import os

try:
    from config import PADDLE_OCR_CONFIG
except ImportError:
    PADDLE_OCR_CONFIG = {
        "use_angle_cls": True, "lang": "en", "use_gpu": False, 
        "show_log": False, "enable_mkldnn": False, "rec_batch_num": 1
    }

try:
    import paddle
    # Force CPU device early
    paddle.set_device('cpu')
    paddle.set_flags({
        'FLAGS_use_mkldnn': False,
        'FLAGS_allocator_strategy': 'naive_best_fit'
    })
    from paddleocr import PaddleOCR
    PADDLE_INSTALLED = True
except ImportError:
    PADDLE_INSTALLED = False

from src.utils.log import get_logger
logger = get_logger("dual_ocr")

if PADDLE_INSTALLED:
    try:
        paddle.set_device('cpu')
    except Exception:
        pass

class OcrEngine:
    def __init__(self):
        if not PADDLE_INSTALLED:
            logger.error("❌ PaddleOCR not installed. Run: uv pip install paddlepaddle paddleocr")
            raise RuntimeError("PaddleOCR missing. Please install paddlepaddle and paddleocr.")
            
        logger.info("📦 Initializing PaddleOCR (PP-OCRv4) engine...")
        
        try:
            import paddle
            paddle.set_device('cpu')
            
            # Centralized config from config.py
            engine_config = PADDLE_OCR_CONFIG.copy()
            
            # Ensure it matches our forced CPU/MKLDNN settings
            engine_config["use_gpu"] = False
            engine_config["enable_mkldnn"] = True
            
            self.reader = PaddleOCR(**engine_config)
            logger.info("✅ OcrEngine (PaddleOCR) initialized successfully on CPU.")
        except Exception as e:
            logger.error(f"❌ OcrEngine initialization failed: {e}")
            self.reader = None
            raise

    def run_single_pass(self, roi: np.ndarray) -> List[Tuple[str, float]]:
        """
        Executes a single, robust OCR pass. 
        Natively handles skew, rotation, and low contrast thermal prints.
        """
        results = []
        if roi is None or roi.size == 0: 
            return results
        
        try:
            ocr_output = self.reader.ocr(roi, cls=True)
            
            if ocr_output and ocr_output[0]:
                for line in ocr_output[0]:
                    text, conf = line[1]
                    if text:
                        text_clean = text.strip()
                        results.append((text_clean, float(conf)))
            
            if results:
                logger.debug(f"PaddleOCR found: {[r[0] for r in results]}")
            else:
                logger.debug("PaddleOCR returned no text for this ROI.")
        except Exception as e:
            logger.error(f"PaddleOCR extraction error: {e}")
            
        return results