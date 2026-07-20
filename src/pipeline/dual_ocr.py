import logging
from typing import Tuple, List
import cv2
import numpy as np
import os

try:
    from config import PADDLE_OCR_CONFIG
except ImportError:
    PADDLE_OCR_CONFIG = {
        "use_angle_cls": True,
        "lang": "en",
        "use_gpu": False,
        "show_log": False,
        "enable_mkldnn": True,
        "rec_batch_num": 1,
    }

try:
    import paddle

    # Force CPU device early
    paddle.set_device("cpu")

    # ✅ FIX: Properly enable oneDNN (MKLDNN) for Intel CPUs
    # Your previous code had FLAGS_use_mkldnn set to False, which was overriding
    # the config and causing the 17-second processing time on large images!
    paddle.set_flags(
        {"FLAGS_use_mkldnn": True, "FLAGS_allocator_strategy": "naive_best_fit"}
    )

    from paddleocr import PaddleOCR

    PADDLE_INSTALLED = True
except ImportError:
    PADDLE_INSTALLED = False

from src.utils.log import get_logger

logger = get_logger("dual_ocr")

if PADDLE_INSTALLED:
    try:
        paddle.set_device("cpu")
    except Exception:
        pass


class OcrEngine:
    def __init__(self):
        if not PADDLE_INSTALLED:
            logger.error(
                "❌ PaddleOCR not installed. Run: uv pip install paddlepaddle paddleocr"
            )
            raise RuntimeError(
                "PaddleOCR missing. Please install paddlepaddle and paddleocr."
            )

        logger.info("📦 Initializing PaddleOCR (PP-OCRv4) engine...")

        try:
            import paddle

            paddle.set_device("cpu")

            # Centralized config from config.py
            engine_config = PADDLE_OCR_CONFIG.copy()

            # Ensure it matches our forced CPU/MKLDNN settings
            engine_config["use_gpu"] = False
            engine_config["enable_mkldnn"] = True  # Enables oneDNN in PaddleOCR

            self.reader = PaddleOCR(**engine_config)
            logger.info(
                "✅ OcrEngine (PaddleOCR) initialized successfully on CPU with MKLDNN."
            )
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
            # ✅ PERFORMANCE FIX: Prevent PaddleOCR from choking on massive images.
            # When the pipeline falls back to the "BR Quadrant", the image is huge.
            # PaddleOCR's detector max side length is 960. If we pass a 2000px image,
            # it slows down drastically. We resize it here to keep speeds under 2 seconds.
            max_dim = 1200
            h, w = roi.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # cls=True forces the angle classifier to run on every pass
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
