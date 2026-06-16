# =============================================================================
# src/pipeline/extractor.py
# =============================================================================
# MACRO-NORMALIZED ORCHESTRATOR - High-Accuracy Bounding Box Parser
# =============================================================================

"""
The Macro-Normalized Orchestrator.
Executes the Step-Scaling Dragnet. Attempts a tight, high-accuracy YOLO crop first.
If it fails, expands drastically horizontally to ensure decapitated double-suffixes 
are caught. Includes prioritized Color-Ink recovery and Red Ink correction flagging.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import cv2

from config import (
    YOLO_MODEL_PATH, YOLO_ENABLED, TITLE_BLOCK_ENABLED, ENABLE_DEBUG_VIZ,
    YOLO_CONF_DETECTION, TITLE_BLOCK_X_START, TITLE_BLOCK_Y_START, 
    TITLE_BLOCK_WIDTH, TITLE_BLOCK_HEIGHT
)

# --- PIPELINE IMPORTS ---
try:
    import torch
    from ultralytics import YOLO
except ImportError:
    pass

from src.pipeline.dual_ocr import OcrEngine
from src.pipeline.yolo_detector import JobDetector
from src.pipeline.confidence_scoring import combine_confidences
from src.utils.debug_utils import DebugVisualizer
from src.pipeline.orientation_corrector import OrientationCorrector
from src.pipeline.hsv_stamp_extractor import extract_blue_stamp
from src.pipeline.blue_stamp_enhancer import BlueStampEnhancer
from src.pipeline.red_ink_corrector import detect_correction_mark

from src.utils.log import get_logger
logger = get_logger("extractor")

_JOB_DETECTOR = None
_OCR_ENGINE = None
_BLUE_STAMP_ENHANCER = None

# =============================================================================
# SINGLETON INITIALIZERS
# =============================================================================

def get_blue_stamp_enhancer():
    global _BLUE_STAMP_ENHANCER
    if _BLUE_STAMP_ENHANCER is None:
        config = {
            "saturation_boost": 1.8,      
            "value_boost": 1.3,           
            "clahe_clip_limit": 4.0,      
            "dilation_iterations": 1,
            "gamma": 0.7,                 
            "debug_mode": False
        }
        _BLUE_STAMP_ENHANCER = BlueStampEnhancer(config)
    return _BLUE_STAMP_ENHANCER

def get_job_detector():
    global _JOB_DETECTOR
    if _JOB_DETECTOR is None and YOLO_ENABLED and YOLO_MODEL_PATH.exists():
        try:
            _JOB_DETECTOR = JobDetector(str(YOLO_MODEL_PATH), conf=YOLO_CONF_DETECTION)
        except Exception as e:
            logger.error(f"❌ Failed to load JobDetector: {e}")
    return _JOB_DETECTOR

def get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            logger.info("Initializing Singleton OcrEngine...")
            _OCR_ENGINE = OcrEngine()
        except Exception as e:
            logger.error(f"❌ Failed to load OcrEngine: {e}")
    return _OCR_ENGINE

# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_job_number(
    image_path: Path, original_file_path: Path
) -> Tuple[Optional[str], float, str, dict]:
    """
    Main extraction pipeline. Integrates YOLO detection, multi-spectrum HSV 
    blue-stamp enhancement, dual-stage OCR, and Red Ink Correction flagging.
    """
    debug_stem = original_file_path.stem
    ocr_engine = get_ocr_engine()
    enhancer = get_blue_stamp_enhancer()  
    
    # Initialize default metadata dictionary
    meta = {"has_red_correction": False, "error": None}
    
    if ocr_engine is None:
        logger.error("OCR Engine failed to initialize. Aborting extraction.")
        return None, 0.0, "failed", {"error": "ocr_engine_missing", "has_red_correction": False}
        
    viz = DebugVisualizer(enabled=ENABLE_DEBUG_VIZ)
    orient_corrector = OrientationCorrector()
    detector = get_job_detector()

    raw_image = cv2.imread(str(image_path))
    if raw_image is None: 
        return None, 0.0, "failed", {"error": "image_load_failed", "has_red_correction": False}

    # Handle page orientation for schematics
    h_raw, w_raw = raw_image.shape[:2]
    oriented_image = cv2.rotate(raw_image, cv2.ROTATE_90_CLOCKWISE) if h_raw > w_raw else raw_image
    h_proc, w_proc = oriented_image.shape[:2]

    # -------------------------------------------------------------------------
    # PHASE 1: STANDARD YOLO (Step-Scaling Dragnet)
    # -------------------------------------------------------------------------
    if detector:
        raw_detections, processed_page = detector.detect_on_image(oriented_image)

        if raw_detections:
            viz.save_yolo_debug(debug_stem, processed_page, raw_detections)
            raw_detections.sort(key=lambda x: x["confidence"], reverse=True)

            for det in raw_detections:
                x1, y1, x2, y2 = det["box"]
                yolo_conf = det["confidence"]
                w_box, h_box = x2 - x1, y2 - y1
                
                best_jn = None
                best_conf = 0.0
                best_method = ""

                # --- STEP 1A: TIGHT CROP (with enhancement) ---
                pad_w_t = int(max(10, w_box * 0.05))
                pad_h_t = int(max(10, h_box * 0.05))
                roi_tight = processed_page[max(0, y1-pad_h_t):min(h_proc, y2+pad_h_t), 
                                           max(0, x1-pad_w_t):min(w_proc, x2+pad_w_t)]
                
                if roi_tight.size > 0:
                    roi_enhanced = enhancer.enhance(roi_tight, method="combined")
                    jn, ocr_conf, used_img, method = orient_corrector.recover_text_orientation(
                        roi_enhanced, ocr_engine, is_tight_crop=True
                    )
                    
                    if jn:
                        # Detect Red Ink inside the Crop
                        has_red = detect_correction_mark(roi_tight)
                        meta["has_red_correction"] = has_red

                        f_conf = combine_confidences(yolo_conf, ocr_conf, method=method)
                        if "color" in method.lower(): f_conf = min(1.0, f_conf + 0.15)

                        viz.save_ocr_debug(debug_stem, f"yolo_tight_{method}_enhanced", jn, f_conf, roi_img=used_img)
                        
                        if f_conf >= 0.75: 
                            return jn, f_conf, f"yolo_tight_{method}_enhanced", meta
                        else:
                            best_jn, best_conf, best_method = jn, f_conf, f"yolo_tight_{method}_enhanced"
                    
                    # Fallback to original (non-enhanced)
                    if not best_jn or best_conf < 0.70:
                        jn_orig, ocr_conf_orig, used_img_orig, method_orig = orient_corrector.recover_text_orientation(
                            roi_tight, ocr_engine, is_tight_crop=True
                        )
                        if jn_orig:
                            has_red = detect_correction_mark(roi_tight)
                            meta["has_red_correction"] = has_red
                            
                            f_conf_orig = combine_confidences(yolo_conf, ocr_conf_orig, method=method_orig)
                            if f_conf_orig > best_conf:
                                best_jn, best_conf, best_method = jn_orig, f_conf_orig, f"yolo_tight_{method_orig}"

                # --- STEP 1B: WIDE DRAGNET (with enhancement) ---
                pad_w_w = int(max(50, w_box * 0.80)) 
                pad_h_w = int(max(30, h_box * 0.40))
                roi_wide = processed_page[max(0, y1-pad_h_w):min(h_proc, y2+pad_h_w), 
                                          max(0, x1-pad_w_w):min(w_proc, x2+pad_w_w)]
                
                if roi_wide.size > 0:
                    roi_wide_enhanced = enhancer.enhance(roi_wide, method="combined")
                    jn, ocr_conf, used_img, method = orient_corrector.recover_text_orientation(
                        roi_wide_enhanced, ocr_engine, is_tight_crop=False
                    )
                    if jn:
                        has_red = detect_correction_mark(roi_wide)
                        meta["has_red_correction"] = has_red
                        
                        f_conf = combine_confidences(yolo_conf, ocr_conf, method=method)
                        if "color" in method.lower(): f_conf = min(1.0, f_conf + 0.15)
                        
                        viz.save_ocr_debug(debug_stem, f"yolo_wide_{method}_enhanced", jn, f_conf, roi_img=used_img)
                        
                        if f_conf > best_conf:
                            best_jn, best_conf, best_method = jn, f_conf, f"yolo_wide_{method}_enhanced"

                if best_jn:
                    return best_jn, best_conf, best_method, meta

    # -------------------------------------------------------------------------
    # PHASE 1.5: MULTI-SPECTRUM HSV SNIPER (Targeted Region - UNRESTRICTED)
    # -------------------------------------------------------------------------
    # Removed the strict coordinate cropping. Let the HSV detector find 
    # blue/green stamps anywhere on the page to prevent rotation failure.
    if oriented_image.size > 0:
        enhanced_image = enhancer.enhance(oriented_image, method="hsv")
        hsv_result = extract_blue_stamp(enhanced_image, ocr_engine)
        
        if hsv_result:
            hsv_job, hsv_conf, hsv_crop = hsv_result
            # Detect Red Ink inside the HSV crop
            has_red = detect_correction_mark(hsv_crop)
            meta["has_red_correction"] = has_red
            
            viz.save_ocr_debug(debug_stem, "hsv_sniper_enhanced", hsv_job, hsv_conf, roi_img=hsv_crop)
            return hsv_job, hsv_conf, "hsv_stamp_sniper_enhanced", meta

    # -------------------------------------------------------------------------
    # PHASE 3: FAST QUADRANT FALLBACK (with enhancement)
    # -------------------------------------------------------------------------
    br_quad = oriented_image[int(h_proc * 0.45):h_proc, int(w_proc * 0.45):w_proc]
    if br_quad.size > 0:
        br_quad_enhanced = enhancer.enhance(br_quad, method="combined")
        jn, conf, used_img, method = orient_corrector.recover_text_orientation(
            br_quad_enhanced, ocr_engine, is_tight_crop=False
        )
        if jn: 
            has_red = detect_correction_mark(br_quad)
            meta["has_red_correction"] = has_red
            return jn, conf, f"title_block_BR_{method}_enhanced", meta

    return None, 0.0, "failed", {"error": "all_methods_failed", "has_red_correction": False}

# =============================================================================
# END OF FILE
# =============================================================================