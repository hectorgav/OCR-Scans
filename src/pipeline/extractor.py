# src/pipeline/extractor.py

"""
The Macro-Normalized Orchestrator.
Executes the Step-Scaling Dragnet. Attempts a tight, high-accuracy YOLO crop first.
If it fails, expands drastically horizontally to ensure decapitated double-suffixes 
are caught. Includes prioritized Color-Ink recovery for handwritten corrections.
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
# STABILITY FIX: Import YOLO/Torch BEFORE PaddleOCR to prevent shm.dll conflicts on Windows
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

from src.utils.log import get_logger
logger = get_logger("extractor")

_JOB_DETECTOR = None
_OCR_ENGINE = None
_BLUE_STAMP_ENHANCER = None

def get_blue_stamp_enhancer():
    global _BLUE_STAMP_ENHANCER
    if _BLUE_STAMP_ENHANCER is None:
        config = {
            "saturation_boost": 1.8,      # Increased from 1.5
            "value_boost": 1.3,           # Increased from 1.2
            "clahe_clip_limit": 4.0,      # Increased from 3.0
            "dilation_iterations": 1,
            "gamma": 0.7,                 # Brighten mid-tones
            "debug_mode": False
        }
        _BLUE_STAMP_ENHANCER = BlueStampEnhancer(config)
    return _BLUE_STAMP_ENHANCER

def get_job_detector():
    global _JOB_DETECTOR
    if _JOB_DETECTOR is None and YOLO_ENABLED and YOLO_MODEL_PATH.exists():
        try:
            # Load the YOLO model with the centralized confidence threshold
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

def extract_job_number(
    image_path: Path, original_file_path: Path
) -> Tuple[Optional[str], float, str, dict]:
    
    debug_stem = original_file_path.stem
    ocr_engine = get_ocr_engine()
    enhancer = get_blue_stamp_enhancer()  # ✅ NEW
    
    if ocr_engine is None:
        logger.error("OCR Engine failed to initialize. Aborting extraction.")
        return None, 0.0, "failed", {"error": "ocr_engine_missing"}
        
    viz = DebugVisualizer(enabled=ENABLE_DEBUG_VIZ)
    orient_corrector = OrientationCorrector()
    detector = get_job_detector()

    raw_image = cv2.imread(str(image_path))
    if raw_image is None: 
        return None, 0.0, "failed", {"error": "image_load_failed"}

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
                    # ✅ ENHANCE FADED BLUE STAMP
                    roi_enhanced = enhancer.enhance(roi_tight, method="combined")
                    
                    # Try enhanced version first
                    jn, ocr_conf, used_img, method = orient_corrector.recover_text_orientation(
                        roi_enhanced, ocr_engine, is_tight_crop=True
                    )
                    
                    if jn:
                        f_conf = combine_confidences(yolo_conf, ocr_conf, method=method)
                        if "color" in method.lower(): f_conf = min(1.0, f_conf + 0.15)
                        viz.save_ocr_debug(debug_stem, f"yolo_tight_{method}_enhanced", jn, f_conf, roi_img=used_img)
                        
                        if f_conf >= 0.75: 
                            return jn, f_conf, f"yolo_tight_{method}_enhanced", {}
                        else:
                            best_jn, best_conf, best_method = jn, f_conf, f"yolo_tight_{method}_enhanced"
                    
                    # Fallback to original (non-enhanced) if enhancement fails
                    if not best_jn or best_conf < 0.70:
                        jn_orig, ocr_conf_orig, used_img_orig, method_orig = orient_corrector.recover_text_orientation(
                            roi_tight, ocr_engine, is_tight_crop=True
                        )
                        if jn_orig:
                            f_conf_orig = combine_confidences(yolo_conf, ocr_conf_orig, method=method_orig)
                            if f_conf_orig > best_conf:
                                best_jn, best_conf, best_method = jn_orig, f_conf_orig, f"yolo_tight_{method_orig}"

                # --- STEP 1B: WIDE DRAGNET (with enhancement) ---
                pad_w_w = int(max(50, w_box * 0.80)) 
                pad_h_w = int(max(30, h_box * 0.40))
                roi_wide = processed_page[max(0, y1-pad_h_w):min(h_proc, y2+pad_h_w), 
                                          max(0, x1-pad_w_w):min(w_proc, x2+pad_w_w)]
                
                if roi_wide.size > 0:
                    # ✅ ENHANCE FADED BLUE STAMP
                    roi_wide_enhanced = enhancer.enhance(roi_wide, method="combined")
                    
                    jn, ocr_conf, used_img, method = orient_corrector.recover_text_orientation(
                        roi_wide_enhanced, ocr_engine, is_tight_crop=False
                    )
                    if jn:
                        f_conf = combine_confidences(yolo_conf, ocr_conf, method=method)
                        if "color" in method.lower(): f_conf = min(1.0, f_conf + 0.10)
                        viz.save_ocr_debug(debug_stem, f"yolo_wide_{method}_enhanced", jn, f_conf, roi_img=used_img)
                        
                        if f_conf > best_conf:
                            best_jn, best_conf, best_method = jn, f_conf, f"yolo_wide_{method}_enhanced"

                if best_jn:
                    return best_jn, best_conf, best_method, {}

    # -------------------------------------------------------------------------
    # PHASE 1.5: MULTI-SPECTRUM HSV SNIPER (Targeted Region)
    # -------------------------------------------------------------------------
    # THE FIX: Restrict the HSV search space to the Bottom-Right Quadrant
    # This eliminates false positives from the rest of the page.
    x_start_hsv = int(w_proc * 0.40)
    y_start_hsv = int(h_proc * 0.40)
    br_search_region = oriented_image[y_start_hsv:h_proc, x_start_hsv:w_proc]

    if br_search_region.size > 0:
        # Enhance only the targeted region
        enhanced_image = enhancer.enhance(br_search_region, method="hsv")
        hsv_result = extract_blue_stamp(enhanced_image, ocr_engine)
        
        if hsv_result:
            hsv_job, hsv_conf, hsv_crop = hsv_result
            viz.save_ocr_debug(debug_stem, "hsv_sniper_enhanced", hsv_job, hsv_conf, roi_img=hsv_crop)
            return hsv_job, hsv_conf, "hsv_stamp_sniper_enhanced", {}

    # -------------------------------------------------------------------------
    # PHASE 2: TARGETED ANCHOR SEARCH (Fallback for massive stamps)
    # -------------------------------------------------------------------------
    # if TITLE_BLOCK_ENABLED:
    #     searcher = AnchorSearcher()
    #     x_start, y_start = int(w_proc * TITLE_BLOCK_X_START), int(h_proc * TITLE_BLOCK_Y_START)
    #     x_end = min(w_proc, x_start + int(w_proc * TITLE_BLOCK_WIDTH))
    #     y_end = min(h_proc, y_start + int(h_proc * TITLE_BLOCK_HEIGHT))
        
    #     tight_br_quad = oriented_image[y_start:y_end, x_start:x_end]
    #     if tight_br_quad.size > 0:
    #         viz.save_debug_roi(debug_stem, tight_br_quad, 0, "anchor_search_zone")
    #         anchor_regions = searcher.find_anchors_and_crop(tight_br_quad, TITLE_BLOCK_ANCHORS)
    #         for region_name, region_img in anchor_regions:
    #             jn, conf, used_img, method = orient_corrector.recover_text_orientation(
    #                 region_img, ocr_engine, is_tight_crop=False
    #             )
    #             if jn:
    #                 # Anchor search assumes a high likelihood of valid text placement
    #                 score = combine_confidences(None, conf, method=method) + 0.15 
    #                 viz.save_ocr_debug(debug_stem, f"anchor_{region_name}_{method}", jn, score, roi_img=used_img)
    #                 if score > 0.70: 
    #                     return jn, score, f"anchor_{region_name}_{method}", {}

    # -------------------------------------------------------------------------
    # PHASE 3: FAST QUADRANT FALLBACK (with enhancement)
    # -------------------------------------------------------------------------
    br_quad = oriented_image[int(h_proc * 0.45):h_proc, int(w_proc * 0.45):w_proc]
    if br_quad.size > 0:
        # ✅ ENHANCE FADED BLUE STAMP
        br_quad_enhanced = enhancer.enhance(br_quad, method="combined")
        jn, conf, used_img, method = orient_corrector.recover_text_orientation(
            br_quad_enhanced, ocr_engine, is_tight_crop=False
        )
        if jn: 
            return jn, conf, f"title_block_BR_{method}_enhanced", {}

    return None, 0.0, "failed", {"error": "all_methods_failed"}