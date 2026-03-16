# src/utils/pdf_utils.py
"""
PDF utilities for conversion and handling.
Integrated with the pipeline's image processing workflow.
"""
import logging
from pathlib import Path
from typing import List, Optional
from pdf2image import convert_from_path

import numpy as np
import cv2

from config import PDF_DPI

logger = logging.getLogger(__name__)

# GENERALIZATION: Prevents massive engineering schematics from locking the CPU.
MAX_SAFE_DIMENSION = 6000 

def pdf_to_images(
    pdf_path: str,
    dpi: int = PDF_DPI,
    max_pages: int = 10,
) -> List[np.ndarray]:
    """
    Convert a PDF to a list of OpenCV images (BGR format).
    Includes a dynamic resolution cap to prevent CPU lockups during OCR.
    """
    pdf_path_obj = Path(pdf_path)
    
    if not pdf_path_obj.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return []
    
    try:
        logger.debug(f"Converting PDF to images: {pdf_path_obj.name} (DPI={dpi}, max_pages={max_pages})")
        
        # Convert PDF pages to PIL images
        pil_images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=1,
            last_page=max_pages,
            fmt="jpeg",
            thread_count=4,
        )
        
        cv2_images = []
        for img in pil_images:
            cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            h, w = cv2_img.shape[:2]
            
            # --- SAFE-RES DOWNSCALING ---
            # If the image exceeds our max dimension, scale it down proportionally
            if max(h, w) > MAX_SAFE_DIMENSION:
                scale_ratio = MAX_SAFE_DIMENSION / max(h, w)
                new_size = (int(w * scale_ratio), int(h * scale_ratio))
                cv2_img = cv2.resize(cv2_img, new_size, interpolation=cv2.INTER_AREA)
                logger.debug(f"Downscaled massive image from {w}x{h} to {new_size[0]}x{new_size[1]}")
                
            cv2_images.append(cv2_img)
        
        logger.info(f"Converted {len(cv2_images)} pages from {pdf_path_obj.name}")
        return cv2_images
    
    except Exception as e:
        logger.error(f"PDF conversion failed for {pdf_path}: {e}")
        return []

def save_image(
    image: np.ndarray,
    output_path: str,
    quality: int = 95
) -> bool:
    try:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success = cv2.imwrite(str(output_path), image, params)
        
        if success:
            logger.debug(f"Saved image: {output_path_obj.name}")
        else:
            logger.error(f"Failed to save image: {output_path}")
        
        return success
    except Exception as e:
        logger.error(f"Error saving image {output_path}: {e}")
        return False

def get_page_dimensions(image: np.ndarray) -> tuple:
    return image.shape

def validate_image(image: Optional[np.ndarray]) -> bool:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return False
    return True

def batch_pdf_to_images(
    pdf_paths: List[str],
    dpi: int = PDF_DPI,
    max_pages: int = 10,
) -> dict:
    results = {}
    for pdf_path in pdf_paths:
        logger.debug(f"Processing: {Path(pdf_path).name}")
        images = pdf_to_images(pdf_path, dpi=dpi, max_pages=max_pages)
        results[pdf_path] = images
    
    logger.info(f"Batch conversion complete: {len(results)} PDFs processed")
    return results