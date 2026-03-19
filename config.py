import os
import sys
from pathlib import Path

# ============================================================================
# --- ENVIRONMENT STABILITY OVERRIDES ---
# --- MUST BE SET BEFORE IMPORTING PADDLE/TORCH ---
# ============================================================================

# FIXED: OneDNN/MKLDNN must be explicitly ENABLED to provide context for fused_conv2d on Windows
os.environ["FLAGS_use_mkldnn"] = "1"
os.environ["FLAGS_enable_mkldnn"] = "1"
os.environ["FLAGS_enable_onednn"] = "1"
os.environ["ENABLE_ONEDNN"] = "1"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"

# DLL & Threading Stability
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_BLOCKTIME"] = "0"

# ============================================================================
# --- EXECUTION MODE (DEV vs PROD) ---
# ============================================================================
APP_MODE = os.getenv("APP_MODE", "DEVELOPMENT").upper()

# ============================================================================
# --- CORE APPLICATION PATHS ---
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR  # FIXED: Restored missing variable

ENV_PATHS = {
    "OFFICE": {
        "INPUT": Path("G:/Scans/preprocess/00-input/0-test"),
        "OUTPUT": Path("G:/Scans/preprocess/"),
    },
    "HOME": {
        "INPUT": Path("C:/Users/Kimera/Downloads/OCR/02-Scans/preprocess/00-input/02-batch"),
        "OUTPUT": Path("C:/Users/Kimera/Downloads/OCR/02-Scans/preprocess/00-output"),
    },
}

WORK_ENV = os.getenv("WORK_ENV", "OFFICE")
if WORK_ENV not in ENV_PATHS:
    raise ValueError(f"Unknown WORK_ENV: '{WORK_ENV}'.")

RAW_OUTPUT_ROOT = ENV_PATHS[WORK_ENV]["OUTPUT"]

# Define Mode-Specific variables ONCE here
if APP_MODE == "DEVELOPMENT":
    OUTPUT_DIR = RAW_OUTPUT_ROOT / "00-output-dev"
    VERBOSE_OUTPUT = True
    ENABLE_DEBUG_VIZ = True
    LOG_LEVEL = "DEBUG"
    YOLO_CONF_DETECTION = 0.12
    TRUST_OCR_THRESHOLD = 1.10  # High threshold for testing
else:
    OUTPUT_DIR = RAW_OUTPUT_ROOT / "00-output"
    VERBOSE_OUTPUT = False
    ENABLE_DEBUG_VIZ = False
    LOG_LEVEL = "INFO"
    YOLO_CONF_DETECTION = 0.25  # Tighter for production
    TRUST_OCR_THRESHOLD = 0.85  # Real-world threshold

# Derived Paths
INPUT_DIR = ENV_PATHS[WORK_ENV]["INPUT"]
REPORTS_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"
DASHBOARD_DIR = OUTPUT_DIR / "dashboard_data"
HOLDING_ZONE_DIR = OUTPUT_DIR / "holding_zone"
DEBUG_BASE_DIR = OUTPUT_DIR / "debug"

DEBUG_FOLDERS = {
    "preprocessed": DEBUG_BASE_DIR / "1_preprocess",
    "macro_vision": DEBUG_BASE_DIR / "2_macro_vision",
    "micro_vision": DEBUG_BASE_DIR / "3_micro_vision",
}
PREPROCESSED_DIR = DEBUG_FOLDERS["preprocessed"]

# ============================================================================
# --- DIRECTORY INITIALIZATION ---
# ============================================================================
_REQUIRED_DIRS = [OUTPUT_DIR, REPORTS_DIR, DASHBOARD_DIR, LOG_DIR, HOLDING_ZONE_DIR]
if APP_MODE == "DEVELOPMENT":
    _REQUIRED_DIRS += [DEBUG_BASE_DIR] + list(DEBUG_FOLDERS.values())

for directory in _REQUIRED_DIRS: 
    directory.mkdir(parents=True, exist_ok=True)
# ============================================================================
# --- LOGGING & PDF SETTINGS ---
# ============================================================================

# VERBOSE_OUTPUT = True
# LOG_LEVEL = "DEBUG"  
LOG_FILENAME = LOG_DIR / "scans_job_extraction.log"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  
LOG_FILE_BACKUP_COUNT = 5

PDF_DPI = 300
OUTPUT_IMAGE_FORMAT = "jpg"
OUTPUT_IMAGE_QUALITY = 95
INCLUDE_ROTATION_IN_FILENAME = True
SUPPORTED_EXTENSIONS = {"*.pdf", "*.jpg", "*.jpeg", "*.png"}
MIN_ROTATION_ANGLE_DETECTION = 5

# ============================================================================
# --- JOB NUMBER DETECTION (YOLO) ---
# ============================================================================

YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "jobnum_detection_V2.pt"
YOLO_CONF_DETECTION = 0.12  # Dropped from 0.25 to ~0.12 to act as a wide net
YOLO_ENABLED = True

# Blue Stamp Enhancement Configuration
BLUE_STAMP_ENHANCEMENT_CONFIG = {
    "enabled": True,
    "saturation_boost": 1.8,        # Boost blue saturation (1.0 = no change, 2.0 = double)
    "value_boost": 1.3,             # Boost blue brightness
    "clahe_clip_limit": 4.0,        # CLAHE contrast limit (higher = more contrast)
    "clahe_tile_size": (8, 8),      # CLAHE grid size
    "dilation_iterations": 1,       # Morphological dilation (restore faded strokes)
    "kernel_size": 3,               # Morphological kernel size
    "gamma": 0.7,                   # Gamma correction (<1.0 brightens, >1.0 darkens)
    "hue_range": (85, 130),         # Blue-green hue range in HSV
    "min_saturation": 20,           # Minimum saturation to consider as blue
    "debug_mode": True,
    "try_multiple_methods": True    # Try all methods and pick best result
}

# ============================================================================
# --- VALIDATION SETTINGS ---
# ============================================================================

VALIDATION_CONFIG = {
    "year_window_past": 4,      # Accept 5 years in past
    "year_window_future": 1,    # Accept 2 years in future (buffer for early scans)
    "trust_ocr_threshold": 0.70, # Confidence to override year validation
}
FUZZY_MATCH_CUTOFF = 85
ALLOWED_SEPARATORS = "-. ="

SMART_FILING_CONFIG = {
    "neighbor_window": 2,          
    "max_physical_distance": 2,    
    "max_digit_dist": 1,
    "debug_mode": True,            
}

# ============================================================================
# --- TITLE BLOCK & HEURISTICS ---
# ============================================================================

TITLE_BLOCK_ENABLED = True
TITLE_BLOCK_X_START = 0.55  
TITLE_BLOCK_Y_START = 0.65  
TITLE_BLOCK_WIDTH = 0.45
TITLE_BLOCK_HEIGHT = 0.35

# ============================================================================
# --- OCR CONFIGURATION (PaddleOCR) ---
# ============================================================================

PADDLE_OCR_CONFIG = {
    "use_angle_cls": True,
    "lang": "en",
    "use_gpu": False,
    "show_log": False,
    "enable_mkldnn": True,  # FIXED: Required for stability on Windows
    "rec_batch_num": 1,
}

# ============================================================================
# --- PIPELINE CONTROL & BATCH PROCESSING ---
# ============================================================================

TEMPLATE_MATCHING_ENABLED = True
ENABLE_MULTI_ROTATION_OCR = True
# ENABLE_DEBUG_VIZ = True

# Parallel workers for batch processing. 
# RECOMMENDED: 2-3 for 16GB RAM, 6-8 for 32GB+ RAM (~3GB per worker)
MAX_WORKERS = 4

# ============================================================================
# --- PRODUCTION HITL SETTINGS ---
# ============================================================================
# Files with confidence below this score will be flagged for manual review
TRUST_OCR_THRESHOLD = 1.10