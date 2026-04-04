import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# --- CORE APPLICATION PATHS & ENV LOADING ---
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR

# Explicitly load from the project root so it works even if launched from /scripts/
load_dotenv(BASE_DIR / ".env")  

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
# --- EXECUTION MODE & PATH ROUTING ---
# ============================================================================
WORK_ENV = os.getenv("WORK_ENV", "OFFICE").upper()
APP_MODE = os.getenv("APP_MODE", "PRODUCTION").upper()

# Dynamically pull the base paths from your .env file to protect privacy
input_key = f"{WORK_ENV}_INPUT_DEV" if APP_MODE == "DEVELOPMENT" else f"{WORK_ENV}_INPUT_PROD"
output_key = f"{WORK_ENV}_OUTPUT_DEV" if APP_MODE == "DEVELOPMENT" else f"{WORK_ENV}_OUTPUT_PROD"

try:
    INPUT_DIR = Path(os.getenv(input_key))
    # We pull the root output directory from .env
    RAW_OUTPUT_ROOT = Path(os.getenv(output_key))
except TypeError:
    raise ValueError(f"CRITICAL: Missing configuration in .env for {input_key} or {output_key}")

# --- SMART PATH REDIRECTION ---
if APP_MODE == "DEVELOPMENT":
    OUTPUT_DIR = RAW_OUTPUT_ROOT / "00-output-dev"
    LOG_LEVEL = "DEBUG"
    VERBOSE_OUTPUT = True
else:
    OUTPUT_DIR = RAW_OUTPUT_ROOT / "00-output-prod"
    LOG_LEVEL = "INFO"
    VERBOSE_OUTPUT = False  # Keep the console cleaner in prod

# ============================================================================
# --- VALIDATION SETTINGS & HITL CONSTANTS ---
# ============================================================================

VALIDATION_CONFIG = {
    "year_window_past": 4,      # Accept 4 years in past
    "year_window_future": 1,    # Accept 1 year in future
    "trust_ocr_threshold": 0.85, # The exact confidence required to bypass human review
}

# Link HITL variables to the single source of truth
TRUST_OCR_THRESHOLD = VALIDATION_CONFIG["trust_ocr_threshold"]

# --- TESTING CONTROLS ---
# Set to True to force 100% of files into the Holding Zone for manual Action Queue review
FORCE_MANUAL_REVIEW = True

FUZZY_MATCH_CUTOFF = 85
ALLOWED_SEPARATORS = "-. ="

SMART_FILING_CONFIG = {
    "neighbor_window": 2,          
    "max_physical_distance": 2,    
    "max_digit_dist": 1,
    "debug_mode": True,            
}

# ============================================================================
# --- DERIVED DIRECTORIES & INITIALIZATION ---
# ============================================================================

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

_REQUIRED_DIRS = [
    OUTPUT_DIR, REPORTS_DIR, DASHBOARD_DIR, LOG_DIR, 
    HOLDING_ZONE_DIR, DEBUG_BASE_DIR
] + list(DEBUG_FOLDERS.values())

for directory in _REQUIRED_DIRS: 
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# --- LOGGING & PDF SETTINGS ---
# ============================================================================

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
# --- YOLO DETECTION ---
# ============================================================================

YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "jobnum_detection_V2.pt"
YOLO_ENABLED = True
YOLO_CONF_DETECTION = 0.12   
ENABLE_DEBUG_VIZ = True      

# Blue Stamp Enhancement Configuration
BLUE_STAMP_ENHANCEMENT_CONFIG = {
    "enabled": True,
    "saturation_boost": 1.8,        
    "value_boost": 1.3,             
    "clahe_clip_limit": 4.0,        
    "clahe_tile_size": (8, 8),      
    "dilation_iterations": 1,       
    "kernel_size": 3,               
    "gamma": 0.7,                   
    "hue_range": (85, 130),         
    "min_saturation": 20,           
    "debug_mode": True,
    "try_multiple_methods": True    
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
    "enable_mkldnn": True,  
    "rec_batch_num": 1,
}

# ============================================================================
# --- PIPELINE CONTROL & BATCH PROCESSING ---
# ============================================================================

TEMPLATE_MATCHING_ENABLED = True
ENABLE_MULTI_ROTATION_OCR = True

# Parallel workers for batch processing. 
MAX_WORKERS = 4