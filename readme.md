# OCR Pipeline Architecture: Job Number Extraction (V3.1.12)

## 🎯 Overview
This project is a high-resilience, high-throughput Python OCR pipeline designed to extract engineering **Job Numbers** (Strict Format: `YY0XXX-SS`) from chaotic, noisy, and degraded schematic scans. 

Upgraded to the **Architecture**, the pipeline achieves 100% format extraction at an average speed of **2.84 seconds/file**. It utilizes a hyper-optimized **PaddleOCR (PP-OCRv4)** engine backed by physics-based image preprocessing, dynamic color-space masking, and a multi-tier topological spatial memory (Smart Filing) to heal missing or optically mutated digits.

---

## Pipeline Architecture

### 1. Vision Layer (`yolo_detector.py` & `hsv_stamp_extractor.py`)
* **Macro-Vision (YOLOv8):** Detects tight bounding boxes around potential text. We apply strict geometric filters (Aspect Ratio > 1.2, Area < 8% of page) to ignore massive decoy stamps and title blocks.
* **Multi-Spectrum HSV Sniper (Phase 1.5):** Captures large blue/green full-page stamps that YOLO geometry filters naturally ignore. It isolates the ink, crops the stamp, and routes it directly to the OCR engine with dynamic confidence scoring.

### 2. Preprocessing Layer (`preprocess.py`)
* **The 65px Upscale Lock:** Standard processing applies CLAHE contrast enhancement and Otsu binarization. Upscaling is locked at a 65px height threshold to prevent thick text from ballooning and fusing together.
* **The Color Mask Fallback:** Strips black CAD lines from the image to rescue overlapping red/blue handwritten corrections or stamps.
* **Faint Ink Recovery (`recover_faint_ink`):** A specialized fallback for faint, fragmented cyan dot-matrix stamps.
    * Uses **LAB Color Space (B-channel)** to isolate cyan ink.
    * Uses **Luminance filtering (L < 30)** to neutralize true-black CAD borders.
    * Uses **B-channel filtering (B > 118)** to neutralize red stamps and dark blue noise.

### 3. Extraction Layer (`dual_ocr.py`)
* **Engine:** PaddleOCR (PP-OCRv4).
* **Single-Pass Efficiency:** Utilizes PaddleOCR's native text-angle classifier (`use_angle_cls=True`) to handle horizontal, vertical, and skewed text in a single pass.
* **Stability Patches:** Forces `enable_mkldnn=False` and `rec_batch_num=1` to bypass memory fragmentation and C++ DLL crashes common in Windows CPU environments.

### 4. Validation Layer (`validation.py`)
* **Strict Format Enforcement:** Uses Regex to enforce the `YY0XXX-SS` format.
* **The Suffix Shield:** Enforces a strictly mandatory 2-digit suffix and utilizes a negative alphanumeric lookahead `(?![0-9A-Za-z])` to reject garbage bleeds or hallucinated double-suffixes.

### 5. Consensus Layer (`smart_filing.py`)
This layer acts as the pipeline's "Spatial Memory," healing OCR errors based on the sequence of files in a packet.
* **Tier 0 - Optical Healer:** Identifies records with exactly 1 mutated character in the base (e.g., an `8` fading into a `0`) by comparing them to surrounding neighbors.
* **Tier 1 - Sandwich Rule:** A deep "Bracket Fill" that looks for identical valid jobs separated by a gap (up to 5 files).
* **Tier 3 - Weighted Consensus:** Fills in completely failed reads based on the nearest valid high-confidence neighbors. Protected by an **Anti-Bulldozing Guardrail** to prevent dragging a number across actual job boundaries.

---

## 🛠️ Core Design Philosophy
* **Generalization over Overfitting:** No hardcoded string patches. All fixes are mathematically or physically grounded.
* **Non-Destructive Processing:** We avoid character-level image slicing and heavy morphological dilations that warp ink geometry.
* **Hardware Sandboxing:** Deep learning backends are explicitly configured for Windows stability to prevent threading lockups.

---

## 📊 Performance Benchmark
* **Accuracy:** ~100% Format Extraction / >95% Ground Truth.
* **Avg Speed:** 2.84 seconds per file.
* **Compute:** Multi-threaded CPU (ProcessPoolExecutor).