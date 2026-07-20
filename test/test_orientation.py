import pytest
import cv2
import numpy as np
import math
from pathlib import Path
from unittest.mock import MagicMock
import sys
import logging

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.extractor import get_stamp_rotation_angle, get_ocr_engine

# Set up logging to see debug prints during pytest
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class TestStampAngleDetection:

    @pytest.fixture(scope="class")
    def ocr_engine(self):
        """Load the OCR engine once per test class to save time."""
        return get_ocr_engine()

    # =========================================================================
    # UNIT TESTS: Verify the Math Logic in Isolation
    # =========================================================================

    def test_math_horizontal_text(self, ocr_engine):
        """Simulates a horizontal page with horizontal text."""
        # Draw horizontal lines to simulate a horizontal engineering drawing
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.line(img, (0, 20), (100, 20), (0, 0, 0), 2)
        cv2.line(img, (0, 50), (100, 50), (0, 0, 0), 2)

        mock_boxes = [
            [
                [
                    [[10.0, 10.0], [100.0, 10.0], [100.0, 30.0], [10.0, 30.0]],
                    ("JOB", 0.99),
                ]
            ]
        ]
        ocr_engine.reader.ocr = MagicMock(return_value=mock_boxes)

        angle = get_stamp_rotation_angle(ocr_engine, img)
        assert angle == 0, f"Expected 0, got {angle}"

    def test_math_90_cw_text(self, ocr_engine):
        """Simulates a vertical page with text reading top-to-bottom."""
        # Draw vertical lines to simulate a vertical engineering drawing
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.line(img, (20, 0), (20, 100), (0, 0, 0), 2)
        cv2.line(img, (50, 0), (50, 100), (0, 0, 0), 2)

        # Text reading top-to-bottom. p1 is top-right, p2 is bottom-right.
        mock_boxes = [
            [[[[90.0, 10.0], [90.0, 90.0], [70.0, 90.0], [70.0, 10.0]], ("JOB", 0.95)]]
        ]
        ocr_engine.reader.ocr = MagicMock(return_value=mock_boxes)

        angle = get_stamp_rotation_angle(ocr_engine, img)
        # Vector points DOWN (+90 deg). To fix, rotate 90 CCW (270 CW).
        assert angle == 270, f"Expected 270, got {angle}"

    def test_math_180_text(self, ocr_engine):
        """
        Simulates an upside-down page.
        Under the new 2-step logic, horizontal lines remain horizontal.
        The function SHOULD return 0, because YOLO will detect the horizontal
        stamp, and PaddleOCR's native angle classifier will fix the 180° text internally!
        """
        # Draw horizontal lines to simulate an upside-down engineering drawing
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.line(img, (0, 20), (100, 20), (0, 0, 0), 2)
        cv2.line(img, (0, 50), (100, 50), (0, 0, 0), 2)

        # p1 is bottom-right, p2 is bottom-left (upside down text)
        mock_boxes = [
            [[[[90.0, 90.0], [10.0, 90.0], [10.0, 70.0], [90.0, 70.0]], ("JOB", 0.90)]]
        ]
        ocr_engine.reader.ocr = MagicMock(return_value=mock_boxes)

        angle = get_stamp_rotation_angle(ocr_engine, img)

        # ✅ EXPECT 0: The page is structurally horizontal.
        # PaddleOCR's use_angle_cls=True will handle the upside-down text internally.
        assert (
            angle == 0
        ), f"Expected 0 (letting Paddle handle 180° internally), got {angle}"

    def test_math_270_cw_text(self, ocr_engine):
        """Simulates a vertical page with text reading bottom-to-top."""
        # Draw vertical lines to simulate a vertical engineering drawing
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.line(img, (20, 0), (20, 100), (0, 0, 0), 2)
        cv2.line(img, (50, 0), (50, 100), (0, 0, 0), 2)

        # Text reading bottom-to-top. p1 is bottom-left, p2 is top-left.
        mock_boxes = [
            [[[[10.0, 90.0], [10.0, 10.0], [30.0, 10.0], [30.0, 90.0]], ("JOB", 0.85)]]
        ]
        ocr_engine.reader.ocr = MagicMock(return_value=mock_boxes)

        angle = get_stamp_rotation_angle(ocr_engine, img)
        # Vector points UP (-90 deg). To fix, rotate 90 CW.
        assert angle == 90, f"Expected 90, got {angle}"

    # =========================================================================
    # INTEGRATION TEST: Diagnose the Real Image Failure
    # =========================================================================

    def test_real_image_diagnosis(self, ocr_engine):
        """
        Runs the actual function on a real failing image to print out
        exactly what PaddleOCR is seeing and where the math is breaking.
        """
        # Adjust this path to point to one of your failing rotated images
        failing_image_path = (
            PROJECT_ROOT
            / "00-output"
            / "debug"
            / "preprocessed"
            / "scan758_rotated_ready.jpg"
        )

        if not failing_image_path.exists():
            # Fallback: try to find ANY rotated image in the debug folder
            debug_dir = PROJECT_ROOT / "00-output" / "debug" / "preprocessed"
            if debug_dir.exists():
                rotated_imgs = list(debug_dir.glob("*rotated*.jpg"))
                if rotated_imgs:
                    failing_image_path = rotated_imgs[0]
                else:
                    pytest.skip("No rotated debug images found to test against.")
            else:
                pytest.skip("Debug directory not found.")

        logger.info(f"\n🔍 DIAGNOSING IMAGE: {failing_image_path.name}")
        img = cv2.imread(str(failing_image_path))
        assert img is not None, "Failed to load image"

        h, w = img.shape[:2]
        br_quad = img[int(h * 0.5) :, int(w * 0.4) :]

        # Run PaddleOCR manually to see the raw output
        logger.info("🏃 Running PaddleOCR detector on bottom-right quadrant...")
        raw_output = ocr_engine.reader.ocr(br_quad, cls=False)

        if not raw_output or not raw_output[0]:
            logger.error("❌ PaddleOCR found ZERO text in the bottom-right quadrant!")
            pytest.fail("PaddleOCR detector failed to find any text.")

        boxes = raw_output[0]
        logger.info(f"📦 PaddleOCR found {len(boxes)} text lines.")

        # Replicate the scoring logic to see which box it chooses
        best_box = None
        best_score = -999
        h_q, w_q = br_quad.shape[:2]

        for i, line in enumerate(boxes):
            box = line[0]
            text = line[1][0]
            conf = line[1][1]

            cx = sum([p[0] for p in box]) / 4
            cy = sum([p[1] for p in box]) / 4

            w_box = max([p[0] for p in box]) - min([p[0] for p in box])
            h_box = max([p[1] for p in box]) - min([p[1] for p in box])
            area = w_box * h_box

            dist_br = math.sqrt(((cx / w_q) - 1.0) ** 2 + ((cy / h_q) - 1.0) ** 2)
            score = (area / (w_q * h_q)) * 10.0 - dist_br * 3.0

            logger.info(
                f"  Line {i}: '{text}' (conf: {conf:.2f}) | Area: {area:.0f} | Dist_BR: {dist_br:.2f} | Score: {score:.2f}"
            )

            if score > best_score:
                best_score = score
                best_box = box
                logger.info(f"    👑 NEW BEST BOX")

        assert best_box is not None, "No best box selected"

        # Calculate the angle of the chosen box
        p1 = best_box[0]  # Top-left
        p2 = best_box[1]  # Top-right

        logger.info(f"📐 Best Box Points: p1={p1}, p2={p2}")

        angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        angle_deg = math.degrees(angle_rad)

        logger.info(f"🧮 Raw Calculated Angle: {angle_deg:.2f} degrees")

        # Run the actual function
        final_angle = get_stamp_rotation_angle(ocr_engine, img)
        logger.info(f"✅ FINAL RETURNED ANGLE: {final_angle}")

        # If it's a rotated stamp, the final angle SHOULD NOT be 0.
        if "rotated" in failing_image_path.name:
            assert (
                final_angle != 0
            ), "Function returned 0 for a rotated image! The math or box selection is flawed."
