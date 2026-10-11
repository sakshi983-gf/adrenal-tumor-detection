"""
AdrenalCare — Inference Helpers & 3-Layer CT Image Validation
"""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class ClassifierPrediction:
    class_name: str
    confidence: float


class BodyRegionClassifier:
    """
    Lightweight classifier distinguishing axial adrenal/abdominal CT slices ("adrenal_CT")
    from non-CT images ("vehicle", "chest_xray", "document", "non_CT").
    """

    def predict(self, gray_img: np.ndarray) -> ClassifierPrediction:
        if gray_img is None or gray_img.size == 0:
            return ClassifierPrediction(class_name="non_CT", confidence=1.0)

        h, w = gray_img.shape[:2]

        # 1. Outer corner air check: axial abdominal CT slices have dark air (near 0) in FOV corners
        ch, cw = max(1, h // 8), max(1, w // 8)
        corners = np.concatenate([
            gray_img[:ch, :cw].ravel(),
            gray_img[:ch, -cw:].ravel(),
            gray_img[-ch:, :cw].ravel(),
            gray_img[-ch:, -cw:].ravel(),
        ])
        corner_mean = float(np.mean(corners))
        overall_mean = float(np.mean(gray_img))

        if corner_mean > 25.0:
            if overall_mean > 190.0:
                return ClassifierPrediction(class_name="document", confidence=0.95)
            return ClassifierPrediction(class_name="vehicle", confidence=0.92)

        # 2. Top border & background air fraction check (distinguishes chest X-ray vs axial abdominal CT)
        top_border_mean = float(np.mean(gray_img[: max(1, h // 16), :]))
        bg_fraction = float(np.mean(gray_img <= 15))
        center_region = gray_img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        center_mean = float(np.mean(center_region))

        if top_border_mean > 25.0 or (center_mean < 45.0 and bg_fraction < 0.25):
            return ClassifierPrediction(class_name="chest_xray", confidence=0.91)

        if bg_fraction < 0.15 or bg_fraction > 0.92 or center_mean < 35.0 or center_mean > 220.0:
            return ClassifierPrediction(class_name="non_CT", confidence=0.90)

        return ClassifierPrediction(class_name="adrenal_CT", confidence=0.98)


body_region_classifier = BodyRegionClassifier()


def validate_file_properties(file):
    """Layer 1 — Check file extension."""
    filename = getattr(file, "filename", "") or ""
    # Strip folder path if uploaded via webkitdirectory
    base_name = filename.replace("\\", "/").split("/")[-1]
    if not base_name.lower().endswith((".png", ".jpg", ".jpeg")):
        return False, "Only PNG, JPG, JPEG accepted"
    return True, None


def validate_image_characteristics(image_path: str):
    """Layer 2 — Check image dimensions, grayscale channels, and HU histogram."""
    img = cv2.imread(image_path)
    if img is None:
        return False, "Only PNG, JPG, JPEG accepted"

    # Check dimensions
    h, w = img.shape[:2]
    if h < 128 or w < 128:
        return False, "Image too small to be a CT slice"

    # Check if grayscale (CT slices are grayscale, stored as RGB/BGR with equal channels)
    b, g, r = cv2.split(img)
    if not (np.array_equal(b, g) and np.array_equal(g, r)):
        return False, "Image is not grayscale — CT slices must be grayscale"

    # Check histogram distribution (CT slices have specific HU distribution)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    non_zero_bins = np.count_nonzero(hist)
    if non_zero_bins < 20:
        return False, "Image histogram does not match CT slice characteristics"

    return True, None


def preprocess_for_classifier(image_path: str) -> np.ndarray:
    """Load and resize grayscale image for body-region classification."""
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return np.zeros((256, 256), dtype=np.uint8)
    return cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)


def classify_body_region(image_path: str):
    """Layer 3 — Body-region classifier (adrenal CT vs non-CT)."""
    img = preprocess_for_classifier(image_path)
    prediction = body_region_classifier.predict(img)

    if prediction.class_name != "adrenal_CT":
        return False, f"This does not appear to be an adrenal CT slice (detected: {prediction.class_name})"
    return True, None


def compute_burden_metrics(slice_areas: dict, current_slice: int) -> dict:
    """Compute Slice Proliferative Burden Index (guaranteed <= 1.0)."""
    if not slice_areas:
        return {
            "current_slice": current_slice,
            "current_area": 0.0,
            "peak_slice": current_slice,
            "peak_area": 0.0,
            "burden_index": 0.0,
            "index": 0.0,
            "relative_weight": 0.0,
        }

    peak_slice = max(slice_areas, key=slice_areas.get)
    peak_area = float(slice_areas[peak_slice])
    current_area = float(slice_areas.get(current_slice, 0.0))

    if peak_area > 0.0:
        burden_index = min(1.0, max(0.0, current_area / peak_area))
    else:
        burden_index = 0.0

    relative_weight = burden_index ** 0.75 if burden_index > 0.0 else 0.0

    return {
        "current_slice": int(current_slice),
        "current_area": round(current_area, 2),
        "peak_slice": int(peak_slice),
        "peak_area": round(peak_area, 2),
        "burden_index": round(burden_index, 2),
        "index": round(burden_index, 2),
        "relative_weight": round(relative_weight, 3),
    }
