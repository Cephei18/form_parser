from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.pipeline_config import PreprocessingConfig


@dataclass(frozen=True)
class PreprocessingResult:
    input_path: Path
    working_path: Path
    diagnostics: dict[str, Any]


def _image_size(image: np.ndarray) -> dict[str, int]:
    height, width = image.shape[:2]
    return {"width": int(width), "height": int(height)}


def _enhance_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _sharpen(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    return cv2.addWeighted(image, 1.45, blurred, -0.45, 0)


def _adaptive_threshold(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    return cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)


def _estimate_skew_degrees(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 50:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if not math.isfinite(angle):
        return 0.0
    return float(angle)


def _deskew(image: np.ndarray, max_skew_degrees: float) -> tuple[np.ndarray, float, bool]:
    angle = _estimate_skew_degrees(image)
    if abs(angle) < 0.25 or abs(angle) > max_skew_degrees:
        return image, angle, False

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    corrected = cv2.warpAffine(
        image,
        rotation,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected, angle, True


def _normalize_long_edge(image: np.ndarray, target_long_edge: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= 0:
        return image, 1.0

    scale = target_long_edge / float(longest)
    if abs(scale - 1.0) < 0.15:
        return image, 1.0

    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return resized, scale


def preprocess_image(
    image_path: Path,
    output_dir: Path,
    config: PreprocessingConfig,
) -> PreprocessingResult:
    diagnostics: dict[str, Any] = {
        "enabled": config.enabled,
        "input_path": str(image_path),
        "working_path": str(image_path),
        "steps": [],
        "config": config.to_dict(),
    }

    if not config.enabled:
        diagnostics["steps"].append({"name": "preprocessing_disabled", "applied": False})
        return PreprocessingResult(image_path, image_path, diagnostics)

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image for preprocessing: {image_path}")

    diagnostics["original_size"] = _image_size(image)
    working = image
    coordinate_scale = 1.0

    if config.dpi_normalization:
        working, coordinate_scale = _normalize_long_edge(working, config.target_long_edge)
        diagnostics["steps"].append(
            {
                "name": "dpi_normalization",
                "applied": coordinate_scale != 1.0,
                "coordinate_scale": round(coordinate_scale, 6),
            }
        )

    if config.denoise:
        working = cv2.fastNlMeansDenoisingColored(working, None, 5, 5, 7, 21)
        diagnostics["steps"].append({"name": "denoise", "applied": True})

    if config.contrast_enhancement:
        working = _enhance_contrast(working)
        diagnostics["steps"].append({"name": "contrast_enhancement", "applied": True})

    if config.sharpen:
        working = _sharpen(working)
        diagnostics["steps"].append({"name": "sharpen", "applied": True})

    if config.skew_correction:
        working, angle, applied = _deskew(working, config.max_skew_degrees)
        diagnostics["steps"].append(
            {
                "name": "skew_correction",
                "applied": applied,
                "estimated_angle_degrees": round(angle, 4),
            }
        )

    if config.adaptive_threshold:
        working = _adaptive_threshold(working)
        diagnostics["steps"].append({"name": "adaptive_threshold", "applied": True})

    output_path = output_dir / "preprocessed.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), working):
        raise RuntimeError(f"Failed to write preprocessed image: {output_path}")

    diagnostics["working_path"] = str(output_path)
    diagnostics["final_size"] = _image_size(working)
    diagnostics["coordinate_scale"] = round(coordinate_scale, 6)
    return PreprocessingResult(image_path, output_path, diagnostics)
