from pathlib import Path

import cv2
import numpy as np


def detect_lines(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection to keep structural boundaries.
    edges = cv2.Canny(gray, 50, 150)

    # Detect line segments that are likely fillable field guides.
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )

    detected_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_lines.append({"start": (int(x1), int(y1)), "end": (int(x2), int(y2))})

    return detected_lines


def is_horizontal(line, y_tolerance: int = 5) -> bool:
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return abs(y1 - y2) < y_tolerance


def line_length(line) -> float:
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def is_right_side(line, x_threshold: int = 300) -> bool:
    x1, _ = line["start"]
    x2, _ = line["end"]
    return (x1 + x2) / 2 > x_threshold


def is_near_text(line, ocr_data, max_vertical_distance: int = 30) -> bool:
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    line_y = (y1 + y2) / 2

    for item in ocr_data:
        text_y = item["center"][1]

        # Keep lines that sit close to a nearby OCR label.
        if abs(text_y - line_y) < max_vertical_distance:
            return True

    return False


def filter_field_lines(
    lines,
    ocr_data,
    min_length: int = 100,
    x_threshold: int = 300,
    max_vertical_distance: int = 30,
):
    filtered = []
    for line in lines:
        if (
            is_horizontal(line)
            and line_length(line) > min_length
            and is_right_side(line, x_threshold=x_threshold)
            and is_near_text(line, ocr_data, max_vertical_distance=max_vertical_distance)
        ):
            filtered.append(line)
    return filtered


def draw_lines(image_path: str, lines, output_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    for line in lines:
        x1, y1 = line["start"]
        x2, y2 = line["end"]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
