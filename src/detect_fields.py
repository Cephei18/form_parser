from pathlib import Path

import cv2
import numpy as np


def _make_line(x1, y1, x2, y2, source: str = "line"):
    return {
        "start": (int(x1), int(y1)),
        "end": (int(x2), int(y2)),
        "field_type": source,
    }


def _line_key(line, y_bucket: int = 6, x_bucket: int = 12):
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return (
        round(min(x1, x2) / x_bucket),
        round(max(x1, x2) / x_bucket),
        round(((y1 + y2) / 2) / y_bucket),
    )


def deduplicate_detected_lines(lines):
    seen = set()
    deduped = []
    for line in sorted(lines, key=lambda item: (item["start"][1], item["start"][0])):
        key = _line_key(line)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    return deduped


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
            detected_lines.append(_make_line(x1, y1, x2, y2, "line"))

    return deduplicate_detected_lines(detected_lines)


def detect_additional_field_candidates(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    candidates = []
    candidates.extend(detect_rectangular_fields(gray))
    candidates.extend(detect_weak_or_dotted_lines(gray))
    return deduplicate_detected_lines(candidates)


def _count_ocr_inside_region(region, ocr_data, padding: float = 4.0):
    count = 0
    for item in ocr_data:
        bounds = _ocr_bounds(item)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if (
            region["x"] - padding <= cx <= region["x"] + region["width"] + padding
            and region["y"] - padding <= cy <= region["y"] + region["height"] + padding
        ):
            count += 1
    return count


def _count_lines_inside_region(region, lines):
    count = 0
    for line in lines:
        cx, cy = _line_center(line)
        if (
            region["x"] <= cx <= region["x"] + region["width"]
            and region["y"] <= cy <= region["y"] + region["height"]
        ):
            count += 1
    return count


def _image_density(binary, x, y, w, h):
    roi = binary[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0
    return cv2.countNonZero(roi) / float(roi.size)


def _avg_ocr_height(ocr_data, default: float = 18.0):
    heights = []
    for item in ocr_data:
        bounds = _ocr_bounds(item)
        if bounds is None:
            continue
        _, y1, _, y2 = bounds
        if y2 > y1:
            heights.append(y2 - y1)
    if not heights:
        return default
    return float(np.median(heights))


def detect_semantic_regions(image_path: str, ocr_data, lines):
    """Classify enclosed regions using structural cues, not label text."""
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_height, image_width = gray.shape[:2]
    page_area = float(image_height * image_width)
    avg_text_height = _avg_ocr_height(ocr_data)
    regions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < max(60, image_width * 0.04) or h < max(18, avg_text_height * 1.1):
            continue
        if w > image_width * 0.98 or h > image_height * 0.85:
            continue

        area = cv2.contourArea(contour)
        rect_area = float(w * h)
        if rect_area <= 0:
            continue

        rectangularity = area / rect_area
        aspect_ratio = w / float(max(h, 1))
        page_area_ratio = rect_area / page_area
        ink_density = _image_density(binary, x, y, w, h)
        ocr_count = _count_ocr_inside_region({"x": x, "y": y, "width": w, "height": h}, ocr_data)
        internal_line_count = _count_lines_inside_region({"x": x, "y": y, "width": w, "height": h}, lines)

        reasons = [
            f"aspect={aspect_ratio:.2f}",
            f"area_ratio={page_area_ratio:.4f}",
            f"ink_density={ink_density:.4f}",
            f"ocr_count={ocr_count}",
            f"internal_lines={internal_line_count}",
            f"rectangularity={rectangularity:.2f}",
        ]

        if rectangularity < 0.18:
            continue

        region_type = "standard_input"
        confidence = 0.55

        is_large_sparse = (
            page_area_ratio >= 0.012
            and h >= avg_text_height * 4.0
            and ocr_count <= 3
            and internal_line_count <= 2
            and ink_density < 0.18
        )
        is_multiline_region = (
            aspect_ratio >= 2.4
            and h >= avg_text_height * 2.2
            and internal_line_count >= 2
            and ocr_count <= max(2, internal_line_count)
        )

        if is_large_sparse:
            region_type = "non_text_sparse_region"
            confidence = 0.86
            reasons.append("large_enclosed_sparse_region")
        elif is_multiline_region:
            region_type = "multiline_text_region"
            confidence = 0.74
            reasons.append("aligned_repeated_field_lines")
        elif h > avg_text_height * 3.5 and ocr_count <= 1:
            region_type = "non_text_candidate"
            confidence = 0.66
            reasons.append("tall_low_text_enclosed_region")
        else:
            reasons.append("standard_field_geometry")

        regions.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "type": region_type,
                "confidence": round(confidence, 4),
                "reasons": reasons,
            }
        )

    return regions


def detect_rectangular_fields(gray):
    """Find input boxes and expose their bottom edge as field candidates."""
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fields = []
    image_height, image_width = gray.shape[:2]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 80 or h < 12 or h > 90:
            continue
        if w > image_width * 0.95 or h > image_height * 0.25:
            continue

        area = cv2.contourArea(contour)
        rect_area = float(w * h)
        if rect_area <= 0 or area / rect_area < 0.35:
            continue

        fields.append(_make_line(x, y + h, x + w, y + h, "box"))

    return fields


def detect_weak_or_dotted_lines(gray):
    """Find faint or dotted horizontal guides with morphology before Hough."""
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        9,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        horizontal,
        1,
        np.pi / 180,
        threshold=45,
        minLineLength=70,
        maxLineGap=25,
    )

    detected = []
    if lines is None:
        return detected

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) <= 8:
            detected.append(_make_line(x1, y1, x2, y2, "weak_line"))

    return detected


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
    excluded_regions=None,
):
    excluded_regions = excluded_regions or []
    filtered = []
    for line in lines:
        if any(_line_inside_region(line, region) for region in excluded_regions):
            continue
        if (
            is_horizontal(line)
            and line_length(line) > min_length
            and is_right_side(line, x_threshold=x_threshold)
            and is_near_text(line, ocr_data, max_vertical_distance=max_vertical_distance)
        ):
            filtered.append(line)
    return filtered


def _line_center(line):
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _line_inside_region(line, region) -> bool:
    cx, cy = _line_center(line)
    return (
        region["x"] <= cx <= region["x"] + region["width"]
        and region["y"] <= cy <= region["y"] + region["height"]
    )


def detect_table_regions(image_path: str):
    """Detect dense table-like areas using intersecting horizontal and vertical rules."""
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        9,
    )

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    table_mask = cv2.dilate(cv2.bitwise_and(horizontal, cv2.dilate(vertical, np.ones((5, 5), np.uint8))), np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    image_height, image_width = gray.shape[:2]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 120 or h < 60:
            continue
        if w > image_width * 0.98 or h > image_height * 0.75:
            continue

        roi_h = horizontal[y:y + h, x:x + w]
        roi_v = vertical[y:y + h, x:x + w]
        horizontal_count = cv2.countNonZero(roi_h) / max(w, 1)
        vertical_count = cv2.countNonZero(roi_v) / max(h, 1)

        if horizontal_count < 2.0 or vertical_count < 2.0:
            continue

        regions.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "region_type": "table",
        })

    return regions


def filter_lines_outside_table_regions(lines, table_regions):
    if not table_regions:
        return lines
    return [
        line for line in lines
        if not any(_line_inside_region(line, region) for region in table_regions)
    ]


def fallback_field_lines_from_ocr(image_path: str, ocr_data, min_width: int = 120):
    """Create conservative field candidates to the right of OCR labels when no lines exist."""
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    image_width = img.shape[1]
    fallback = []

    for item in ocr_data:
        bbox = item.get("bbox")
        center = item.get("center")
        if not bbox or not center or center[0] > image_width * 0.75:
            continue

        try:
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
        except Exception:
            continue

        start_x = int(max(xs) + 24)
        end_x = min(image_width - 40, start_x + 320)
        if end_x - start_x < min_width:
            continue

        y = int(max(ys) + 4)
        fallback.append(_make_line(start_x, y, end_x, y, "fallback_text_region"))

    return fallback


def _bbox_center(box):
    return (box["x"] + box["width"] / 2.0, box["y"] + box["height"] / 2.0)


def _nearest_checkbox_label(box, ocr_data, max_y_distance: float = 35.0, max_distance: float = 260.0):
    box_cx, box_cy = _bbox_center(box)
    best = None
    best_score = float("inf")

    for item in ocr_data:
        center = item.get("center")
        text = item.get("text")
        if not center or not text:
            continue

        dx = abs(float(center[0]) - box_cx)
        dy = abs(float(center[1]) - box_cy)
        if dy > max_y_distance:
            continue

        score = dx + (dy * 2)
        if score < best_score and score <= max_distance:
            best = item
            best_score = score

    return best


def _ocr_bounds(item):
    bbox = item.get("bbox")
    if not bbox:
        return None
    try:
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
    except Exception:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _overlaps_ocr_text(box, ocr_data, padding: float = 2.0) -> bool:
    x1 = box["x"] - padding
    y1 = box["y"] - padding
    x2 = box["x"] + box["width"] + padding
    y2 = box["y"] + box["height"] + padding

    for item in ocr_data:
        bounds = _ocr_bounds(item)
        if bounds is None:
            continue
        bx1, by1, bx2, by2 = bounds
        if x1 <= bx2 and x2 >= bx1 and y1 <= by2 and y2 >= by1:
            return True

    return False


def detect_checkbox_mappings(image_path: str, ocr_data):
    """Detect small square checkbox regions and attach them to nearby OCR labels."""
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 10 or h < 10 or w > 32 or h > 32:
            continue

        aspect = w / float(h)
        if aspect < 0.85 or aspect > 1.18:
            continue

        area = cv2.contourArea(contour)
        rect_area = float(w * h)
        if rect_area <= 0 or area / rect_area < 0.18 or area / rect_area > 0.95:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) < 4 or len(approx) > 6:
            continue

        box = {"x": float(x), "y": float(y), "width": float(w), "height": float(h)}
        if _overlaps_ocr_text(box, ocr_data):
            continue

        candidates.append(box)

    mappings = []
    seen = set()
    for box in candidates:
        key = (round(box["x"] / 4), round(box["y"] / 4), round(box["width"] / 4), round(box["height"] / 4))
        if key in seen:
            continue
        seen.add(key)

        label = _nearest_checkbox_label(box, ocr_data)
        if label is None:
            continue

        mappings.append({
            "label": label.get("text", "checkbox"),
            "label_pos": label.get("center", _bbox_center(box)),
            "field_lines": [],
            "field_bboxes": [{**box, "field_type": "checkbox"}],
            "field_type": "checkbox",
        })

    return mappings


def draw_lines(image_path: str, lines, output_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    for line in lines:
        x1, y1 = line["start"]
        x2, y2 = line["end"]
        color = (255, 0, 0)
        if line.get("field_type") == "box":
            color = (0, 180, 255)
        elif line.get("field_type") == "weak_line":
            color = (255, 0, 255)
        elif line.get("field_type") == "fallback_text_region":
            color = (0, 165, 255)
        cv2.line(img, (x1, y1), (x2, y2), color, 2)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
