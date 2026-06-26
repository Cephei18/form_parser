from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable

import cv2
import numpy as np

from src.answer_region_engine import answer_region_v2_enabled, build_answer_regions
from src.assignment_solver import (
    global_assignment_enabled,
    region_identity,
    solve_global_assignment,
)
from src.checkbox_validator import (
    checkbox_validation_enabled,
    checkbox_validation_observe_enabled,
    validate_checkboxes,
)
from src.comb_detector import apply_comb_detection
from src.comb_diagnostics import analyze_comb_groups, comb_diagnostics_enabled
from src.confidence_pipeline import confidence_pipeline_enabled, review_queue_enabled
from src.dotted_leader_diagnostics import analyze_dotted_leaders, dotted_leader_diagnostics_enabled
from src.dotted_underline_detector import detect_dotted_underlines, split_inline_answer_region
from src.radio_grouper import apply_radio_grouping
from src.section_detector import SectionIndex, detect_sections, qualify_label, section_summary
from src.semantic_classifier import (
    apply_semantic_priors,
    build_semantic_diagnostics,
    candidate_region_family,
    classify_field,
    semantics_enabled,
)
from src.table_intelligence import build_tables, table_intelligence_enabled

logger = logging.getLogger("form_parser.field_anchor")


PHOTO_KEYWORDS = {
    "photo",
    "photograph",
    "passport size",
    "affix",
}

SIGNATURE_KEYWORDS = {
    "signature",
    "signed",
    "sign here",
}

MULTILINE_KEYWORDS = {
    "address",
    "remarks",
    "details",
    "description",
    "particulars",
    "reason",
    "comments",
    "experience",
    "qualification",
}

CHECKBOX_VALUE_TOKENS = {"[x]", "[ ]", "x", "yes", "no"}

# --- Textract-pipeline-only tunables (safe to revert; OCR path never reads these) ---
# Answer regions anchored on an already-printed feature must NOT get a second
# artificial border drawn over them, otherwise the printed line and the widget
# underline stack into a cluttered double line (Batch A).
PRINTED_FEATURE_ANCHORS = {
    "underline",
    "empty_rectangle",
    "table_cell",
    "dotted_underline",
    "broken_underline",
    "inline_dotted_field",
}

# Photo / decorative placeholder detection, as fractions of the page (Batch B).
PHOTO_REGION_MIN_AREA = 0.012
PHOTO_REGION_MIN_HEIGHT = 0.07
PHOTO_REGION_MIN_ASPECT = 0.5
PHOTO_REGION_MAX_ASPECT = 1.7
PHOTO_REGION_MAX_TEXT = 1
PHOTO_OVERLAP_SUPPRESS = 0.55

# Confidence floor below which an unresolved (purely estimated) field is dropped
# instead of rendered as clutter.
UNRESOLVED_DROP_FLOOR = 0.2

# Whether to drop checkboxes Textract could not associate with any owner.
# IMPORTANT: defaults to False. Standalone / agreement / option-list checkboxes
# are frequently reported by Textract as "unassociated" yet are real, fillable
# widgets — dropping them made checkboxes disappear from many forms. We now flag
# them in diagnostics instead. Set to True only for a deliberate audit pass.
DROP_UNASSOCIATED_CHECKBOXES = False

# Deduplicate "[X]"/"[ ]" token checkboxes that KEY_VALUE_SET parsing emits as
# field items against the canonical SELECTION_ELEMENT checkbox at the same spot.
# A token checkbox is suppressed ONLY when it overlaps a real SELECTION_ELEMENT
# region (text values like "yes"/"no" never overlap one, so they are untouched).
# Reversible via this flag.
DEDUPE_TOKEN_CHECKBOXES = True
SELECTION_DEDUPE_OVERLAP = 0.4


def _relationship_ids(block: dict[str, Any], relationship_type: str | None = None) -> list[str]:
    ids: list[str] = []
    for relationship in block.get("Relationships", []) or []:
        if relationship_type is not None and relationship.get("Type") != relationship_type:
            continue
        ids.extend(relationship.get("Ids", []) or [])
    return ids


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _round_box(box: dict[str, float]) -> dict[str, float]:
    return {
        "x": round(float(box["x"]), 6),
        "y": round(float(box["y"]), 6),
        "width": round(float(box["width"]), 6),
        "height": round(float(box["height"]), 6),
    }


def _normalize_box(box: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(box, dict):
        return None
    try:
        x = float(box.get("x", box.get("Left", 0.0)))
        y = float(box.get("y", box.get("Top", 0.0)))
        width = float(box.get("width", box.get("Width", 0.0)))
        height = float(box.get("height", box.get("Height", 0.0)))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    x = _clamp(x)
    y = _clamp(y)
    width = max(0.0001, min(width, 1.0 - x))
    height = max(0.0001, min(height, 1.0 - y))
    return {"x": x, "y": y, "width": width, "height": height}


def _block_bbox(block: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(block, dict):
        return None
    geometry = block.get("Geometry") or {}
    return _normalize_box(geometry.get("BoundingBox"))


def _parsed_geometry_bbox(geometry: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(geometry, dict):
        return None
    return _normalize_box(geometry.get("bounding_box") or geometry.get("BoundingBox"))


def _box_right(box: dict[str, float]) -> float:
    return float(box["x"]) + float(box["width"])


def _box_bottom(box: dict[str, float]) -> float:
    return float(box["y"]) + float(box["height"])


def _box_center(box: dict[str, float]) -> tuple[float, float]:
    return (float(box["x"]) + float(box["width"]) / 2.0, float(box["y"]) + float(box["height"]) / 2.0)


def _box_area(box: dict[str, float]) -> float:
    return max(0.0, float(box["width"])) * max(0.0, float(box["height"]))


def _intersection_area(a: dict[str, float], b: dict[str, float]) -> float:
    x1 = max(float(a["x"]), float(b["x"]))
    y1 = max(float(a["y"]), float(b["y"]))
    x2 = min(_box_right(a), _box_right(b))
    y2 = min(_box_bottom(a), _box_bottom(b))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _overlap_ratio(a: dict[str, float], b: dict[str, float]) -> float:
    return _intersection_area(a, b) / max(_box_area(a), 0.000001)


def _bbox_union(boxes: list[dict[str, float]]) -> dict[str, float] | None:
    clean = [box for box in boxes if _normalize_box(box)]
    if not clean:
        return None
    x1 = min(float(box["x"]) for box in clean)
    y1 = min(float(box["y"]) for box in clean)
    x2 = max(_box_right(box) for box in clean)
    y2 = max(_box_bottom(box) for box in clean)
    return _normalize_box({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1})


def _expand_box(box: dict[str, float], x_pad: float = 0.0, y_pad: float = 0.0) -> dict[str, float]:
    return _normalize_box(
        {
            "x": float(box["x"]) - x_pad,
            "y": float(box["y"]) - y_pad,
            "width": float(box["width"]) + x_pad * 2.0,
            "height": float(box["height"]) + y_pad * 2.0,
        }
    ) or dict(box)


def _box_to_pixel_box(box: dict[str, float], image_width: int, image_height: int) -> dict[str, float]:
    return {
        "x": float(box["x"]) * image_width,
        "y": float(box["y"]) * image_height,
        "width": float(box["width"]) * image_width,
        "height": float(box["height"]) * image_height,
    }


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalized_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean_text(value).lower())


def _field_label_slug(value: Any) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"^\s*\d+[\s.)\-:]*", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _contains_any(text: str, keywords: set[str]) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in keywords)


def _confidence_class(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.82:
        return "high"
    if score >= 0.62:
        return "medium"
    if score >= 0.42:
        return "low"
    return "unresolved"


def _build_parent_map(blocks: list[dict[str, Any]]) -> dict[str, list[str]]:
    parents: dict[str, list[str]] = defaultdict(list)
    for block in blocks:
        parent_id = block.get("Id")
        if not parent_id:
            continue
        for child_id in _relationship_ids(block, "CHILD"):
            parents[child_id].append(parent_id)
        for value_id in _relationship_ids(block, "VALUE"):
            parents[value_id].append(parent_id)
    return parents


def _build_page_index(blocks: list[dict[str, Any]]) -> dict[str, int]:
    page_by_id: dict[str, int] = {}
    page_blocks = [block for block in blocks if block.get("BlockType") == "PAGE"]
    for index, block in enumerate(page_blocks, start=1):
        block_id = block.get("Id")
        if block_id:
            page_by_id[block_id] = index
        for child_id in _relationship_ids(block, "CHILD"):
            page_by_id[child_id] = index

    for block in blocks:
        block_id = block.get("Id")
        if not block_id:
            continue
        page = block.get("Page")
        if page is not None:
            try:
                page_by_id[block_id] = int(page)
            except (TypeError, ValueError):
                pass

    # Propagate page ownership through child relationships. Textract response
    # shapes differ across APIs, so this keeps the anchoring layer page-aware
    # without assuming all child blocks carry a Page field.
    changed = True
    while changed:
        changed = False
        for block in blocks:
            block_id = block.get("Id")
            if not block_id or block_id not in page_by_id:
                continue
            page = page_by_id[block_id]
            for child_id in _relationship_ids(block, "CHILD") + _relationship_ids(block, "VALUE"):
                if child_id not in page_by_id:
                    page_by_id[child_id] = page
                    changed = True

    return page_by_id


def _block_page(block: dict[str, Any] | None, page_by_id: dict[str, int], default: int = 1) -> int:
    if not isinstance(block, dict):
        return default
    block_id = block.get("Id")
    if block_id and block_id in page_by_id:
        return page_by_id[block_id]
    try:
        return int(block.get("Page") or default)
    except (TypeError, ValueError):
        return default


def _block_text(block_id: str, blocks_by_id: dict[str, dict[str, Any]], cache: dict[str, str]) -> str:
    if block_id in cache:
        return cache[block_id]
    block = blocks_by_id.get(block_id)
    if not block:
        cache[block_id] = ""
        return ""
    block_type = block.get("BlockType")
    if block_type == "WORD":
        text = str(block.get("Text") or "")
    elif block_type == "LINE":
        text = str(block.get("Text") or "")
    elif block_type == "SELECTION_ELEMENT":
        text = "[X]" if block.get("SelectionStatus") == "SELECTED" else "[ ]"
    else:
        parts = [_block_text(child_id, blocks_by_id, cache) for child_id in _relationship_ids(block, "CHILD")]
        text = " ".join(part for part in parts if part).strip()
    cache[block_id] = text
    return text


def _median(values: list[float], default: float) -> float:
    clean = [float(value) for value in values if isinstance(value, (int, float)) and value > 0]
    if not clean:
        return default
    return float(median(clean))


def _visual_text_boxes(blocks: list[dict[str, Any]], page_by_id: dict[str, int]) -> list[dict[str, Any]]:
    text_boxes: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("BlockType") not in {"LINE", "WORD"}:
            continue
        bbox = _block_bbox(block)
        if not bbox:
            continue
        text_boxes.append(
            {
                "bbox": bbox,
                "text": _clean_text(block.get("Text")),
                "page": _block_page(block, page_by_id),
            }
        )
    return text_boxes


def _line_boxes(blocks: list[dict[str, Any]], page_by_id: dict[str, int]) -> list[dict[str, Any]]:
    """LINE-level text boxes used for section-heading detection. LINE blocks
    carry the full heading text (their WORD children joined), which is what the
    section detector needs."""
    lines: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("BlockType") != "LINE":
            continue
        bbox = _block_bbox(block)
        if not bbox:
            continue
        lines.append(
            {
                "block_id": block.get("Id"),
                "text": _clean_text(block.get("Text")),
                "bbox": bbox,
                "page": _block_page(block, page_by_id),
            }
        )
    return lines


def _build_page_metrics(text_boxes: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in text_boxes:
        by_page[int(item.get("page") or 1)].append(item)
    if not by_page:
        by_page[1] = []

    metrics_by_page: dict[int, dict[str, float]] = {}
    for page, boxes in by_page.items():
        heights = [float(item["bbox"]["height"]) for item in boxes]
        widths = [float(item["bbox"]["width"]) for item in boxes]
        right_edges = [_box_right(item["bbox"]) for item in boxes]
        left_edges = [float(item["bbox"]["x"]) for item in boxes]
        line_height = _median(heights, 0.014)
        metrics_by_page[page] = {
            "line_height": line_height,
            "text_width": _median(widths, 0.12),
            "row_tolerance": max(line_height * 1.85, 0.014),
            "wide_row_tolerance": max(line_height * 3.0, 0.024),
            "x_gap": max(line_height * 0.75, 0.008),
            "page_left": max(0.02, min(left_edges) - 0.015) if left_edges else 0.05,
            "page_right": min(0.98, max(right_edges) + 0.04) if right_edges else 0.95,
        }
    if 1 not in metrics_by_page:
        metrics_by_page[1] = {
            "line_height": 0.014,
            "text_width": 0.12,
            "row_tolerance": 0.026,
            "wide_row_tolerance": 0.042,
            "x_gap": 0.01,
            "page_left": 0.05,
            "page_right": 0.95,
        }
    return metrics_by_page


def _feature_box_from_pixels(x: int, y: int, width: int, height: int, page_width: int, page_height: int) -> dict[str, float]:
    return _round_box(
        {
            "x": x / float(page_width),
            "y": y / float(page_height),
            "width": width / float(page_width),
            "height": height / float(page_height),
        }
    )


def _text_count_inside(box: dict[str, float], text_boxes: list[dict[str, Any]], page: int, padding: float = 0.0) -> int:
    count = 0
    padded = _expand_box(box, padding, padding)
    for item in text_boxes:
        if int(item.get("page") or 1) != page:
            continue
        text_box = item["bbox"]
        center_x, center_y = _box_center(text_box)
        if padded["x"] <= center_x <= _box_right(padded) and padded["y"] <= center_y <= _box_bottom(padded):
            count += 1
    return count


def _detect_visual_features(image_path: Path, text_boxes: list[dict[str, Any]], page: int = 1) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("[anchor] unable to read image for visual feature detection: %s", image_path)
        return {"underlines": [], "empty_boxes": [], "synthetic_underlines": [], "dotted_underline_debug": {}, "image_size": None}

    image_height, image_width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_kernel_width = max(18, int(image_width * 0.028))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    underlines: list[dict[str, Any]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < max(24, image_width * 0.045):
            continue
        if height > max(8, image_height * 0.012):
            continue
        box = _feature_box_from_pixels(x, y, width, max(height, 1), image_width, image_height)
        underlines.append(
            {
                "bbox": box,
                "page": page,
                "type": "underline",
                "confidence": 0.76,
                "length": round(box["width"], 6),
            }
        )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    empty_boxes: list[dict[str, Any]] = []
    page_area = float(image_width * image_height)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = float(width * height)
        if area < page_area * 0.00025 or area > page_area * 0.22:
            continue
        if width < max(16, image_width * 0.015) or height < max(10, image_height * 0.008):
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        rectangularity = cv2.contourArea(contour) / max(area, 1.0)
        if rectangularity < 0.16:
            continue
        box = _feature_box_from_pixels(x, y, width, height, image_width, image_height)
        text_count = _text_count_inside(box, text_boxes, page=page, padding=0.002)
        aspect_ratio = width / max(float(height), 1.0)
        empty_boxes.append(
            {
                "bbox": box,
                "page": page,
                "type": "empty_rectangle",
                "confidence": round(_clamp(0.62 + rectangularity * 0.28 - min(text_count, 3) * 0.08), 4),
                "text_count": text_count,
                "aspect_ratio": round(aspect_ratio, 4),
                "rectangularity": round(rectangularity, 4),
            }
        )

    dotted_debug = detect_dotted_underlines(image_path, page=page)
    synthetic_underlines = dotted_debug.get("detected", []) or []

    return {
        "underlines": _dedupe_feature_boxes(underlines),
        "empty_boxes": _dedupe_feature_boxes(empty_boxes),
        "synthetic_underlines": _dedupe_feature_boxes(synthetic_underlines),
        "dotted_underline_debug": {
            "source_image": str(image_path),
            "page": page,
            "detected": synthetic_underlines,
            "rejected": dotted_debug.get("rejected", []) or [],
        },
        "image_size": {"width": image_width, "height": image_height},
    }


def _detect_visual_features_for_pages(
    page_images: dict[int, Path],
    text_boxes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run CV underline/rectangle detection once per page and tag every feature
    with its page number. Page N's features come from Page N's raster, so a
    field on page N can only ever anchor onto page N visual features (the
    candidate builders already filter by ``feature["page"] == page``).

    Returns the same shape as ``_detect_visual_features`` but with an
    ``image_sizes`` map (page -> {width, height}) instead of a single
    ``image_size``.
    """
    underlines: list[dict[str, Any]] = []
    empty_boxes: list[dict[str, Any]] = []
    synthetic_underlines: list[dict[str, Any]] = []
    dotted_debug_pages: list[dict[str, Any]] = []
    image_sizes: dict[int, dict[str, int]] = {}

    for page in sorted(page_images):
        features = _detect_visual_features(Path(page_images[page]), text_boxes, page=page)
        underlines.extend(features.get("underlines", []) or [])
        empty_boxes.extend(features.get("empty_boxes", []) or [])
        synthetic_underlines.extend(features.get("synthetic_underlines", []) or [])
        dotted_debug = features.get("dotted_underline_debug") or {}
        if dotted_debug:
            dotted_debug_pages.append(dotted_debug)
        size = features.get("image_size")
        if size:
            image_sizes[page] = size

    return {
        "underlines": underlines,
        "empty_boxes": empty_boxes,
        "synthetic_underlines": synthetic_underlines,
        "dotted_underline_debug": dotted_debug_pages,
        "image_sizes": image_sizes,
    }


def _dedupe_feature_boxes(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(features, key=lambda item: (item.get("page", 1), item["bbox"]["y"], item["bbox"]["x"]))
    deduped: list[dict[str, Any]] = []
    for feature in ordered:
        box = feature["bbox"]
        if any(_intersection_area(box, existing["bbox"]) / max(min(_box_area(box), _box_area(existing["bbox"])), 0.000001) > 0.78 for existing in deduped):
            continue
        deduped.append(feature)
    return deduped


def _split_photo_regions(empty_boxes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate large near-square decorative/photo placeholders from genuine
    input rectangles so no field widget is ever anchored on top of them."""
    photo_regions: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for feature in empty_boxes or []:
        box = feature.get("bbox") or {}
        try:
            width = float(box["width"])
            height = float(box["height"])
        except (KeyError, TypeError, ValueError):
            remaining.append(feature)
            continue
        area = width * height
        aspect = width / max(height, 1e-6)
        text_count = int(feature.get("text_count") or 0)
        if (
            area >= PHOTO_REGION_MIN_AREA
            and height >= PHOTO_REGION_MIN_HEIGHT
            and PHOTO_REGION_MIN_ASPECT <= aspect <= PHOTO_REGION_MAX_ASPECT
            and text_count <= PHOTO_REGION_MAX_TEXT
        ):
            enriched = dict(feature)
            enriched["type"] = "photo_region"
            photo_regions.append(enriched)
        else:
            remaining.append(feature)
    return photo_regions, remaining


def _overlaps_photo_region(box: dict[str, float], page: int, photo_regions: list[dict[str, Any]]) -> bool:
    for feature in photo_regions or []:
        if int(feature.get("page") or 1) != page:
            continue
        if _overlap_ratio(box, feature["bbox"]) >= PHOTO_OVERLAP_SUPPRESS:
            return True
    return False


def _selection_regions(blocks: list[dict[str, Any]], page_by_id: dict[str, int]) -> list[dict[str, Any]]:
    """Canonical checkbox regions: the geometry of every SELECTION_ELEMENT."""
    regions: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("BlockType") != "SELECTION_ELEMENT":
            continue
        bbox = _block_bbox(block)
        if not bbox:
            continue
        regions.append({"page": _block_page(block, page_by_id, 1), "bbox": bbox})
    return regions


def _overlaps_selection_region(box: dict[str, float], page: int, selection_regions: list[dict[str, Any]]) -> bool:
    """True when ``box`` covers a SELECTION_ELEMENT (so a token checkbox at this
    location is a duplicate of that native checkbox). Uses center-containment OR
    strong overlap relative to the small selection glyph — deterministic and
    conservative so independent checkboxes elsewhere are never matched."""
    for region in selection_regions:
        if int(region.get("page") or 1) != page:
            continue
        selection = region["bbox"]
        center_x, center_y = _box_center(selection)
        if box["x"] <= center_x <= _box_right(box) and box["y"] <= center_y <= _box_bottom(box):
            return True
        if _intersection_area(box, selection) / max(_box_area(selection), 1e-9) >= SELECTION_DEDUPE_OVERLAP:
            return True
    return False


def _table_cells(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for table_index, table in enumerate(parsed.get("tables", []) or [], start=1):
        page = int(table.get("page") or 1)
        table_id = table.get("table_block_id") or f"table_{table_index}"
        for cell in table.get("cells", []) or []:
            bbox = _parsed_geometry_bbox(cell.get("geometry") or {})
            if not bbox:
                continue
            cells.append(
                {
                    "table_id": table_id,
                    "cell_id": cell.get("cell_id"),
                    "page": page,
                    "row_index": int(cell.get("row_index") or 0),
                    "column_index": int(cell.get("column_index") or 0),
                    "row_span": int(cell.get("row_span") or 1),
                    "column_span": int(cell.get("column_span") or 1),
                    "text": _clean_text(cell.get("text")),
                    "bbox": bbox,
                    "confidence": _safe_float(cell.get("confidence"), 0.0),
                }
            )
    return cells


def _classify_field(label: str, value: str, answer_box: dict[str, float] | None, metrics: dict[str, float]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    lower_label = label.lower()
    lower_value = value.lower().strip()

    if _contains_any(lower_label, PHOTO_KEYWORDS):
        reasons.append("photo_keyword")
        return "photo", reasons
    if _contains_any(lower_label, SIGNATURE_KEYWORDS):
        reasons.append("signature_keyword")
        return "signature", reasons
    if lower_value in CHECKBOX_VALUE_TOKENS:
        reasons.append("checkbox_value")
        return "checkbox", reasons

    multiline = False
    if "\n" in value:
        multiline = True
        reasons.append("value_has_line_breaks")
    if _contains_any(lower_label, MULTILINE_KEYWORDS):
        multiline = True
        reasons.append("multiline_label_keyword")
    if answer_box is not None and float(answer_box["height"]) >= metrics["line_height"] * 2.1:
        multiline = True
        reasons.append("answer_region_is_tall")

    if multiline:
        return "multiline", reasons
    return "text", reasons


def _candidate(
    *,
    bbox: dict[str, float],
    anchor_type: str,
    score: float,
    reasons: list[str],
    source_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "bbox": _round_box(bbox),
        "anchor_type": anchor_type,
        "score": round(_clamp(score), 4),
        "reasons": reasons,
    }
    if source_id:
        payload["source_id"] = source_id
    if extra:
        payload.update(extra)
    return payload


def _inline_candidate_from_split(split: dict[str, Any] | None) -> dict[str, Any] | None:
    if not split:
        return None
    bbox = _normalize_box(split.get("answer_bbox"))
    if not bbox:
        return None
    return _candidate(
        bbox=bbox,
        anchor_type="inline_dotted_field",
        score=float(split.get("confidence") or 0.82),
        reasons=list(split.get("reasons") or ["inline_leader_sequence"]),
        extra={
            "source_type": split.get("source_type"),
            "leader_length": int(split.get("leader_length") or 0),
        },
    )


def _value_block_candidates(
    field_item: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
    page_by_id: dict[str, int],
    label_box: dict[str, float],
    label_page: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for value_id in field_item.get("value_block_ids", []) or []:
        value_block = blocks_by_id.get(value_id)
        value_box = _block_bbox(value_block)
        if not value_box:
            continue
        page = _block_page(value_block, page_by_id, label_page)
        if page != label_page:
            continue
        overlap = _overlap_ratio(value_box, label_box)
        score = 0.9
        reasons = ["textract_key_value_relationship", "value_block_geometry"]
        if overlap > 0.05:
            score -= min(0.45, overlap)
            reasons.append("value_region_overlaps_label")
        if value_box["x"] >= _box_right(label_box) or value_box["y"] >= label_box["y"]:
            score += 0.04
            reasons.append("directionally_plausible")
        candidates.append(
            _candidate(
                bbox=value_box,
                anchor_type="value_block",
                score=score,
                reasons=reasons,
                source_id=value_id,
            )
        )
    return candidates


def _cell_in_section_band(cell: dict[str, Any], section_band: dict[str, Any] | None) -> bool:
    """True when a table cell's vertical centre falls inside the field's section
    band (with a small tolerance). Used to scope label-cell matching so a label
    only matches cells inside its own section, not an identically-named cell in
    another section."""
    if not section_band:
        return True
    if int(cell.get("page") or 1) != int(section_band.get("page") or 1):
        return False
    box = cell.get("bbox") or {}
    center_y = float(box.get("y", 0.0)) + float(box.get("height", 0.0)) / 2.0
    tolerance = 0.01
    return (float(section_band["y_start"]) - tolerance) <= center_y <= (float(section_band["y_end"]) + tolerance)


def _matching_label_cell(
    label: str,
    page: int,
    label_box: dict[str, float],
    cells: list[dict[str, Any]],
    section_band: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    label_norm = _field_label_slug(label)
    # Scope to the field's section band when known; fall back to the full cell
    # set if scoping leaves nothing (never lose a match we would have made).
    if section_band:
        scoped = [cell for cell in cells if _cell_in_section_band(cell, section_band)]
        cells = scoped or cells
    best: tuple[float, dict[str, Any]] | None = None
    for cell in cells:
        if int(cell.get("page") or 1) != page:
            continue
        cell_text = cell.get("text") or ""
        cell_norm = _field_label_slug(cell_text)
        if not cell_norm:
            continue
        text_match = cell_norm == label_norm or (label_norm and label_norm in cell_norm) or (cell_norm and cell_norm in label_norm)
        geom_match = _intersection_area(cell["bbox"], label_box) / max(_box_area(label_box), 0.000001)
        if not text_match and geom_match < 0.45:
            continue
        score = (0.7 if text_match else 0.0) + min(geom_match, 1.0) * 0.3
        if best is None or score > best[0]:
            best = (score, cell)
    return best[1] if best else None


def _table_cell_candidates(
    label: str,
    page: int,
    label_box: dict[str, float],
    cells: list[dict[str, Any]],
    multiline_hint: bool,
    section_band: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    label_cell = _matching_label_cell(label, page, label_box, cells, section_band)
    if not label_cell:
        return []

    same_table = [cell for cell in cells if cell.get("table_id") == label_cell.get("table_id")]
    row = int(label_cell.get("row_index") or 0)
    col = int(label_cell.get("column_index") or 0)
    same_row_right = sorted(
        [
            cell
            for cell in same_table
            if int(cell.get("row_index") or 0) == row and int(cell.get("column_index") or 0) > col
        ],
        key=lambda item: int(item.get("column_index") or 0),
    )
    if not same_row_right:
        return []

    answer_cell = next((cell for cell in same_row_right if not _clean_text(cell.get("text"))), same_row_right[0])
    boxes = [answer_cell["bbox"]]
    reasons = ["table_label_cell_match", "right_sibling_answer_cell"]

    if multiline_hint:
        answer_col = int(answer_cell.get("column_index") or 0)
        for cell in sorted(same_table, key=lambda item: (int(item.get("row_index") or 0), int(item.get("column_index") or 0))):
            cell_row = int(cell.get("row_index") or 0)
            if cell_row <= row:
                continue
            if int(cell.get("column_index") or 0) != answer_col:
                continue

            left_side_text = [
                _clean_text(other.get("text"))
                for other in same_table
                if int(other.get("row_index") or 0) == cell_row and int(other.get("column_index") or 0) <= col
            ]
            if any(text for text in left_side_text):
                break
            if _clean_text(cell.get("text")):
                break
            boxes.append(cell["bbox"])
            reasons.append("empty_table_continuation_cell")

    union = _bbox_union(boxes) or answer_cell["bbox"]
    confidence = _safe_float(answer_cell.get("confidence"), 80.0) / 100.0
    return [
        _candidate(
            bbox=union,
            anchor_type="table_cell",
            score=0.86 + min(confidence, 1.0) * 0.1,
            reasons=reasons,
            source_id=str(answer_cell.get("cell_id") or ""),
            extra={
                "table_id": answer_cell.get("table_id"),
                "row_index": row,
                "column_index": int(answer_cell.get("column_index") or 0),
                "continuation_cell_count": len(boxes) - 1,
            },
        )
    ]


def _synthetic_covers_solid_fragment(
    line_box: dict[str, float],
    page: int,
    visual_features: dict[str, Any],
) -> bool:
    for feature in visual_features.get("synthetic_underlines", []) or []:
        if int(feature.get("page") or 1) != page:
            continue
        if feature.get("source_type") != "broken":
            continue
        synthetic_box = feature.get("bbox") or {}
        if not _normalize_box(synthetic_box):
            continue
        same_row = abs(_box_center(line_box)[1] - _box_center(synthetic_box)[1]) <= max(line_box["height"] * 4.0, synthetic_box["height"] * 4.0, 0.01)
        if not same_row:
            continue
        overlap = _intersection_area(line_box, synthetic_box) / max(_box_area(line_box), 1e-9)
        if overlap >= 0.72:
            return True
    return False


def _underline_candidates(
    label_box: dict[str, float],
    page: int,
    visual_features: dict[str, Any],
    metrics: dict[str, float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    label_center_x, label_center_y = _box_center(label_box)
    label_bottom = _box_bottom(label_box)
    for feature in visual_features.get("underlines", []) or []:
        if int(feature.get("page") or 1) != page:
            continue
        line_box = feature["bbox"]
        if _synthetic_covers_solid_fragment(line_box, page, visual_features):
            continue
        line_center_x, line_center_y = _box_center(line_box)
        if _overlap_ratio(line_box, label_box) > 0.3:
            continue
        right_of_label = float(line_box["x"]) >= _box_right(label_box) - metrics["x_gap"]
        same_row = abs(line_center_y - label_center_y) <= metrics["wide_row_tolerance"]
        just_below = 0 <= line_center_y - label_bottom <= metrics["wide_row_tolerance"] * 1.25
        below_label = line_center_y > label_center_y and abs(line_center_x - label_center_x) <= max(label_box["width"], 0.1)
        if not (right_of_label and (same_row or just_below)) and not below_label:
            continue
        field_height = max(metrics["line_height"] * 1.45, line_box["height"] * 5.0)
        bbox = _normalize_box(
            {
                "x": line_box["x"],
                "y": max(0.0, line_box["y"] - field_height * 0.88),
                "width": line_box["width"],
                "height": field_height,
            }
        )
        if not bbox:
            continue
        vertical_score = _clamp(1.0 - abs(line_center_y - label_center_y) / max(metrics["wide_row_tolerance"] * 2.0, 0.001))
        score = 0.58 + vertical_score * 0.24 + (0.08 if right_of_label else 0.0)
        candidates.append(
            _candidate(
                bbox=bbox,
                anchor_type="underline",
                score=score,
                reasons=["visual_underline", "right_of_label" if right_of_label else "below_label"],
            )
        )
    return candidates


def _synthetic_underline_candidates(
    label_box: dict[str, float],
    page: int,
    visual_features: dict[str, Any],
    metrics: dict[str, float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    label_center_x, label_center_y = _box_center(label_box)
    label_bottom = _box_bottom(label_box)
    label_right = _box_right(label_box)
    for feature in visual_features.get("synthetic_underlines", []) or []:
        if int(feature.get("page") or 1) != page:
            continue
        line_box = feature.get("bbox") or {}
        if not _normalize_box(line_box):
            continue
        line_center_x, line_center_y = _box_center(line_box)
        if _overlap_ratio(line_box, label_box) > 0.3:
            continue

        right_gap = float(line_box["x"]) - label_right
        right_of_label = right_gap >= -metrics["x_gap"]
        same_row = abs(line_center_y - label_center_y) <= metrics["wide_row_tolerance"]
        just_below = 0 <= line_center_y - label_bottom <= metrics["wide_row_tolerance"] * 1.25
        below_label = line_center_y > label_center_y and abs(line_center_x - label_center_x) <= max(label_box["width"], 0.1)
        if not (right_of_label and (same_row or just_below)) and not below_label:
            continue

        field_height = max(metrics["line_height"] * 1.45, line_box["height"] * 5.0)
        bbox = _normalize_box(
            {
                "x": line_box["x"],
                "y": max(0.0, line_box["y"] - field_height * 0.88),
                "width": line_box["width"],
                "height": field_height,
            }
        )
        if not bbox:
            continue

        source_type = str(feature.get("source_type") or "dotted")
        anchor_type = "broken_underline" if source_type == "broken" else "dotted_underline"
        vertical_score = _clamp(1.0 - abs(line_center_y - label_center_y) / max(metrics["wide_row_tolerance"] * 2.0, 0.001))
        gap_score = _clamp(1.0 - max(right_gap, 0.0) / max(metrics["text_width"] * 2.4, 0.08))
        feature_confidence = float(feature.get("confidence") or 0.62)
        if anchor_type == "broken_underline":
            score = 0.50 + vertical_score * 0.10 + gap_score * 0.08 + feature_confidence * 0.12
        else:
            score = 0.58 + vertical_score * 0.11 + gap_score * 0.08 + feature_confidence * 0.10
        candidates.append(
            _candidate(
                bbox=bbox,
                anchor_type=anchor_type,
                score=score,
                reasons=[
                    f"synthetic_{source_type}_underline",
                    "right_of_label" if right_of_label else "below_label",
                    "phase_f_detector",
                ],
                extra={
                    "source_type": source_type,
                    "segment_count": int(feature.get("segment_count") or 0),
                    "visual_confidence": round(feature_confidence, 4),
                },
            )
        )
    return candidates


def _rectangle_candidates(
    label_box: dict[str, float],
    page: int,
    visual_features: dict[str, Any],
    metrics: dict[str, float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    label_center_x, label_center_y = _box_center(label_box)
    for feature in visual_features.get("empty_boxes", []) or []:
        if int(feature.get("page") or 1) != page:
            continue
        rect = feature["bbox"]
        rect_center_x, rect_center_y = _box_center(rect)
        if _overlap_ratio(rect, label_box) > 0.25:
            continue
        right_of_label = rect["x"] >= _box_right(label_box) - metrics["x_gap"]
        same_row = abs(rect_center_y - label_center_y) <= metrics["wide_row_tolerance"]
        below_label = 0 <= rect["y"] - _box_bottom(label_box) <= metrics["line_height"] * 4.0
        if not ((right_of_label and same_row) or below_label):
            continue
        text_count = int(feature.get("text_count") or 0)
        score = 0.62 + float(feature.get("confidence", 0.0)) * 0.2 + (0.08 if text_count == 0 else 0.0)
        candidates.append(
            _candidate(
                bbox=rect,
                anchor_type="empty_rectangle",
                score=score,
                reasons=["visual_empty_rectangle", "right_of_label" if right_of_label else "below_label"],
            )
        )
    return candidates


def _adjacent_whitespace_candidate(
    label_box: dict[str, float],
    page: int,
    text_boxes: list[dict[str, Any]],
    metrics: dict[str, float],
    multiline_hint: bool,
) -> dict[str, Any] | None:
    x_gap = metrics["x_gap"]
    page_right = metrics["page_right"]
    label_center_y = _box_center(label_box)[1]
    row_top = label_box["y"] - metrics["wide_row_tolerance"] * 0.35
    row_bottom = _box_bottom(label_box) + metrics["wide_row_tolerance"] * 0.35

    obstacles: list[dict[str, float]] = []
    for item in text_boxes:
        if int(item.get("page") or 1) != page:
            continue
        text_box = item["bbox"]
        center_y = _box_center(text_box)[1]
        if not (row_top <= center_y <= row_bottom):
            continue
        if float(text_box["x"]) <= _box_right(label_box):
            continue
        obstacles.append(text_box)

    start_x = min(page_right, _box_right(label_box) + x_gap)
    nearest_right_text = min((float(box["x"]) for box in obstacles), default=page_right)
    end_x = max(start_x, min(page_right, nearest_right_text - x_gap))

    if end_x - start_x < max(metrics["text_width"] * 0.35, 0.08):
        if multiline_hint:
            start_x = max(metrics["page_left"], float(label_box["x"]))
            end_x = page_right
            y = _box_bottom(label_box) + metrics["x_gap"]
            height = max(metrics["line_height"] * 3.4, 0.04)
            if y + height > 0.98:
                height = max(metrics["line_height"] * 1.8, 0.98 - y)
        else:
            return None
    else:
        y = max(0.0, label_center_y - metrics["line_height"] * 0.7)
        height = metrics["line_height"] * (3.2 if multiline_hint else 1.55)
        if not multiline_hint:
            # Cap an otherwise-unbounded single-line writing area so the widget
            # aligns tightly to a realistic answer width instead of stretching
            # to the page margin when there is no obstacle to the right.
            max_width = max(metrics["text_width"] * 3.0, 0.30)
            end_x = min(end_x, start_x + max_width)

    bbox = _normalize_box({"x": start_x, "y": y, "width": end_x - start_x, "height": height})
    if not bbox:
        return None
    return _candidate(
        bbox=bbox,
        anchor_type="adjacent_whitespace",
        score=0.48 + (0.08 if multiline_hint else 0.0),
        reasons=["estimated_adjacent_whitespace", "no_direct_value_region"],
    )


def _photo_region_candidate(
    label_box: dict[str, float],
    page: int,
    visual_features: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    label_center_x, label_center_y = _box_center(label_box)
    containing: list[dict[str, Any]] = []
    for feature in visual_features.get("empty_boxes", []) or []:
        if int(feature.get("page") or 1) != page:
            continue
        box = feature["bbox"]
        if box["x"] <= label_center_x <= _box_right(box) and box["y"] <= label_center_y <= _box_bottom(box):
            containing.append(feature)
    if containing:
        best = max(containing, key=lambda item: _box_area(item["bbox"]))
        return _candidate(
            bbox=best["bbox"],
            anchor_type="photo_region",
            score=0.86,
            reasons=["photo_keyword", "visual_container"],
        )

    expanded = _expand_box(label_box, metrics["line_height"] * 3.5, metrics["line_height"] * 4.0)
    return _candidate(
        bbox=expanded,
        anchor_type="photo_region",
        score=0.58,
        reasons=["photo_keyword", "fallback_label_region"],
    )


def _signature_region_candidate(
    label_box: dict[str, float],
    page: int,
    visual_features: dict[str, Any],
    metrics: dict[str, float],
    text_boxes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = _underline_candidates(label_box, page, visual_features, metrics)
    candidates.extend(_rectangle_candidates(label_box, page, visual_features, metrics))
    whitespace = _adjacent_whitespace_candidate(label_box, page, text_boxes, metrics, multiline_hint=False)
    if whitespace:
        candidates.append(whitespace)
    if not candidates:
        return None
    selected = max(candidates, key=lambda item: item["score"])
    selected["anchor_type"] = "signature_region"
    selected["reasons"] = list(dict.fromkeys(selected.get("reasons", []) + ["signature_keyword"]))
    return selected


def _guard_against_label_overlap(
    selected: dict[str, Any],
    label_box: dict[str, float],
    metrics: dict[str, float],
) -> dict[str, Any]:
    box = dict(selected["bbox"])
    overlap = _overlap_ratio(box, label_box)
    if selected.get("anchor_type") == "inline_dotted_field":
        selected["bbox"] = _round_box(box)
        selected["label_overlap_ratio"] = round(overlap, 4)
        selected.setdefault("reasons", []).append("inline_overlap_allowed")
        return selected
    if overlap <= 0.03 or selected.get("anchor_type") in {"photo_region"}:
        selected["bbox"] = _round_box(box)
        selected["label_overlap_ratio"] = round(overlap, 4)
        return selected

    right = _box_right(box)
    new_x = _box_right(label_box) + metrics["x_gap"]
    if new_x < right and right - new_x >= max(metrics["text_width"] * 0.28, 0.05):
        box["x"] = new_x
        box["width"] = right - new_x
        selected["bbox"] = _round_box(box)
        selected["label_overlap_ratio"] = round(_overlap_ratio(box, label_box), 4)
        selected.setdefault("reasons", []).append("trimmed_label_overlap")
        return selected

    if float(box["y"]) < _box_bottom(label_box):
        box["y"] = _box_bottom(label_box) + metrics["x_gap"] * 0.5
        box["height"] = max(metrics["line_height"] * 1.4, box["height"])
    selected["bbox"] = _round_box(_normalize_box(box) or box)
    selected["label_overlap_ratio"] = round(_overlap_ratio(selected["bbox"], label_box), 4)
    selected.setdefault("reasons", []).append("label_overlap_guard_applied")
    selected["score"] = round(float(selected.get("score", 0.0)) * 0.82, 4)
    return selected


# Phase I: map generalized AnswerRegion types onto the EXISTING anchor-type
# vocabulary so the renderer, field classifier, confidence pipeline and Phase H
# region identity never see a novel string. region_type lives in candidate
# metadata + diagnostics only.
_ANSWER_REGION_ANCHOR_TYPE = {
    "UNDERLINE": "underline",
    "DOTTED": "dotted_underline",
    "BROKEN": "broken_underline",
    "MULTILINE": "underline",  # tall combined bbox -> _classify_field types it multiline
    "COMB": "empty_rectangle",
    "GRID": "empty_rectangle",
    "WHITESPACE": "adjacent_whitespace",
    "FREEFORM": "adjacent_whitespace",
    "TABLE_CELL": "table_cell",
    "SIGNATURE": "signature_region",
}
_ANSWER_REGION_BASE_SCORE = {
    "UNDERLINE": 0.60,
    "DOTTED": 0.60,
    "BROKEN": 0.55,
    "MULTILINE": 0.72,
    "COMB": 0.66,
    "GRID": 0.64,
    "WHITESPACE": 0.48,
    "FREEFORM": 0.50,
    "TABLE_CELL": 0.70,
    "SIGNATURE": 0.72,
}


def _answer_region_candidates(
    label_box: dict[str, float],
    page: int,
    answer_regions: list[Any] | None,
    metrics: dict[str, float],
) -> list[dict[str, Any]]:
    """Adapter (Phase I): turn nearby generalized AnswerRegions into candidates
    using the same right-of / same-row / below gating the other builders use."""
    if not answer_regions:
        return []
    candidates: list[dict[str, Any]] = []
    label_center_x, label_center_y = _box_center(label_box)
    label_right = _box_right(label_box)
    label_bottom = _box_bottom(label_box)
    for region in answer_regions:
        if int(getattr(region, "page", 1)) != page:
            continue
        rbox = region.bbox_dict()
        if _overlap_ratio(rbox, label_box) > 0.3:
            continue
        rcx, rcy = _box_center(rbox)
        right_of_label = float(rbox["x"]) >= label_right - metrics["x_gap"]
        same_row = abs(rcy - label_center_y) <= metrics["wide_row_tolerance"]
        below_label = (
            0 <= float(rbox["y"]) - label_bottom <= metrics["wide_row_tolerance"] * 3.0
            and abs(rcx - label_center_x) <= max(float(label_box["width"]), 0.2)
        )
        if not ((right_of_label and same_row) or below_label):
            continue
        region_type = str(getattr(region, "region_type", ""))
        anchor_type = _ANSWER_REGION_ANCHOR_TYPE.get(region_type, "adjacent_whitespace")
        base = _ANSWER_REGION_BASE_SCORE.get(region_type, 0.55)
        proximity = _clamp(1.0 - abs(rcy - label_center_y) / max(metrics["wide_row_tolerance"] * 2.0, 0.001))
        score = base + proximity * 0.10 + float(getattr(region, "confidence", 0.0)) * 0.10
        extra: dict[str, Any] = {
            "answer_region_type": region_type,
            "answer_region_id": getattr(region, "region_id", None),
        }
        line_count = (getattr(region, "metadata", {}) or {}).get("line_count")
        if line_count is not None:
            extra["line_count"] = line_count
        candidates.append(
            _candidate(
                bbox=rbox,
                anchor_type=anchor_type,
                score=score,
                reasons=["answer_region_v2", f"region_{region_type.lower()}", "right_of_label" if right_of_label else "below_label"],
                extra=extra,
            )
        )
    return candidates


def _table_input_candidates(
    label_box: dict[str, float],
    page: int,
    table_input_cells: list[Any] | None,
    metrics: dict[str, float],
) -> list[dict[str, Any]]:
    """Adapter (Phase J): turn detected table INPUT cells near the label into
    ``table_cell`` candidates. Existing anchor vocabulary + cell_id source_id, so
    the renderer/confidence/Phase-H-identity see nothing new and exclusivity
    (one input cell -> one field) is enforced by the assignment engine."""
    if not table_input_cells:
        return []
    candidates: list[dict[str, Any]] = []
    label_center_x, label_center_y = _box_center(label_box)
    label_right = _box_right(label_box)
    label_bottom = _box_bottom(label_box)
    for cell in table_input_cells:
        if int((getattr(cell, "metadata", {}) or {}).get("page") or 1) != page:
            continue
        rbox = cell.bbox_dict()
        if _overlap_ratio(rbox, label_box) > 0.3:
            continue
        rcx, rcy = _box_center(rbox)
        right_of_label = float(rbox["x"]) >= label_right - metrics["x_gap"]
        same_row = abs(rcy - label_center_y) <= metrics["wide_row_tolerance"]
        below_label = (
            0 <= float(rbox["y"]) - label_bottom <= metrics["wide_row_tolerance"] * 3.0
            and abs(rcx - label_center_x) <= max(float(label_box["width"]), 0.2)
        )
        if not ((right_of_label and same_row) or below_label):
            continue
        proximity = _clamp(1.0 - abs(rcy - label_center_y) / max(metrics["wide_row_tolerance"] * 2.0, 0.001))
        score = 0.80 + proximity * 0.08 + float(getattr(cell, "confidence", 0.0)) * 0.08
        meta = getattr(cell, "metadata", {}) or {}
        candidates.append(
            _candidate(
                bbox=rbox,
                anchor_type="table_cell",
                score=score,
                reasons=["table_intelligence_input_cell", "right_of_label" if right_of_label else "below_label"],
                source_id=str(meta.get("cell_id") or ""),
                extra={
                    "table_id": meta.get("table_id"),
                    "row_index": cell.row,
                    "column_index": cell.column,
                    "table_input_cell": True,
                },
            )
        )
    return candidates


def _select_answer_region(
    label: str,
    value: str,
    field_item: dict[str, Any],
    label_box: dict[str, float],
    page: int,
    blocks_by_id: dict[str, dict[str, Any]],
    page_by_id: dict[str, int],
    cells: list[dict[str, Any]],
    visual_features: dict[str, Any],
    text_boxes: list[dict[str, Any]],
    metrics: dict[str, float],
    section_band: dict[str, Any] | None = None,
    inline_candidate: dict[str, Any] | None = None,
    answer_regions: list[Any] | None = None,
    table_input_cells: list[Any] | None = None,
    semantic_field: Any | None = None,
) -> tuple[dict[str, Any], str, list[str], list[dict[str, Any]]]:
    preliminary_multiline = "\n" in value or _contains_any(label.lower(), MULTILINE_KEYWORDS)
    field_type, type_reasons = _classify_field(label, value, None, metrics)

    if field_type == "photo":
        selected = _photo_region_candidate(label_box, page, visual_features, metrics)
        return selected, field_type, type_reasons, [selected]

    if field_type == "signature":
        signature = _signature_region_candidate(label_box, page, visual_features, metrics, text_boxes)
        if signature is not None:
            return signature, field_type, type_reasons, [signature]

    candidates: list[dict[str, Any]] = []
    candidates.extend(_table_cell_candidates(label, page, label_box, cells, preliminary_multiline, section_band))
    candidates.extend(_value_block_candidates(field_item, blocks_by_id, page_by_id, label_box, page))
    candidates.extend(_underline_candidates(label_box, page, visual_features, metrics))
    candidates.extend(_synthetic_underline_candidates(label_box, page, visual_features, metrics))
    candidates.extend(_rectangle_candidates(label_box, page, visual_features, metrics))
    candidates.extend(_answer_region_candidates(label_box, page, answer_regions, metrics))
    candidates.extend(_table_input_candidates(label_box, page, table_input_cells, metrics))
    if inline_candidate is not None:
        candidates.append(inline_candidate)
    whitespace = _adjacent_whitespace_candidate(label_box, page, text_boxes, metrics, preliminary_multiline)
    if whitespace:
        candidates.append(whitespace)

    if not candidates:
        fallback = _candidate(
            bbox=_expand_box(label_box, x_pad=metrics["line_height"] * 2.0, y_pad=metrics["line_height"] * 0.4),
            anchor_type="unresolved_label_region",
            score=0.18,
            reasons=["no_answer_region_candidate"],
        )
        return fallback, field_type, type_reasons, [fallback]

    candidates = [_guard_against_label_overlap(candidate, label_box, metrics) for candidate in candidates]
    # Phase K: fold semantic region preferences into candidate scores BEFORE
    # selection so both the legacy argmax and the Phase H assignment consume the
    # semantically-adjusted scores (the assignment engine itself is unchanged).
    if semantic_field is not None:
        candidates = apply_semantic_priors(candidates, semantic_field)
    selected = max(candidates, key=lambda item: float(item.get("score", 0.0)))
    field_type, type_reasons = _finalize_field_type(label, value, selected, metrics)
    return selected, field_type, type_reasons, sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)


def _finalize_field_type(
    label: str,
    value: str,
    selected: dict[str, Any],
    metrics: dict[str, float],
) -> tuple[str, list[str]]:
    """Classify the field from the *selected* answer region. Shared by the
    legacy argmax path and the Phase H reassignment path so a globally
    reassigned field is typed exactly as if it had been selected directly."""
    field_type, type_reasons = _classify_field(label, value, selected.get("bbox"), metrics)
    explicit_multiline = "\n" in value or _contains_any(label.lower(), MULTILINE_KEYWORDS)
    if (
        field_type == "multiline"
        and not explicit_multiline
        and selected.get("anchor_type") == "table_cell"
        and int(selected.get("continuation_cell_count") or 0) == 0
    ):
        field_type = "text"
        type_reasons = ["single_table_cell"]
    return field_type, type_reasons


def _field_item_page(
    field_item: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
    page_by_id: dict[str, int],
    default: int = 1,
) -> int:
    key_id = field_item.get("key_block_id")
    key_block = blocks_by_id.get(key_id)
    if key_block:
        return _block_page(key_block, page_by_id, default)
    for value_id in field_item.get("value_block_ids", []) or []:
        value_block = blocks_by_id.get(value_id)
        if value_block:
            return _block_page(value_block, page_by_id, default)
    try:
        return int(field_item.get("page") or default)
    except (TypeError, ValueError):
        return default


def _field_item_label_box(field_item: dict[str, Any], blocks_by_id: dict[str, dict[str, Any]]) -> dict[str, float] | None:
    key_block = blocks_by_id.get(field_item.get("key_block_id"))
    return _block_bbox(key_block) or _parsed_geometry_bbox(field_item.get("geometry") or {})


def _checkbox_mapping(
    checkbox: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
    page_by_id: dict[str, int],
    page_px: Callable[[int], tuple[int, int]],
    index: int,
) -> dict[str, Any] | None:
    selection_id = checkbox.get("selection_element_id")
    block = blocks_by_id.get(selection_id)
    bbox = _block_bbox(block) or _parsed_geometry_bbox(checkbox.get("geometry") or {})
    if not bbox:
        return None
    ownership = checkbox.get("ownership") or {}
    label = _clean_text(ownership.get("key_text")) or f"Checkbox {index}"
    confidence = _safe_float(checkbox.get("confidence"), 0.0) / 100.0
    page = _block_page(block, page_by_id, int(checkbox.get("page") or 1) if checkbox.get("page") else 1)
    return {
        "field_id": f"textract_checkbox_{index}",
        "label": label,
        "value": "[X]" if checkbox.get("is_selected") else "[ ]",
        "field_type": "checkbox",
        "bbox": _round_box(bbox),
        "label_bbox": None,
        "answer_region": {
            "bbox": _round_box(bbox),
            "type": "checkbox_region",
            "confidence": round(confidence or 0.72, 4),
        },
        "page": page,
        "confidence": round(confidence or 0.72, 4),
        "candidate_score": round(confidence or 0.72, 4),
        "confidence_class": _confidence_class(confidence or 0.72),
        "multiline_group_size": 1,
        "field_bboxes": [_box_to_pixel_box(bbox, *page_px(page))],
        "source": "textract_anchor",
        "anchoring": {
            "anchor_type": "checkbox_region",
            "reasons": ["selection_element_geometry", ownership.get("ownership_type") or "checkbox"],
        },
    }


def _resolve_field(
    index: int,
    field_item: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
    page_by_id: dict[str, int],
    metrics_by_page: dict[int, dict[str, float]],
    section_index: SectionIndex,
    cells: list[dict[str, Any]],
    visual_features: dict[str, Any],
    text_boxes: list[dict[str, Any]],
    answer_regions: list[Any] | None = None,
    table_input_cells: list[Any] | None = None,
    semantics_on: bool = False,
) -> dict[str, Any] | None:
    """Resolve one field to its candidate set + legacy argmax selection.

    Returns ``None`` when the field has no label geometry (the historical skip).
    Lifted out of ``build_anchored_mappings`` so the same preparation feeds both
    the build loop and the Phase H pre-pass with no divergence.
    """
    field_id = f"anchored_field_{index}"
    raw_label = _clean_text(field_item.get("key")) or f"Field {index}"
    label = raw_label
    value = str(field_item.get("value") or "")
    key_id = str(field_item.get("key_block_id") or "")

    label_box = _field_item_label_box(field_item, blocks_by_id)
    if not label_box:
        logger.info("[anchor] skip field without label geometry label=%r", label)
        return None
    page = _field_item_page(field_item, blocks_by_id, page_by_id, default=1)
    metrics = metrics_by_page.get(page, metrics_by_page.get(1, {}))
    inline_split = split_inline_answer_region(
        label,
        label_box,
        page=page,
        line_height=metrics.get("line_height"),
    )
    inline_candidate = _inline_candidate_from_split(inline_split)
    if inline_split and inline_candidate:
        label = _clean_text(inline_split.get("label")) or label
        label_box = _normalize_box(inline_split.get("label_bbox")) or label_box

    # Section ownership is resolved from the label's reading position.
    owner_section = section_index.owner(page, float(label_box["y"]))
    section_band = section_index.band(owner_section)

    # Phase K: classify the field's meaning (label + section + value) so its
    # region preferences can steer candidate scoring.
    semantic_field = None
    if semantics_on:
        semantic_field = classify_field(
            label,
            section_title=owner_section.get("title"),
            value=value,
        )

    selected, field_type, type_reasons, candidates = _select_answer_region(
        label,
        value,
        field_item,
        label_box,
        page,
        blocks_by_id,
        page_by_id,
        cells,
        visual_features,
        text_boxes,
        metrics,
        section_band,
        inline_candidate=inline_candidate,
        answer_regions=answer_regions,
        table_input_cells=table_input_cells,
        semantic_field=semantic_field,
    )
    return {
        "field_id": field_id,
        "semantic_field": semantic_field,
        "raw_label": raw_label,
        "label": label,
        "value": value,
        "key_id": key_id,
        "label_box": label_box,
        "page": page,
        "metrics": metrics,
        "owner_section": owner_section,
        "inline_split": inline_split,
        "selected": selected,
        "field_type": field_type,
        "type_reasons": type_reasons,
        "candidates": candidates,
    }


def build_anchored_mappings(
    raw_response: dict[str, Any],
    parsed: dict[str, Any],
    image_path: str | Path,
    page_images: dict[int, str | Path] | None = None,
) -> dict[str, Any]:
    """Estimate answer regions between Textract parsing and PDF rendering.

    The output keeps the API-compatible mapping contract while making `bbox`
    and `field_bboxes` refer to the answer/input region, not the label.

    ``page_images`` maps a 1-based page number to the rasterised image for that
    page. When omitted, the engine behaves exactly as before (single page-1
    image), preserving the legacy/rollback contract. When supplied, CV visual
    features are detected per page and every page's fields only ever anchor onto
    that page's features.
    """
    image_path = Path(image_path)
    # Normalise the page->image map. Default keeps the historical single-image
    # (page 1) behaviour byte-for-byte.
    if page_images:
        page_images_norm: dict[int, Path] = {int(p): Path(img) for p, img in page_images.items()}
    else:
        page_images_norm = {1: image_path}

    blocks = raw_response.get("Blocks", []) or []
    blocks_by_id = {block.get("Id"): block for block in blocks if block.get("Id")}
    page_by_id = _build_page_index(blocks)
    text_cache: dict[str, str] = {}
    text_boxes = _visual_text_boxes(blocks, page_by_id)
    metrics_by_page = _build_page_metrics(text_boxes)

    # --- Document hierarchy (Phase 2.2) ---------------------------------------
    # Detect section headings and build a reading-order owner index. Field keys
    # are excluded from heading candidacy so a label is never mistaken for a
    # section. Heading-less documents yield zero sections, so every item falls to
    # the document root and behaviour is identical to before (backward compatible).
    lines = _line_boxes(blocks, page_by_id)
    field_key_slugs = {
        _field_label_slug(field_item.get("key"))
        for field_item in (parsed.get("field_items") or [])
        if field_item.get("key")
    }
    sections = detect_sections(lines, field_key_slugs, metrics_by_page)
    section_index = SectionIndex(sections)

    detected = _detect_visual_features_for_pages(page_images_norm, text_boxes)
    visual_features: dict[str, Any] = {
        "underlines": detected.get("underlines", []),
        "empty_boxes": detected.get("empty_boxes", []),
        "synthetic_underlines": detected.get("synthetic_underlines", []),
    }
    image_sizes: dict[int, dict[str, int]] = detected.get("image_sizes", {}) or {}
    dotted_debug_pages = detected.get("dotted_underline_debug", []) or []
    source_images_by_page = {int(page): str(path) for page, path in page_images_norm.items()}
    # Batch B: peel decorative/photo placeholders out of the empty-rectangle pool
    # so they can never be selected as an input region for a neighbouring label.
    photo_regions, visual_features["empty_boxes"] = _split_photo_regions(visual_features.get("empty_boxes", []))
    visual_features["photo_regions"] = photo_regions
    cells = _table_cells(parsed)
    # Canonical checkbox geometry for token-checkbox deduplication (see below).
    selection_regions = _selection_regions(blocks, page_by_id)

    # Per-page pixel dimensions for the (diagnostic) pixel `field_bboxes`. The
    # rendered widgets use the page-local fraction `bbox`, so a wrong size here
    # never moves a widget — but we still resolve the correct page so previews
    # and overlays line up. Fall back to page 1, then to reading the image.
    if image_sizes:
        _default_size = image_sizes.get(1) or next(iter(image_sizes.values()))
        image_width = int(_default_size["width"])
        image_height = int(_default_size["height"])
    else:
        image = cv2.imread(str(image_path))
        if image is None:
            image_width, image_height = 1, 1
        else:
            image_height, image_width = image.shape[:2]

    def _page_px(page: int) -> tuple[int, int]:
        size = image_sizes.get(int(page))
        if size:
            return int(size["width"]), int(size["height"])
        return image_width, image_height

    mappings: list[dict[str, Any]] = []
    anchor_records: list[dict[str, Any]] = []
    answer_region_debug_records: list[dict[str, Any]] = []
    deduped_token_checkbox_count = 0

    # --- Phase I: generalized answer-region engine ----------------------------
    # Build one unified set of AnswerRegions (multiline stacks, merged broken
    # underlines, signatures, freeform whitespace, table cells, ...) from the
    # already-detected features. Gated OFF by default; when OFF the engine is
    # never invoked and no extra candidates are produced.
    answer_region_v2_on = answer_region_v2_enabled()
    answer_region_result = None
    answer_regions: list[Any] = []
    if answer_region_v2_on:
        answer_region_result = build_answer_regions(
            visual_features,
            text_boxes=text_boxes,
            table_cells=cells,
            metrics_by_page=metrics_by_page,
        )
        answer_regions = answer_region_result.regions

    # --- Phase J: table intelligence ------------------------------------------
    # Classify each Textract table (input / repeating / matrix / line-item /
    # layout) and surface its empty INPUT cells. Gated OFF by default; when OFF
    # the engine is never invoked and no table-input candidates are produced.
    table_intelligence_on = table_intelligence_enabled()
    table_result = None
    table_input_cells: list[Any] = []
    if table_intelligence_on:
        table_result = build_tables(cells, sections=sections, metrics_by_page=metrics_by_page)
        table_input_cells = table_result.input_cells

    # --- Phase K: semantic classification -------------------------------------
    # Classify each field's meaning and let its region preferences steer
    # candidate scoring. Gated OFF by default; when OFF no field is classified
    # and no scores change.
    semantics_on = semantics_enabled()
    semantic_records: list[dict[str, Any]] = []

    # --- Per-field resolution (candidate generation + legacy argmax) ----------
    # Resolve every field once. Each record carries the field's candidate set
    # and the candidate the greedy argmax would have chosen. This is the same
    # computation as before, lifted out so it can also feed the Phase H pre-pass
    # without recomputation or drift.
    resolved_records: list[dict[str, Any]] = []
    for index, field_item in enumerate(parsed.get("field_items", []) or [], start=1):
        rec = _resolve_field(
            index,
            field_item,
            blocks_by_id,
            page_by_id,
            metrics_by_page,
            section_index,
            cells,
            visual_features,
            text_boxes,
            answer_regions=answer_regions,
            table_input_cells=table_input_cells,
            semantics_on=semantics_on,
        )
        if rec is not None:
            resolved_records.append(rec)

    # --- Phase H: global assignment ------------------------------------------
    # Replace per-field greedy argmax with one document-wide max-weight
    # bipartite assignment so two fields can never claim the same physical
    # region. Gated OFF by default; when OFF, overrides is empty and every field
    # keeps its legacy argmax selection (behaviour identical to before).
    global_assignment_on = global_assignment_enabled()
    if global_assignment_on:
        assignment_overrides, assignment_diagnostics = solve_global_assignment(
            [
                {
                    "field_id": rec["field_id"],
                    "page": rec["page"],
                    "candidates": rec["candidates"],
                    "legacy_selected": rec["selected"],
                }
                for rec in resolved_records
            ]
        )
    else:
        assignment_overrides, assignment_diagnostics = {}, None

    for rec in resolved_records:
        field_id = rec["field_id"]
        raw_label = rec["raw_label"]
        label = rec["label"]
        value = rec["value"]
        key_id = rec["key_id"]
        label_box = rec["label_box"]
        page = rec["page"]
        metrics = rec["metrics"]
        owner_section = rec["owner_section"]
        inline_split = rec["inline_split"]
        selected = rec["selected"]
        field_type = rec["field_type"]
        type_reasons = rec["type_reasons"]
        candidates = rec["candidates"]
        semantic_field = rec.get("semantic_field")

        # When the solver hands this field a different physical region than the
        # greedy argmax did, swap to that candidate and re-type from its box.
        reassigned_from: dict[str, Any] | None = None
        if global_assignment_on:
            target_region_id = assignment_overrides.get(field_id)
            if target_region_id:
                match = next(
                    (c for c in candidates if region_identity(c, page, field_id) == target_region_id),
                    None,
                )
                if match is not None and match is not selected:
                    reassigned_from = {
                        "anchor_type": selected.get("anchor_type"),
                        "score": round(float(selected.get("score", 0.0)), 4),
                    }
                    selected = match
                    field_type, type_reasons = _finalize_field_type(label, value, selected, metrics)

        answer_box = selected["bbox"]
        score = float(selected.get("score", 0.0))
        anchor_type = selected.get("anchor_type")
        debug_entry = {
            "field_id": field_id,
            "raw_label": raw_label,
            "label": label,
            "page": page,
            "source_image": source_images_by_page.get(page),
            "selected_type": anchor_type,
            "selected_confidence": round(score, 4),
            "selected_bbox": _round_box(answer_box),
            "candidate_count": len(candidates),
            "inline_split": inline_split or None,
            "reassigned_from": reassigned_from,
            "candidates": [
                {
                    "type": candidate.get("anchor_type"),
                    "confidence": candidate.get("score"),
                    "bbox": candidate.get("bbox"),
                    "page": page,
                    "source_image": source_images_by_page.get(page),
                    "reasons": candidate.get("reasons", []),
                    "rejection_reasons": []
                    if candidate is selected
                    else ["lower_score_than_selected"],
                }
                for candidate in candidates
            ],
            "outcome": "selected",
        }
        answer_region_debug_records.append(debug_entry)

        # Batch C (moderate): drop purely-estimated fields we could not resolve
        # to any plausible region rather than rendering them as low-value clutter.
        if (
            anchor_type == "unresolved_label_region"
            and score < UNRESOLVED_DROP_FLOOR
            and field_type not in {"photo", "signature", "checkbox"}
        ):
            if not (confidence_pipeline_enabled() and review_queue_enabled()):
                logger.info("[anchor] drop unresolved low-confidence field label=%r score=%.3f", label, score)
                debug_entry["outcome"] = "dropped_unresolved_low_confidence"
                continue
            logger.info("[anchor] preserve unresolved low-confidence field for review label=%r score=%.3f", label, score)
            debug_entry["outcome"] = "preserved_for_review"
            confidence = max(0.0, min(1.0, score))
            review_mapping = {
                "field_id": field_id,
                "label": label,
                "qualified_label": qualify_label(owner_section, label),
                "section": section_summary(owner_section),
                "value": value,
                "field_type": field_type,
                "bbox": _round_box(answer_box),
                "label_bbox": _round_box(label_box),
                "answer_region": {
                    "bbox": _round_box(answer_box),
                    "type": selected.get("anchor_type", "answer_region"),
                    "confidence": round(confidence, 4),
                },
                "page": page,
                "confidence": round(confidence, 4),
                "candidate_score": round(confidence, 4),
                "confidence_class": _confidence_class(confidence),
                "multiline_group_size": 1,
                "field_bboxes": [_box_to_pixel_box(answer_box, *_page_px(page))],
                "source": "textract_anchor_low_confidence_review",
                "render_border": False,
                "anchoring": {
                    "anchor_type": selected.get("anchor_type"),
                    "type_reasons": type_reasons,
                    "selection_reasons": selected.get("reasons", []),
                    "label_overlap_ratio": selected.get("label_overlap_ratio", round(_overlap_ratio(answer_box, label_box), 4)),
                    "candidate_count": len(candidates),
                    "top_candidates": candidates[:5],
                    "key_block_id": key_id or None,
                    "value_block_ids": field_item.get("value_block_ids", []) or [],
                    "preserved_for_review": True,
                    "drop_floor": UNRESOLVED_DROP_FLOOR,
                },
            }
            mappings.append(review_mapping)
            anchor_records.append(
                {
                    "field_id": review_mapping["field_id"],
                    "label": label,
                    "field_type": field_type,
                    "page": page,
                    "section_id": owner_section.get("section_id"),
                    "section_title": owner_section.get("title"),
                    "label_bbox": review_mapping["label_bbox"],
                    "answer_bbox": review_mapping["bbox"],
                    "anchor_type": selected.get("anchor_type"),
                    "confidence": review_mapping["confidence"],
                    "label_overlap_ratio": review_mapping["anchoring"]["label_overlap_ratio"],
                    "candidate_count": len(candidates),
                    "preserved_for_review": True,
                }
            )
            continue

        # Batch B: never draw an input widget on top of a detected photo region.
        # Strong structural anchors (real Textract value/table geometry) are kept;
        # only floaty estimated boxes landing on a photo are suppressed. Checkboxes
        # are exempt: a checkbox glyph near a photo is still a real control.
        if (
            field_type not in {"photo", "signature", "checkbox"}
            and anchor_type not in {"value_block", "table_cell"}
            and _overlaps_photo_region(answer_box, page, photo_regions)
        ):
            logger.info("[anchor] suppress field overlapping photo region label=%r anchor=%s", label, anchor_type)
            debug_entry["outcome"] = "suppressed_photo_overlap"
            continue

        # Checkbox deduplication: a token checkbox ("[X]"/"[ ]") emitted from
        # KEY_VALUE_SET parsing that overlaps a real SELECTION_ELEMENT is the same
        # logical checkbox. Prefer the structurally-native SELECTION_ELEMENT (it
        # renders below) and suppress only this overlapping token duplicate.
        if (
            DEDUPE_TOKEN_CHECKBOXES
            and field_type == "checkbox"
            and _overlaps_selection_region(answer_box, page, selection_regions)
        ):
            deduped_token_checkbox_count += 1
            logger.info("[anchor] dedupe token checkbox into SELECTION_ELEMENT label=%r", label)
            debug_entry["outcome"] = "deduped_token_checkbox"
            continue

        multiline_group_size = max(1, value.count("\n") + 1 if value else 1)
        if field_type == "multiline":
            multiline_group_size = max(multiline_group_size, int(math.ceil(answer_box["height"] / max(metrics["line_height"] * 1.35, 0.001))))

        confidence = max(0.0, min(1.0, score))
        mapping = {
            "field_id": field_id,
            "label": label,
            "qualified_label": qualify_label(owner_section, label),
            "section": section_summary(owner_section),
            "value": value,
            "field_type": field_type,
            "bbox": _round_box(answer_box),
            "label_bbox": _round_box(label_box),
            "answer_region": {
                "bbox": _round_box(answer_box),
                "type": selected.get("anchor_type", "answer_region"),
                "confidence": round(confidence, 4),
            },
            "page": page,
            "confidence": round(confidence, 4),
            "candidate_score": round(confidence, 4),
            "confidence_class": _confidence_class(confidence),
            "multiline_group_size": multiline_group_size,
            "field_bboxes": [_box_to_pixel_box(answer_box, *_page_px(page))],
            "source": "textract_anchor",
            # Batch A: when the answer sits on an already-printed line/box/cell,
            # signal the renderer to stay borderless so it does not stack a
            # second line over the existing one.
            "render_border": anchor_type not in PRINTED_FEATURE_ANCHORS,
            "anchoring": {
                "anchor_type": selected.get("anchor_type"),
                "type_reasons": type_reasons,
                "selection_reasons": selected.get("reasons", []),
                "label_overlap_ratio": selected.get("label_overlap_ratio", round(_overlap_ratio(answer_box, label_box), 4)),
                "candidate_count": len(candidates),
                "top_candidates": candidates[:5],
                "key_block_id": key_id or None,
                "value_block_ids": field_item.get("value_block_ids", []) or [],
            },
        }
        # Phase K: attach semantic metadata + record the (possibly score-shifting)
        # prior on the selected region for diagnostics.
        if semantic_field is not None:
            mapping["semantic"] = semantic_field.to_dict()
            applied_prior = float(selected.get("semantic_prior") or 0.0)
            semantic_records.append(
                {
                    "field_id": field_id,
                    "label": label,
                    "semantic": mapping["semantic"],
                    "applied_prior": applied_prior,
                    "selected_family": candidate_region_family(selected),
                }
            )
        mappings.append(mapping)
        anchor_records.append(
            {
                "field_id": mapping["field_id"],
                "label": label,
                "field_type": field_type,
                "page": page,
                "section_id": owner_section.get("section_id"),
                "section_title": owner_section.get("title"),
                "label_bbox": mapping["label_bbox"],
                "answer_bbox": mapping["bbox"],
                "anchor_type": selected.get("anchor_type"),
                "confidence": mapping["confidence"],
                "label_overlap_ratio": mapping["anchoring"]["label_overlap_ratio"],
                "candidate_count": len(candidates),
            }
        )

    unassociated_checkbox_count = 0
    for index, checkbox in enumerate(parsed.get("checkboxes", []) or [], start=1):
        is_unassociated = (checkbox.get("ownership") or {}).get("ownership_type") == "unassociated"
        if is_unassociated:
            unassociated_checkbox_count += 1
            # Flag for diagnostics but DO NOT drop by default: standalone / agreement
            # checkboxes are real fillable widgets. Opt-in drop only.
            if DROP_UNASSOCIATED_CHECKBOXES:
                logger.info("[anchor] drop unassociated checkbox id=%s (opt-in)", checkbox.get("selection_element_id"))
                continue
        checkbox_mapping = _checkbox_mapping(checkbox, blocks_by_id, page_by_id, _page_px, index)
        if checkbox_mapping is None:
            continue
        # Checkbox ownership inherits its section from the glyph's reading
        # position, so option lists ("Single / Joint / Anyone or Survivor")
        # resolve under their section ("Holding Mode") even when Textract leaves
        # the checkbox unassociated.
        cb_owner = section_index.owner(checkbox_mapping["page"], float(checkbox_mapping["bbox"]["y"]))
        checkbox_mapping["section"] = section_summary(cb_owner)
        checkbox_mapping["qualified_label"] = qualify_label(cb_owner, checkbox_mapping["label"])
        mappings.append(checkbox_mapping)
        anchor_records.append(
            {
                "field_id": checkbox_mapping["field_id"],
                "label": checkbox_mapping["label"],
                "field_type": "checkbox",
                "page": checkbox_mapping["page"],
                "section_id": cb_owner.get("section_id"),
                "section_title": cb_owner.get("title"),
                "label_bbox": None,
                "answer_bbox": checkbox_mapping["bbox"],
                "anchor_type": "checkbox_region",
                "confidence": checkbox_mapping["confidence"],
                "label_overlap_ratio": 0.0,
                "candidate_count": 1,
            }
        )

    # Batch B: record detected photo/decorative regions as explicit non-fillable
    # markers (the renderer already skips field_type == "photo"). This documents
    # the suppressed zone without drawing any widget, and avoids re-emitting a
    # region already covered by a keyword-detected photo field above.
    existing_photo_boxes = [m["bbox"] for m in mappings if m.get("field_type") == "photo"]
    for p_index, feature in enumerate(photo_regions, start=1):
        box = feature["bbox"]
        if any(_overlap_ratio(box, existing) > 0.5 for existing in existing_photo_boxes):
            continue
        page = int(feature.get("page") or 1)
        conf = round(_clamp(float(feature.get("confidence") or 0.7)), 4)
        photo_owner = section_index.owner(page, float(box["y"]))
        photo_mapping = {
            "field_id": f"photo_region_{p_index}",
            "label": "Photograph",
            "qualified_label": qualify_label(photo_owner, "Photograph"),
            "section": section_summary(photo_owner),
            "value": "",
            "field_type": "photo",
            "bbox": _round_box(box),
            "label_bbox": None,
            "answer_region": {"bbox": _round_box(box), "type": "photo_region", "confidence": conf},
            "page": page,
            "confidence": conf,
            "candidate_score": conf,
            "confidence_class": _confidence_class(conf),
            "multiline_group_size": 1,
            "field_bboxes": [_box_to_pixel_box(box, *_page_px(page))],
            "source": "textract_anchor",
            "render_border": False,
            "anchoring": {
                "anchor_type": "photo_region",
                "type_reasons": ["visual_photo_region"],
                "selection_reasons": ["decorative_region_suppressed"],
            },
        }
        mappings.append(photo_mapping)
        existing_photo_boxes.append(photo_mapping["bbox"])
        anchor_records.append(
            {
                "field_id": photo_mapping["field_id"],
                "label": "Photograph",
                "field_type": "photo",
                "page": page,
                "section_id": photo_owner.get("section_id"),
                "section_title": photo_owner.get("title"),
                "label_bbox": None,
                "answer_bbox": photo_mapping["bbox"],
                "anchor_type": "photo_region",
                "confidence": conf,
                "label_overlap_ratio": 0.0,
                "candidate_count": 1,
            }
        )

    mappings, comb_diagnostics = apply_comb_detection(
        mappings,
        visual_features,
        table_cells=cells,
        selection_regions=selection_regions,
        photo_regions=photo_regions,
    )
    mappings, radio_diagnostics = apply_radio_grouping(
        mappings,
        text_boxes=text_boxes,
    )

    # --- False-checkbox rejection (Phase L0.1) --------------------------------
    # Drop "checkboxes" that are really narrow OCR glyphs (11 / II / |). Two
    # independent flags, both default OFF: ENFORCE drops; OBSERVE only scores +
    # diagnoses without dropping (precision measurement). ENFORCE wins if both
    # are set. When neither is set mappings are unchanged.
    checkbox_validation_diag: dict[str, Any] = {"enabled": False}
    _cb_enforce = checkbox_validation_enabled()
    if _cb_enforce or checkbox_validation_observe_enabled():
        mappings, checkbox_validation_diag = validate_checkboxes(
            mappings,
            page_images=page_images_norm,
            text_boxes=text_boxes,
            image_sizes=image_sizes,
            observe=not _cb_enforce,
        )

    anchor_records_by_id = {
        str(record.get("field_id")): record for record in anchor_records if record.get("field_id")
    }
    for mapping in mappings:
        page = int(mapping.get("page") or 1)
        bbox = _normalize_box(mapping.get("bbox"))
        if bbox and mapping.get("widget_type") in {"comb", "radio"}:
            mapping["field_bboxes"] = [_box_to_pixel_box(bbox, *_page_px(page))]
        record = anchor_records_by_id.get(str(mapping.get("field_id")))
        if record and mapping.get("widget_type") == "comb":
            record["answer_bbox"] = mapping.get("bbox")
            record["anchor_type"] = "comb_region"
            record["confidence"] = mapping.get("comb_confidence", record.get("confidence"))
            record["comb_cells"] = mapping.get("comb_cells")
            record["widget_type"] = "comb"
        elif record and mapping.get("widget_type") == "radio":
            record["answer_bbox"] = mapping.get("bbox")
            record["anchor_type"] = "radio_region"
            record["confidence"] = mapping.get("radio_confidence", record.get("confidence"))
            record["radio_group"] = mapping.get("radio_group")
            record["export_value"] = mapping.get("export_value")
            record["widget_type"] = "radio"

    anchor_type_counts = Counter(record["anchor_type"] for record in anchor_records)
    field_type_counts = Counter(record["field_type"] for record in anchor_records)
    fields_per_page = Counter(int(record.get("page") or 1) for record in anchor_records)
    analyzed_pages = sorted({int(item.get("page") or 1) for item in text_boxes}) or [1]

    # --- Section hierarchy diagnostics (Task 7 table ownership + Task 8) -------
    # Tables inherit a section from their top edge's reading position. This is
    # ownership only — no table semantics are computed in this phase.
    table_sections: dict[str, dict[str, Any]] = {}
    for table in parsed.get("tables", []) or []:
        table_box = _parsed_geometry_bbox(table.get("geometry") or {})
        if not table_box:
            continue
        table_owner = section_index.owner(int(table.get("page") or 1), float(table_box["y"]))
        table_sections[str(table.get("table_block_id"))] = section_summary(table_owner)

    section_field_counts: dict[str, int] = defaultdict(int)
    section_checkbox_counts: dict[str, int] = defaultdict(int)
    section_pages: dict[str, set] = defaultdict(set)
    for record in anchor_records:
        sid = record.get("section_id") or "document"
        if record.get("field_type") == "checkbox":
            section_checkbox_counts[sid] += 1
        else:
            section_field_counts[sid] += 1
        section_pages[sid].add(int(record.get("page") or 1))

    orphan_field_count = sum(
        1 for record in anchor_records if (record.get("section_id") or "document") == "document"
    )

    section_summaries: list[dict[str, Any]] = []
    for section in sections:
        sid = section["section_id"]
        section_summaries.append(
            {
                **section_summary(section),
                "y_start": round(float(section["y_start"]), 4),
                "y_end": round(float(section["y_end"]), 4),
                "reasons": section.get("reasons", []),
                "field_count": section_field_counts.get(sid, 0),
                "checkbox_count": section_checkbox_counts.get(sid, 0),
                "table_count": sum(1 for value in table_sections.values() if value.get("section_id") == sid),
                "pages": sorted(section_pages.get(sid, set())),
            }
        )

    section_diagnostics = {
        "section_count": len(sections),
        "section_types_detected": sorted({section["type"] for section in sections}),
        "sections": section_summaries,
        "fields_per_section": {
            section["title"]: section_field_counts.get(section["section_id"], 0)
            + section_checkbox_counts.get(section["section_id"], 0)
            for section in sections
        },
        "orphan_field_count": orphan_field_count,
        "pages_per_section": {
            section["section_id"]: sorted(section_pages.get(section["section_id"], set())) for section in sections
        },
        "table_sections": table_sections,
    }
    label_overlap_count = sum(
        1
        for record in anchor_records
        if record.get("field_type") != "photo" and float(record.get("label_overlap_ratio") or 0.0) > 0.03
    )
    photo_label_overlap_count = sum(
        1
        for record in anchor_records
        if record.get("field_type") == "photo" and float(record.get("label_overlap_ratio") or 0.0) > 0.03
    )
    legacy_overlap_count = sum(1 for record in anchor_records if record.get("label_bbox") and record.get("field_type") != "photo")
    answer_region_selected_counts = Counter(
        str(record.get("selected_type") or "unknown") for record in answer_region_debug_records
    )
    dotted_detected = visual_features.get("synthetic_underlines", []) or []
    dotted_rejected = []
    for page_debug in dotted_debug_pages:
        for rejected in page_debug.get("rejected", []) or []:
            enriched = dict(rejected)
            enriched.setdefault("source_image", page_debug.get("source_image"))
            dotted_rejected.append(enriched)

    # --- Diagnostics-only analyzers (Phase L0.2 / L0.4) -----------------------
    # Both are pure observation: gated OFF by default and never mutate mappings.
    comb_diagnostics_v2: dict[str, Any] = {"enabled": False}
    if comb_diagnostics_enabled():
        comb_diagnostics_v2 = analyze_comb_groups(
            mappings,
            text_boxes=text_boxes,
            selection_regions=selection_regions,
        )
    dotted_leader_source: dict[str, Any] = {"enabled": False}
    if dotted_leader_diagnostics_enabled():
        leader_field_labels = [
            {"page": rec.get("page"), "bbox": rec.get("label_box")}
            for rec in resolved_records
            if isinstance(rec.get("label_box"), dict)
        ]
        dotted_leader_source = analyze_dotted_leaders(
            dotted_debug_pages,
            text_boxes=text_boxes,
            field_labels=leader_field_labels,
            mappings=mappings,
        )

    diagnostics = {
        "engine": "semantic_visual_anchor",
        "field_count": len(anchor_records),
        "page_count": len(page_images_norm),
        "pages_rasterized": sorted(page_images_norm.keys()),
        "analyzed_pages": analyzed_pages,
        "fields_per_page": {str(page): count for page, count in sorted(fields_per_page.items())},
        "hierarchy": section_diagnostics,
        "fillable_field_count": sum(1 for mapping in mappings if mapping.get("field_type") != "photo"),
        "photo_field_count": field_type_counts.get("photo", 0),
        "checkbox_field_count": field_type_counts.get("checkbox", 0),
        "unassociated_checkbox_count": unassociated_checkbox_count,
        "dropped_unassociated_checkboxes": DROP_UNASSOCIATED_CHECKBOXES,
        "deduped_token_checkbox_count": deduped_token_checkbox_count,
        "field_type_counts": dict(field_type_counts),
        "anchor_type_counts": dict(anchor_type_counts),
        "label_overlap_count": label_overlap_count,
        "photo_label_overlap_count": photo_label_overlap_count,
        "legacy_direct_label_overlap_estimate": legacy_overlap_count,
        "label_overlap_reduction_estimate": max(0, legacy_overlap_count - label_overlap_count),
        "visual_features": {
            "underline_count": len(visual_features.get("underlines", []) or []),
            "synthetic_underline_count": len(dotted_detected),
            "empty_box_count": len(visual_features.get("empty_boxes", []) or []),
            "photo_region_count": len(photo_regions),
        },
        "dotted_underlines": {
            "detected_count": len(dotted_detected),
            "rejected_count": len(dotted_rejected),
            "detected": dotted_detected,
            "rejected": dotted_rejected[:120],
            "pages": dotted_debug_pages,
        },
        "answer_regions": {
            "candidate_count": sum(int(item.get("candidate_count") or 0) for item in answer_region_debug_records),
            "selected_type_counts": dict(answer_region_selected_counts),
            "regions": answer_region_debug_records,
        },
        "comb_fields": comb_diagnostics,
        "radio_groups": radio_diagnostics,
        "checkbox_validation": checkbox_validation_diag,
        "comb_diagnostics": comb_diagnostics_v2,
        "dotted_leader_source": dotted_leader_source,
        "table_cell_count": len(cells),
        "text_box_count": len(text_boxes),
        "anchors": anchor_records,
    }
    diagnostics["global_assignment"] = (
        assignment_diagnostics if assignment_diagnostics is not None else {"enabled": False}
    )
    diagnostics["answer_region_engine"] = (
        answer_region_result.to_debug_dict() if answer_region_result is not None else {"enabled": False}
    )
    diagnostics["table_intelligence"] = (
        table_result.to_debug_dict() if table_result is not None else {"enabled": False}
    )
    diagnostics["semantic_classifier"] = (
        build_semantic_diagnostics(semantic_records) if semantics_on else {"enabled": False}
    )

    return {
        "mappings": mappings,
        "field_objects": anchor_records,
        "diagnostics": diagnostics,
        "sections": sections,
        "visual_features": {
            "underlines": visual_features.get("underlines", []),
            "synthetic_underlines": visual_features.get("synthetic_underlines", []),
            "empty_boxes": visual_features.get("empty_boxes", []),
            "photo_regions": photo_regions,
        },
    }


def draw_anchor_debug_overlay(
    image_path: str | Path,
    mappings: list[dict[str, Any]],
    output_path: str | Path,
    visual_features: dict[str, Any] | None = None,
) -> None:
    image_path = Path(image_path)
    output_path = Path(output_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unable to read source image for anchor debug overlay: {image_path}")

    image_height, image_width = image.shape[:2]

    def to_px(box: dict[str, Any]) -> tuple[int, int, int, int]:
        x = int(float(box["x"]) * image_width)
        y = int(float(box["y"]) * image_height)
        w = int(float(box["width"]) * image_width)
        h = int(float(box["height"]) * image_height)
        return x, y, w, h

    for feature in (visual_features or {}).get("underlines", []) or []:
        if int(feature.get("page") or 1) != 1:
            continue
        x, y, w, h = to_px(feature["bbox"])
        cv2.rectangle(image, (x, y), (x + w, y + max(h, 2)), (160, 160, 160), 1)

    for feature in (visual_features or {}).get("synthetic_underlines", []) or []:
        if int(feature.get("page") or 1) != 1:
            continue
        x, y, w, h = to_px(feature["bbox"])
        color = (60, 170, 220) if feature.get("source_type") != "broken" else (60, 120, 230)
        cv2.rectangle(image, (x, y), (x + w, y + max(h, 3)), color, 2)

    for feature in (visual_features or {}).get("empty_boxes", []) or []:
        if int(feature.get("page") or 1) != 1:
            continue
        x, y, w, h = to_px(feature["bbox"])
        cv2.rectangle(image, (x, y), (x + w, y + h), (140, 140, 220), 1)

    for feature in (visual_features or {}).get("photo_regions", []) or []:
        if int(feature.get("page") or 1) != 1:
            continue
        x, y, w, h = to_px(feature["bbox"])
        cv2.rectangle(image, (x, y), (x + w, y + h), (190, 80, 190), 2)
        cv2.putText(image, "photo/suppressed", (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 80, 190), 1, cv2.LINE_AA)

    for mapping in mappings:
        if int(mapping.get("page") or 1) != 1:
            continue
        label_box = mapping.get("label_bbox")
        answer_box = (mapping.get("answer_region") or {}).get("bbox") or mapping.get("bbox")
        if not answer_box:
            continue

        field_type = mapping.get("field_type")
        score = float(mapping.get("candidate_score") or 0.0)
        intensity = int(80 + _clamp(score) * 175)
        answer_color = (70, intensity, 70)
        if field_type == "checkbox":
            answer_color = (0, intensity, 255)
        elif field_type == "photo":
            answer_color = (190, 80, 190)
        elif field_type == "signature":
            answer_color = (210, 100, 40)

        ax, ay, aw, ah = to_px(answer_box)
        cv2.rectangle(image, (ax, ay), (ax + aw, ay + ah), answer_color, 3)
        cv2.putText(
            image,
            f"{mapping.get('label', '')[:44]} {score:.2f}",
            (ax, max(12, ay - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            answer_color,
            1,
            cv2.LINE_AA,
        )

        if label_box:
            lx, ly, lw, lh = to_px(label_box)
            cv2.rectangle(image, (lx, ly), (lx + lw, ly + lh), (255, 150, 40), 2)
            label_center = (lx + lw // 2, ly + lh // 2)
            answer_center = (ax + aw // 2, ay + ah // 2)
            cv2.line(image, label_center, answer_center, (30, 30, 220), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write anchor debug overlay: {output_path}")
