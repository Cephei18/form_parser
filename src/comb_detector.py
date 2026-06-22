from __future__ import annotations

import os
import re
from collections import Counter
from statistics import median
from typing import Any


DEFAULT_COMB_CONFIDENCE_THRESHOLD = 0.84
MIN_GEOMETRY_CELLS = 5
MAX_GEOMETRY_CELLS = 24


LABEL_PATTERNS: tuple[dict[str, Any], ...] = (
    {
        "kind": "pan",
        "expected_length": 10,
        "confidence": 0.90,
        "patterns": (r"\bpan\b", r"permanent\s+account\s+number"),
    },
    {
        "kind": "aadhaar",
        "expected_length": 12,
        "confidence": 0.91,
        "patterns": (r"\baadhaar\b", r"\badhar\b", r"\baadhar\b", r"unique\s+identification"),
    },
    {
        "kind": "date_of_birth",
        "expected_length": 8,
        "confidence": 0.88,
        "patterns": (r"\bdob\b", r"date\s+of\s+birth", r"birth\s+date"),
    },
    {
        "kind": "ifsc",
        "expected_length": 11,
        "confidence": 0.87,
        "patterns": (r"\bifsc\b", r"ifsc\s+code"),
    },
    {
        "kind": "date",
        "expected_length": 8,
        "confidence": 0.76,
        "patterns": (r"^\s*date\s*$", r"\bdate\s*\(dd", r"\bdd\s*/\s*mm\s*/\s*yyyy\b"),
    },
    {
        "kind": "account_number",
        "expected_length": None,
        "confidence": 0.72,
        "patterns": (
            r"account\s+(number|no|#)",
            r"\ba/c\s*(number|no|#)?\b",
            r"\bacct\s*(number|no|#)?\b",
        ),
    },
)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def comb_detection_enabled() -> bool:
    return _bool_env("FORM_PARSER_COMB_DETECTION_ENABLED", False)


def comb_confidence_threshold() -> float:
    return max(0.0, min(1.0, _float_env("FORM_PARSER_COMB_MIN_CONFIDENCE", DEFAULT_COMB_CONFIDENCE_THRESHOLD)))


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize_label(value: Any) -> str:
    return _clean_text(value).lower()


def expected_length_for_label(label: Any) -> dict[str, Any] | None:
    text = _normalize_label(label)
    if not text:
        return None
    for rule in LABEL_PATTERNS:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in rule["patterns"]):
            return {
                "kind": rule["kind"],
                "expected_length": rule["expected_length"],
                "confidence": float(rule["confidence"]),
                "reason": f"label_pattern_{rule['kind']}",
            }
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return parsed


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _normalize_box(box: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(box, dict):
        return None
    x = _safe_float(box.get("x", box.get("Left")), -1.0)
    y = _safe_float(box.get("y", box.get("Top")), -1.0)
    width = _safe_float(box.get("width", box.get("Width")), 0.0)
    height = _safe_float(box.get("height", box.get("Height")), 0.0)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        return None
    x = _clamp(x)
    y = _clamp(y)
    width = max(0.0001, min(width, 1.0 - x))
    height = max(0.0001, min(height, 1.0 - y))
    return {"x": x, "y": y, "width": width, "height": height}


def _box_right(box: dict[str, float]) -> float:
    return float(box["x"]) + float(box["width"])


def _box_bottom(box: dict[str, float]) -> float:
    return float(box["y"]) + float(box["height"])


def _box_area(box: dict[str, float]) -> float:
    return max(0.0, float(box["width"])) * max(0.0, float(box["height"]))


def _box_center(box: dict[str, float]) -> tuple[float, float]:
    return float(box["x"]) + float(box["width"]) / 2.0, float(box["y"]) + float(box["height"]) / 2.0


def _intersection_area(a: dict[str, float], b: dict[str, float]) -> float:
    x1 = max(float(a["x"]), float(b["x"]))
    y1 = max(float(a["y"]), float(b["y"]))
    x2 = min(_box_right(a), _box_right(b))
    y2 = min(_box_bottom(a), _box_bottom(b))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _overlap_ratio(a: dict[str, float], b: dict[str, float]) -> float:
    return _intersection_area(a, b) / max(_box_area(a), 1e-9)


def _bbox_union(boxes: list[dict[str, float]]) -> dict[str, float] | None:
    if not boxes:
        return None
    x1 = min(float(box["x"]) for box in boxes)
    y1 = min(float(box["y"]) for box in boxes)
    x2 = max(_box_right(box) for box in boxes)
    y2 = max(_box_bottom(box) for box in boxes)
    return _normalize_box({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1})


def _round_box(box: dict[str, float]) -> dict[str, float]:
    return {
        "x": round(float(box["x"]), 6),
        "y": round(float(box["y"]), 6),
        "width": round(float(box["width"]), 6),
        "height": round(float(box["height"]), 6),
    }


def _box_from_mapping(mapping: dict[str, Any]) -> dict[str, float] | None:
    answer_region = mapping.get("answer_region") if isinstance(mapping.get("answer_region"), dict) else {}
    return _normalize_box(answer_region.get("bbox")) or _normalize_box(mapping.get("bbox"))


def _page(mapping: dict[str, Any]) -> int:
    try:
        return max(1, int(mapping.get("page") or 1))
    except (TypeError, ValueError):
        return 1


def _feature_boxes(visual_features: dict[str, Any] | None, page: int) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for index, feature in enumerate((visual_features or {}).get("empty_boxes", []) or []):
        if int(feature.get("page") or 1) != page:
            continue
        box = _normalize_box((feature.get("bbox") or feature))
        if box is None:
            continue
        features.append({"index": index, "bbox": box, "feature": feature})
    return features


def _overlaps_any(box: dict[str, float], regions: list[dict[str, Any]] | None, page: int, threshold: float = 0.25) -> bool:
    for region in regions or []:
        if int(region.get("page") or 1) != page:
            continue
        region_box = _normalize_box(region.get("bbox") or region)
        if region_box and _intersection_area(box, region_box) / max(min(_box_area(box), _box_area(region_box)), 1e-9) >= threshold:
            return True
    return False


def _is_plausible_cell(box: dict[str, float], feature: dict[str, Any]) -> tuple[bool, str | None]:
    width = float(box["width"])
    height = float(box["height"])
    area = width * height
    aspect = width / max(height, 1e-6)
    if area > 0.006:
        return False, "cell_too_large"
    if width < 0.006 or height < 0.008:
        return False, "cell_too_small"
    if not 0.45 <= aspect <= 1.85:
        return False, "cell_aspect_not_character_like"
    if int(feature.get("text_count") or 0) > 0:
        return False, "cell_contains_text"
    return True, None


def _row_groups(features: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ordered = sorted(features, key=lambda item: (_box_center(item["bbox"])[1], _box_center(item["bbox"])[0]))
    rows: list[list[dict[str, Any]]] = []
    for feature in ordered:
        box = feature["bbox"]
        _, cy = _box_center(box)
        placed = False
        for row in rows:
            row_cy = median(_box_center(item["bbox"])[1] for item in row)
            row_h = median(float(item["bbox"]["height"]) for item in row)
            if abs(cy - row_cy) <= max(row_h * 0.55, 0.008):
                row.append(feature)
                placed = True
                break
        if not placed:
            rows.append([feature])
    return [sorted(row, key=lambda item: float(item["bbox"]["x"])) for row in rows]


def _coefficient(values: list[float]) -> float:
    if not values:
        return 0.0
    center = median(values)
    if center <= 0:
        return 1.0
    return median(abs(value - center) for value in values) / center


def _sequence_quality(row: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(row) < MIN_GEOMETRY_CELLS:
        return None
    widths = [float(item["bbox"]["width"]) for item in row]
    heights = [float(item["bbox"]["height"]) for item in row]
    gaps = [
        max(0.0, float(right["bbox"]["x"]) - _box_right(left["bbox"]))
        for left, right in zip(row, row[1:])
    ]
    width_cv = _coefficient(widths)
    height_cv = _coefficient(heights)
    gap_cv = _coefficient(gaps)
    if width_cv > 0.28 or height_cv > 0.22 or gap_cv > 0.65:
        return None

    median_width = median(widths)
    median_height = median(heights)
    median_gap = median(gaps) if gaps else 0.0
    if median_gap > median_width * 1.35:
        return None

    return {
        "width_cv": round(width_cv, 4),
        "height_cv": round(height_cv, 4),
        "gap_cv": round(gap_cv, 4),
        "median_width": round(median_width, 6),
        "median_height": round(median_height, 6),
        "median_gap": round(median_gap, 6),
    }


def _distance_to_box(a: dict[str, float], b: dict[str, float]) -> float:
    ax, ay = _box_center(a)
    bx, by = _box_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _sequence_relevance(
    union: dict[str, float],
    mapping_box: dict[str, float] | None,
    label_box: dict[str, float] | None,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    if mapping_box:
        overlap = _intersection_area(union, mapping_box) / max(min(_box_area(union), _box_area(mapping_box)), 1e-9)
        if overlap >= 0.25:
            score += 0.42
            reasons.append("overlaps_answer_region")
        elif _distance_to_box(union, mapping_box) <= max(float(union["height"]) * 2.2, 0.035):
            score += 0.24
            reasons.append("near_answer_region")
    if label_box:
        label_center_y = _box_center(label_box)[1]
        union_center_y = _box_center(union)[1]
        right_of_label = float(union["x"]) >= _box_right(label_box) - max(float(label_box["height"]) * 0.75, 0.006)
        same_row = abs(label_center_y - union_center_y) <= max(float(label_box["height"]) * 1.8, float(union["height"]) * 1.1, 0.02)
        below_label = 0 <= float(union["y"]) - _box_bottom(label_box) <= max(float(label_box["height"]) * 3.2, 0.04)
        if right_of_label and same_row:
            score += 0.36
            reasons.append("right_of_label_same_row")
        elif below_label:
            score += 0.22
            reasons.append("below_label")
    return _clamp(score), reasons


def detect_geometry_comb(
    mapping: dict[str, Any],
    visual_features: dict[str, Any] | None,
    *,
    expected_length: int | None = None,
    table_cells: list[dict[str, Any]] | None = None,
    selection_regions: list[dict[str, Any]] | None = None,
    photo_regions: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(mapping, dict):
        return None
    if mapping.get("field_type") in {"checkbox", "photo", "signature"}:
        return None
    if (mapping.get("answer_region") or {}).get("type") == "table_cell":
        return None

    page = _page(mapping)
    usable: list[dict[str, Any]] = []
    rejected_cells: list[dict[str, Any]] = []
    for feature in _feature_boxes(visual_features, page):
        box = feature["bbox"]
        ok, reason = _is_plausible_cell(box, feature["feature"])
        if not ok:
            rejected_cells.append({"bbox": _round_box(box), "reason": reason})
            continue
        if _overlaps_any(box, selection_regions, page, threshold=0.35):
            rejected_cells.append({"bbox": _round_box(box), "reason": "overlaps_selection_element"})
            continue
        if _overlaps_any(box, photo_regions, page, threshold=0.30):
            rejected_cells.append({"bbox": _round_box(box), "reason": "overlaps_photo_region"})
            continue
        if _overlaps_any(box, table_cells, page, threshold=0.30):
            rejected_cells.append({"bbox": _round_box(box), "reason": "overlaps_table_cell"})
            continue
        usable.append(feature)

    mapping_box = _box_from_mapping(mapping)
    label_box = _normalize_box(mapping.get("label_bbox"))
    best: dict[str, Any] | None = None
    rejections: list[dict[str, Any]] = []

    for row in _row_groups(usable):
        if len(row) < MIN_GEOMETRY_CELLS:
            continue
        quality = _sequence_quality(row)
        if quality is None:
            rejections.append({"cell_count": len(row), "reason": "poor_size_or_spacing_consistency"})
            continue
        if len(row) > MAX_GEOMETRY_CELLS:
            rejections.append({"cell_count": len(row), "reason": "too_many_cells_for_comb"})
            continue
        boxes = [item["bbox"] for item in row]
        union = _bbox_union(boxes)
        if union is None:
            continue
        relevance, relevance_reasons = _sequence_relevance(union, mapping_box, label_box)
        if relevance < 0.30:
            rejections.append({"cell_count": len(row), "reason": "not_near_mapping", "bbox": _round_box(union)})
            continue

        count = len(row)
        count_score = 0.68
        count_reason = "geometry_cell_count"
        if expected_length:
            delta = abs(count - expected_length)
            if delta == 0:
                count_score = 0.96
                count_reason = "geometry_matches_expected_length"
            elif delta <= 1:
                count_score = 0.82
                count_reason = "geometry_near_expected_length"
            else:
                count_score = max(0.38, 0.75 - delta * 0.08)
                count_reason = "geometry_length_differs_from_label"
        elif count < 6:
            rejections.append({"cell_count": count, "reason": "too_few_cells_without_label"})
            continue

        consistency = _clamp(
            1.0
            - float(quality["width_cv"]) * 1.1
            - float(quality["height_cv"]) * 1.2
            - float(quality["gap_cv"]) * 0.35
        )
        confidence = _clamp(0.32 * consistency + 0.36 * count_score + 0.32 * relevance)
        candidate = {
            "source": "geometry",
            "confidence": round(confidence, 4),
            "expected_length": expected_length,
            "comb_cells": count,
            "comb_boxes": [_round_box(box) for box in boxes],
            "bbox": _round_box(union),
            "reason": count_reason,
            "reasons": [count_reason, *relevance_reasons, "equal_size_boxes", "equal_spacing"],
            "geometry_evidence": {
                **quality,
                "cell_count": count,
                "relevance": round(relevance, 4),
            },
            "rejected_cells": rejected_cells[:20],
        }
        if best is None or candidate["confidence"] > best["confidence"]:
            best = candidate

    if best is not None:
        best["rejected_candidates"] = rejections[:20]
    return best


def _label_only_candidate(mapping: dict[str, Any], label_hit: dict[str, Any]) -> dict[str, Any] | None:
    expected_length = label_hit.get("expected_length")
    if not expected_length:
        return None
    mapping_box = _box_from_mapping(mapping)
    if mapping_box is None:
        return None
    aspect = float(mapping_box["width"]) / max(float(mapping_box["height"]), 1e-6)
    if aspect < max(2.2, float(expected_length) * 0.42):
        return {
            "source": "label",
            "confidence": round(float(label_hit["confidence"]) * 0.62, 4),
            "expected_length": expected_length,
            "comb_cells": expected_length,
            "comb_boxes": [],
            "bbox": _round_box(mapping_box),
            "reason": "label_hit_but_answer_region_too_narrow",
            "rejected": True,
        }
    return {
        "source": "label",
        "confidence": round(float(label_hit["confidence"]), 4),
        "expected_length": expected_length,
        "comb_cells": expected_length,
        "comb_boxes": [],
        "bbox": _round_box(mapping_box),
        "reason": label_hit["reason"],
        "reasons": [label_hit["reason"], "wide_answer_region"],
        "geometry_evidence": {
            "answer_aspect_ratio": round(aspect, 4),
            "cell_count": 0,
        },
        "rejected_candidates": [],
    }


def detect_comb_candidate(
    mapping: dict[str, Any],
    visual_features: dict[str, Any] | None,
    *,
    table_cells: list[dict[str, Any]] | None = None,
    selection_regions: list[dict[str, Any]] | None = None,
    photo_regions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    label_hit = expected_length_for_label(mapping.get("label") if isinstance(mapping, dict) else "")
    expected_length = int(label_hit["expected_length"]) if label_hit and label_hit.get("expected_length") else None
    mapping_box = _box_from_mapping(mapping) if isinstance(mapping, dict) else None
    page = _page(mapping) if isinstance(mapping, dict) else 1
    suppress_label_only = False
    suppress_reasons: list[dict[str, Any]] = []
    if isinstance(mapping, dict) and (mapping.get("answer_region") or {}).get("type") == "table_cell":
        suppress_label_only = True
        suppress_reasons.append({"reason": "answer_region_is_table_cell"})
    if mapping_box and _overlaps_any(mapping_box, table_cells, page, threshold=0.25):
        suppress_label_only = True
        suppress_reasons.append({"reason": "answer_region_overlaps_table_cell", "bbox": _round_box(mapping_box)})
    if mapping_box and _overlaps_any(mapping_box, photo_regions, page, threshold=0.25):
        suppress_label_only = True
        suppress_reasons.append({"reason": "answer_region_overlaps_photo_region", "bbox": _round_box(mapping_box)})

    geometry = detect_geometry_comb(
        mapping,
        visual_features,
        expected_length=expected_length,
        table_cells=table_cells,
        selection_regions=selection_regions,
        photo_regions=photo_regions,
    )
    label_only = _label_only_candidate(mapping, label_hit) if label_hit and not suppress_label_only else None

    candidates = [candidate for candidate in (geometry, label_only) if candidate and not candidate.get("rejected")]
    rejected = [candidate for candidate in (geometry, label_only) if candidate and candidate.get("rejected")]
    rejected.extend(suppress_reasons)
    if geometry and geometry.get("rejected_candidates"):
        rejected.extend(geometry.get("rejected_candidates") or [])
    if not candidates:
        return {
            "accepted": False,
            "confidence": 0.0,
            "reason": "no_comb_candidate",
            "label_evidence": label_hit,
            "rejected_candidates": rejected,
        }

    # Prefer geometry when it is available; it carries the real cell count and
    # union box. Label evidence boosts it only when lengths agree.
    best = max(candidates, key=lambda item: (item.get("source") == "geometry", float(item.get("confidence") or 0.0)))
    if geometry and label_hit and geometry.get("comb_cells") == label_hit.get("expected_length"):
        best = dict(geometry)
        best["confidence"] = round(_clamp(float(geometry["confidence"]) + 0.07), 4)
        best["reasons"] = list(dict.fromkeys((geometry.get("reasons") or []) + [label_hit["reason"]]))
        best["expected_length"] = label_hit.get("expected_length")
    elif geometry and label_hit:
        best = dict(geometry)
        best["confidence"] = round(_clamp(float(geometry["confidence"]) + 0.05), 4)
        best["reasons"] = list(dict.fromkeys((geometry.get("reasons") or []) + [label_hit["reason"]]))
        if label_hit.get("expected_length"):
            best["expected_length"] = label_hit.get("expected_length")
    elif geometry:
        best = geometry

    best = dict(best)
    best["accepted"] = True
    best["label_evidence"] = label_hit
    best.setdefault("rejected_candidates", rejected)
    return best


def enrich_mapping_with_comb(mapping: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(mapping)
    comb_cells = int(candidate.get("comb_cells") or candidate.get("expected_length") or 0)
    enriched["widget_type"] = "comb"
    enriched["comb_cells"] = comb_cells
    enriched["comb_boxes"] = candidate.get("comb_boxes") or []
    enriched["expected_length"] = candidate.get("expected_length") or comb_cells
    enriched["comb_confidence"] = round(float(candidate.get("confidence") or 0.0), 4)
    enriched["comb_reason"] = candidate.get("reason") or "comb_detected"
    enriched["comb_detection"] = {
        "confidence": enriched["comb_confidence"],
        "reason": enriched["comb_reason"],
        "reasons": candidate.get("reasons", []),
        "source": candidate.get("source"),
        "geometry_evidence": candidate.get("geometry_evidence", {}),
        "label_evidence": candidate.get("label_evidence"),
    }
    if candidate.get("bbox"):
        bbox = _round_box(candidate["bbox"])
        enriched["bbox"] = bbox
        answer_region = dict(enriched.get("answer_region") or {})
        answer_region["bbox"] = bbox
        answer_region["type"] = "comb_region"
        answer_region["confidence"] = enriched["comb_confidence"]
        enriched["answer_region"] = answer_region
        enriched["render_border"] = False
    return enriched


def apply_comb_detection(
    mappings: list[dict[str, Any]],
    visual_features: dict[str, Any] | None,
    *,
    table_cells: list[dict[str, Any]] | None = None,
    selection_regions: list[dict[str, Any]] | None = None,
    photo_regions: list[dict[str, Any]] | None = None,
    enabled: bool | None = None,
    threshold: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enabled = comb_detection_enabled() if enabled is None else bool(enabled)
    threshold = comb_confidence_threshold() if threshold is None else max(0.0, min(1.0, float(threshold)))
    if not enabled:
        return list(mappings or []), {
            "enabled": False,
            "feature_flag": "FORM_PARSER_COMB_DETECTION_ENABLED",
            "threshold": threshold,
            "detected_count": 0,
            "detected_fields": [],
            "rejected_candidates": [],
            "reasoning": "Comb detection is disabled; mappings are unchanged.",
        }

    enriched_mappings: list[dict[str, Any]] = []
    detected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for mapping in mappings or []:
        if not isinstance(mapping, dict):
            enriched_mappings.append(mapping)
            continue
        candidate = detect_comb_candidate(
            mapping,
            visual_features,
            table_cells=table_cells,
            selection_regions=selection_regions,
            photo_regions=photo_regions,
        )
        if candidate.get("accepted") and float(candidate.get("confidence") or 0.0) >= threshold:
            enriched = enrich_mapping_with_comb(mapping, candidate)
            enriched_mappings.append(enriched)
            detected.append(
                {
                    "field_id": mapping.get("field_id"),
                    "label": mapping.get("label"),
                    "page": mapping.get("page", 1),
                    "comb_cells": enriched.get("comb_cells"),
                    "expected_length": enriched.get("expected_length"),
                    "confidence": enriched.get("comb_confidence"),
                    "reason": enriched.get("comb_reason"),
                    "reasons": candidate.get("reasons", []),
                    "bbox": enriched.get("bbox"),
                    "geometry_evidence": candidate.get("geometry_evidence", {}),
                    "label_evidence": candidate.get("label_evidence"),
                }
            )
            continue

        enriched_mappings.append(mapping)
        if candidate.get("accepted") or candidate.get("label_evidence") or candidate.get("rejected_candidates"):
            rejected.append(
                {
                    "field_id": mapping.get("field_id"),
                    "label": mapping.get("label"),
                    "page": mapping.get("page", 1),
                    "confidence": round(float(candidate.get("confidence") or 0.0), 4),
                    "reason": candidate.get("reason", "below_threshold"),
                    "label_evidence": candidate.get("label_evidence"),
                    "threshold": threshold,
                    "rejected_candidates": candidate.get("rejected_candidates", [])[:12],
                }
            )

    counts = Counter(str(item.get("reason") or "unknown") for item in detected)
    return enriched_mappings, {
        "enabled": True,
        "feature_flag": "FORM_PARSER_COMB_DETECTION_ENABLED",
        "threshold": threshold,
        "detected_count": len(detected),
        "detected_fields": detected,
        "rejected_count": len(rejected),
        "rejected_candidates": rejected[:50],
        "reason_counts": dict(sorted(counts.items())),
        "reasoning": (
            "Comb fields are emitted only when label or geometry evidence reaches "
            "the configured confidence threshold; field_type remains unchanged."
        ),
    }
