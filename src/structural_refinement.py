from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

import cv2


EXCLUDED_REGION_TYPES = {
    "photo_region",
    "signature_area",
    "stamp_area",
    "logo_region",
    "decorative_box",
    "non_text_sparse_region",
    "non_text_candidate",
    "table_like_region",
    "checkbox_region",
}


@dataclass(frozen=True)
class FieldCandidateRefinementResult:
    lines: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _median(values, default: float) -> float:
    clean = [float(value) for value in values if isinstance(value, (int, float)) and value > 0]
    if not clean:
        return default
    return float(median(clean))


def _line_bounds(line: dict[str, Any]) -> tuple[float, float, float, float]:
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _line_center(line: dict[str, Any]) -> tuple[float, float]:
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def _line_length(line: dict[str, Any]) -> float:
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _ocr_bounds(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = item.get("bbox")
    if not bbox:
        return None
    try:
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
    except Exception:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _box_center(bounds: tuple[float, float, float, float] | list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bounds
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_overlap(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    overlap_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    overlap_y = max(0.0, min(ay2, by2) - max(ay1, by1))
    if overlap_x <= 0 or overlap_y <= 0:
        return 0.0
    return (overlap_x * overlap_y) / max((ax2 - ax1) * (ay2 - ay1), 1.0)


def _point_in_bounds(point, bounds, padding: float = 0.0) -> bool:
    x, y = point
    x1, y1, x2, y2 = bounds
    return x1 - padding <= x <= x2 + padding and y1 - padding <= y <= y2 + padding


def _region_bounds(region: dict[str, Any]) -> list[float]:
    x = float(region.get("x", 0.0))
    y = float(region.get("y", 0.0))
    return [
        x,
        y,
        x + float(region.get("width", 0.0)),
        y + float(region.get("height", 0.0)),
    ]


def _page_metrics(image_path: str, ocr_data, lines) -> dict[str, float]:
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    page_height, page_width = img.shape[:2]
    text_heights = []
    text_widths = []
    for item in ocr_data or []:
        bounds = _ocr_bounds(item)
        if bounds is None:
            continue
        text_heights.append(bounds[3] - bounds[1])
        text_widths.append(bounds[2] - bounds[0])
    line_lengths = [_line_length(line) for line in lines or []]
    return {
        "page_width": float(page_width),
        "page_height": float(page_height),
        "avg_text_height": _median(text_heights, 18.0),
        "avg_text_width": _median(text_widths, 80.0),
        "avg_line_length": _median(line_lengths, max(float(page_width) * 0.22, 180.0)),
        "row_tolerance": max(18.0, _median(text_heights, 18.0) * 1.35, float(page_height) * 0.008),
    }


def _group_lines_by_y(lines, tolerance: float) -> list[list[dict[str, Any]]]:
    groups: list[dict[str, Any]] = []
    for line in sorted(lines or [], key=lambda item: (_line_center(item)[1], _line_center(item)[0])):
        y = _line_center(line)[1]
        if not groups or abs(groups[-1]["center_y"] - y) > tolerance:
            groups.append({"center_y": y, "items": [line]})
            continue
        group = groups[-1]
        group["items"].append(line)
        group["center_y"] = sum(_line_center(item)[1] for item in group["items"]) / len(group["items"])
    return [group["items"] for group in groups]


def detect_dense_line_regions(lines, metrics: dict[str, float]) -> list[dict[str, Any]]:
    """Find repeated dense horizontal bands that should be treated as table structures."""
    if not lines:
        return []

    row_tolerance = max(8.0, metrics["avg_text_height"] * 0.55)
    row_groups = _group_lines_by_y(lines, row_tolerance)
    dense_rows = []
    page_width = max(metrics["page_width"], 1.0)
    for group in row_groups:
        bounds = [_line_bounds(line) for line in group]
        x1 = min(bound[0] for bound in bounds)
        x2 = max(bound[2] for bound in bounds)
        total_width = sum(bound[2] - bound[0] for bound in bounds)
        coverage = min((x2 - x1) / page_width, 1.0)
        segmented_coverage = min(total_width / page_width, 1.0)
        if len(group) >= 4 or (len(group) >= 3 and max(coverage, segmented_coverage) >= 0.45):
            dense_rows.append(
                {
                    "center_y": sum(_line_center(line)[1] for line in group) / len(group),
                    "lines": group,
                    "coverage": max(coverage, segmented_coverage),
                }
            )

    if not dense_rows:
        return []

    row_gap = max(metrics["avg_text_height"] * 3.2, 48.0)
    bands: list[list[dict[str, Any]]] = []
    current = [dense_rows[0]]
    for row in dense_rows[1:]:
        if row["center_y"] - current[-1]["center_y"] <= row_gap:
            current.append(row)
            continue
        if len(current) >= 3:
            bands.append(current)
        current = [row]
    if len(current) >= 3:
        bands.append(current)

    regions = []
    for index, band in enumerate(bands):
        band_lines = [line for row in band for line in row["lines"]]
        bounds = [_line_bounds(line) for line in band_lines]
        x1 = min(bound[0] for bound in bounds)
        y1 = min(bound[1] for bound in bounds)
        x2 = max(bound[2] for bound in bounds)
        y2 = max(bound[3] for bound in bounds)
        row_count = len(band)
        line_count = len(band_lines)
        coverage = sum(row["coverage"] for row in band) / max(row_count, 1)
        regions.append(
            {
                "id": f"dense-table-{index}",
                "type": "table_like_region",
                "bounds": [
                    round(max(0.0, x1 - metrics["avg_text_height"]), 2),
                    round(max(0.0, y1 - metrics["avg_text_height"]), 2),
                    round(min(metrics["page_width"], x2 + metrics["avg_text_height"]), 2),
                    round(min(metrics["page_height"], y2 + metrics["avg_text_height"]), 2),
                ],
                "row_count": row_count,
                "line_count": line_count,
                "coverage": round(coverage, 4),
                "confidence": round(_clamp01(0.46 + 0.08 * min(row_count, 5) + 0.12 * coverage), 4),
                "reasons": ["dense_repeated_horizontal_lines", "table_border_discrimination"],
            }
        )
    return regions


def _line_overlaps_ocr_text(line, ocr_data, padding: float = 2.0) -> bool:
    line_left, line_top, line_right, line_bottom = _line_bounds(line)
    line_y = (line_top + line_bottom) / 2.0
    line_width = max(line_right - line_left, 1.0)

    for item in ocr_data or []:
        bounds = _ocr_bounds(item)
        if bounds is None:
            continue
        text_left, text_top, text_right, text_bottom = bounds
        text_width = max(text_right - text_left, 1.0)
        text_height = max(text_bottom - text_top, 1.0)
        if not (text_top - padding <= line_y <= text_bottom + padding):
            continue
        overlap = max(0.0, min(line_right, text_right + padding) - max(line_left, text_left - padding))
        if overlap >= min(text_width * 0.35, line_width * 0.55) or overlap >= 24.0:
            return True
    return False


def _nearby_label_features(line, ocr_data, metrics: dict[str, float]) -> dict[str, Any]:
    line_bounds = _line_bounds(line)
    line_center = _line_center(line)
    best: dict[str, Any] | None = None
    best_score = -1.0

    for item in ocr_data or []:
        bounds = _ocr_bounds(item)
        if bounds is None:
            continue
        text_center = _box_center(bounds)
        row_gap = abs(text_center[1] - line_center[1])
        vertical_above_gap = max(0.0, line_bounds[1] - bounds[3])
        horizontal_gap = line_bounds[0] - bounds[2]
        overlap = max(0.0, min(bounds[2], line_bounds[2]) - max(bounds[0], line_bounds[0]))
        overlap_ratio = overlap / max(min(bounds[2] - bounds[0], line_bounds[2] - line_bounds[0]), 1.0)
        same_row = row_gap <= metrics["row_tolerance"] * 1.25 and horizontal_gap >= -metrics["avg_text_width"] * 0.25
        above = 0 <= vertical_above_gap <= metrics["avg_text_height"] * 1.8 and overlap_ratio > 0.08
        if not same_row and not above:
            continue
        distance_score = _clamp01(1.0 - ((max(horizontal_gap, 0.0) + row_gap) / max(metrics["page_width"] * 0.55, 1.0)))
        alignment_score = 0.35 if same_row else 0.18
        alignment_score += 0.18 if above else 0.0
        alignment_score += 0.20 * overlap_ratio
        score = _clamp01(distance_score * 0.55 + alignment_score)
        if score > best_score:
            best_score = score
            best = {
                "text": item.get("text", ""),
                "center": [round(text_center[0], 2), round(text_center[1], 2)],
                "same_row": same_row,
                "above": above,
                "horizontal_gap": round(horizontal_gap, 2),
                "row_gap": round(row_gap, 2),
                "overlap_ratio": round(overlap_ratio, 4),
                "score": round(score, 4),
            }

    return best or {"score": 0.0}


def _line_inside_structural_region(line, regions, padding: float = 0.0) -> dict[str, Any] | None:
    center = _line_center(line)
    bounds = _line_bounds(line)
    for region in regions or []:
        region_bounds = region.get("bounds") if "bounds" in region else _region_bounds(region)
        if _point_in_bounds(center, region_bounds, padding=padding) or _box_overlap(bounds, region_bounds) > 0:
            return region
    return None


def _quality_for_line(
    line,
    ocr_data,
    semantic_regions,
    dense_table_regions,
    metrics: dict[str, float],
    table_aware_enabled: bool,
) -> dict[str, Any]:
    length = _line_length(line)
    x1, y1, x2, y2 = _line_bounds(line)
    page_width = max(metrics["page_width"], 1.0)
    page_height = max(metrics["page_height"], 1.0)
    length_ratio = length / page_width
    center = _line_center(line)
    reasons = []
    penalties = []
    score = 0.24

    if abs(y2 - y1) <= max(4.0, metrics["avg_text_height"] * 0.25):
        score += 0.10
        reasons.append("horizontal_line_semantics")
    else:
        penalties.append("non_horizontal_candidate")
        score -= 0.18

    if 0.08 <= length_ratio <= 0.68:
        score += 0.18
        reasons.append("field_length_page_relative")
    elif length_ratio > 0.82:
        score -= 0.34
        penalties.append("page_width_separator")
    elif length_ratio < 0.045:
        score -= 0.16
        penalties.append("too_short_for_text_field")
    else:
        score += 0.04
        reasons.append("acceptable_field_length")

    label_features = _nearby_label_features(line, ocr_data, metrics)
    if label_features.get("score", 0.0) >= 0.35:
        score += 0.28 * float(label_features["score"])
        reasons.append("nearby_label_support")
        if label_features.get("same_row"):
            score += 0.08
            reasons.append("same_row_label_support")
        if label_features.get("above"):
            score += 0.06
            reasons.append("above_label_support")
    else:
        score -= 0.08
        penalties.append("no_nearby_label_support")

    source = line.get("field_type", "line")
    if source in {"box", "weak_line", "fallback_text_region"}:
        score += 0.05
        reasons.append(f"source_{source}")

    if center[0] > page_width * 0.18:
        score += 0.04
        reasons.append("right_side_field_position")

    if x1 < page_width * 0.04 and x2 > page_width * 0.78 and not label_features.get("same_row"):
        score -= 0.22
        penalties.append("decorative_separator_position")

    if y1 < page_height * 0.08 and length_ratio > 0.52:
        score -= 0.14
        penalties.append("header_separator_risk")

    if _line_overlaps_ocr_text(line, ocr_data):
        score -= 0.32
        penalties.append("overlaps_ocr_text")

    semantic_region = _line_inside_structural_region(line, semantic_regions, padding=2.0)
    semantic_type = semantic_region.get("type") if semantic_region else None
    if semantic_type in EXCLUDED_REGION_TYPES:
        score -= 0.42
        penalties.append(f"inside_{semantic_type}")
    elif semantic_region is not None:
        score += 0.05
        reasons.append(f"semantic_region_{semantic_type}")

    table_region = _line_inside_structural_region(line, dense_table_regions, padding=metrics["avg_text_height"] * 0.35)
    if table_region is not None:
        if table_aware_enabled:
            score -= 0.55
            penalties.append("inside_dense_table_structure")
        else:
            penalties.append("dense_table_structure_detected")

    if "inside_checkbox_region" in penalties or semantic_type == "checkbox_region":
        score -= 0.20
        penalties.append("checkbox_candidate_not_text_line")

    classification = "input_candidate"
    if table_region is not None and table_aware_enabled:
        classification = "table_border"
    elif "page_width_separator" in penalties or "decorative_separator_position" in penalties:
        classification = "decorative_separator"
    elif semantic_type == "checkbox_region":
        classification = "checkbox_artifact"
    elif "overlaps_ocr_text" in penalties:
        classification = "text_overlap_artifact"
    elif score < 0.32:
        classification = "weak_candidate"

    return {
        "score": round(_clamp01(score), 4),
        "classification": classification,
        "length": round(length, 2),
        "length_ratio": round(length_ratio, 4),
        "nearby_label": label_features,
        "semantic_region_type": semantic_type,
        "table_region_id": table_region.get("id") if table_region else None,
        "reasons": reasons,
        "penalties": penalties,
    }


def refine_field_candidates(
    image_path: str,
    field_lines,
    ocr_data,
    semantic_regions,
    config,
) -> FieldCandidateRefinementResult:
    metrics = _page_metrics(image_path, ocr_data, field_lines)
    table_aware_enabled = bool(getattr(config, "table_aware_enabled", True))
    dense_table_regions = detect_dense_line_regions(field_lines, metrics) if table_aware_enabled else []
    min_score = float(getattr(config, "min_field_quality_score", 0.32))
    refined = []
    candidate_diagnostics = []
    removed_by_classification: dict[str, int] = {}

    for index, line in enumerate(field_lines or []):
        quality = _quality_for_line(
            line,
            ocr_data,
            semantic_regions,
            dense_table_regions,
            metrics,
            table_aware_enabled,
        )
        keep = quality["score"] >= min_score and quality["classification"] not in {
            "decorative_separator",
            "table_border",
            "checkbox_artifact",
            "text_overlap_artifact",
        }
        enriched = {
            **line,
            "candidate_quality": quality["score"],
            "line_semantics": quality["classification"],
            "quality_reasons": quality["reasons"],
            "quality_penalties": quality["penalties"],
        }
        if keep:
            refined.append(enriched)
        else:
            removed_by_classification[quality["classification"]] = removed_by_classification.get(quality["classification"], 0) + 1

        candidate_diagnostics.append(
            {
                "index": index,
                "line": line,
                "kept": keep,
                **quality,
            }
        )

    diagnostics = {
        "enabled": True,
        "input_count": len(field_lines or []),
        "output_count": len(refined),
        "removed_count": max(0, len(field_lines or []) - len(refined)),
        "min_field_quality_score": round(min_score, 4),
        "page_metrics": {key: round(value, 4) for key, value in metrics.items()},
        "removed_by_classification": removed_by_classification,
        "table_structures": dense_table_regions,
        "candidates": candidate_diagnostics,
    }
    return FieldCandidateRefinementResult(refined, diagnostics)


def _entity_lookup(layout_structure: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entities = layout_structure.get("entities", {}) if isinstance(layout_structure, dict) else {}
    lookup = {}
    for group in ["ocr", "fields", "regions"]:
        for item in entities.get(group, []) or []:
            lookup[item.get("id")] = item
    return lookup


def _band_texts(band: dict[str, Any], entity_by_id: dict[str, dict[str, Any]]) -> list[str]:
    texts = []
    for item_id in band.get("label_item_ids", []) or []:
        text = entity_by_id.get(item_id, {}).get("text")
        if text:
            texts.append(str(text))
    return texts


def _looks_like_section_header(band: dict[str, Any], entity_by_id: dict[str, dict[str, Any]], next_band: dict[str, Any] | None) -> tuple[bool, list[str]]:
    texts = _band_texts(band, entity_by_id)
    joined = " ".join(texts).strip()
    if not joined:
        return False, []

    reasons = []
    alpha = [char for char in joined if char.isalpha()]
    uppercase_ratio = sum(1 for char in alpha if char.isupper()) / len(alpha) if alpha else 0.0
    has_following_fields = bool(next_band and int(next_band.get("field_count", 0)) > 0)
    no_fields_in_band = int(band.get("field_count", 0)) == 0
    wide_header = (band["bounds"][2] - band["bounds"][0]) >= 180

    if no_fields_in_band and has_following_fields:
        reasons.append("label_only_band_before_fields")
    if uppercase_ratio >= 0.68 and wide_header:
        reasons.append("uppercase_wide_header")
    if len(texts) >= 2 and no_fields_in_band:
        reasons.append("multi_text_label_band")

    return bool(no_fields_in_band and reasons), reasons


def _section_from_bands(
    section_id: str,
    bands: list[dict[str, Any]],
    entity_by_id: dict[str, dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    bounds = [
        min(band["bounds"][0] for band in bands),
        min(band["bounds"][1] for band in bands),
        max(band["bounds"][2] for band in bands),
        max(band["bounds"][3] for band in bands),
    ]
    label_zone_ids = []
    field_zone_ids = []
    header_texts = _band_texts(bands[0], entity_by_id)
    for band in bands:
        label_zone_ids.extend(band.get("label_zone_ids", []) or [])
        field_zone_ids.extend(band.get("field_zone_ids", []) or [])
    confidence = _clamp01(0.52 + 0.04 * min(len(bands), 5) + 0.05 * bool(field_zone_ids))
    return {
        "id": section_id,
        "type": "logical_section",
        "bounds": [round(value, 2) for value in bounds],
        "center": [round((bounds[0] + bounds[2]) / 2.0, 2), round((bounds[1] + bounds[3]) / 2.0, 2)],
        "band_ids": [band["id"] for band in bands],
        "label_zone_ids": sorted(set(label_zone_ids)),
        "field_zone_ids": sorted(set(field_zone_ids)),
        "header_texts": header_texts,
        "band_count": len(bands),
        "confidence": round(confidence, 4),
        "reasons": [reason],
    }


def _build_sections(layout_structure: dict[str, Any]) -> list[dict[str, Any]]:
    bands = list(layout_structure.get("layout_bands", []) or [])
    if not bands:
        return []

    entity_by_id = _entity_lookup(layout_structure)
    zone_by_band: dict[str, dict[str, list[str]]] = {}
    for zone in layout_structure.get("label_zones", []) or []:
        zone_by_band.setdefault(zone.get("band_id"), {"labels": [], "fields": []})["labels"].append(zone["id"])
    for zone in layout_structure.get("field_zones", []) or []:
        zone_by_band.setdefault(zone.get("band_id"), {"labels": [], "fields": []})["fields"].append(zone["id"])
    for band in bands:
        membership = zone_by_band.get(band["id"], {"labels": [], "fields": []})
        band["label_zone_ids"] = membership["labels"]
        band["field_zone_ids"] = membership["fields"]

    start_indexes = [0]
    metrics = layout_structure.get("metrics", {})
    gap_threshold = max(float(metrics.get("region_gap", 32.0)) * 1.7, float(metrics.get("avg_text_height", 18.0)) * 3.6)
    for index, band in enumerate(bands):
        previous = bands[index - 1] if index > 0 else None
        next_band = bands[index + 1] if index + 1 < len(bands) else None
        is_header, _ = _looks_like_section_header(band, entity_by_id, next_band)
        if index > 0 and is_header:
            start_indexes.append(index)
            continue
        if previous is not None:
            gap = band["bounds"][1] - previous["bounds"][3]
            if gap >= gap_threshold and int(band.get("label_count", 0)) > 0:
                start_indexes.append(index)

    start_indexes = sorted(set(start_indexes))
    sections = []
    for section_index, start in enumerate(start_indexes):
        stop = start_indexes[section_index + 1] if section_index + 1 < len(start_indexes) else len(bands)
        section_bands = bands[start:stop]
        if not section_bands:
            continue
        reason = "section_header" if start != 0 else "document_start"
        sections.append(_section_from_bands(f"section-{section_index}", section_bands, entity_by_id, reason))
    return sections


def _section_indexes(sections: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    by_band = {}
    by_label_zone = {}
    by_field_zone = {}
    for section in sections:
        for band_id in section.get("band_ids", []) or []:
            by_band[band_id] = section["id"]
        for zone_id in section.get("label_zone_ids", []) or []:
            by_label_zone[zone_id] = section["id"]
        for zone_id in section.get("field_zone_ids", []) or []:
            by_field_zone[zone_id] = section["id"]
    return {
        "section_by_band_id": by_band,
        "section_by_label_zone_id": by_label_zone,
        "section_by_field_zone_id": by_field_zone,
    }


def _section_for_zone(zone: dict[str, Any], index: dict[str, dict[str, str]], zone_kind: str) -> str | None:
    if not zone:
        return None
    key = "section_by_label_zone_id" if zone_kind == "label" else "section_by_field_zone_id"
    return index.get(key, {}).get(zone.get("id"))


def _build_ownership_chains(layout_structure: dict[str, Any], sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = layout_structure.get("metrics", {})
    band_lookup = (layout_structure.get("indexes", {}) or {}).get("band_by_id", {})
    section_index = _section_indexes(sections)
    chains = []

    for label_zone in layout_structure.get("label_zones", []) or []:
        label_section_id = _section_for_zone(label_zone, section_index, "label")
        candidate_fields = []
        for field_zone in layout_structure.get("field_zones", []) or []:
            field_section_id = _section_for_zone(field_zone, section_index, "field")
            if label_section_id and field_section_id and label_section_id != field_section_id:
                continue
            label_band = band_lookup.get(label_zone.get("band_id"), {})
            field_band = band_lookup.get(field_zone.get("band_id"), {})
            band_gap = abs(float(field_band.get("center_y", 0.0)) - float(label_band.get("center_y", 0.0)))
            right_of_label = field_zone["bounds"][0] >= label_zone["bounds"][2] - float(metrics.get("column_tolerance", 40.0)) * 0.2
            vertical_score = _clamp01(1.0 - band_gap / max(float(metrics.get("band_tolerance", 36.0)) * 2.2, 1.0))
            horizontal_score = 0.75 if right_of_label else 0.28
            same_band = label_zone.get("band_id") == field_zone.get("band_id")
            score = _clamp01(
                vertical_score * 0.42
                + horizontal_score * 0.30
                + float(field_zone.get("confidence", 0.0)) * 0.16
                + float(label_zone.get("confidence", 0.0)) * 0.12
                + (0.10 if same_band else 0.0)
            )
            candidate_fields.append((score, field_zone, same_band, band_gap, right_of_label))

        if not candidate_fields:
            continue
        score, field_zone, same_band, band_gap, right_of_label = max(candidate_fields, key=lambda item: item[0])
        chains.append(
            {
                "id": f"ownership-chain-{len(chains)}",
                "type": "ownership_chain",
                "label_zone_id": label_zone["id"],
                "field_zone_id": field_zone["id"],
                "section_id": label_section_id,
                "score": round(score, 4),
                "same_band": bool(same_band),
                "band_gap": round(band_gap, 2),
                "right_of_label": bool(right_of_label),
                "reasons": ["contextual_candidate_ranking", "neighboring_field_reasoning"],
            }
        )
    return chains


def _build_row_continuity(layout_structure: dict[str, Any]) -> list[dict[str, Any]]:
    clusters = layout_structure.get("field_clusters", []) or []
    if not clusters:
        return []

    rows = []
    for cluster in clusters:
        if cluster.get("line_count", 0) <= 0:
            continue
        rows.append(
            {
                "center_y": float(cluster.get("center_y", 0.0)),
                "min_x": float(cluster.get("min_x", 0.0) or 0.0),
                "max_x": float(cluster.get("max_x", 0.0) or 0.0),
                "line_count": int(cluster.get("line_count", 0) or 0),
                "cluster_type": cluster.get("type", "field_row"),
            }
        )
    rows.sort(key=lambda item: item["center_y"])

    continuity = []
    for index, row in enumerate(rows):
        previous = rows[index - 1] if index > 0 else None
        next_row = rows[index + 1] if index + 1 < len(rows) else None
        neighbors = [item for item in [previous, next_row] if item is not None]
        if not neighbors:
            score = 0.0
            reasons = ["isolated_field_row"]
        else:
            span = max(row["max_x"] - row["min_x"], 1.0)
            scores = []
            for neighbor in neighbors:
                neighbor_span = max(neighbor["max_x"] - neighbor["min_x"], 1.0)
                left_similarity = 1.0 - min(abs(row["min_x"] - neighbor["min_x"]) / max(span, neighbor_span), 1.0)
                right_similarity = 1.0 - min(abs(row["max_x"] - neighbor["max_x"]) / max(span, neighbor_span), 1.0)
                count_similarity = 1.0 - min(abs(row["line_count"] - neighbor["line_count"]) / max(row["line_count"], neighbor["line_count"], 1), 1.0)
                scores.append(_clamp01(left_similarity * 0.35 + right_similarity * 0.35 + count_similarity * 0.30))
            score = sum(scores) / len(scores)
            reasons = ["row_continuity_reasoning"] if score >= 0.55 else ["row_shape_shift"]
        continuity.append(
            {
                "row_index": index,
                "center_y": round(row["center_y"], 2),
                "line_count": row["line_count"],
                "continuity_score": round(score, 4),
                "reasons": reasons,
            }
        )
    return continuity


def _semantic_table_structures(semantic_regions) -> list[dict[str, Any]]:
    structures = []
    for index, region in enumerate(semantic_regions or []):
        if region.get("type") != "table_like_region":
            continue
        structures.append(
            {
                "id": f"semantic-table-{index}",
                "type": "table_like_region",
                "bounds": _region_bounds(region),
                "confidence": float(region.get("confidence", 0.0) or 0.0),
                "reasons": list(region.get("reasons", [])) + ["semantic_table_region"],
            }
        )
    return structures


def refine_layout_structure(
    layout_structure: dict[str, Any],
    ocr_data,
    field_lines,
    semantic_regions,
    config,
    field_candidate_diagnostics: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    refined = dict(layout_structure or {})
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "section_grouping_enabled": bool(getattr(config, "section_grouping_enabled", True)),
        "ownership_propagation_enabled": bool(getattr(config, "ownership_propagation_enabled", True)),
        "table_aware_enabled": bool(getattr(config, "table_aware_enabled", True)),
    }

    sections: list[dict[str, Any]] = []
    if getattr(config, "section_grouping_enabled", True):
        sections = _build_sections(refined)
        refined["sections"] = sections
        refined["region_segments"] = [
            {
                "id": f"segment-{index}",
                "type": "logical_block",
                "section_id": section["id"],
                "bounds": section["bounds"],
                "band_count": section["band_count"],
                "label_zone_count": len(section.get("label_zone_ids", []) or []),
                "field_zone_count": len(section.get("field_zone_ids", []) or []),
                "confidence": section["confidence"],
            }
            for index, section in enumerate(sections)
        ]
        diagnostics["section_count"] = len(sections)
        diagnostics["region_segment_count"] = len(refined["region_segments"])
    else:
        refined.setdefault("sections", [])
        refined.setdefault("region_segments", [])
        diagnostics["section_count"] = 0
        diagnostics["region_segment_count"] = 0

    row_continuity = _build_row_continuity(refined)
    refined["row_continuity"] = row_continuity
    diagnostics["row_continuity_count"] = len(row_continuity)

    table_structures = []
    if getattr(config, "table_aware_enabled", True):
        table_structures.extend(_semantic_table_structures(semantic_regions))
        table_structures.extend((field_candidate_diagnostics or {}).get("table_structures", []) or [])
        refined["table_structures"] = table_structures
        existing_ids = {region.get("id") for region in refined.get("excluded_regions", []) or []}
        for table in table_structures:
            excluded_id = f"structural-table-excluded-{table['id']}"
            if excluded_id in existing_ids:
                continue
            refined.setdefault("excluded_regions", []).append(
                {
                    "id": excluded_id,
                    "type": "table_like_region",
                    "bounds": table["bounds"],
                    "confidence": table.get("confidence", 0.0),
                    "reasons": list(table.get("reasons", [])) + ["table_aware_reasoning"],
                }
            )
        diagnostics["table_structure_count"] = len(table_structures)
    else:
        refined.setdefault("table_structures", [])
        diagnostics["table_structure_count"] = 0

    if getattr(config, "ownership_propagation_enabled", True):
        chains = _build_ownership_chains(refined, sections)
        refined["ownership_chains"] = chains
        diagnostics["ownership_chain_count"] = len(chains)
    else:
        refined.setdefault("ownership_chains", [])
        diagnostics["ownership_chain_count"] = 0

    graph = dict(refined.get("graph", {}) or {})
    graph_nodes = list(graph.get("nodes", []) or [])
    graph_edges = list(graph.get("edges", []) or [])
    known_node_ids = {node.get("id") for node in graph_nodes if isinstance(node, dict)}
    for section in refined.get("sections", []) or []:
        if section["id"] not in known_node_ids:
            graph_nodes.append(section)
            known_node_ids.add(section["id"])
        for band_id in section.get("band_ids", []) or []:
            graph_edges.append(
                {
                    "source": band_id,
                    "target": section["id"],
                    "relationship": "section_membership",
                    "score": section.get("confidence", 0.0),
                    "reasons": section.get("reasons", []),
                }
            )
    for chain in refined.get("ownership_chains", []) or []:
        if chain["id"] not in known_node_ids:
            graph_nodes.append(chain)
            known_node_ids.add(chain["id"])
        graph_edges.append(
            {
                "source": chain["label_zone_id"],
                "target": chain["field_zone_id"],
                "relationship": "ownership_chain",
                "score": chain.get("score", 0.0),
                "reasons": chain.get("reasons", []),
            }
        )
    adjacency: dict[str, list[dict[str, Any]]] = {}
    for edge in graph_edges:
        adjacency.setdefault(edge["source"], []).append(edge)
    graph["nodes"] = graph_nodes
    graph["edges"] = graph_edges
    graph["adjacency"] = adjacency
    refined["graph"] = graph

    section_indexes = _section_indexes(sections)
    indexes = dict(refined.get("indexes", {}) or {})
    indexes.update(section_indexes)
    indexes["section_by_id"] = {section["id"]: section for section in sections}
    indexes["ownership_chain_by_label_zone_id"] = {
        chain["label_zone_id"]: chain for chain in refined.get("ownership_chains", []) or []
    }
    indexes["ownership_chain_by_pair"] = {
        f"{chain['label_zone_id']}::{chain['field_zone_id']}": chain
        for chain in refined.get("ownership_chains", []) or []
    }
    refined["indexes"] = indexes

    page_priors = dict(refined.get("page_priors", {}) or {})
    page_priors.update(
        {
            "structural_refinement_enabled": True,
            "section_count": len(refined.get("sections", []) or []),
            "region_segment_count": len(refined.get("region_segments", []) or []),
            "ownership_chain_count": len(refined.get("ownership_chains", []) or []),
            "table_structure_count": len(refined.get("table_structures", []) or []),
            "row_continuity_count": len(refined.get("row_continuity", []) or []),
        }
    )
    refined["page_priors"] = page_priors
    diagnostics["page_priors"] = page_priors
    return refined, diagnostics
