import math
from collections import defaultdict

import numpy as np


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _median(values, default: float) -> float:
    clean = [float(value) for value in values if isinstance(value, (int, float)) and value > 0]
    if not clean:
        return default
    return float(np.median(clean))


def _ocr_bounds(item):
    bbox = item.get("bbox")
    if not bbox:
        return None
    try:
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
    except Exception:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _line_center(line):
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _line_bounds(line):
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _line_length(line):
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _box_center(bounds):
    x1, y1, x2, y2 = bounds
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_size(bounds):
    x1, y1, x2, y2 = bounds
    return max(x2 - x1, 1.0), max(y2 - y1, 1.0)


def _box_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    overlap_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    overlap_y = max(0.0, min(ay2, by2) - max(ay1, by1))
    if overlap_x <= 0 or overlap_y <= 0:
        return 0.0
    return (overlap_x * overlap_y) / max((ax2 - ax1) * (ay2 - ay1), 1.0)


def _group_by_y(items, y_getter, tolerance):
    groups = []
    for item in sorted(items, key=y_getter):
        y = y_getter(item)
        if not groups or abs(groups[-1]["center_y"] - y) > tolerance:
            groups.append({"center_y": y, "items": [item]})
            continue
        group = groups[-1]
        group["items"].append(item)
        group["center_y"] = sum(y_getter(entry) for entry in group["items"]) / len(group["items"])
    return groups


def _page_bounds(ocr_data, field_lines, semantic_regions):
    xs = []
    ys = []
    for item in ocr_data or []:
        bounds = _ocr_bounds(item)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    for line in field_lines or []:
        x1, y1, x2, y2 = _line_bounds(line)
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    for region in semantic_regions or []:
        xs.extend([float(region.get("x", 0)), float(region.get("x", 0)) + float(region.get("width", 0))])
        ys.extend([float(region.get("y", 0)), float(region.get("y", 0)) + float(region.get("height", 0))])

    return {
        "width": max(xs) if xs else 0.0,
        "height": max(ys) if ys else 0.0,
    }


def build_layout_metrics(ocr_data, field_lines, semantic_regions=None):
    text_heights = []
    text_widths = []
    for item in ocr_data or []:
        bounds = _ocr_bounds(item)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        text_heights.append(y2 - y1)
        text_widths.append(x2 - x1)

    line_lengths = [_line_length(line) for line in field_lines or []]
    avg_text_height = _median(text_heights, 18.0)
    avg_text_width = _median(text_widths, 80.0)
    avg_line_length = _median(line_lengths, 220.0)
    page_bounds = _page_bounds(ocr_data, field_lines, semantic_regions)
    page_width = max(page_bounds["width"], 1.0)
    page_height = max(page_bounds["height"], 1.0)
    page_area = page_width * page_height
    ocr_density = len(ocr_data or []) / max(page_area / 100000.0, 1.0)

    return {
        "page_width": round(page_width, 2),
        "page_height": round(page_height, 2),
        "page_area": round(page_area, 2),
        "page_aspect_ratio": round(page_width / page_height, 4),
        "avg_text_height": round(avg_text_height, 2),
        "avg_text_width": round(avg_text_width, 2),
        "avg_line_length": round(avg_line_length, 2),
        "row_tolerance": round(max(18.0, avg_text_height * 1.5, page_height * 0.008), 2),
        "column_tolerance": round(max(28.0, avg_text_width * 0.8), 2),
        "multiline_y_gap": round(max(34.0, avg_text_height * 2.6, page_height * 0.03), 2),
        "indent_tolerance": round(max(42.0, avg_text_height * 3.0, page_width * 0.04), 2),
        "region_gap": round(max(24.0, avg_text_height * 1.9), 2),
        "distance_limit": round(max(420.0, avg_line_length * 3.2, page_width * 0.55), 2),
        "ocr_density": round(ocr_density, 4),
        "ocr_count": len(ocr_data or []),
        "field_line_count": len(field_lines or []),
        "semantic_region_count": len(semantic_regions or []),
    }


def _build_entities(ocr_data, field_lines, semantic_regions):
    ocr_entities = []
    for index, item in enumerate(ocr_data or []):
        bounds = _ocr_bounds(item)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        ocr_entities.append(
            {
                "id": f"ocr-{index}",
                "type": "ocr_block",
                "text": item.get("text", ""),
                "bounds": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "center": [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)],
                "confidence": item.get("confidence"),
                "source_index": index,
            }
        )

    field_entities = []
    for index, line in enumerate(field_lines or []):
        x1, y1, x2, y2 = _line_bounds(line)
        field_entities.append(
            {
                "id": f"field-{index}",
                "type": "field_line",
                "bounds": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "center": [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)],
                "length": round(_line_length(line), 2),
                "field_type": line.get("field_type", "line"),
                "source_index": index,
            }
        )

    region_entities = []
    for index, region in enumerate(semantic_regions or []):
        x = float(region.get("x", 0))
        y = float(region.get("y", 0))
        width = float(region.get("width", 0))
        height = float(region.get("height", 0))
        region_entities.append(
            {
                "id": f"region-{index}",
                "type": "semantic_region",
                "region_type": region.get("type", "region"),
                "bounds": [round(x, 2), round(y, 2), round(x + width, 2), round(y + height, 2)],
                "center": [round(x + width / 2.0, 2), round(y + height / 2.0, 2)],
                "confidence": region.get("confidence", 0.0),
                "reasons": region.get("reasons", []),
                "source_index": index,
            }
        )

    return {
        "ocr": ocr_entities,
        "fields": field_entities,
        "regions": region_entities,
    }


def _build_edges(entities, metrics):
    edges = []
    adjacency: dict[str, list[dict]] = defaultdict(list)

    def add_edge(source_id: str, target_id: str, relationship: str, score: float, reasons: list[str]):
        payload = {
            "source": source_id,
            "target": target_id,
            "relationship": relationship,
            "score": round(_clamp01(score), 4),
            "reasons": reasons,
        }
        edges.append(payload)
        adjacency[source_id].append(payload)

    for label in entities["ocr"]:
        label_bounds = label["bounds"]
        label_center = label["center"]
        for field in entities["fields"]:
            field_bounds = field["bounds"]
            field_center = field["center"]

            horizontal_gap = field_center[0] - label_center[0]
            vertical_gap = field_center[1] - label_center[1]
            same_row = abs(vertical_gap) <= metrics["row_tolerance"]
            same_column = abs(field_center[0] - label_center[0]) <= metrics["column_tolerance"]
            overlap = _box_overlap(label_bounds, field_bounds)
            proximity = math.sqrt(horizontal_gap**2 + vertical_gap**2)
            continuation = field_center[1] > label_center[1] and vertical_gap <= metrics["multiline_y_gap"]

            if same_row:
                add_edge(label["id"], field["id"], "same_row", 0.85 if horizontal_gap >= 0 else 0.55, ["row_alignment"])
            if same_column:
                add_edge(label["id"], field["id"], "same_column", 0.64, ["column_alignment"])
            if overlap > 0:
                add_edge(label["id"], field["id"], "spatial_overlap", overlap, ["box_overlap"])
            if continuation:
                add_edge(label["id"], field["id"], "vertical_continuation", _clamp01(1.0 - (vertical_gap / max(metrics["multiline_y_gap"], 1.0))), ["downward_flow"])
            if proximity <= metrics["distance_limit"]:
                add_edge(label["id"], field["id"], "proximity", _clamp01(1.0 - (proximity / max(metrics["distance_limit"], 1.0))), ["nearby_entities"])

        for region in entities["regions"]:
            region_bounds = region["bounds"]
            if _box_overlap(label_bounds, region_bounds) <= 0:
                continue
            add_edge(label["id"], region["id"], "enclosure", region.get("confidence", 0.0), [region.get("region_type", "region")])

    for region in entities["regions"]:
        region_bounds = region["bounds"]
        for field in entities["fields"]:
            if _box_overlap(field["bounds"], region_bounds) <= 0:
                continue
            add_edge(field["id"], region["id"], "enclosed_in", region.get("confidence", 0.0), [region.get("region_type", "region")])

    return edges, {key: value for key, value in adjacency.items()}


def _group_ocr_entities(ocr_entities, metrics):
    left_column_split = max(metrics["page_width"] * 0.45, metrics["page_width"] * 0.35)
    left_column = [entity for entity in ocr_entities if entity["center"][0] <= left_column_split]
    left_column.sort(key=lambda entity: (entity["center"][1], entity["center"][0]))

    groups = []
    current = []
    for entity in left_column:
        if not current:
            current.append(entity)
            continue
        prev = current[-1]
        vertical_gap = entity["center"][1] - prev["center"][1]
        indentation_delta = abs(entity["center"][0] - prev["center"][0])
        if vertical_gap <= metrics["row_tolerance"] * 1.3 and indentation_delta <= metrics["column_tolerance"] * 1.4:
            current.append(entity)
        else:
            groups.append(current)
            current = [entity]
    if current:
        groups.append(current)

    ownership_groups = []
    for index, group in enumerate(groups):
        bounds = [
            min(entity["bounds"][0] for entity in group),
            min(entity["bounds"][1] for entity in group),
            max(entity["bounds"][2] for entity in group),
            max(entity["bounds"][3] for entity in group),
        ]
        ownership_groups.append(
            {
                "id": f"ownership-{index}",
                "type": "label_group",
                "item_ids": [entity["id"] for entity in group],
                "texts": [entity.get("text", "") for entity in group],
                "bounds": [round(value, 2) for value in bounds],
                "center": [round((bounds[0] + bounds[2]) / 2.0, 2), round((bounds[1] + bounds[3]) / 2.0, 2)],
                "group_type": "multiline_label" if len(group) > 1 else "single_label",
            }
        )

    return ownership_groups


def relationship_features_for(label_item, field_line, layout_structure):
    metrics = (layout_structure or {}).get("metrics", {})
    label_bounds = _ocr_bounds(label_item)
    if label_bounds is None:
        return {
            "same_row": 0.0,
            "same_column": 0.0,
            "vertical_continuation": 0.0,
            "x_overlap_ratio": 0.0,
            "spacing_similarity": 0.0,
            "size_consistency": 0.0,
            "ocr_confidence": float(label_item.get("confidence") or 0.0),
            "ownership_strength": 0.0,
            "region_support": 0.0,
            "alignment_score": 0.0,
        }

    line_bounds = _line_bounds(field_line)
    label_center = _box_center(label_bounds)
    line_center = _line_center(field_line)
    label_width, label_height = _box_size(label_bounds)
    line_width = max(line_bounds[2] - line_bounds[0], 1.0)
    line_height = max(line_bounds[3] - line_bounds[1], 1.0)

    row_tolerance = max(float(metrics.get("row_tolerance", 24.0)), label_height * 1.25)
    multiline_gap = max(float(metrics.get("multiline_y_gap", 50.0)), label_height * 2.4)
    column_tolerance = max(float(metrics.get("column_tolerance", 40.0)), label_width * 0.8)

    vertical_gap = line_center[1] - label_center[1]
    horizontal_gap = line_center[0] - label_center[0]
    same_row = _clamp01(1.0 - (abs(vertical_gap) / max(row_tolerance, 1.0)))
    same_column = _clamp01(1.0 - (abs(horizontal_gap) / max(column_tolerance, 1.0)))
    continuation = _clamp01(1.0 - (max(0.0, vertical_gap) / max(multiline_gap, 1.0))) if vertical_gap >= 0 else 0.0

    span_overlap = max(0.0, min(label_bounds[2], line_bounds[2]) - max(label_bounds[0], line_bounds[0]))
    x_overlap_ratio = _clamp01(span_overlap / max(min(label_width, line_width), 1.0))
    spacing_similarity = _clamp01(1.0 - (abs(vertical_gap) / max(multiline_gap, 1.0)))
    size_consistency = _clamp01(1.0 - (abs(line_width - float(metrics.get("avg_line_length", line_width))) / max(float(metrics.get("avg_line_length", line_width)) * 1.4, 1.0)))
    alignment_score = _clamp01((same_row * 0.35) + (same_column * 0.2) + (continuation * 0.25) + (spacing_similarity * 0.2))

    ownership_strength = _clamp01((same_row * 0.32) + (continuation * 0.44) + (alignment_score * 0.24))
    for group in (layout_structure or {}).get("ownership_groups", []) or []:
        bounds = group.get("bounds")
        if not bounds or len(bounds) != 4:
            continue
        if label_bounds[0] < bounds[2] and label_bounds[2] > bounds[0] and label_bounds[1] >= bounds[1] - row_tolerance:
            ownership_strength = max(ownership_strength, 0.55 if len(group.get("item_ids", [])) > 1 else 0.42)

    region_support = 0.0
    for region in (layout_structure or {}).get("semantic_regions", []) or []:
        if region.get("type") not in {"multiline_text_region", "standard_input", "table_like_region"}:
            continue
        region_bounds = [float(region.get("x", 0)), float(region.get("y", 0)), float(region.get("x", 0)) + float(region.get("width", 0)), float(region.get("y", 0)) + float(region.get("height", 0))]
        if _box_overlap(label_bounds, region_bounds) > 0 and _box_overlap(line_bounds, region_bounds) > 0:
            region_support = max(region_support, float(region.get("confidence", 0.0)))

    return {
        "same_row": round(same_row, 4),
        "same_column": round(same_column, 4),
        "vertical_continuation": round(continuation, 4),
        "x_overlap_ratio": round(x_overlap_ratio, 4),
        "spacing_similarity": round(spacing_similarity, 4),
        "size_consistency": round(size_consistency, 4),
        "ocr_confidence": round(float(label_item.get("confidence") or 0.0), 4),
        "ownership_strength": round(_clamp01(ownership_strength), 4),
        "region_support": round(_clamp01(region_support), 4),
        "alignment_score": round(alignment_score, 4),
    }


def infer_layout_structure(ocr_data, field_lines, semantic_regions):
    metrics = build_layout_metrics(ocr_data, field_lines, semantic_regions)
    text_rows = _group_by_y(ocr_data or [], lambda item: item.get("center", (0, 0))[1], metrics["row_tolerance"])
    field_rows = _group_by_y(field_lines or [], lambda line: _line_center(line)[1], metrics["row_tolerance"])

    field_clusters = []
    for group in field_rows:
        lines = group["items"]
        xs = [_line_center(line)[0] for line in lines]
        ys = [_line_center(line)[1] for line in lines]
        field_clusters.append(
            {
                "center_y": round(group["center_y"], 2),
                "line_count": len(lines),
                "min_x": round(min(xs), 2) if xs else None,
                "max_x": round(max(xs), 2) if xs else None,
                "min_y": round(min(ys), 2) if ys else None,
                "max_y": round(max(ys), 2) if ys else None,
                "avg_line_length": round(_median([_line_length(line) for line in lines], metrics["avg_line_length"]), 2),
                "type": "repeated_row" if len(lines) >= 3 else "field_row",
            }
        )

    entities = _build_entities(ocr_data, field_lines, semantic_regions)
    ownership_groups = _group_ocr_entities(entities["ocr"], metrics)
    edges, adjacency = _build_edges(entities, metrics)

    return {
        "metrics": metrics,
        "text_rows": [
            {
                "center_y": round(group["center_y"], 2),
                "item_count": len(group["items"]),
                "texts": [item.get("text", "") for item in group["items"]],
            }
            for group in text_rows
        ],
        "field_clusters": field_clusters,
        "semantic_regions": semantic_regions or [],
        "entities": entities,
        "ownership_groups": ownership_groups,
        "graph": {
            "nodes": entities["ocr"] + entities["fields"] + entities["regions"] + ownership_groups,
            "edges": edges,
            "adjacency": adjacency,
        },
    }
